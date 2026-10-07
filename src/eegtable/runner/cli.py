"""The ``eegtable`` command: start a recipe, check it, run it, see where it stands.

Exit status is 0 when everything succeeded, 1 when a run finished but some
recordings failed (or ``check``'s trial recording failed, or it found recordings
that lack channels the recipe names), and 2 when nothing could start: an invalid
recipe, missing inputs, or earlier results in the way. ``status`` exits 0 whatever
state the recordings are in.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
from collections.abc import Sequence
from importlib import resources
from pathlib import Path

from eegtable.runner.batch import (
    STATES,
    CheckReport,
    RecordingStatus,
    RunError,
    TrialError,
    check,
    run,
    status,
)
from eegtable.runner.progress import (
    CHECK,
    CROSS,
    JsonReporter,
    TextReporter,
    human_duration,
    shell_command,
)
from eegtable.runner.recipe import RecipeError, load_recipe

_LABEL_WIDTH = 15
_TEMPLATES = {
    "basic": "template.toml",
    "task": "template_task.toml",
    "resting": "template_resting.toml",
}
_SLOWEST = 5


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line and return its exit status."""
    for stream in (sys.stdout, sys.stderr):
        # Windows gives redirected output its ANSI code page, often cp1252, which cannot
        # encode the arrows and check marks in reports; degrade them instead of failing.
        if isinstance(stream, io.TextIOWrapper):
            stream.reconfigure(errors="replace")
    args = _parser().parse_args(argv)
    status: int = args.handler(args)
    return status


def _parser() -> argparse.ArgumentParser:
    from eegtable import __version__

    parser = argparse.ArgumentParser(
        prog="eegtable",
        description="Preprocess raw EEG into epochs, and compute features from epochs, "
        "as recipes describe.",
    )
    parser.add_argument("--version", action="version", version=f"eegtable {__version__}")
    commands = parser.add_subparsers(dest="command", required=True, metavar="command")

    run_parser = commands.add_parser(
        "run", help="compute features for every recording a recipe selects"
    )
    run_parser.add_argument("recipe", type=Path, help="the recipe file")
    run_parser.add_argument(
        "--overwrite", action="store_true", help="replace results from an earlier run"
    )
    run_parser.add_argument(
        "--resume",
        action="store_true",
        help="compute only the recordings whose results are not up to date",
    )
    _add_n_jobs(run_parser)
    _add_workers(run_parser, "recordings computed at once, each in its own process (default 1)")
    run_parser.add_argument(
        "--progress-json",
        action="store_true",
        help="report progress as one JSON event per line, for a front end",
    )
    run_parser.set_defaults(handler=_run)

    check_parser = commands.add_parser(
        "check", help="validate a recipe and try it on the first recording, writing nothing"
    )
    check_parser.add_argument("recipe", type=Path, help="the recipe file")
    _add_n_jobs(check_parser)
    check_parser.add_argument(
        "--quick",
        action="store_true",
        help="compute only the first few epochs, and scale the timing from them",
    )
    _add_workers(check_parser, "the --workers the projected run time assumes")
    check_parser.set_defaults(handler=_check)

    status_parser = commands.add_parser(
        "status", help="say which recordings are done, missing, failed, stale or partial"
    )
    status_parser.add_argument("recipe", type=Path, help="the recipe file")
    status_parser.add_argument(
        "--json", action="store_true", help="one JSON object, for a front end"
    )
    status_parser.set_defaults(handler=_status)

    init_parser = commands.add_parser("init", help="write a commented recipe to start from")
    init_parser.add_argument(
        "path", type=Path, nargs="?", default=Path("recipe.toml"), help="default: recipe.toml"
    )
    init_parser.add_argument(
        "--template",
        choices=list(_TEMPLATES),
        default="basic",
        help="basic: a few spectral measures; task: every family, for event-related "
        "epochs; resting: every family that needs no event (default basic)",
    )
    init_parser.set_defaults(handler=_init)
    report_parser = commands.add_parser("report", help="write an auditable cohort quality report")
    report_parser.add_argument("recipe", type=Path, help="the recipe whose results to report")
    report_parser.add_argument("output", type=Path, help="the HTML report to write")
    report_parser.add_argument(
        "--rows",
        choices=("epochs", "groups"),
        default="epochs",
        help="per-epoch features or cross-trial group rows (default epochs)",
    )
    report_parser.add_argument(
        "--by",
        nargs="+",
        default=["recording"],
        help="descriptor columns to summarize quality by (default recording)",
    )
    report_parser.add_argument(
        "--min-coverage",
        type=float,
        default=0.0,
        help="mask cells resting on less finite input than this fraction (default 0)",
    )
    report_parser.add_argument(
        "--min-support",
        type=float,
        default=0.0,
        help="mask Morlet cells resting on less of their window than this fraction (default 0)",
    )
    report_parser.add_argument(
        "--reject-flag",
        action="append",
        default=[],
        help="mask cells carrying this flag; repeat for several",
    )
    report_parser.set_defaults(handler=_report)
    from eegtable.preprocessing.cli import register

    register(commands)
    from eegtable.runner.model_cli import register as register_model

    register_model(commands)
    return parser


def _report(args: argparse.Namespace) -> int:
    from eegtable.group import GroupDataset, read_group_dataset
    from eegtable.io import FeatureDataset, read_dataset
    from eegtable.quality import QualityPolicy
    from eegtable.report import recording_quality, write_quality_report

    try:
        recipe = load_recipe(args.recipe)
        entries = status(recipe)
        incomplete = [entry for entry in entries if entry.state != "done"]
        if incomplete:
            raise RunError(
                f"Report requires current results: {incomplete[0].label} "
                f"is {incomplete[0].state} ({incomplete[0].reason})."
            )
        dataset: FeatureDataset | GroupDataset
        if args.rows == "epochs":
            paths = [entry.recording.features_path for entry in entries]
            dataset = read_dataset(paths)
        else:
            paths = [entry.recording.crosstrial_path for entry in entries]
            dataset = read_group_dataset(paths)
        policy = QualityPolicy(
            args.min_coverage, tuple(args.reject_flag), min_support=args.min_support
        )
        write_quality_report(
            dataset.table,
            dataset.targets,
            args.output,
            by=tuple(args.by),
            quality=policy,
            recording_summary=recording_quality(paths),
        )
    except (RecipeError, RunError, OSError, ValueError) as exc:
        return _fail(exc, None)
    print(f"Wrote {args.output}")
    return 0


def _add_n_jobs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        metavar="N",
        help="parallel jobs for MNE filtering and spectral estimation (default 1)",
    )


def _add_workers(parser: argparse.ArgumentParser, help: str) -> None:
    parser.add_argument("--workers", type=_positive, default=None, metavar="N", help=help)


def _positive(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {value}")
    return value


def _run(args: argparse.Namespace) -> int:
    reporter = JsonReporter() if args.progress_json else None
    try:
        recipe = load_recipe(args.recipe)
        result = run(
            recipe,
            overwrite=args.overwrite,
            resume=args.resume,
            n_jobs=args.n_jobs,
            workers=args.workers or 1,
            reporter=reporter if reporter is not None else TextReporter(),
        )
    except (RecipeError, RunError, OSError) as exc:
        return _fail(exc, reporter)
    return 0 if result.ok else 1


def _check(args: argparse.Namespace) -> int:
    try:
        recipe = load_recipe(args.recipe)
        report = check(recipe, n_jobs=args.n_jobs, quick=args.quick)
    except (RecipeError, RunError, OSError) as exc:
        return _fail(exc, None)
    except TrialError as exc:
        print(f"eegtable: the trial recording failed: {exc}", file=sys.stderr)
        return 1
    workers = args.workers or min(len(report.recordings), os.cpu_count() or 1)
    for line in _check_lines(
        recipe.path, len(recipe.features), recipe.output.root, report, workers
    ):
        print(line)
    return 1 if report.missing_channels else 0


def _status(args: argparse.Namespace) -> int:
    try:
        recipe = load_recipe(args.recipe)
        entries = status(recipe)
    except (RecipeError, RunError, OSError) as exc:
        return _fail(exc, None)
    counts: dict[str, int] = {s: sum(entry.state == s for entry in entries) for s in STATES}
    after = _next_run(str(args.recipe), counts)
    if args.json:
        report = {
            "recipe": str(args.recipe),
            "output_root": str(recipe.output.root),
            "counts": counts,
            "recordings": [_status_record(entry) for entry in entries],
            "next": after,
        }
        print(json.dumps(report))
        return 0
    width = max(len(entry.label) for entry in entries)
    print(f"eegtable status · {args.recipe} → {recipe.output.root}")
    for entry in entries:
        reason = "" if entry.state == "done" else f"  {entry.reason}"
        print(f"  {entry.label:<{width}}  {entry.state:<7}{reason}".rstrip())
    tally = " · ".join(f"{n} {state}" for state, n in counts.items() if n)
    print(f"{len(entries)} recording{'s' if len(entries) != 1 else ''}: {tally}")
    if after is None:
        print(f"{CHECK} Every recording is up to date.")
    else:
        print("Next: " + shell_command(["eegtable", *after]))
    return 0


def _next_run(recipe: str, counts: dict[str, int]) -> list[str] | None:
    if counts["done"] == sum(counts.values()):
        return None
    command = ["run", recipe, "--resume"]
    if counts["stale"] or counts["partial"]:
        command.append("--overwrite")
    return command


def _status_record(entry: RecordingStatus) -> dict[str, object]:
    return {
        "label": entry.label,
        "input": str(entry.recording.source),
        "state": entry.state,
        "reason": entry.reason,
        "outputs": [str(path) for path in entry.outputs],
    }


def _init(args: argparse.Namespace) -> int:
    path: Path = args.path
    template = resources.files("eegtable.runner").joinpath(_TEMPLATES[args.template])
    try:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(template.read_text(encoding="utf-8"))
    except FileExistsError:
        print(f"eegtable: error: {path} already exists; choose another path.", file=sys.stderr)
        return 2
    except OSError as exc:
        return _fail(exc, None)
    print(
        f"Wrote {path}. Set inputs.root and output.root, then run: "
        f"{shell_command(['eegtable', 'check', str(path)])}"
    )
    return 0


def _fail(exc: Exception, reporter: JsonReporter | None) -> int:
    code = "recipe" if isinstance(exc, RecipeError) else "setup"
    if reporter is not None:
        reporter.error(code, str(exc))
    print(f"eegtable: error: {exc}", file=sys.stderr)
    return 2


def _timing_rows(report: CheckReport, workers: int) -> list[tuple[str, str]]:
    trial = report.trial
    assert trial is not None
    # A quick check's computing is scaled up as if it grew with the epoch count, which
    # holds for per-epoch measures and roughly for the rest; reading the file is not.
    scale = trial.epochs_total / trial.n_epochs if trial.n_epochs else 1.0
    seconds = trial.read_seconds + (trial.seconds - trial.read_seconds) * scale
    scaled = (
        f" (scaled from {trial.n_epochs} of {trial.epochs_total} epochs)" if scale != 1.0 else ""
    )
    timings = sorted(trial.features.timings, key=lambda t: t.seconds, reverse=True)
    measured = sum(t.seconds for t in timings) or 1.0
    slowest = " · ".join(
        f"{t.measure} {human_duration(t.seconds * scale)} ({t.seconds / measured:.0%})"
        for t in timings[:_SLOWEST]
    )
    n = len(report.recordings)
    whole = seconds * n
    projected = f"{n} recording{'s' if n != 1 else ''} ≈ {human_duration(whole)} one at a time"
    if workers > 1:
        # Assumes a full core per worker; efficiency cores or shared memory bandwidth
        # make the real figure longer.
        batches = -(-n // workers)
        projected += f", ≈ {human_duration(seconds * batches)} with --workers {workers}"
    return [
        ("Time", f"{human_duration(seconds)} for this recording{scaled}"),
        ("Slowest", slowest),
        ("Projected", f"{projected}, if the others are like it"),
    ]


def _check_lines(
    recipe_path: Path, n_entries: int, output_root: Path, report: CheckReport, workers: int
) -> list[str]:
    def row(label: str, value: str) -> str:
        return f"  {label:<{_LABEL_WIDTH}}{value}"

    n = len(report.recordings)
    existing = (
        f"{len(report.existing)} result files already there; "
        f"{shell_command(['eegtable', 'status', str(recipe_path)])} shows which are current"
        if report.existing
        else "no earlier results"
    )
    lines = [
        f"eegtable check · {recipe_path}",
        row("Recipe", f"valid · {n_entries} feature entries"),
        row("Inputs", f"{n} recording{'s' if n != 1 else ''}"),
        row("Output", f"{output_root} · {existing}"),
    ]
    if report.trial is not None:
        trial = report.trial
        computed = (
            f"first {trial.n_epochs} of {trial.epochs_total} epochs"
            if trial.n_epochs < trial.epochs_total
            else f"{trial.n_epochs} epochs"
        )
        lines.append(
            row("Trial", f"{trial.recording.label} · {computed} · {len(trial.channels)} channels")
        )
        if trial.features.epochs is not None:
            table = trial.features.epochs
            lines.append(row("Per epoch", f"{len(table.meta)} features × {table.n_rows} epochs"))
        if trial.features.crosstrial is not None:
            table = trial.features.crosstrial
            groups = ", ".join(table.row_labels or ())
            lines.append(row("Across trials", f"{len(table.meta)} features × groups {groups}"))
        lines.extend(row(label, value) for label, value in _timing_rows(report, workers))
    if report.missing_channels:
        n_failing = len(report.missing_channels)
        counted = "1 recording lacks" if n_failing == 1 else f"{n_failing} recordings lack"
        them = "it" if n_failing == 1 else "them"
        lines.append(f"{CROSS} {counted} channels the recipe names, so a run would fail {them}:")
        for label, problems in report.missing_channels.items():
            lines.extend(f"  {label}: {problem}" for problem in problems)
        return lines
    lines.extend(f"! {warning}" for warning in report.warnings)
    lines.append(f"{CHECK} Ready: {shell_command(['eegtable', 'run', str(recipe_path)])}")
    return lines
