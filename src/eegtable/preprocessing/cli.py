"""The ``eegtable preprocess`` command family: a recipe in, reviewed epochs out."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import traceback
import webbrowser
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import dataclass
from functools import partial
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING, Any

from eegtable.runner.progress import CHECK, CROSS, JsonReporter, Reporter, TextReporter

if TYPE_CHECKING:
    from .config import PreprocessingConfig
    from .execution import StepStatus, Workflow

PER_RECORDING = ("step", "next", "inspect", "reset")
RECIPE_ERRORS = (ValueError, TypeError, ModuleNotFoundError)
_COMMAND = re.compile(r"\b(run|reset|review|next) CONFIG")


def register(commands: Any) -> None:
    parser = commands.add_parser(
        "preprocess",
        help="turn raw EEG recordings into epochs, one checkpointed stage at a time",
        description="Preprocess the recordings a YAML recipe selects. Every stage is a checkpoint. "
        "A review gate stops the run until a decision is saved, unless the recipe sets a policy.",
    )
    sub = parser.add_subparsers(dest="preprocess_command", required=True, metavar="command")

    def command(name: str, help: str) -> argparse.ArgumentParser:
        p: argparse.ArgumentParser = sub.add_parser(name, help=help)
        p.add_argument("config", type=Path, help="the preprocessing recipe (YAML)")
        p.set_defaults(handler=_handle)
        return p

    def recording(p: argparse.ArgumentParser) -> None:
        p.add_argument("--recording", metavar="LABEL", help="one recording of a cohort, by label")

    def jobs(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--n-jobs", type=int, default=1, metavar="N", help="parallel jobs for filtering"
        )
        p.add_argument("--overwrite", action="store_true", help="republish a matching export")

    p = command("init", "write a commented recipe to start from")
    p.add_argument(
        "--mode",
        choices=("events", "resting"),
        default="events",
        help="event-locked epochs (default) or fixed-length resting epochs",
    )
    recording(command("check", "validate the recipe, channels and events; writes nothing"))
    command("steps", "list every stage and whether this recipe enables it")
    p = command("status", "where each recording stands, or one recording's stage list")
    recording(p)
    p.add_argument("--verify", action="store_true", help="re-read that recording's checkpoints")
    p.add_argument("--json", action="store_true", help="one JSON object, for a program")
    p = command("step", "run one stage of one recording; its parents must be complete")
    p.add_argument("stage", help="a stage name from 'steps'")
    recording(p)
    jobs(p)
    p = command("next", "run the next pending stage of one recording")
    recording(p)
    jobs(p)
    p = command("run", "run each recording up to --until, stopping at review gates")
    p.add_argument("--until", default="export", metavar="STAGE", help="last stage (default export)")
    recording(p)
    jobs(p)
    p.add_argument("--progress-json", action="store_true", help="one JSON event per line")
    p = command("inspect", "open one recording's checkpoint in the MNE viewer, or its report")
    p.add_argument("stage", help="the checkpoint to open")
    recording(p)
    p.add_argument("--report", action="store_true", help="build and open an HTML report instead")
    p.add_argument("--json", action="store_true", help="describe a review gate as JSON instead")
    p = command(
        "review", "save a gate decision from its filled pending file, the viewer, or a file"
    )
    p.add_argument("target", choices=("raw", "artifact", "epochs"), help="which gate")
    recording(p)
    p.add_argument("--decisions", type=Path, metavar="FILE", help="a filled decision file")
    p.add_argument("--suggested", action="store_true", help="save the detectors' own suggestions")
    p = command("reset", "retire a stage and everything after it; payloads stay on disk")
    p.add_argument("--from", dest="stage", required=True, metavar="STAGE", help="first stage")
    recording(p)


def _render(text: str, config: Path, label: str | None) -> str:
    # Engine messages name commands as "run CONFIG ..."; the user sees one they can paste.
    target = f"{config} --recording {label}" if label else str(config)
    return _COMMAND.sub(lambda match: f"eegtable preprocess {match.group(1)} {target}", text)


class StageReporter(TextReporter):
    # A recording can take minutes; on a terminal, show the stage it just finished.
    def step(self, label: str, step: str, current: int, total: int) -> None:
        if self.stream.isatty():
            print(f"\r      · {step} ({current}/{total})", end="", file=self.stream, flush=True)

    def recording_done(self, label: str, success: bool, message: str) -> None:
        if self.stream.isatty():
            print("\r\x1b[K", end="", file=self.stream)
        super().recording_done(label, success, message)


@dataclass(frozen=True)
class Context:
    args: argparse.Namespace
    recordings: dict[str, PreprocessingConfig]
    selected: dict[str, PreprocessingConfig]

    @property
    def cohort(self) -> bool:
        return len(self.recordings) > 1

    def one(self) -> tuple[str, PreprocessingConfig]:
        return next(iter(self.selected.items()))

    def render(self, text: str, label: str | None = None) -> str:
        return _render(text, self.args.config, label if self.cohort else None)


def _handle(args: argparse.Namespace) -> int:
    try:
        return _execute(args)
    except RECIPE_ERRORS as exc:
        return _fail(args, exc, 2)
    except OSError as exc:
        return _fail(args, exc, 1)


def _fail(args: argparse.Namespace, exc: Exception, code: int) -> int:
    message = _render(str(exc), args.config, getattr(args, "recording", None))
    print(f"eegtable preprocess: {message}", file=sys.stderr)
    return code


def _execute(args: argparse.Namespace) -> int:
    if args.preprocess_command == "init":
        return _init(args)
    # Imported here so registering the command never loads MNE or optional extras.
    import mne  # type: ignore[import-untyped]

    from .config import load_recipe
    from .stages import get_stage

    # A wrong stage name fails once here, not once per recording.
    for name in ("until", "stage"):
        if getattr(args, name, None) is not None:
            get_stage(getattr(args, name))
    recordings = load_recipe(args.config)
    command = args.preprocess_command
    label = getattr(args, "recording", None)
    if label is not None and label not in recordings:
        raise ValueError(
            f"--recording {label}: not in this recipe; choose from {', '.join(recordings)}"
        )
    if label is None and len(recordings) > 1 and command in PER_RECORDING:
        raise ValueError(
            f"{command}: the recipe selects {len(recordings)} recordings; add --recording"
        )
    selected = recordings if label is None else {label: recordings[label]}
    # MNE's per-call narration would bury the one line per recording that matters,
    # except on the JSON path, where a front end has a pane to scroll it in.
    with mne.use_log_level("info" if getattr(args, "progress_json", False) else "warning"):
        return COMMANDS[command](Context(args, recordings, selected))


def _init(args: argparse.Namespace) -> int:
    if args.config.exists():
        raise ValueError(f"{args.config} already exists; choose another path")
    content = (
        resources.files("eegtable.preprocessing")
        .joinpath("template.yaml")
        .read_text(encoding="utf-8")
    )
    if args.mode == "resting":
        content = (
            content[: content.index("epochs:\n")]
            + "epochs:\n  kind: fixed\n  duration: 2.0\n  overlap: 0.0\n"
            "  padding: 0.0\n  baseline: null\n  detrend: null\n"
        )
    with args.config.open("x", encoding="utf-8") as stream:
        stream.write(content)
    print(
        f"Wrote {args.config}. Set input and output, then: eegtable preprocess check {args.config}"
    )
    return 0


def _check(ctx: Context) -> int:
    from .execution import _input_provenance, load_source, open_workflow
    from .pipeline import StageData, execute_numeric

    failed = 0
    for label, config in ctx.selected.items():
        try:
            state = StageData(
                load_source(open_workflow(config)), provenance=_input_provenance(config)
            )
            for stage in ("load", "prepare", "events"):
                state = execute_numeric(stage, state, config.processing)
        except Exception as exc:  # one recording's problem must not hide the others'
            failed += 1
            print(f"{CROSS} {label}: {exc}")
        else:
            print(f"{CHECK} {label}")
    print(
        f"{failed} of {len(ctx.selected)} recordings failed"
        if failed
        else "Recipe, channels, source and event geometry validated; nothing written."
    )
    return 1 if failed else 0


def _steps(ctx: Context) -> int:
    from .stages import STAGES, enabled

    _, config = ctx.one()
    for stage in STAGES:
        state = "enabled" if enabled(stage.name, config.processing, config.workflow) else "disabled"
        print(
            f"{stage.name}: {state} | parents={stage.parents} | config={stage.fields} | "
            f"output={stage.output}"
        )
    return 0


def _status(ctx: Context) -> int:
    from .execution import list_steps, open_workflow
    from .stages import STAGES

    if ctx.args.json:
        return _status_json(ctx)
    if len(ctx.selected) > 1:
        return _status_all(ctx)
    label, config = ctx.one()
    workflow = open_workflow(config)
    statuses = list_steps(workflow)
    if ctx.args.verify:
        _verify(workflow, statuses)
    for definition, status in zip(STAGES, statuses, strict=True):
        print(
            f"{status.stage}: {status.state} | parents={definition.parents} | "
            f"config={definition.fields} | output={definition.output} | {status.reason}"
        )
    _, pending = _progress(statuses)
    if pending is not None:
        print("Next:", ctx.render(pending.next_action, label))
    return 0


def _status_all(ctx: Context) -> int:
    from .execution import list_steps, open_workflow

    rows = {}
    for label, config in ctx.selected.items():
        workflow = open_workflow(config)
        statuses = list_steps(workflow)
        if ctx.args.verify:
            _verify(workflow, statuses)
        rows[label] = _progress(statuses)
    width = max(map(len, rows))
    for label, (text, _) in rows.items():
        print(f"{label:<{width}}  {text}")
    # Reviews first, then stale checkpoints, then plain resumption.
    for state in ("needs-review", "stale", "pending"):
        for label, (_, pending) in rows.items():
            if pending is not None and pending.state == state:
                if state == "pending":
                    print("Next:", ctx.render("run CONFIG"))
                else:
                    print(
                        "Next:",
                        ctx.render(pending.next_action, label if state == "stale" else None),
                    )
                return 0
    return 0


def _status_json(ctx: Context) -> int:
    from .execution import list_steps, open_workflow

    recordings = []
    for label, config in ctx.selected.items():
        workflow = open_workflow(config)
        statuses = list_steps(workflow)
        if ctx.args.verify:
            _verify(workflow, statuses)
        summary, pending = _progress(statuses)
        recordings.append(
            {
                "label": label,
                "summary": summary,
                "stages": [
                    {"stage": s.stage, "state": s.state, "reason": s.reason} for s in statuses
                ],
                "next": None if pending is None else _next_action(pending),
            }
        )
    print(json.dumps({"recordings": recordings}))
    return 0


def _verify(workflow: Workflow, statuses: Sequence[StepStatus]) -> None:
    from .execution import read_checkpoint

    for status in statuses:
        if status.state == "completed":
            read_checkpoint(workflow, status.stage)


def _next_action(pending: StepStatus) -> dict[str, str]:
    if pending.state == "needs-review":
        target = pending.stage.removeprefix("review-")
        return {"kind": "review", "stage": pending.stage, "target": target}
    return {"kind": "reset" if pending.state == "stale" else "run", "stage": pending.stage}


def _progress(statuses: tuple[StepStatus, ...]) -> tuple[str, StepStatus | None]:
    enabled = [status for status in statuses if status.state != "disabled"]
    pending = next((status for status in enabled if status.state != "completed"), None)
    if pending is None:
        return "exported", None
    if pending.state == "needs-review":
        return f"awaiting {pending.stage}", pending
    if pending.state == "stale":
        return f"stale at {pending.stage}", pending
    done = sum(status.state == "completed" for status in enabled)
    return f"{done} of {len(enabled)} stages", pending


def _run(ctx: Context) -> int:
    args = ctx.args
    reporter: Reporter = JsonReporter(sys.stdout) if args.progress_json else StageReporter()
    root = Path(os.path.commonpath([c.output.directory for c in ctx.recordings.values()]))
    started = time.perf_counter()
    reporter.start(list(ctx.selected), root)
    failed: list[str] = []
    waiting: dict[str, str] = {}
    outputs: list[str] = []
    with _narration(root, args.progress_json) as log:
        for position, (label, config) in enumerate(ctx.selected.items(), 1):
            reporter.recording_start(label, position, len(ctx.selected))
            _run_one(ctx, reporter, label, config, failed, waiting, outputs)
    n = len(ctx.selected)
    if failed:
        message = f"{CROSS} {len(failed)} of {n} recordings failed: {', '.join(failed)}"
    elif waiting:
        message = f"{len(waiting)} of {n} recordings await review"
        message += f"\nNext: {ctx.render(next(iter(waiting.values())))}"
    else:
        message = f"{CHECK} {n} recording{'s' if n != 1 else ''} reached {args.until}"
        if args.until == "export":
            message += f'\nNext: eegtable init recipe.toml  and set inputs.root = "{root}"'
    # A quiet run leaves an empty file: worth neither keeping nor announcing.
    if log.stat().st_size:
        message += f"\nLog: {log}"
    else:
        log.unlink()
    reporter.complete(not failed, time.perf_counter() - started, message, outputs)
    return 1 if failed else 3 if waiting else 0


class _Tee:
    """Writes to a stream and to a file, so the log keeps what the pane drops."""

    def __init__(self, stream: Any, handle: Any) -> None:
        self._stream, self._handle = stream, handle

    def write(self, text: str) -> int:
        self._handle.write(text)
        return int(self._stream.write(text))

    def flush(self) -> None:
        self._handle.flush()
        self._stream.flush()

    def __getattr__(self, name: str) -> Any:
        # isatty, fileno, encoding: whatever MNE or a progress bar asks the stream.
        return getattr(self._stream, name)


@contextmanager
def _narration(root: Path, progress_json: bool) -> Iterator[Path]:
    """Keep the whole run's narration in a file beside the outputs.

    A front end scrolls a bounded pane and a terminal scrolls back only so far;
    the file is the run entire. Nothing but events may reach a front end's
    stdout, so on the JSON path the narration is folded into stderr first.
    """
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"preprocess-run-{time.strftime('%Y%m%d-%H%M%S')}.log"
    with path.open("w", encoding="utf-8") as handle:
        narration = sys.stderr if progress_json else sys.stdout
        with redirect_stdout(_Tee(narration, handle)), redirect_stderr(_Tee(sys.stderr, handle)):
            yield path


def _run_one(
    ctx: Context,
    reporter: Reporter,
    label: str,
    config: PreprocessingConfig,
    failed: list[str],
    waiting: dict[str, str],
    outputs: list[str],
) -> None:
    from .execution import open_workflow, run_until

    args = ctx.args
    try:
        outcome = run_until(
            open_workflow(config),
            args.until,
            n_jobs=args.n_jobs,
            overwrite=args.overwrite,
            on_step=partial(_step_event, reporter, label),
        )
    except Exception as exc:  # one recording's failure must not stop the others
        failed.append(label)
        if not isinstance(exc, (*RECIPE_ERRORS, OSError)):
            traceback.print_exc(file=sys.stderr)  # anything else is a bug; keep its trace
        reporter.recording_done(label, False, ctx.render(str(exc) or type(exc).__name__, label))
        return
    message = ctx.render(outcome.next_action, label)
    if outcome.state == "needs-review":
        waiting[label] = outcome.next_action
        message = f"awaiting {outcome.steps[-1].stage} · {message}"
    elif args.until == "export":
        outputs.append(str(config.output.directory / f"{config.output.name}_epo.fif"))
    reporter.recording_done(label, True, message)


def _step_event(reporter: Reporter, label: str, result: Any, current: int, total: int) -> None:
    reporter.step(label, result.stage, current, total)


def _step(ctx: Context) -> int:
    from .execution import open_workflow, run_next, run_step

    label, config = ctx.one()
    workflow, args = open_workflow(config), ctx.args
    if args.preprocess_command == "step":
        result = run_step(workflow, args.stage, n_jobs=args.n_jobs, overwrite=args.overwrite)
    else:
        result = run_next(workflow, n_jobs=args.n_jobs, overwrite=args.overwrite)
    print(result.state, ctx.render(result.next_action, label))
    return 3 if result.state == "needs-review" else 0


def _inspect(ctx: Context) -> int:
    from .execution import open_workflow, read_checkpoint

    _, config = ctx.one()
    workflow = open_workflow(config)
    if ctx.args.json:
        if ctx.args.report:
            raise ValueError("inspect: --json and --report are exclusive")
        from .review import gate_view

        print(json.dumps(gate_view(workflow, ctx.args.stage)))
        return 0
    checkpoint = read_checkpoint(workflow, ctx.args.stage)
    if ctx.args.report:
        from .report import build_checkpoint_report

        # Inspection artifacts live outside the immutable checkpoint inventory.
        path = workflow.workspace / f"{ctx.args.stage}-{checkpoint.artifact_id}-inspection.html"
        if not path.exists():
            build_checkpoint_report(checkpoint.state).save(path, open_browser=False)
        webbrowser.open(path.as_uri())
        return 0
    from .review import open_viewer

    open_viewer(checkpoint.state)
    return 0


def _review(ctx: Context) -> int:
    from .execution import list_steps, open_workflow
    from .review import save_review

    args, stage = ctx.args, f"review-{ctx.args.target}"
    if args.recording is None and ctx.cohort:
        selected = {
            label: config
            for label, config in ctx.selected.items()
            if any(
                s.stage == stage and s.state == "needs-review"
                for s in list_steps(open_workflow(config))
            )
        }
        if not selected:
            raise ValueError(f"review: no recording awaits {stage}")
        if args.decisions is not None and len(selected) > 1:
            raise ValueError(
                f"--decisions: {len(selected)} recordings await {stage}; add --recording"
            )
        follow_up = ctx.render("run CONFIG")
    else:
        selected = ctx.selected
        follow_up = ctx.render("run CONFIG", ctx.one()[0])
    for label, config in selected.items():
        saved = save_review(
            open_workflow(config), args.target, args.decisions, suggested=args.suggested
        )
        print(f"Saved {label}: {saved}")
    print("Next:", follow_up)
    return 0


def _reset(ctx: Context) -> int:
    from .execution import open_workflow, reset_from

    _, config = ctx.one()
    print("Pending:", ", ".join(reset_from(open_workflow(config), ctx.args.stage)))
    return 0


COMMANDS: dict[str, Callable[[Context], int]] = {
    "check": _check,
    "steps": _steps,
    "status": _status,
    "step": _step,
    "next": _step,
    "run": _run,
    "inspect": _inspect,
    "review": _review,
    "reset": _reset,
}
