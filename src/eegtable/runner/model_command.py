"""Handlers for ``eegtable model init/check/run``."""

from __future__ import annotations

import argparse
import json
import sys
from importlib import resources, util
from pathlib import Path

from eegtable.runner.model_cli import register as register


def _require_dependencies() -> None:
    for dependency in ("yaml", "sklearn", "filelock"):
        if util.find_spec(dependency) is None:
            raise ModuleNotFoundError(
                f"Model commands require {dependency}. "
                "Install it with: pip install 'eegtable[model]'",
                name=dependency,
            )


# The headline scores of each task; every metric is in the bundle's JSON files.
_HEADLINE = ("pearson_r", "r2", "mean_absolute_error", "balanced_accuracy", "auc")


def _score_lines(path: Path) -> list[str]:
    summaries = json.loads(path.read_text(encoding="utf-8"))
    lines = ["Held-out scores, pooled over folds:"]
    for name, summary in summaries.items():
        overall = summary["overall"]
        shown = [
            f"{key} {overall[key]:.3f}" for key in _HEADLINE if isinstance(overall.get(key), float)
        ]
        lines.append(f"  {name:<12} {', '.join(shown)}")
    return lines


def handle(args: argparse.Namespace) -> int:
    """Dispatch a model command, reporting expected setup errors as exit status 2."""
    try:
        if args.model_command == "init":
            template = (
                resources.files("eegtable.runner")
                .joinpath("model_template.yaml")
                .read_text(encoding="utf-8")
            )
            with args.recipe.open("x", encoding="utf-8") as stream:
                stream.write(template)
            print(
                f"Wrote {args.recipe}. Set inputs, target and output, then run: "
                f"eegtable model check {args.recipe}"
            )
            return 0
        _require_dependencies()
        from eegtable.runner.model_recipe import load_model_recipe
        from eegtable.runner.model_run import check_model, run_model

        recipe = load_model_recipe(args.recipe)
        if args.model_command == "check":
            report = check_model(recipe)
            print(
                f"Validated {report.n_rows} samples, {report.n_features} features, "
                f"{report.n_groups} groups and {report.n_folds} outer folds. "
                f"Output: {report.output}"
            )
        elif args.model_command == "run":
            result = run_model(recipe)
            print(
                f"Saved {result.n_rows} held-out predictions from {result.n_folds} "
                f"outer folds to {result.output}"
            )
            for line in _score_lines(result.output / "benchmark_metrics.json"):
                print(line)
        else:
            raise ValueError(f"unknown model command {args.model_command!r}.")
    except (ValueError, OSError, ModuleNotFoundError) as error:
        print(f"eegtable model: {error}", file=sys.stderr)
        return 2
    return 0
