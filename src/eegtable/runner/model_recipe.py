"""Strict YAML recipes for grouped predictive modeling."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml

from eegtable.model.design import Selection
from eegtable.model.transformers import PreprocessingConfig
from eegtable.quality import QualityPolicy


@dataclass(frozen=True)
class ModelInputs:
    paths: tuple[Path, ...]
    rows: Literal["epochs", "groups"]
    columns: Literal["identical", "union"]
    targets: Path | None


@dataclass(frozen=True)
class ModelAnalysis:
    task: Literal["regression", "classification"]
    target: str
    groups: str
    covariates: tuple[str, ...]


@dataclass(frozen=True)
class EstimatorSettings:
    estimator: str
    grid: dict[str, tuple[object, ...]]


@dataclass(frozen=True)
class ValidationSettings:
    outer: Literal["loso", "group_kfold"]
    outer_splits: int | None
    inner_splits: int
    scoring: str
    seed: int


@dataclass(frozen=True)
class ModelRecipe:
    path: Path
    recipe_hash: str
    inputs: ModelInputs
    analysis: ModelAnalysis
    model: EstimatorSettings
    validation: ValidationSettings
    selection: Selection
    quality: QualityPolicy
    preprocessing: PreprocessingConfig
    output: Path


class _StrictLoader(yaml.SafeLoader):
    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict[Any, Any]:
        self.flatten_mapping(node)
        result = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, str) or key in result:
                raise ValueError(f"YAML keys must be unique strings; found {key!r}.")
            result[key] = self.construct_object(value_node, deep=deep)
        return result


def _mapping(
    value: object, section: str, allowed: set[str], required: set[str] | None = None
) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{section} must be a mapping with string keys.")
    unknown, missing = set(value) - allowed, (set() if required is None else required) - set(value)
    if unknown or missing:
        raise ValueError(
            f"{section}: unknown keys {sorted(unknown)}; missing keys {sorted(missing)}."
        )
    return value


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string.")
    return value


def _strings(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise ValueError(f"{label} must be a list of nonempty strings.")
    if len(set(value)) != len(value):
        raise ValueError(f"{label} must contain unique names.")
    return tuple(value)


def _integer(value: object, label: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer of at least {minimum}.")
    return value


def _fraction(value: object, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (float, int))
        or not np.isfinite(value)
        or not 0 <= value <= 1
    ):
        raise ValueError(f"{label} must be a finite number in [0, 1].")
    return float(value)


def _path(value: object, base: Path, label: str) -> Path:
    path = Path(_text(value, label)).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def _inputs(value: object, base: Path) -> ModelInputs:
    record = _mapping(value, "inputs", {"paths", "rows", "columns", "targets"}, {"paths"})
    names = _strings(record["paths"], "inputs.paths")
    if not names:
        raise ValueError("inputs.paths must contain at least one feature table.")
    paths = tuple(_path(name, base, "inputs.paths") for name in names)
    if len(set(paths)) != len(paths) or any(path.suffix != ".tsv" for path in paths):
        raise ValueError("inputs.paths must resolve to distinct .tsv feature tables.")
    rows, columns = record.get("rows", "epochs"), record.get("columns", "union")
    if rows not in ("epochs", "groups") or columns not in ("identical", "union"):
        raise ValueError("inputs.rows must be epochs/groups and columns must be identical/union.")
    targets = record.get("targets")
    target_path = None if targets is None else _path(targets, base, "inputs.targets")
    if target_path is not None and target_path.suffix != ".tsv":
        raise ValueError("inputs.targets must be a .tsv descriptor table.")
    return ModelInputs(paths, rows, columns, target_path)


def _analysis(value: object) -> ModelAnalysis:
    record = _mapping(
        value, "analysis", {"task", "target", "groups", "covariates"}, {"task", "target"}
    )
    task = record["task"]
    if task not in ("regression", "classification"):
        raise ValueError("analysis.task must be regression/classification.")
    return ModelAnalysis(
        task,
        _text(record["target"], "analysis.target"),
        _text(record.get("groups", "subject_id"), "analysis.groups"),
        _strings(record.get("covariates", []), "analysis.covariates"),
    )


def _model(value: object, task: str) -> EstimatorSettings:
    record = _mapping(value, "model", {"estimator", "grid"}, {"estimator", "grid"})
    choices = {
        "regression": {"ridge", "scaled_ridge", "elasticnet", "random_forest"},
        "classification": {"logistic", "svm", "random_forest"},
    }
    estimator = _text(record["estimator"], "model.estimator")
    if estimator not in choices[task]:
        raise ValueError(f"model.estimator {estimator!r} is unsupported for {task}.")
    raw_grid = record["grid"]
    if not isinstance(raw_grid, dict) or not raw_grid:
        raise ValueError("model.grid must be a nonempty mapping of parameter names to candidates.")
    grid: dict[str, tuple[object, ...]] = {}
    for name, values in raw_grid.items():
        _text(name, "model.grid parameter")
        if not isinstance(values, list) or not values:
            raise ValueError(f"model.grid.{name} must be a nonempty candidate list.")
        for value in values:
            if not isinstance(value, (str, int, float, bool, type(None))):
                raise ValueError("grid candidates must be scalar YAML values.")
            if isinstance(value, float) and not np.isfinite(value):
                raise ValueError("grid candidates must be finite.")
        grid[name] = tuple(values)
    return EstimatorSettings(estimator, grid)


def _validation(value: object, task: str) -> ValidationSettings:
    record = _mapping(
        value, "validation", {"outer", "outer_splits", "inner_splits", "scoring", "seed"}
    )
    outer = record.get("outer", "loso")
    if outer not in ("loso", "group_kfold"):
        raise ValueError("validation.outer must be loso/group_kfold.")
    outer_splits = record.get("outer_splits")
    if outer == "loso" and outer_splits is not None:
        raise ValueError("outer_splits is only meaningful for group_kfold.")
    if outer == "group_kfold":
        outer_splits = _integer(outer_splits, "validation.outer_splits", 2)
    scoring = record.get(
        "scoring", "neg_mean_squared_error" if task == "regression" else "balanced_accuracy"
    )
    scorers = {
        "regression": {"neg_mean_squared_error", "neg_mean_absolute_error", "r2", "subject_r"},
        "classification": {"balanced_accuracy", "accuracy", "roc_auc", "average_precision"},
    }
    if scoring not in scorers[task]:
        raise ValueError(f"validation.scoring {scoring!r} is unsupported for {task}.")
    return ValidationSettings(
        outer,
        outer_splits,
        _integer(record.get("inner_splits", 3), "inner_splits", 2),
        scoring,
        _integer(record.get("seed", 42), "seed", 0),
    )


def _selection(value: object) -> Selection:
    names = {field.name for field in fields(Selection)}
    record = _mapping(value, "selection", names)
    parameters: dict[str, Any] = {
        name: _strings(items, f"selection.{name}")
        for name, items in record.items()
        if name != "exclude"
    }
    if "exclude" in record:
        parameters["exclude"] = _selection(record["exclude"])
    return Selection(**parameters)


def _preprocessing(value: object, seed: int) -> PreprocessingConfig:
    record = _mapping(
        value,
        "preprocessing",
        {
            "max_feature_missingness",
            "max_subject_missingness",
            "feature_selection_percentile",
            "pca_components",
        },
    )
    percentile = record.get("feature_selection_percentile")
    if percentile is not None and (
        isinstance(percentile, bool)
        or not isinstance(percentile, (int, float))
        or not np.isfinite(percentile)
        or not 0 < percentile <= 100
    ):
        raise ValueError("feature_selection_percentile must be finite and in (0, 100].")
    components = record.get("pca_components")
    if components is not None and (
        isinstance(components, bool)
        or not isinstance(components, (int, float))
        or not np.isfinite(components)
        or not (
            isinstance(components, int)
            and components >= 1
            or isinstance(components, float)
            and 0 < components < 1
        )
    ):
        raise ValueError(
            "pca_components must be a positive integer or variance fraction in (0, 1)."
        )
    return PreprocessingConfig(
        max_feature_missingness=_fraction(
            record.get("max_feature_missingness", 0.2), "max_feature_missingness"
        ),
        max_subject_missingness=_fraction(
            record.get("max_subject_missingness", 0.5), "max_subject_missingness"
        ),
        feature_selection_percentile=percentile,
        pca_enabled=components is not None,
        pca_n_components=components,
        pca_svd_solver="full",
        pca_random_state=seed,
    )


def load_model_recipe(path: str | Path) -> ModelRecipe:
    """Read schema version 1; reject unknown/duplicate keys and invalid values."""
    source = Path(path).resolve()
    content = source.read_bytes()
    try:
        document = yaml.load(content, Loader=_StrictLoader)
    except yaml.YAMLError as error:
        raise ValueError(f"invalid model recipe YAML: {error}") from error
    record = _mapping(
        document,
        "model recipe",
        {
            "version",
            "inputs",
            "analysis",
            "model",
            "validation",
            "selection",
            "quality",
            "preprocessing",
            "output",
        },
        {"version", "inputs", "analysis", "model", "output"},
    )
    if _integer(record["version"], "model recipe version", 1) != 1:
        raise ValueError("model recipe version must be 1.")
    inputs = _inputs(record["inputs"], source.parent)
    analysis = _analysis(record["analysis"])
    if inputs.rows == "groups" and analysis.covariates:
        raise ValueError("group-row models do not support covariate columns.")
    validation = _validation(record.get("validation", {}), analysis.task)
    quality_record = _mapping(
        record.get("quality", {}), "quality", {"min_coverage", "rejected_flags"}
    )
    return ModelRecipe(
        source,
        hashlib.sha256(content).hexdigest(),
        inputs,
        analysis,
        _model(record["model"], analysis.task),
        validation,
        _selection(record.get("selection", {})),
        QualityPolicy(
            _fraction(quality_record.get("min_coverage", 0), "quality.min_coverage"),
            _strings(quality_record.get("rejected_flags", []), "quality.rejected_flags"),
        ),
        _preprocessing(record.get("preprocessing", {}), validation.seed),
        _path(record["output"], source.parent, "output"),
    )
