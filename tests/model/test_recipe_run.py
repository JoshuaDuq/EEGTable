import argparse
import hashlib
import json
from dataclasses import replace
from importlib.util import find_spec

import numpy as np
import pandas as pd
import pytest
import yaml

from eegtable.io import write_table
from eegtable.table import ComputationSpec, FeatureMeta, FeatureTable


def _api():
    assert find_spec("eegtable.runner.model_recipe") is not None, "model recipe is not implemented"
    assert find_spec("eegtable.runner.model_run") is not None, "model run is not implemented"
    from eegtable.runner.model_recipe import load_model_recipe
    from eegtable.runner.model_run import check_model, run_model

    return load_model_recipe, check_model, run_model


def _feature_meta(name):
    return FeatureMeta(
        measure=name,
        band=None,
        space="C3",
        space_kind="channel",
        window="all",
        normalization="raw",
        unit="a.u.",
        source="synthetic",
        window_bounds=(0.0, 1.0),
        computation=ComputationSpec.create("synthetic"),
    )


def _inputs(tmp_path, task="regression", rows="epochs"):
    paths = []
    rng = np.random.default_rng(31)
    for subject in range(4):
        n_rows = 12
        target = (
            np.tile([0, 1], n_rows // 2) if task == "classification" else rng.normal(size=n_rows)
        )
        values = np.column_stack(
            [target + rng.normal(scale=0.1, size=n_rows), rng.normal(size=n_rows)]
        )
        coverage = np.ones(values.shape)
        coverage[0, 1] = 0.4
        table = FeatureTable(
            values=values,
            coverage=coverage,
            meta=(_feature_meta("signal"), _feature_meta("noise")),
            row_ids=(
                tuple((f"sub-{subject}", row, "event") for row in range(n_rows))
                if rows == "epochs"
                else None
            ),
            row_labels=(
                tuple(f"condition-{row}" for row in range(n_rows)) if rows == "groups" else None
            ),
            flags={"bad": np.zeros(values.shape, dtype=bool)},
        )
        descriptors = pd.DataFrame(
            {
                "recording": [f"sub-{subject}"] * n_rows,
                "subject_id": [f"s{subject}"] * n_rows,
                "outcome": target,
            }
        )
        if rows == "epochs":
            descriptors["event"] = "event"
        else:
            descriptors["n_trials"] = 20
        path = tmp_path / f"sub-{subject}_features.tsv"
        write_table(table, path, rows=descriptors)
        paths.append(path.name)
    return paths


def _recipe(tmp_path, task="regression", rows="epochs"):
    paths = _inputs(tmp_path, task, rows)
    record = {
        "version": 1,
        "inputs": {"paths": paths, "rows": rows},
        "analysis": {"task": task, "target": "outcome", "groups": "subject_id"},
        "model": {
            "estimator": "ridge" if task == "regression" else "logistic",
            "grid": {"regressor__alpha" if task == "regression" else "lr__C": [0.1, 1.0]},
        },
        "validation": {"outer": "loso", "inner_splits": 2, "seed": 13},
        "quality": {"min_coverage": 0.8, "rejected_flags": ["bad"]},
        "output": "results",
    }
    path = tmp_path / "model.yaml"
    path.write_text(yaml.safe_dump(record), encoding="utf-8")
    return path, record


def test_model_recipe_resolves_paths_and_quality(tmp_path):
    path, _ = _recipe(tmp_path)
    load, _, _ = _api()
    recipe = load(path)
    assert recipe.inputs.paths[0] == tmp_path / "sub-0_features.tsv"
    assert recipe.output == tmp_path / "results"
    assert recipe.quality.min_coverage == 0.8
    assert recipe.validation.scoring == "neg_mean_squared_error"


def test_model_recipe_reads_a_minimum_support(tmp_path):
    path, record = _recipe(tmp_path)
    record["quality"]["min_support"] = 0.5
    path.write_text(yaml.safe_dump(record), encoding="utf-8")
    load, _, _ = _api()
    assert load(path).quality.min_support == 0.5


@pytest.mark.parametrize("rows", ["epochs", "groups"])
def test_model_bundle_retains_selected_support_and_excludes_unsupported_values(tmp_path, rows):
    from eegtable.group import read_group_dataset
    from eegtable.io import read_dataset, read_table

    path, record = _recipe(tmp_path, rows=rows)
    reader = read_dataset if rows == "epochs" else read_group_dataset
    sources = [tmp_path / name for name in record["inputs"]["paths"]]
    for source in sources:
        table = read_table(source)
        descriptors = reader([source]).targets.drop(
            columns="epoch" if rows == "epochs" else "group"
        )
        support = np.full(table.values.shape, 0.9)
        support[0, 0] = 0.25
        write_table(replace(table, support=support), source, rows=descriptors)
    record["selection"] = {"measure": ["signal"]}
    record["quality"]["min_support"] = 0.5
    path.write_text(yaml.safe_dump(record), encoding="utf-8")

    load, _, run = _api()
    result = run(load(path))
    restored = read_table(result.output / "design_features.tsv")

    assert restored.support is not None
    expected_support = np.full((48, 1), 0.9)
    expected_support[::12] = 0.25
    np.testing.assert_array_equal(restored.support, expected_support)
    np.testing.assert_array_equal(np.flatnonzero(np.isnan(restored.values[:, 0])), [0, 12, 24, 36])
    ledger = pd.read_csv(result.output / "quality_ledger.tsv", sep="\t")
    assert ledger.reason.tolist() == ["low_support"] * 4
    manifest = json.loads((result.output / "manifest.json").read_text(encoding="utf-8"))
    for source in sources:
        support_path = source.with_name(source.stem + "_support.tsv")
        assert str(support_path) in manifest["provenance"]["inputs"]


def test_model_run_refuses_support_modified_during_fitting(tmp_path, monkeypatch):
    from eegtable.io import read_dataset
    from eegtable.runner import model_run

    path, record = _recipe(tmp_path)
    source = tmp_path / record["inputs"]["paths"][0]
    dataset = read_dataset([source])
    table = replace(dataset.table, support=np.full(dataset.table.values.shape, 0.9))
    write_table(table, source, rows=dataset.targets.drop(columns="epoch"))
    predict = model_run._predictions

    def mutate_after_fitting(recipe, prepared):
        predictions = predict(recipe, prepared)
        source.with_name(source.stem + "_support.tsv").write_text("changed", encoding="utf-8")
        return predictions

    monkeypatch.setattr(model_run, "_predictions", mutate_after_fitting)
    load, _, run = _api()
    with pytest.raises(ValueError, match="checksum mismatch"):
        run(load(path))
    assert not (tmp_path / "results").exists()


@pytest.mark.parametrize(
    "change", ["unknown", "duplicate", "bool_seed", "invalid_estimator", "bad_grid"]
)
def test_model_recipe_refuses_invalid_settings(tmp_path, change):
    path, record = _recipe(tmp_path)
    if change == "unknown":
        record["fallback"] = True
    if change == "bool_seed":
        record["validation"]["seed"] = True
    if change == "invalid_estimator":
        record["model"]["estimator"] = "svm"
    if change == "bad_grid":
        record["model"]["grid"] = {"regressor__alpha": []}
    path.write_text(yaml.safe_dump(record), encoding="utf-8")
    if change == "duplicate":
        path.write_text(path.read_text(encoding="utf-8") + "version: 1\n", encoding="utf-8")
    load, _, _ = _api()
    with pytest.raises(ValueError):
        load(path)


def test_model_check_validates_nested_group_splits_without_writing(tmp_path):
    path, _ = _recipe(tmp_path)
    load, check, _ = _api()
    report = check(load(path))
    assert report.n_rows == 48
    assert report.n_groups == 4
    assert report.n_features == 2
    assert report.n_folds == 4
    assert not (tmp_path / "results").exists()


@pytest.mark.parametrize("rows", ["epochs", "groups"])
def test_model_check_keeps_numeric_looking_groups_distinct(tmp_path, rows):
    from eegtable.io import read_table

    path, settings = _recipe(tmp_path, rows=rows)
    row_key = "epoch" if rows == "epochs" else "group"
    for name, label in zip(settings["inputs"]["paths"], ["01", "1", "02", "2"], strict=True):
        source = tmp_path / name
        table = read_table(source)
        sidecar = json.loads(source.with_suffix(".json").read_text(encoding="utf-8"))
        descriptors = pd.read_csv(source, sep="\t", usecols=sidecar["row_columns"])
        descriptors["subject_id"] = label
        write_table(table, source, rows=descriptors.drop(columns=row_key))
    load, check, _ = _api()

    assert check(load(path)).n_groups == 4


@pytest.mark.parametrize(
    "task,estimator,grid",
    [
        ("regression", "ridge", {"regressor__alpha": [0.1, 1.0]}),
        ("classification", "logistic", {"lr__C": [0.1, 1.0]}),
        (
            "regression",
            "hist_gradient_boosting",
            {"hgb__max_iter": [5, 10], "hgb__min_samples_leaf": [2]},
        ),
        (
            "classification",
            "hist_gradient_boosting",
            {"hgb__max_iter": [5, 10], "hgb__min_samples_leaf": [2]},
        ),
        ("regression", "svr", {"svr__regressor__C": [0.1, 1.0]}),
        ("classification", "lda", {"lda__shrinkage": ["auto", 0.5]}),
    ],
)
def test_model_run_writes_verifiable_prediction_and_quality_bundle(tmp_path, task, estimator, grid):
    path, record = _recipe(tmp_path, task)
    record["model"] = {"estimator": estimator, "grid": grid}
    path.write_text(yaml.safe_dump(record), encoding="utf-8")
    load, _, run = _api()
    result = run(load(path))
    predictions = pd.read_csv(result.output / "predictions.tsv", sep="\t")
    assert len(predictions) == 48
    assert not predictions.duplicated(["recording", "epoch", "event"]).any()
    assert predictions.y_pred.notna().all()
    manifest = json.loads((result.output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["task"] == task
    assert manifest["provenance"]["implementation_hash"]
    assert manifest["provenance"]["software"]["scikit-learn"]
    for name, expected_hash in manifest["files"].items():
        assert hashlib.sha256((result.output / name).read_bytes()).hexdigest() == expected_hash
    folds = json.loads((result.output / "folds.json").read_text(encoding="utf-8"))
    assert len(folds) == 4
    for fold in folds:
        assert not set(fold["train_groups"]) & set(fold["test_groups"])
        assert fold["best_params"]
        for split in fold["inner_splits"]:
            assert not set(split["train_groups"]) & set(split["test_groups"])
    quality = pd.read_csv(result.output / "quality_ledger.tsv", sep="\t")
    assert len(quality) == 4
    assert (result.output / "design_features_coverage.tsv").is_file()
    assert (result.output / "design_features.json").is_file()
    if task == "classification":
        np.testing.assert_allclose(predictions.probability_0 + predictions.probability_1, 1)
    assert "overall" in json.loads((result.output / "metrics.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("task,estimator", [("regression", "lda"), ("classification", "svr")])
def test_model_recipe_rejects_additional_estimators_for_the_wrong_task(tmp_path, task, estimator):
    path, record = _recipe(tmp_path, task)
    record["model"]["estimator"] = estimator
    path.write_text(yaml.safe_dump(record), encoding="utf-8")
    load, _, _ = _api()
    with pytest.raises(ValueError, match="unsupported"):
        load(path)


@pytest.mark.parametrize("task", ["regression", "classification"])
@pytest.mark.parametrize("early_stopping", [True, "auto"])
def test_model_check_rejects_boosting_internal_validation(tmp_path, task, early_stopping):
    path, record = _recipe(tmp_path, task)
    record["model"] = {
        "estimator": "hist_gradient_boosting",
        "grid": {"hgb__early_stopping": [early_stopping]},
    }
    path.write_text(yaml.safe_dump(record), encoding="utf-8")
    load, check, _ = _api()
    with pytest.raises(ValueError, match="early_stopping=False"):
        check(load(path))
    assert not (tmp_path / "results").exists()


@pytest.mark.parametrize(
    "task,estimator,grid",
    [
        ("regression", "hist_gradient_boosting", {"hgb__max_iter": [0]}),
        ("classification", "hist_gradient_boosting", {"hgb__max_iter": [0]}),
        ("regression", "svr", {"svr__regressor__C": [-1.0]}),
        ("classification", "lda", {"lda__shrinkage": [2.0]}),
    ],
)
def test_additional_estimator_fit_errors_leave_no_completed_bundle(tmp_path, task, estimator, grid):
    path, record = _recipe(tmp_path, task)
    record["model"] = {"estimator": estimator, "grid": grid}
    path.write_text(yaml.safe_dump(record), encoding="utf-8")
    load, _, run = _api()
    with pytest.raises(ValueError, match="Fold"):
        run(load(path))
    assert not (tmp_path / "results").exists()


def test_model_run_handles_group_rows_without_epoch_broadcast(tmp_path):
    path, _ = _recipe(tmp_path, rows="groups")
    load, _, run = _api()
    result = run(load(path))
    frame = pd.read_csv(result.output / "predictions.tsv", sep="\t")
    assert "group" in frame and "epoch" not in frame
    assert len(frame) == 48


def test_model_run_supports_one_group_sample_per_participant(tmp_path):
    paths = []
    rng = np.random.default_rng(31)
    for subject in range(8):
        values = rng.normal(size=(1, 2))
        table = FeatureTable(
            values=values,
            coverage=np.ones((1, 2)),
            meta=(_feature_meta("signal"), _feature_meta("noise")),
            row_labels=("rest",),
        )
        source = tmp_path / f"sub-{subject}_features.tsv"
        write_table(
            table,
            source,
            rows=pd.DataFrame(
                {
                    "recording": [f"sub-{subject}"],
                    "subject_id": [f"s{subject}"],
                    "outcome": [values[0, 0] + 0.2 * values[0, 1]],
                    "n_trials": [20],
                }
            ),
        )
        paths.append(source.name)
    record = {
        "version": 1,
        "inputs": {"paths": paths, "rows": "groups"},
        "analysis": {"task": "regression", "target": "outcome"},
        "model": {"estimator": "ridge", "grid": {"regressor__alpha": [0.1, 1.0]}},
        "validation": {"inner_splits": 2},
        "output": "results",
    }
    path = tmp_path / "model.yaml"
    path.write_text(yaml.safe_dump(record), encoding="utf-8")
    load, check, run = _api()
    recipe = load(path)
    assert check(recipe).n_rows == 8
    result = run(recipe)
    metrics = json.loads((result.output / "metrics.json").read_text(encoding="utf-8"))
    assert np.isfinite(metrics["overall"]["mean_absolute_error"])
    assert len(metrics["per_group"]) == 8
    assert np.isfinite(metrics["group_mean"]["mean_absolute_error"])


def test_model_recipe_svm_reports_auc_without_probabilities(tmp_path):
    path, record = _recipe(tmp_path, task="classification")
    record["model"] = {"estimator": "svm", "grid": {"svm__C": [1.0]}}
    record["validation"]["scoring"] = "roc_auc"
    path.write_text(yaml.safe_dump(record), encoding="utf-8")
    load, _, run = _api()
    result = run(load(path))
    predictions = pd.read_csv(result.output / "predictions.tsv", sep="\t")
    assert "decision_score" in predictions
    assert "probability_1" not in predictions
    metrics = json.loads((result.output / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["group_mean"]["auc"] > 0.9


@pytest.mark.parametrize("task", ["regression", "classification"])
def test_model_benchmark_uses_identical_held_out_rows_and_training_only_dummy(tmp_path, task):
    path, _ = _recipe(tmp_path, task=task)
    load, _, run = _api()
    result = run(load(path))
    predictions = pd.read_csv(result.output / "predictions.tsv", sep="\t")
    baselines = pd.read_csv(result.output / "baseline_predictions.tsv", sep="\t")
    dummy = baselines[baselines.model == "dummy"]
    pd.testing.assert_frame_equal(
        dummy[["row", "fold", "y_true"]].reset_index(drop=True),
        predictions[["row", "fold", "y_true"]],
    )
    for fold, rows in dummy.groupby("fold"):
        training_target = predictions.loc[predictions.fold != fold, "y_true"]
        expected = training_target.mean() if task == "regression" else training_target.mode()[0]
        np.testing.assert_allclose(rows.y_pred, expected)
    folds = json.loads((result.output / "baseline_folds.json").read_text(encoding="utf-8"))
    assert all(not fold["inner_splits"] and not fold["best_params"] for fold in folds["dummy"])
    metrics = json.loads((result.output / "benchmark_metrics.json").read_text(encoding="utf-8"))
    if task == "regression":
        error = np.mean((dummy.y_true - dummy.y_pred) ** 2)
        assert metrics["dummy"]["overall"]["mean_squared_error"] == pytest.approx(error)
        assert all(group["pearson_r"] == "NaN" for group in metrics["dummy"]["per_group"])
    else:
        from sklearn.metrics import roc_auc_score

        assert metrics["dummy"]["overall"]["auc"] == pytest.approx(
            roc_auc_score(dummy.y_true, dummy.probability_1)
        )
    comparison = pd.read_csv(result.output / "benchmarks.tsv", sep="\t")
    assert set(comparison.model) == {"model", "dummy"}
    assert not (result.output / "report.html").exists()
    manifest = json.loads((result.output / "manifest.json").read_text(encoding="utf-8"))
    assert {"benchmarks.tsv", "baseline_predictions.tsv"} <= set(manifest["files"])
    assert "report.html" not in manifest["files"]


@pytest.mark.parametrize("task", ["regression", "classification"])
@pytest.mark.parametrize("constant", [False, True])
def test_model_benchmark_includes_covariate_only_predictions(tmp_path, task, constant):
    from eegtable.io import read_table

    path, record = _recipe(tmp_path, task)
    for name in record["inputs"]["paths"]:
        source = tmp_path / name
        table = read_table(source)
        descriptors = pd.read_csv(source, sep="\t")[["recording", "subject_id", "outcome", "event"]]
        descriptors["stimulus"] = (
            1.0 if constant else descriptors.outcome + np.arange(len(descriptors)) / 100
        )
        write_table(table, source, rows=descriptors)
    record["analysis"]["covariates"] = ["stimulus"]
    path.write_text(yaml.safe_dump(record), encoding="utf-8")
    load, _, run = _api()
    result = run(load(path))
    baselines = pd.read_csv(result.output / "baseline_predictions.tsv", sep="\t")
    covariates = baselines[baselines.model == "covariates"]
    assert len(covariates) == 48
    assert np.isfinite(covariates.y_pred).all()
    assert set(baselines.model) == {"dummy", "covariates"}


def test_model_recipe_accepts_training_scaled_ridge_and_subject_r(tmp_path):
    path, record = _recipe(tmp_path)
    record["model"]["estimator"] = "scaled_ridge"
    record["validation"]["scoring"] = "subject_r"
    path.write_text(yaml.safe_dump(record), encoding="utf-8")
    load, check, run = _api()
    recipe = load(path)
    assert check(recipe).n_folds == 4
    result = run(recipe)
    metrics = json.loads((result.output / "metrics.json").read_text(encoding="utf-8"))
    assert np.isfinite(metrics["group_mean"]["subject_level_r"])
    assert "subject_level_r" not in metrics["overall"]


def test_regression_summary_weights_groups_equally_and_labels_subject_r(tmp_path):
    from eegtable.runner.model_run import _metrics

    path, record = _recipe(tmp_path)
    record["validation"]["scoring"] = "subject_r"
    path.write_text(yaml.safe_dump(record), encoding="utf-8")
    load, _, _ = _api()
    frame = pd.DataFrame(
        {
            "model_group": ["a"] * 3 + ["b"] * 9,
            "y_true": np.r_[np.arange(3), np.arange(9)],
            "y_pred": np.r_[np.arange(3), np.arange(8, -1, -1)],
        }
    )
    metrics = _metrics(load(path), frame)
    assert metrics["group_mean"]["subject_level_r"] == pytest.approx(0.0)
    assert metrics["group_mean"]["mean_squared_error"] == pytest.approx(40 / 3)
    assert metrics["overall"]["mean_squared_error"] == pytest.approx(20.0)
    assert "subject_level_r" not in metrics["overall"]


def test_subject_r_check_refuses_constant_targets_within_a_group(tmp_path):
    from eegtable.io import read_table

    path, record = _recipe(tmp_path)
    source = tmp_path / record["inputs"]["paths"][0]
    table = read_table(source)
    descriptors = pd.read_csv(source, sep="\t")[["recording", "subject_id", "outcome", "event"]]
    descriptors["outcome"] = 1.0
    write_table(table, source, rows=descriptors)
    record["validation"]["scoring"] = "subject_r"
    path.write_text(yaml.safe_dump(record), encoding="utf-8")
    load, check, _ = _api()
    with pytest.raises(ValueError, match="subject_r.*vary"):
        check(load(path))


@pytest.mark.parametrize("n_rows", [2, 3])
def test_subject_r_check_requires_three_varying_targets_per_group(tmp_path, n_rows):
    from eegtable.io import read_table

    path, record = _recipe(tmp_path)
    for name in record["inputs"]["paths"]:
        source = tmp_path / name
        table = read_table(source).take(np.arange(n_rows))
        descriptors = pd.read_csv(source, sep="\t")[
            ["recording", "subject_id", "outcome", "event"]
        ].iloc[:n_rows]
        write_table(table, source, rows=descriptors)
    record["validation"]["scoring"] = "subject_r"
    path.write_text(yaml.safe_dump(record), encoding="utf-8")
    load, check, _ = _api()
    if n_rows == 2:
        with pytest.raises(ValueError, match="subject_r.*three"):
            check(load(path))
    else:
        assert check(load(path)).n_rows == 12


def test_model_run_rejects_existing_results(tmp_path):
    path, _ = _recipe(tmp_path)
    load, _, run = _api()
    (tmp_path / "results").mkdir()
    sentinel = tmp_path / "results" / "sentinel.txt"
    sentinel.write_text("preserve", encoding="utf-8")
    with pytest.raises(FileExistsError):
        run(load(path))
    assert sentinel.read_text(encoding="utf-8") == "preserve"


def test_model_check_rejects_insufficient_nested_training_groups(tmp_path):
    path, record = _recipe(tmp_path)
    record["validation"]["inner_splits"] = 4
    path.write_text(yaml.safe_dump(record), encoding="utf-8")
    load, check, _ = _api()
    with pytest.raises(ValueError, match="training groups"):
        check(load(path))


def test_model_check_rejects_nonbinary_classification_labels(tmp_path):
    path, _ = _recipe(tmp_path, task="classification")
    file = tmp_path / "sub-0_features.tsv"
    from eegtable.io import read_table

    table = read_table(tmp_path / "sub-1_features.tsv")
    table = FeatureTable(
        values=table.values,
        coverage=table.coverage,
        meta=table.meta,
        row_ids=tuple(("sub-0", index, "event") for index in range(12)),
        flags=table.flags,
    )
    descriptors = pd.DataFrame(
        {
            "recording": ["sub-0"] * 12,
            "subject_id": ["s0"] * 12,
            "event": ["event"] * 12,
            "outcome": np.r_[0.5, np.tile([1, 0], 6)[:11]],
        }
    )
    write_table(table, file, rows=descriptors)
    load, check, _ = _api()
    with pytest.raises(ValueError, match="0/1"):
        check(load(path))


def test_model_cli_init_and_check(tmp_path, capsys):
    assert find_spec("eegtable.runner.model_command") is not None, "model CLI is not implemented"
    from eegtable.runner.model_command import register

    parser = argparse.ArgumentParser()
    register(parser.add_subparsers(required=True))
    output = tmp_path / "starter.yaml"
    args = parser.parse_args(["model", "init", str(output)])
    assert args.handler(args) == 0
    assert output.is_file()
    path, _ = _recipe(tmp_path)
    args = parser.parse_args(["model", "check", str(path)])
    assert args.handler(args) == 0
    assert "48" in capsys.readouterr().out


def test_model_recipe_requires_integer_version(tmp_path):
    path, record = _recipe(tmp_path)
    record["version"] = 1.0
    path.write_text(yaml.safe_dump(record), encoding="utf-8")
    load, _, _ = _api()
    with pytest.raises(ValueError, match="version"):
        load(path)


def test_external_targets_preserve_float_precision(tmp_path):
    from eegtable.runner.model_run import _merge_targets

    descriptors = pd.DataFrame(
        {"recording": ["rec", "rec"], "epoch": [0, 1], "event": ["left", "right"]}
    )
    outcome = np.array([0.30000000000000004, 0.12345678901234567])
    source = tmp_path / "targets.tsv"
    descriptors.assign(outcome=outcome, subject_id="01").to_csv(source, sep="\t", index=False)

    targets = _merge_targets(descriptors, source, "epochs", "subject_id")

    np.testing.assert_array_equal(targets.outcome.to_numpy(), outcome)


def test_external_targets_join_by_exact_sample_identity(tmp_path):
    from eegtable.io import read_dataset

    path, record = _recipe(tmp_path)
    dataset = read_dataset([tmp_path / name for name in record["inputs"]["paths"]])
    targets = dataset.targets[["recording", "epoch", "event", "outcome"]].copy()
    targets = targets.rename(columns={"outcome": "external_score"}).iloc[::-1]
    targets.to_csv(tmp_path / "targets.tsv", sep="\t", index=False)
    record["inputs"]["targets"] = "targets.tsv"
    record["analysis"]["target"] = "external_score"
    path.write_text(yaml.safe_dump(record), encoding="utf-8")
    load, _, run = _api()
    result = run(load(path))
    frame = pd.read_csv(result.output / "predictions.tsv", sep="\t")
    np.testing.assert_allclose(frame.y_true, dataset.targets.outcome)


def test_external_targets_refuse_missing_identity(tmp_path):
    from eegtable.io import read_dataset

    path, record = _recipe(tmp_path)
    dataset = read_dataset([tmp_path / name for name in record["inputs"]["paths"]])
    targets = dataset.targets[["recording", "epoch", "event", "outcome"]].iloc[:-1]
    targets.rename(columns={"outcome": "external_score"}).to_csv(
        tmp_path / "targets.tsv", sep="\t", index=False
    )
    record["inputs"]["targets"] = "targets.tsv"
    record["analysis"]["target"] = "external_score"
    path.write_text(yaml.safe_dump(record), encoding="utf-8")
    load, check, _ = _api()
    with pytest.raises(ValueError, match="exactly"):
        check(load(path))


def test_external_targets_keyed_by_another_name_show_both_sides(tmp_path):
    # Keyed by subject, where feature samples carry the epochs file's path, every row fails
    # to match; the error has to show what each side holds for the mistake to be visible.
    from eegtable.io import read_dataset

    path, record = _recipe(tmp_path)
    dataset = read_dataset([tmp_path / name for name in record["inputs"]["paths"]])
    targets = dataset.targets[["recording", "epoch", "event", "outcome"]].copy()
    targets["recording"] = "sub-01"
    targets.rename(columns={"outcome": "external_score"}).drop_duplicates(
        ["recording", "epoch", "event"]
    ).to_csv(tmp_path / "targets.tsv", sep="\t", index=False)
    record["inputs"]["targets"] = "targets.tsv"
    record["analysis"]["target"] = "external_score"
    path.write_text(yaml.safe_dump(record), encoding="utf-8")
    load, check, _ = _api()
    with pytest.raises(ValueError) as error:
        check(load(path))
    message = str(error.value)
    assert "sub-01" in message and dataset.targets["recording"].iloc[0] in message


def test_model_cli_run_reports_its_scores_beside_the_baselines(tmp_path, capsys):
    # The scores were only in metrics.json; a run should say how it did.
    from eegtable.runner.model_command import register

    parser = argparse.ArgumentParser()
    register(parser.add_subparsers(required=True))
    path, _ = _recipe(tmp_path)
    args = parser.parse_args(["model", "run", str(path)])
    assert args.handler(args) == 0
    lines = capsys.readouterr().out.splitlines()
    assert any(line.lstrip().startswith("model") and "r2" in line for line in lines)
    assert any(line.lstrip().startswith("dummy") and "r2" in line for line in lines)


def test_external_group_labels_preserve_distinct_numeric_strings(tmp_path):
    from eegtable.io import read_dataset

    path, record = _recipe(tmp_path)
    dataset = read_dataset([tmp_path / name for name in record["inputs"]["paths"]])
    targets = dataset.targets[["recording", "epoch", "event"]].copy()
    labels = ["001", "01", "1", "2"]
    targets["external_group"] = np.repeat(labels, 12)
    targets.to_csv(tmp_path / "targets.tsv", sep="\t", index=False)
    record["inputs"]["targets"] = "targets.tsv"
    record["analysis"]["groups"] = "external_group"
    path.write_text(yaml.safe_dump(record), encoding="utf-8")
    load, check, _ = _api()
    assert check(load(path)).n_groups == 4


def test_model_run_leaves_no_published_bundle_after_backend_error(tmp_path):
    path, record = _recipe(tmp_path)
    record["model"]["grid"] = {"regressor__alpha": [-1.0]}
    path.write_text(yaml.safe_dump(record), encoding="utf-8")
    load, _, run = _api()
    with pytest.raises(ValueError):
        run(load(path))
    assert not (tmp_path / "results").exists()


def test_model_check_refuses_unknown_pipeline_grid_parameter(tmp_path):
    path, record = _recipe(tmp_path)
    record["model"]["grid"] = {"regressor__unknown": [1.0]}
    path.write_text(yaml.safe_dump(record), encoding="utf-8")
    load, check, _ = _api()
    with pytest.raises(ValueError, match="unknown"):
        check(load(path))


def test_group_kfold_predictions_are_complete_and_reproducible(tmp_path):
    path, record = _recipe(tmp_path, task="classification")
    record["validation"].update(outer="group_kfold", outer_splits=2)
    path.write_text(yaml.safe_dump(record), encoding="utf-8")
    load, _, run = _api()
    first = run(load(path))
    frame = pd.read_csv(first.output / "predictions.tsv", sep="\t")
    record["output"] = "second-results"
    path.write_text(yaml.safe_dump(record), encoding="utf-8")
    second = run(load(path))
    repeated = pd.read_csv(second.output / "predictions.tsv", sep="\t")
    pd.testing.assert_frame_equal(frame, repeated)
    assert frame.fold.nunique() == 2


@pytest.mark.parametrize("changed", ["recipe", "implementation", "software"])
def test_model_run_refuses_changed_provenance_during_fitting(tmp_path, monkeypatch, changed):
    from eegtable.runner import model_run

    path, _ = _recipe(tmp_path)
    load, _, run = _api()
    predict = model_run._predictions

    def mutate_after_fitting(recipe, prepared):
        predictions = predict(recipe, prepared)
        if changed == "recipe":
            path.write_text(
                path.read_text(encoding="utf-8") + "\n# modified during execution\n",
                encoding="utf-8",
            )
        elif changed == "implementation":
            monkeypatch.setattr(model_run, "implementation_hash", lambda: "modified")
        else:
            monkeypatch.setattr(model_run, "software_versions", lambda: {"python": "modified"})
        return predictions

    monkeypatch.setattr(model_run, "_predictions", mutate_after_fitting)
    with pytest.raises(ValueError, match="changed"):
        run(load(path))
    assert not (tmp_path / "results").exists()


def test_model_run_refuses_recipe_modified_after_loading(tmp_path):
    path, _ = _recipe(tmp_path)
    load, _, run = _api()
    recipe = load(path)
    path.write_text(
        path.read_text(encoding="utf-8") + "\n# modified after loading\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="recipe.*changed"):
        run(recipe)
    assert not (tmp_path / "results").exists()


@pytest.mark.parametrize(
    "rows,role,name",
    [
        ("epochs", "target", "event"),
        ("epochs", "target", "epoch"),
        ("epochs", "groups", "epoch"),
        ("groups", "target", "group"),
        ("groups", "groups", "group"),
    ],
)
def test_model_bundle_preserves_canonical_identity_when_used_in_analysis(
    tmp_path, rows, role, name
):
    from eegtable.group import read_group_dataset
    from eegtable.io import read_dataset

    task = "classification" if name == "event" else "regression"
    path, record = _recipe(tmp_path, task=task, rows=rows)
    reader = read_dataset if rows == "epochs" else read_group_dataset
    paths = [tmp_path / filename for filename in record["inputs"]["paths"]]
    if name in ("event", "group"):
        for source in paths:
            dataset = reader([source])
            descriptors = dataset.targets.copy()
            labels = tuple(
                str(row % 2 if name == "event" else row) for row in range(dataset.table.n_rows)
            )
            descriptors[name] = labels
            if name == "event":
                assert dataset.table.row_ids is not None
                identities = tuple(
                    (recording, epoch, label)
                    for (recording, epoch, _), label in zip(
                        dataset.table.row_ids, labels, strict=True
                    )
                )
                table = replace(dataset.table, row_ids=identities)
            else:
                table = replace(dataset.table, row_labels=labels)
            descriptors = descriptors.drop(columns="epoch" if rows == "epochs" else "group")
            write_table(table, source, rows=descriptors)
    record["analysis"][role] = name
    path.write_text(yaml.safe_dump(record), encoding="utf-8")
    load, check, run = _api()
    recipe = load(path)
    original = reader(paths)
    assert check(recipe).n_rows == original.table.n_rows
    result = run(recipe)
    restored = reader([result.output / "design_features.tsv"])
    keys = ["recording", "epoch", "event"] if rows == "epochs" else ["recording", "group"]
    pd.testing.assert_frame_equal(restored.targets[keys], original.targets[keys])
    matrix = np.load(result.output / "design_matrix.npz")
    np.testing.assert_array_equal(
        matrix["y"], pd.to_numeric(original.targets[record["analysis"]["target"]])
    )
