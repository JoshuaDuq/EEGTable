from __future__ import annotations

import numpy as np
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import eegtable.model.nulls as nulls
from eegtable.model import _ridge_null
from eegtable.model.aggregate import AggregationConfig, subject_r_scorer
from eegtable.model.crossfit import cross_fit_regression
from eegtable.model.estimators import ridge_grid, ridge_pipeline
from eegtable.model.nulls import NullConfig, _prediction_statistic, permutation_test
from eegtable.model.splits import InnerSplit, loso_folds, within_subject_folds
from eegtable.model.transformers import PreprocessingConfig

GROUPS = np.repeat([f"s{i}" for i in range(6)], 12).astype(object)
RUNS = np.tile(np.repeat(["r1", "r2", "r3"], 4), 6).astype(object)


def test_batched_subject_correlation_is_zero_for_constant_fold_cells() -> None:
    values = np.array([1e-6] * 10 + [2e-6] * 3)[:, None]
    actual = _ridge_null._subject_r(
        values, values, np.full(13, "s1"), np.array([1] * 10 + [2] * 3), AggregationConfig()
    )
    np.testing.assert_array_equal(actual, [0.0])


def test_batched_subject_correlation_refuses_two_points_left_by_fold_centring() -> None:
    values = np.array([[0.3], [1.9], [0.7]])
    with pytest.raises(ValueError, match="at least 3"):
        _ridge_null._subject_r(
            values, values[::-1], np.full(3, "s1"), np.array([1, 1, 2]), AggregationConfig()
        )


@pytest.mark.parametrize("greater_is_better", [True, False])
@pytest.mark.parametrize("batched", [True, False])
def test_identity_permutations_count_as_ties(batched, greater_is_better, monkeypatch) -> None:
    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(9, 2)), rng.normal(size=9)
    groups = np.repeat(["A", "B", "C"], 3).astype(object)
    folds = loso_folds(groups)
    pipeline = Pipeline([("regressor", Ridge(alpha=1.0, solver="svd"))])
    inner = InnerSplit("subject", n_splits=2)
    aggregation = AggregationConfig(ci_method="none")
    predictions = cross_fit_regression(folds, X, y, groups, pipeline, {}, inner=inner, seed=0)
    observed = _prediction_statistic(predictions, groups, aggregation, None)
    # Independent floating-point calculations of the same statistic can differ by an ULP.
    observed = np.nextafter(observed, np.inf if greater_is_better else -np.inf)
    if batched:
        estimate = _ridge_null.ridge_null

        def perturbed_identity_statistics(*args, **kwargs):
            statistics = estimate(*args, **kwargs)
            sources = args[3]
            identity = (sources == np.arange(len(y))).all(axis=1)
            # BLAS kernels can round single-target and batched fits differently.
            statistics[identity] += 1e-12
            return statistics

        monkeypatch.setattr(_ridge_null, "ridge_null", perturbed_identity_statistics)
    else:
        monkeypatch.setattr(_ridge_null, "ridge_penalty", lambda *args, **kwargs: None)
    result = permutation_test(
        folds,
        X,
        y,
        groups,
        np.ones(9, dtype=object),
        pipeline,
        {},
        observed,
        config=NullConfig(
            scheme="circular_shift_within_run", min_retained_trials=3, n_permutations=100
        ),
        trial_indices=np.tile(np.arange(3), 3),
        inner=inner,
        seed=0,
        aggregation=aggregation,
        greater_is_better=greater_is_better,
    )
    identity = result.changed_fractions == 0.0
    assert identity.any()
    np.testing.assert_allclose(result.null[identity], observed, rtol=0, atol=1e-14)
    strict_tail = result.null >= observed if greater_is_better else result.null <= observed
    expected = (np.count_nonzero(strict_tail | identity) + 1) / 101
    assert result.p_value == expected


def _design(missing: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    nuisance = rng.normal(size=GROUPS.size)
    X = rng.normal(size=(GROUPS.size, 20)) + np.outer(nuisance, rng.normal(size=20))
    y = X[:, 0] - 0.5 * X[:, 1] + 2.0 * nuisance + rng.normal(size=GROUPS.size)
    if missing:
        X[rng.random(X.shape) < 0.05] = np.nan
        X[GROUPS == "s2", 3] = np.nan
    return X, y, nuisance.reshape(-1, 1)


def _case(name: str) -> dict[str, object]:
    X, y, nuisance = _design(missing=name == "missing")
    case: dict[str, object] = {
        "folds": loso_folds(GROUPS),
        "X": X,
        "y": y,
        "runs": None,
        "pipeline": ridge_pipeline(PreprocessingConfig(), seed=0),
        "grid": {"regressor__alpha": ridge_grid(X)["regressor__alpha"][2:7:2]},
        "config": NullConfig(n_permutations=9),
        "inner": InnerSplit(grouping="subject", n_splits=2),
        "options": {},
    }
    if name == "missing":
        case["options"] = {"harmonization": "intersection"}
    elif name == "pooled_nuisance":
        case["options"] = {"covariates": nuisance, "residualize_on": ("n",)}
    elif name == "subject_nuisance":
        case["options"] = {
            "covariates": nuisance,
            "residualize_on": ("n",),
            "residualize_within": "subject",
        }
    elif name == "within_subject_folds":
        case["folds"] = within_subject_folds(GROUPS, RUNS, inner_splits=2, outer_splits=3)
        case["runs"] = RUNS
        case["inner"] = InnerSplit(grouping="run", n_splits=2)
        case["config"] = NullConfig(scheme="within_subject_within_run", n_permutations=9)
    elif name == "trial_count":
        case["aggregation"] = AggregationConfig(subject_weighting="trial_count")
    elif name in ("deconfounded", "harmonized_covariates"):
        case["X"] = np.column_stack([X, nuisance])
        case["pipeline"] = ridge_pipeline(
            PreprocessingConfig(deconfound=True), seed=0, n_covariates=1
        )
        if name == "harmonized_covariates":
            case["X"][GROUPS == "s2", -1] = np.nan
            case["options"] = {"harmonization": "intersection"}
    elif name == "untuned":
        case["grid"] = {}
    elif name == "supervised_remainder":
        case["pipeline"] = Pipeline(
            [
                (
                    "columns",
                    ColumnTransformer(
                        [("scale", StandardScaler(), [0])],
                        remainder=SelectKBest(f_regression, k=1),
                    ),
                ),
                ("regressor", Ridge()),
            ]
        )
        case["grid"] = {}
        case["config"] = NullConfig(n_permutations=3)
    return case


def _null(case: dict[str, object]):
    options = dict(case["options"])  # type: ignore[call-overload]
    aggregation = case.get("aggregation", AggregationConfig())
    predictions = cross_fit_regression(
        case["folds"],
        case["X"],
        case["y"],
        GROUPS,
        case["pipeline"],
        case["grid"],
        inner=case["inner"],
        seed=3,
        runs=case["runs"],
        **options,
    )
    observed = _prediction_statistic(predictions, GROUPS, aggregation, None)
    return permutation_test(
        case["folds"],
        case["X"],
        case["y"],
        GROUPS,
        case["runs"],
        case["pipeline"],
        case["grid"],
        observed,
        config=case["config"],
        inner=case["inner"],
        seed=3,
        aggregation=aggregation,
        **options,
    )


CASES = [
    "loso",
    "missing",
    "pooled_nuisance",
    "subject_nuisance",
    "within_subject_folds",
    "trial_count",
    "deconfounded",
    "harmonized_covariates",
    "untuned",
    "supervised_remainder",
]


@pytest.mark.parametrize("name", CASES)
def test_the_closed_form_ridge_null_equals_refitting_every_draw(name: str, monkeypatch) -> None:
    # The null is a claim about the procedure that produced the observed statistic, so the
    # shortcut must reproduce every refitted draw, not just a similar distribution.
    fast = _null(_case(name))
    monkeypatch.setattr(_ridge_null, "ridge_penalty", lambda *args, **kwargs: None)
    refitted = _null(_case(name))
    np.testing.assert_allclose(fast.null, refitted.null, rtol=0, atol=1e-9)
    assert fast.p_value == refitted.p_value


def test_ridge_null_harmonization_preserves_missing_covariates() -> None:
    case = _case("harmonized_covariates")
    harmonized = _null(case)
    case["options"] = {}
    reference = _null(case)
    np.testing.assert_allclose(harmonized.null, reference.null, rtol=0, atol=1e-9)
    assert harmonized.p_value == reference.p_value


@pytest.mark.parametrize("target_scale", [1e-160, 1e160])
def test_ridge_null_correlation_is_independent_of_target_units(target_scale, monkeypatch) -> None:
    case = _case("untuned")
    case["y"] *= target_scale
    case["pipeline"] = Pipeline([("regressor", Ridge(alpha=1.0, solver="svd"))])
    fast = _null(case)
    monkeypatch.setattr(_ridge_null, "ridge_penalty", lambda *args, **kwargs: None)
    refitted = _null(case)
    np.testing.assert_allclose(fast.null, refitted.null, rtol=0, atol=1e-9)
    assert fast.p_value == refitted.p_value


@pytest.mark.parametrize("solver", ["svd", "cholesky", "auto"])
@pytest.mark.parametrize("alpha", [1e-12, 1e-20])
def test_ridge_null_preserves_low_variance_predictors(alpha: float, solver: str, monkeypatch):
    rng = np.random.default_rng(0)
    case = _case("untuned")
    X = rng.normal(size=(GROUPS.size, 2))
    X[:, 1] *= 1e-8
    case["X"] = X
    case["y"] = rng.normal(size=GROUPS.size)
    case["pipeline"] = Pipeline([("regressor", Ridge(alpha=alpha, solver=solver))])

    fast = _null(case)
    monkeypatch.setattr(_ridge_null, "ridge_penalty", lambda *args, **kwargs: None)
    refitted = _null(case)

    np.testing.assert_allclose(fast.null, refitted.null, rtol=0, atol=1e-9)
    assert fast.p_value == refitted.p_value


@pytest.mark.parametrize("grid", [{}, {"regressor__alpha": [0.0, 1.0]}])
def test_batched_ridge_null_rejects_unregularized_fits(grid) -> None:
    # With duplicate predictors, batching an unregularized SVD solve changes the
    # null statistic and can change p, despite using the same solver as each refit.
    case = _case("untuned")
    rng = np.random.default_rng(6)
    first = rng.normal(size=GROUPS.size)
    case["X"] = np.column_stack([first, first, rng.normal(size=GROUPS.size)])
    case["pipeline"] = Pipeline([("regressor", Ridge(alpha=0.0, solver="svd"))])
    case["grid"] = grid

    with pytest.raises(RuntimeError, match="No p-value") as error:
        _null(case)
    assert "strictly positive" in str(error.value.__cause__)


def _count_refits(monkeypatch) -> list[int]:
    calls: list[int] = []
    engine = nulls._cross_fit_engine

    def counting(*args, **kwargs):
        calls.append(1)
        return engine(*args, **kwargs)

    monkeypatch.setattr(nulls, "_cross_fit_engine", counting)
    return calls


def test_a_ridge_null_refits_the_procedure_once_not_once_per_draw(monkeypatch) -> None:
    case = _case("loso")
    calls = _count_refits(monkeypatch)
    _null(case)
    assert len(calls) == 1  # the observed statistic's own check


def test_multi_metric_null_checks_every_metric_on_permuted_targets() -> None:
    case = _case("loso")
    targets = {tuple(row): value for row, value in zip(case["X"], case["y"], strict=True)}

    def alignment_score(estimator, X, y):
        del estimator
        expected = np.array([targets[tuple(row)] for row in X])
        return 0.0 if np.array_equal(y, expected) else np.nan

    case["options"] = {
        "scoring": {"r": subject_r_scorer(), "alignment": alignment_score},
        "refit": "r",
    }

    with pytest.raises(RuntimeError, match="No p-value"):
        _null(case)


def test_a_step_that_learns_from_the_target_falls_back_to_refitting(monkeypatch) -> None:
    # Univariate selection scores features against y, so preprocessing differs by draw.
    case = _case("loso")
    case["pipeline"] = ridge_pipeline(PreprocessingConfig(feature_selection_percentile=50), seed=0)
    calls = _count_refits(monkeypatch)
    _null(case)
    assert len(calls) == 1 + 9
