from __future__ import annotations

import numpy as np
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression

from eegtable.model.scoring import _selection_pearsonr, pearsonr_scorer, safe_pearsonr, scoring_dict


@pytest.mark.parametrize("score", [safe_pearsonr, _selection_pearsonr])
@pytest.mark.parametrize("argument", [0, 1])
def test_correlations_reject_complex_values(score, argument) -> None:
    inputs = [np.arange(4.0), np.arange(4.0)]
    inputs[argument] = inputs[argument].astype(complex) + 1j
    with pytest.raises(ValueError, match="real"):
        score(*inputs)


def test_perfect_correlation_is_one() -> None:
    r, _ = safe_pearsonr(np.arange(5.0), 2.0 * np.arange(5.0))
    assert r == pytest.approx(1.0)


@pytest.mark.parametrize("scale", [1e-12, 1e-7, 1.0, 1e7])
def test_correlation_and_selection_do_not_depend_on_measurement_units(scale) -> None:
    X = np.arange(10.0).reshape(-1, 1)
    y = X[:, 0] * scale
    model = LinearRegression().fit(X, y)
    assert safe_pearsonr(y, model.predict(X))[0] == pytest.approx(1.0)
    assert pearsonr_scorer()(model, X, y) == pytest.approx(1.0)


@pytest.mark.parametrize("constant", [0.0, 0.1])
def test_a_constant_predictor_gives_nan_rather_than_a_divide_by_zero(constant) -> None:
    # Zero variance makes Pearson undefined. Returning NaN keeps the fold in the record
    # as unscored; returning 0.0 would claim a measured absence of correlation.
    r, p = safe_pearsonr(np.full(3, constant), np.arange(3.0))
    assert np.isnan(r) and np.isnan(p)


def test_fewer_than_two_finite_pairs_gives_nan() -> None:
    r, p = safe_pearsonr(np.array([1.0, np.nan, np.nan]), np.array([1.0, 2.0, 3.0]))
    assert np.isnan(r) and np.isnan(p)


def test_non_finite_pairs_are_dropped_not_propagated() -> None:
    r, _ = safe_pearsonr(np.array([1.0, 2.0, 3.0, np.nan]), np.array([2.0, 4.0, 6.0, 1.0]))
    assert r == pytest.approx(1.0)


def test_scoring_dict_exposes_the_correlation_scorer() -> None:
    scores = scoring_dict()
    assert "r" in scores


def test_pearsonr_scorer_callable() -> None:
    scorer = pearsonr_scorer()
    assert callable(scorer)


def test_the_selection_scorer_scores_a_constant_predictor_as_no_correlation() -> None:
    # A candidate that predicts one value everywhere (an elastic net whose penalty zeroed
    # every coefficient) has no linear association with the target. Scoring it NaN made
    # the non-finite-score guard abort the whole fold instead of ranking the candidate last.
    X = np.arange(10.0).reshape(-1, 1)
    y = np.arange(10.0)
    constant = DummyRegressor(strategy="constant", constant=3.0).fit(X, y)
    assert pearsonr_scorer()(constant, X, y) == 0.0


def test_the_selection_scorer_leaves_a_constant_target_undefined() -> None:
    # A validation split whose target does not vary is a data fault, not a weak candidate.
    X = np.arange(10.0).reshape(-1, 1)
    model = LinearRegression().fit(X, np.arange(10.0))
    assert np.isnan(pearsonr_scorer()(model, X, np.full(10, 2.0)))


def test_model_selection_cannot_drop_failed_predictions() -> None:
    with pytest.raises(ValueError, match="finite"):
        _selection_pearsonr(np.arange(4.0), np.array([0.0, 1.0, np.nan, np.nan]))


def test_model_selection_recognizes_inexact_constant_predictions() -> None:
    assert _selection_pearsonr(np.arange(3.0), np.full(3, 0.1)) == 0.0
