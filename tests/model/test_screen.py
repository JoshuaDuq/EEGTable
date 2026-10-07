from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from eegtable.model.screen import univariate_screen

GROUPS = np.repeat([f"s{i:02d}" for i in range(12)], 30).astype(object)


@pytest.mark.parametrize("missing", [None, np.nan])
def test_screen_requires_a_subject_label_for_every_trial(missing) -> None:
    target = np.tile(np.arange(4.0), 4)
    groups = np.repeat(np.arange(4), 4).astype(object)
    groups[-4:] = missing
    with pytest.raises(ValueError, match="subject label for every trial"):
        univariate_screen(target[:, None], target, groups, n_flips=31, seed=0)


@pytest.mark.parametrize("n_subjects", [5, 10, 11])
@pytest.mark.parametrize("direction", [-1.0, 1.0])
def test_identical_subject_effects_have_valid_significance(n_subjects, direction) -> None:
    target = np.tile(np.arange(4.0), n_subjects)
    groups = np.repeat(np.arange(n_subjects), 4)

    screen = univariate_screen(
        direction * target[:, None], target, groups, n_flips=31, seed=0
    ).iloc[0]

    assert direction * screen["t"] > 1e12
    assert screen["p"] == 0.0 and screen["q"] == 0.0
    assert 0.0 < screen["p_fwer"] <= 1.0


def test_nearly_identical_subject_effects_match_scipy_ttest() -> None:
    n_subjects = 5
    target = np.tile(np.arange(4.0), n_subjects)
    values = np.tile([0.0, 2.0, 1.0, 3.0], n_subjects)
    values[::4] += np.arange(n_subjects) * 1e-7
    groups = np.repeat(np.arange(n_subjects), 4)
    correlations = [
        np.corrcoef(values[groups == subject], target[groups == subject])[0, 1]
        for subject in range(n_subjects)
    ]
    reference = stats.ttest_1samp(np.arctanh(correlations), 0.0)

    screen = univariate_screen(values[:, None], target, groups, n_flips=31, seed=0).iloc[0]

    assert screen["t"] == pytest.approx(reference.statistic, rel=1e-7)
    assert screen["p"] == pytest.approx(reference.pvalue, rel=1e-6)
    assert np.isfinite(screen["q"])


def test_maximum_statistic_counts_sign_flip_ties_despite_roundoff() -> None:
    target = np.array([-1.0, 0.0, 1.0]) / np.sqrt(2.0)
    orthogonal = np.array([1.0, -2.0, 1.0]) / np.sqrt(6.0)

    def feature(z):
        correlation = np.tanh(z)
        return correlation * target + np.sqrt(1.0 - correlation**2) * orthogonal

    first, cancelling, last = 0.049054613825311656, 2.002392583645255, 0.18851919251246557
    values = np.concatenate(
        [feature(first), feature(cancelling), -feature(cancelling), feature(last)]
    )[:, None]
    n_flips = 1000
    signs = np.random.default_rng(0).choice([-1.0, 1.0], size=(n_flips, 4))
    # Opposite signs break the cancelling pair; equal outer signs tie the observed
    # sum in magnitude. |t| increases with |sum(z)| because sum(z**2) is fixed.
    extreme = (signs[:, 1] != signs[:, 2]) | (signs[:, 0] == signs[:, 3])
    expected = (1 + extreme.sum()) / (n_flips + 1)

    screen = univariate_screen(
        values, np.tile(target, 4), np.repeat(np.arange(4), 3), n_flips=n_flips, seed=0
    )

    assert screen["p_fwer"].iloc[0] == pytest.approx(expected)


def test_zero_subject_effects_have_undefined_inference_without_breaking_the_family() -> None:
    target = np.tile([-1.0, 0.0, 1.0], 5)
    orthogonal = np.tile([1.0, -2.0, 1.0], 5)
    groups = np.repeat(np.arange(5), 3)
    values = np.column_stack([target, orthogonal])

    screen = univariate_screen(values, target, groups, n_flips=31, seed=0)

    assert screen["p"].iloc[0] == 0.0
    assert screen["q"].iloc[0] == 0.0
    assert screen["r"].iloc[1] == 0.0
    assert screen.iloc[1][["t", "p", "q", "p_fwer"]].isna().all()


def _noise(seed: int, n_features: int = 40) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(GROUPS.size, n_features)), rng.normal(size=GROUPS.size)


def test_a_feature_that_tracks_the_target_within_subjects_survives_the_family() -> None:
    X, y = _noise(0)
    X[:, 0] = y + np.random.default_rng(1).normal(size=GROUPS.size)
    screen = univariate_screen(X, y, GROUPS, n_flips=999, seed=0)
    assert screen["p_fwer"].iloc[0] < 0.01
    assert (screen["p_fwer"].iloc[1:] > 0.05).all()
    assert screen["r"].iloc[0] == pytest.approx(1 / np.sqrt(2), abs=0.1)


def test_differences_between_subjects_do_not_count_as_tracking() -> None:
    # Subjects with a higher mean target also have a higher feature, but within every
    # subject the feature follows nothing: pooled r is large, within-subject r is not.
    X, y = _noise(2, n_features=1)
    offset = np.repeat(np.arange(12) * 5.0, 30)
    y, X[:, 0] = y + offset, X[:, 0] + offset
    screen = univariate_screen(X, y, GROUPS, n_flips=999, seed=0)
    assert np.corrcoef(X[:, 0], y)[0, 1] > 0.9
    assert abs(screen["r"].iloc[0]) < 0.1
    assert screen["p"].iloc[0] > 0.05


def test_the_family_wise_error_is_held_at_its_level_under_the_null() -> None:
    # Features share variance, as channels do, which a max-statistic null keeps.
    hits = 0
    for seed in range(200):
        rng = np.random.default_rng(seed)
        shared = rng.normal(size=(GROUPS.size, 1))
        X = shared + rng.normal(size=(GROUPS.size, 20))
        y = rng.normal(size=GROUPS.size)
        hits += bool(
            (univariate_screen(X, y, GROUPS, n_flips=199, seed=seed)["p_fwer"] <= 0.05).any()
        )
    assert hits / 200 <= 0.08


def test_a_shared_nuisance_is_removed_from_both_sides_within_each_subject() -> None:
    rng = np.random.default_rng(3)
    stimulus = rng.choice([-1.0, 0.0, 1.0], size=GROUPS.size)
    slope = np.repeat(rng.uniform(0.5, 2.0, 12), 30)
    y = slope * stimulus + 0.5 * rng.normal(size=GROUPS.size)
    X = np.column_stack([slope * stimulus + 0.5 * rng.normal(size=GROUPS.size)])
    raw = univariate_screen(X, y, GROUPS, n_flips=199, seed=0)
    adjusted = univariate_screen(
        X,
        y,
        GROUPS,
        covariates=stimulus.reshape(-1, 1),
        residualize_on=("stimulus",),
        n_flips=199,
        seed=0,
    )
    assert raw["r"].iloc[0] > 0.5
    assert abs(adjusted["r"].iloc[0]) < 0.15


def test_a_feature_constant_within_every_subject_is_not_tested_after_residualizing() -> None:
    X, y = _noise(5, n_features=2)
    X[:, 1] = 3.7
    stimulus = np.random.default_rng(6).choice([-1.0, 0.0, 1.0], size=GROUPS.size)
    screen = univariate_screen(
        X,
        y,
        GROUPS,
        covariates=stimulus.reshape(-1, 1),
        residualize_on=("stimulus",),
        n_flips=99,
        seed=0,
    )
    assert screen["n_subjects"].iloc[1] == 0 and np.isnan(screen["p"].iloc[1])


def test_a_subject_missing_a_feature_drops_out_of_that_feature_only() -> None:
    X, y = _noise(4, n_features=3)
    X[GROUPS == "s00", 1] = np.nan
    X[~np.isin(GROUPS, ["s00", "s01"]), 2] = np.nan
    screen = univariate_screen(X, y, GROUPS, feature_names=["a", "b", "c"], n_flips=99, seed=0)
    assert screen.loc["a", "n_subjects"] == 12
    assert screen.loc["b", "n_subjects"] == 11
    # Two subjects cannot estimate a spread across subjects.
    assert screen.loc["c", "n_subjects"] == 2 and np.isnan(screen.loc["c", "p_fwer"])
