import numpy as np
import pytest

from eegtable.microstates import (
    _coverage,
    _duration,
    _occurrence,
    _peak_topographies,
    _runs,
    _smooth,
    _transitions,
    microstate_coverage,
    microstate_duration,
    microstate_occurrence,
    microstate_transitions,
    segment,
)
from eegtable.signal import Signal
from eegtable.spectra import Window

# segment() clusters with scikit-learn, the optional "microstates" extra; the
# metric definitions below do not need it.
requires_sklearn = pytest.mark.skipif(
    __import__("importlib.util", fromlist=["util"]).find_spec("sklearn") is None,
    reason="scikit-learn is not installed in this environment",
)

SFREQ = 100.0
WINDOW = Window("all", 0.0, 3.99)
N_CHANNELS = 8


def _planted(n_epochs: int = 6, n_times: int = 400, noise: float = 0.05) -> tuple:
    """Blocks of four fixed topographies, each held for 25 samples."""
    rng = np.random.RandomState(0)
    maps = rng.randn(4, N_CHANNELS)
    maps -= maps.mean(axis=1, keepdims=True)
    maps /= np.linalg.norm(maps, axis=1, keepdims=True)
    data = np.zeros((n_epochs, N_CHANNELS, n_times))
    truth = np.zeros((n_epochs, n_times), dtype=int)
    shape = 1.0 + 0.5 * np.sin(np.linspace(0, np.pi, 25))
    for epoch in range(n_epochs):
        for block in range(n_times // 25):
            state = (block + epoch) % 4
            data[epoch, :, block * 25 : (block + 1) * 25] = maps[state][:, None] * shape[None, :]
            truth[epoch, block * 25 : (block + 1) * 25] = state
    data += rng.randn(n_epochs, N_CHANNELS, n_times) * noise
    signal = Signal.from_arrays(
        data=data,
        times=np.arange(n_times) / SFREQ,
        ch_names=tuple(f"E{i}" for i in range(N_CHANNELS)),
        sfreq=SFREQ,
        row_ids=tuple(("test", index, "event") for index in range(n_epochs)),
    )
    return signal, truth


@requires_sklearn
@pytest.mark.parametrize("gain", [1.0, 1e-6, -3.0])
def test_templates_preserve_gfp_weighting_in_the_scatter_objective(gain: float) -> None:
    first = np.array([1.0, -1.0, 0.0, 0.0]) / np.sqrt(2.0)
    tilted_axis = np.array([1.0, 1.0, -2.0, 0.0]) / np.sqrt(6.0)
    separate = np.array([1.0, 1.0, 1.0, -3.0]) / np.sqrt(12.0)
    angle = np.deg2rad(30.0)
    stronger = 10.0 * (np.cos(angle) * first + np.sin(angle) * tilted_axis)
    peak_maps = np.repeat(np.stack([first, stronger, separate]), 20, axis=0)
    # Mixed polarities cannot change the scatter matrix or the fitted maps.
    peak_maps[::2] *= -1.0
    pulse = np.array([0.8, 1.0, 0.8])
    data = (gain * peak_maps.T[:, :, None] * pulse).reshape(1, 4, -1)
    signal = Signal.from_arrays(
        data=data,
        times=np.arange(data.shape[-1]) / SFREQ,
        ch_names=("F3", "F4", "P3", "P4"),
        sfreq=SFREQ,
        row_ids=(("test", 0, "event"),),
    )

    segmentation = segment(signal, n_states=2, min_duration_ms=0.0)

    # The leading direction of uu.T + 100 vv.T is 29.753 degrees from u,
    # rather than the 15-degree midpoint produced by unit-normalized peak maps.
    expected_angle = 0.5 * np.arctan2(
        100.0 * np.sin(2.0 * angle), 1.0 + 100.0 * np.cos(2.0 * angle)
    )
    expected = np.cos(expected_angle) * first + np.sin(expected_angle) * tilted_axis
    assert np.max(np.abs(segmentation.templates @ expected)) == pytest.approx(1.0, abs=1e-10)
    assert segmentation.global_explained_variance > 0.997


@requires_sklearn
def test_segmentation_respects_fractional_sample_minimum_duration() -> None:
    signal, _ = _planted(noise=0.0)
    result = segment(signal, min_duration_ms=254.0)
    for states in result.states:
        assert all(stop - start >= 26 for start, stop, _ in _runs(states))


def test_peak_separation_is_not_rounded_down() -> None:
    strength = np.array([0.0, 1.0, 0.0, 2.0, 0.0])
    epoch = np.array([-1.0, 1.0])[:, None] * strength
    maps = _peak_topographies(epoch, 100.0, 24.0, 10, None)
    assert maps.shape[0] == 1


@requires_sklearn
def test_segmentation_rejects_epochs_without_qualifying_gfp_peaks() -> None:
    signal, _ = _planted()
    with pytest.raises(ValueError, match="only 0 global field power peaks"):
        segment(signal, n_states=2, peak_prominence=1e12)


@requires_sklearn
def test_segmentation_rejects_collapsed_microstate_clusters() -> None:
    amplitudes = np.tile([1.0, 2.0, 1.0], 30)
    topography = np.array([1.0, 2.0, -1.0, -2.0])
    signal = Signal.from_arrays(
        data=(topography[:, None] * amplitudes)[None],
        times=np.arange(amplitudes.size) / SFREQ,
        ch_names=("F3", "F4", "P3", "P4"),
        sfreq=SFREQ,
        row_ids=(("test", 0, "event"),),
    )
    with pytest.raises(ValueError, match="empty.*microstate cluster"):
        segment(signal, n_states=2, min_duration_ms=0.0)


def _alternating_ring_maps(seed: int) -> tuple:
    """Four orthogonal maps whose polarity alternates at 10 Hz, in 60-120 ms segments."""
    rng = np.random.default_rng(seed)
    n_channels, n_times, sfreq = 32, 1000, 250.0
    theta = np.linspace(0, 2 * np.pi, n_channels, endpoint=False)
    maps = np.stack([np.cos(theta), np.sin(theta), np.cos(2 * theta), np.sin(2 * theta)])
    maps /= np.linalg.norm(maps, axis=1, keepdims=True)
    times = np.arange(n_times) / sfreq
    data = np.zeros((10, n_channels, n_times))
    for epoch in range(10):
        start = 0
        while start < n_times:
            span = slice(start, start + int(rng.integers(15, 30)))
            phase = rng.uniform(0, 2 * np.pi)
            oscillation = np.sin(2 * np.pi * 10 * times[span] + phase)
            data[epoch, :, span] = maps[rng.integers(4)][:, None] * oscillation
            start = span.stop
    data += 0.2 * rng.standard_normal(data.shape) / np.sqrt(n_channels)
    signal = Signal.from_arrays(
        data=data,
        times=times,
        ch_names=tuple(f"E{i}" for i in range(n_channels)),
        sfreq=sfreq,
        row_ids=tuple(("test", index, "event") for index in range(10)),
    )
    return signal, maps


@requires_sklearn
@pytest.mark.parametrize("seed", [0, 5, 7])
def test_distinct_states_are_found_whichever_start_leaves_a_cluster_empty(seed) -> None:
    # Euclidean k-means on sign-flipped maps can return a blended centre no map is
    # closest to; one start like that used to end the fit instead of being discarded.
    signal, maps = _alternating_ring_maps(seed)
    templates = segment(signal, n_states=4, min_duration_ms=0.0).templates
    assert np.abs(templates @ maps.T).max(axis=0).min() > 0.99


def test_peak_topographies_never_substitutes_an_extremum() -> None:
    strength = np.array([1.0, 2.0, 3.0, 4.0])
    epoch = np.array([-1.0, 1.0])[:, None] * strength
    assert _peak_topographies(epoch, 100.0, 10.0, 10, None).shape == (0, 2)


@requires_sklearn
def test_planted_topographies_are_recovered() -> None:
    signal, truth = _planted()
    seg = segment(signal, n_states=4)
    purity = np.mean(
        [
            np.bincount(seg.states[truth == k], minlength=4).max() / (truth == k).sum()
            for k in range(4)
        ]
    )
    assert purity == pytest.approx(1.0)


@requires_sklearn
def test_cluster_indices_never_claim_canonical_microstate_labels() -> None:
    signal, _ = _planted()
    assert segment(signal, n_states=4).labels == ("state1", "state2", "state3", "state4")
    assert segment(signal, n_states=3).labels == ("state1", "state2", "state3")


@requires_sklearn
def test_segmentation_exposes_global_explained_variance() -> None:
    signal, _ = _planted(noise=0.01)
    segmentation = segment(signal, n_states=4)
    assert 0.0 <= segmentation.global_explained_variance <= 1.0


@requires_sklearn
def test_a_topography_and_its_inversion_are_the_same_state() -> None:
    signal, _ = _planted(noise=0.0)
    seg = segment(signal, n_states=4)
    flipped = Signal.from_arrays(
        data=-signal.data,
        times=signal.times,
        ch_names=signal.ch_names,
        sfreq=signal.sfreq,
        row_ids=signal.row_ids,
    )
    from eegtable.microstates import _assign

    original = _assign(signal.data[0], seg.templates)
    inverted = _assign(flipped.data[0], seg.templates)
    np.testing.assert_array_equal(original, inverted)


@requires_sklearn
def test_coverage_sums_to_one_across_states() -> None:
    signal, _ = _planted()
    table = microstate_coverage(segment(signal, n_states=4), windows=[WINDOW])
    np.testing.assert_allclose(table.values.sum(axis=1), 1.0)


@requires_sklearn
def test_measures_have_one_row_per_epoch() -> None:
    signal, _ = _planted(n_epochs=6)
    seg = segment(signal, n_states=4)
    for fn in (microstate_coverage, microstate_duration, microstate_occurrence):
        table = fn(seg, windows=[WINDOW])
        assert table.values.shape == (6, 4)
        assert table.row_labels is None
        assert all(m.space_kind == "state" for m in table.meta)


@requires_sklearn
def test_duration_recovers_the_planted_block_length() -> None:
    signal, _ = _planted(noise=0.0)
    table = microstate_duration(segment(signal, n_states=4), windows=[WINDOW])
    # Blocks are 25 samples at 100 Hz = 250 ms.
    assert np.nanmean(table.values) == pytest.approx(250.0, rel=0.1)


@requires_sklearn
def test_transitions_are_ordered_pairs_excluding_self() -> None:
    signal, _ = _planted()
    table = microstate_transitions(segment(signal, n_states=4), windows=[WINDOW])
    spaces = [m.space for m in table.meta]
    assert len(spaces) == 12  # 4 states, ordered, no self-transitions
    assert "state1-to-state1" not in spaces
    assert all(m.space_kind == "pair" for m in table.meta)


@requires_sklearn
def test_fit_on_restricts_which_trials_inform_the_templates() -> None:
    signal, _ = _planted(n_epochs=6)
    mask = np.array([True, True, True, False, False, False])
    restricted = segment(signal, n_states=4, fit_on=mask)
    # Every epoch is still segmented, including those excluded from fitting.
    assert restricted.states.shape[0] == 6


@requires_sklearn
def test_a_fit_mask_of_the_wrong_length_raises() -> None:
    signal, _ = _planted(n_epochs=6)
    with pytest.raises(ValueError, match="one entry per epoch"):
        segment(signal, n_states=4, fit_on=np.array([True, False]))


@requires_sklearn
def test_excluding_every_epoch_raises() -> None:
    signal, _ = _planted(n_epochs=4)
    with pytest.raises(ValueError, match="nothing to cluster"):
        segment(signal, n_states=4, fit_on=np.zeros(4, dtype=bool))


@requires_sklearn
def test_a_non_boolean_fit_mask_cannot_admit_unselected_trials() -> None:
    signal, _ = _planted(n_epochs=6)
    mask = np.array([1.0, 1.0, 1.0, np.nan, np.nan, np.nan])
    with pytest.raises(ValueError, match="fit_on.*boolean"):
        segment(signal, fit_on=mask)


@requires_sklearn
@pytest.mark.parametrize("n_states", [1, 13])
def test_an_unsupported_state_count_raises(n_states: int) -> None:
    signal, _ = _planted()
    with pytest.raises(ValueError, match="n_states"):
        segment(signal, n_states=n_states)


def test_segment_reports_its_missing_dependency_clearly() -> None:
    import importlib.util

    if importlib.util.find_spec("sklearn") is not None:
        pytest.skip("scikit-learn is installed here")
    signal, _ = _planted()
    with pytest.raises(ImportError, match=r"eegtable\[microstates\]"):
        segment(signal)


@requires_sklearn
@pytest.mark.parametrize("invalid", [np.nan, np.inf, 0.0])
def test_invalid_topographies_are_not_assigned_to_state_one(invalid) -> None:
    from dataclasses import replace

    signal, _ = _planted()
    data = signal.data.copy()
    data[0, :, 0] = invalid
    with pytest.raises(ValueError, match="finite.*nonzero spatial variance"):
        segment(replace(signal, data=data))


@requires_sklearn
def test_independently_fitted_maps_have_distinct_feature_identities() -> None:
    from dataclasses import replace

    signal, _ = _planted()
    other = replace(signal, data=signal.data[:, ::-1, :].copy())
    first = segment(signal)
    second = segment(other)
    for measure in (microstate_coverage, microstate_transitions):
        first_table = measure(first, windows=[WINDOW])
        second_table = measure(second, windows=[WINDOW])
        assert set(first_table.names).isdisjoint(second_table.names)


# --- metric definitions ---------------------------------------------------------------


def test_smoothing_absorbs_a_short_run_into_the_longer_neighbour() -> None:
    # The single sample of state 1 goes to state 0, whose run is longer.
    states = np.array([0, 0, 0, 0, 1, 2, 2])
    np.testing.assert_array_equal(_smooth(states, 2), [0, 0, 0, 0, 0, 2, 2])


def test_a_tie_splits_the_run_and_an_odd_sample_goes_to_the_later_state() -> None:
    # Neighbours are equally long, so the run is halved. A one-sample run has an
    # empty first half, so it lands entirely on the following state.
    np.testing.assert_array_equal(_smooth(np.array([0, 0, 1, 2, 2]), 2), [0, 0, 2, 2, 2])
    np.testing.assert_array_equal(
        _smooth(np.array([0, 0, 0, 1, 1, 2, 2, 2]), 3), [0, 0, 0, 0, 2, 2, 2, 2]
    )


def test_a_short_run_at_the_edge_takes_its_only_neighbour() -> None:
    np.testing.assert_array_equal(_smooth(np.array([1, 0, 0, 0, 0]), 2), [0, 0, 0, 0, 0])


def test_consecutive_short_runs_are_absorbed_until_none_is_left() -> None:
    # Flicker 1 2 1 between a 6-sample state 0 and a 7-sample state 2. A single pass over
    # the original runs relabels it 0 1 2, leaving a one-sample run of state 1. Absorbing
    # the shortest run first and re-reading its neighbours after every merge hands all
    # three samples to state 0, which is each time the longer neighbour (7, 8, then 8 > 7).
    states = np.array([0] * 6 + [1, 2, 1] + [2] * 7)
    np.testing.assert_array_equal(_smooth(states, 3), [0] * 9 + [2] * 7)


def test_no_run_shorter_than_the_minimum_survives_rapid_flicker() -> None:
    states = np.random.default_rng(0).integers(0, 4, size=500)
    lengths = [stop - start for start, stop, _ in _runs(_smooth(states, 10))]
    assert len(lengths) == 1 or min(lengths) >= 10


def test_duration_is_nan_for_an_absent_state_but_occurrence_is_zero() -> None:
    states = np.array([0, 0, 1, 1])
    assert np.isnan(_duration(states, 3, SFREQ)[2])
    assert _occurrence(states, 3, SFREQ)[2] == 0.0
    assert _coverage(states, 3, SFREQ)[2] == 0.0


def test_transitions_count_segments_not_samples() -> None:
    # Two long runs: one transition, not one per sample pair.
    states = np.array([0, 0, 0, 1, 1, 1])
    matrix = _transitions(states, 2)
    assert matrix[0, 1] == pytest.approx(1.0)
    assert np.isnan(matrix[1, 0])  # state 1 is never left


def test_transition_rows_sum_to_one_where_defined() -> None:
    rng = np.random.RandomState(1)
    matrix = _transitions(rng.randint(0, 3, size=200), 3)
    rows = np.nansum(matrix, axis=1)
    assert np.allclose(rows[np.isfinite(matrix).any(axis=1)], 1.0)


def _ambiguous_polarity(n_epochs: int = 4, n_times: int = 400) -> Signal:
    """Two states, one of them a dipole whose two extrema are equal in magnitude.

    Which extremum is largest is then decided by noise, so a rule that orients a
    map by the sign of its strongest channel sends otherwise identical maps of
    that state in opposite directions.
    """
    rng = np.random.RandomState(1)
    ambiguous = np.zeros(N_CHANNELS)
    ambiguous[0], ambiguous[1] = 1.0, -1.0
    distinct = np.array([1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0])
    states = [m - m.mean() for m in (ambiguous, distinct)]
    states = [m / np.linalg.norm(m) for m in states]

    data = np.zeros((n_epochs, N_CHANNELS, n_times))
    shape = 1.0 + 0.5 * np.sin(np.linspace(0, np.pi, 25))
    for epoch in range(n_epochs):
        for block in range(n_times // 25):
            which = states[(block + epoch) % 2]
            data[epoch, :, block * 25 : (block + 1) * 25] = which[:, None] * shape[None, :]
    data += rng.randn(n_epochs, N_CHANNELS, n_times) * 0.02
    return Signal.from_arrays(
        data=data,
        times=np.arange(n_times) / SFREQ,
        ch_names=tuple(f"E{i}" for i in range(N_CHANNELS)),
        sfreq=SFREQ,
        row_ids=tuple(("test", index, "event") for index in range(n_epochs)),
    )


@requires_sklearn
def test_no_two_templates_are_the_same_topography_inverted() -> None:
    # A map and its inversion are one state, so spending two of the four classes
    # on one topography means the clustering is not polarity invariant.
    seg = segment(_ambiguous_polarity(), n_states=2, random_state=0)
    similarity = np.abs(seg.templates @ seg.templates.T)
    off_diagonal = similarity[~np.eye(2, dtype=bool)]
    assert off_diagonal.max() < 0.9


@requires_sklearn
def test_an_ambiguous_dipole_does_not_cost_explained_variance() -> None:
    seg = segment(_ambiguous_polarity(), n_states=2, random_state=0)
    assert seg.global_explained_variance > 0.95


@requires_sklearn
def test_fitted_microstate_model_segments_a_new_recording_without_refitting() -> None:
    from dataclasses import replace

    from eegtable import microstates

    training, _ = _planted(n_epochs=3)
    held_out, _ = _planted(n_epochs=2)
    held_out = replace(held_out, row_ids=(("new", 0, "event"), ("new", 1, "event")))
    model = microstates.MicrostateModel.fit(training)
    templates = model.templates.copy()
    result = model.segment(held_out)

    np.testing.assert_array_equal(model.templates, templates)
    assert result.row_ids == held_out.row_ids
    assert result.states.shape == held_out.data.shape[::2]
    np.testing.assert_array_equal(result.templates, templates)
    with pytest.raises(ValueError, match="read-only"):
        model.templates[0, 0] = 0.0


@requires_sklearn
def test_microstate_model_fits_only_selected_rows() -> None:
    from dataclasses import replace

    from eegtable import microstates

    signal, _ = _planted(n_epochs=6)
    rows = np.arange(3)
    first = microstates.MicrostateModel.fit(signal, rows=rows)
    altered = signal.data.copy()
    altered[3:] = np.nan
    second = microstates.MicrostateModel.fit(replace(signal, data=altered), rows=rows)
    np.testing.assert_array_equal(first.templates, second.templates)


@requires_sklearn
def test_reference_matching_reorders_states_and_carries_identified_labels() -> None:
    from eegtable import microstates

    signal, _ = _planted()
    model = microstates.MicrostateModel.fit(signal)
    order = np.array([2, 0, 3, 1])
    reference = microstates.MicrostateModel.from_templates(
        -model.templates[order],
        ch_names=signal.ch_names,
        labels=("A", "B", "C", "D"),
        reference_name="identified-reference",
    )
    matched = model.match_reference(reference)
    result = matched.segment(signal, min_duration_ms=0.0)
    original = model.segment(signal, min_duration_ms=0.0)
    expected = np.argsort(order)[original.states]

    np.testing.assert_array_equal(result.states, expected)
    assert result.labels == reference.labels
    assert matched.computation.parameters["reference_name"] == "identified-reference"
    assert model.labels == ("state1", "state2", "state3", "state4")


@requires_sklearn
def test_shared_templates_reject_reordered_channels() -> None:
    from dataclasses import replace

    from eegtable import microstates

    signal, _ = _planted()
    model = microstates.MicrostateModel.fit(signal)
    with pytest.raises(ValueError, match="channels.*order"):
        model.segment(replace(signal, ch_names=signal.ch_names[::-1]))


def test_reference_templates_are_owned_and_spatially_validated() -> None:
    from eegtable import microstates

    templates = np.array([[1.0, -1.0, 0.0], [1.0, 1.0, -2.0]])
    model = microstates.MicrostateModel.from_templates(
        templates,
        ch_names=("F3", "F4", "Cz"),
        labels=("A", "B"),
        reference_name="external-study",
    )
    templates[:] = 0.0
    np.testing.assert_allclose(np.linalg.norm(model.templates, axis=1), 1.0)
    with pytest.raises(ValueError, match="spatial variance"):
        microstates.MicrostateModel.from_templates(
            templates,
            ch_names=("F3", "F4", "Cz"),
            labels=("A", "B"),
            reference_name="external-study",
        )


def test_reference_templates_refuse_complex_topographies() -> None:
    from eegtable import microstates

    templates = np.array([[1.0, -1.0, 0.0], [1.0, 1.0, -2.0]]) + 1j
    with pytest.raises(TypeError, match="real"):
        microstates.MicrostateModel.from_templates(
            templates,
            ch_names=("F3", "F4", "Cz"),
            labels=("A", "B"),
            reference_name="external-study",
        )


def test_direct_microstate_models_refuse_complex_topographies() -> None:
    from eegtable import microstates
    from eegtable.table import ComputationSpec

    templates = np.array([[1.0, -1.0, 0.0], [1.0, 1.0, -2.0]])
    templates /= np.linalg.norm(templates, axis=1, keepdims=True)
    with pytest.raises(TypeError, match="real"):
        microstates.MicrostateModel(
            templates + 1j,
            ch_names=("F3", "F4", "Cz"),
            labels=("A", "B"),
            computation=ComputationSpec.create("provided_microstate_templates"),
        )


@requires_sklearn
def test_segmentation_rejects_negative_peak_separation() -> None:
    signal, _ = _planted()
    with pytest.raises(ValueError, match="min_peak_distance_ms.*non-negative"):
        segment(signal, min_peak_distance_ms=-1.0)
