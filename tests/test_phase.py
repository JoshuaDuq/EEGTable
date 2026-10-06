from dataclasses import replace

import numpy as np
import pytest

import eegtable as ef
from eegtable.bands import Band
from eegtable.phase import itpc, pac, ppc
from eegtable.signal import BandSignal
from eegtable.spectra import Window
from eegtable.table import ComputationSpec, FeatureTable

SFREQ = 100.0
ALPHA, GAMMA = Band("alpha", 8.0, 13.0), Band("gamma", 30.0, 45.0)
WINDOW = Window("all", 0.0, 2.0)


def _from_phase(
    phase: np.ndarray, amplitude: float | np.ndarray = 1.0, band: Band = ALPHA
) -> BandSignal:
    analytic = amplitude * np.exp(1j * phase)
    return BandSignal.from_arrays(
        analytic=analytic,
        times=np.arange(phase.shape[-1]) / SFREQ,
        ch_names=("C3",),
        band=band,
        sfreq=SFREQ,
        row_ids=tuple(("test", index, "event") for index in range(phase.shape[0])),
    )


def test_identical_phase_across_trials_gives_coherence_one() -> None:
    n_epochs, n_times = 20, 201
    ramp = np.linspace(0.0, 8 * np.pi, n_times)
    phase = np.tile(ramp, (n_epochs, 1, 1)).reshape(n_epochs, 1, n_times)
    table = itpc([_from_phase(phase)], windows=[WINDOW], include_global=False)
    assert table.values.item() == pytest.approx(1.0)


def test_trials_are_averaged_before_time_not_after() -> None:
    # Phase sweeps over time but is identical across trials. Averaging trials
    # first gives 1.0; averaging time first would cancel the sweep to near zero.
    n_epochs, n_times = 20, 201
    ramp = np.linspace(0.0, 8 * np.pi, n_times)
    phase = np.tile(ramp, (n_epochs, 1, 1)).reshape(n_epochs, 1, n_times)
    table = itpc([_from_phase(phase)], windows=[WINDOW], include_global=False)
    time_first = np.abs(np.mean(np.exp(1j * ramp)))
    assert table.values.item() == pytest.approx(1.0)
    assert time_first < 0.1


def test_uniformly_random_phase_gives_low_coherence() -> None:
    rng = np.random.RandomState(0)
    phase = rng.uniform(-np.pi, np.pi, (200, 1, 201))
    table = itpc([_from_phase(phase)], windows=[WINDOW], include_global=False)
    # The expected value under the null is about 1/sqrt(N).
    assert table.values.item() < 3.0 / np.sqrt(200)


def test_without_trial_labels_there_is_one_row_named_all() -> None:
    rng = np.random.RandomState(1)
    table = itpc(
        [_from_phase(rng.uniform(-np.pi, np.pi, (8, 1, 201)))],
        windows=[WINDOW],
        include_global=False,
    )
    assert table.row_labels == ("all",)
    assert table.values.shape[0] == 1


def test_trial_labels_give_one_row_per_group_in_sorted_order() -> None:
    n_times = 201
    locked = np.tile(np.linspace(0, 8 * np.pi, n_times), (10, 1, 1)).reshape(10, 1, n_times)
    rng = np.random.RandomState(2)
    scattered = rng.uniform(-np.pi, np.pi, (10, 1, n_times))
    phase = np.concatenate([locked, scattered], axis=0)
    labels = ["locked"] * 10 + ["scattered"] * 10
    table = itpc([_from_phase(phase)], windows=[WINDOW], trials=labels, include_global=False)
    assert table.row_labels == ("locked", "scattered")
    values = dict(zip(table.row_labels, table.values[:, 0], strict=True))
    assert values["locked"] == pytest.approx(1.0)
    assert values["scattered"] < 0.5


def test_a_cross_trial_table_cannot_be_joined_to_a_per_epoch_one() -> None:
    rng = np.random.RandomState(3)
    signal = _from_phase(rng.uniform(-np.pi, np.pi, (4, 1, 201)))
    coherence = itpc([signal], windows=[WINDOW], include_global=False)
    per_epoch = ef.variance([signal], windows=[WINDOW], include_global=False)
    with pytest.raises(ValueError, match="row semantics"):
        ef.concat([coherence, per_epoch])


def test_group_rows_and_epoch_rows_differ_even_at_the_same_count() -> None:
    # One trial group and one epoch happen to give one row each; they still must
    # not be joined, because the rows mean different things.
    rng = np.random.RandomState(9)
    signal = _from_phase(rng.uniform(-np.pi, np.pi, (1, 1, 201)))
    coherence = itpc([signal], windows=[WINDOW], include_global=False)
    per_epoch = ef.variance([signal], windows=[WINDOW], include_global=False)
    assert coherence.n_rows == per_epoch.n_rows == 1
    with pytest.raises(ValueError, match="row semantics"):
        ef.concat([coherence, per_epoch])


def test_mismatched_trial_labels_raise() -> None:
    rng = np.random.RandomState(4)
    signal = _from_phase(rng.uniform(-np.pi, np.pi, (4, 1, 201)))
    with pytest.raises(ValueError, match="one label per epoch"):
        itpc([signal], windows=[WINDOW], trials=["a", "b"])


@pytest.mark.parametrize("measure", [itpc, ppc])
@pytest.mark.parametrize("trials", [[None, None], [1, 1], ["", "locked"], ["locked", 1]])
def test_phase_measures_require_nonempty_string_trial_identities(measure, trials) -> None:
    signal = _from_phase(np.zeros((2, 1, 201)))
    with pytest.raises(ValueError, match="non-empty strings"):
        measure([signal], windows=[WINDOW], trials=trials)


@pytest.mark.parametrize("measure", [itpc, ppc])
@pytest.mark.parametrize("minimum", [2.5, np.nan, np.inf])
def test_phase_measures_require_integer_minimum_trial_counts(measure, minimum) -> None:
    signal = _from_phase(np.zeros((3, 1, 201)))
    with pytest.raises(ValueError, match="integer.*at least 2"):
        measure([signal], windows=[WINDOW], min_valid_trials=minimum)


def test_coherence_is_bounded() -> None:
    rng = np.random.RandomState(5)
    table = itpc(
        [_from_phase(rng.uniform(-np.pi, np.pi, (30, 1, 201)))],
        windows=[WINDOW],
        include_global=False,
    )
    assert 0.0 <= table.values.item() <= 1.0


def test_itpc_requires_at_least_two_valid_trials() -> None:
    phase = np.zeros((1, 1, 201))
    table = itpc([_from_phase(phase)], windows=[WINDOW], include_global=False)
    assert np.isnan(table.values).all()
    assert table.flags["insufficient_trials"].all()


def test_ppc_is_a_distinct_sample_size_unbiased_estimator() -> None:
    rng = np.random.RandomState(15)
    phase = rng.uniform(-np.pi, np.pi, (40, 1, 201))
    table = ppc([_from_phase(phase)], windows=[WINDOW], include_global=False)
    assert table.meta[0].measure == "ppc"
    assert abs(table.values.item()) < 0.05


# --- phase-amplitude coupling ---------------------------------------------------------


def _coupled(strength: float, n_epochs: int = 6, n_times: int = 401) -> tuple:
    times = np.arange(n_times) / SFREQ
    slow_phase = np.tile(2 * np.pi * 5.0 * times, (n_epochs, 1, 1)).reshape(n_epochs, 1, n_times)
    envelope = 1.0 + strength * np.cos(slow_phase)
    slow = _from_phase(slow_phase, band=ALPHA)
    fast = BandSignal.from_arrays(
        analytic=envelope * np.exp(1j * 2 * np.pi * 40.0 * times),
        times=times,
        ch_names=("C3",),
        band=GAMMA,
        sfreq=SFREQ,
        row_ids=slow.row_ids,
    )
    return slow, fast


def test_normalized_coupling_does_not_depend_on_the_amplitude_unit() -> None:
    # Normalized by the summed amplitude, the value is dimensionless; an absolute
    # threshold on that sum withheld an envelope this small instead.
    slow, fast = _coupled(0.5)
    tiny = BandSignal.from_arrays(
        analytic=fast.analytic * 1e-25,
        times=fast.times,
        ch_names=fast.ch_names,
        band=GAMMA,
        sfreq=SFREQ,
        row_ids=fast.row_ids,
    )
    native, rescaled = (
        pac(slow, amplitude, windows=[WINDOW], include_global=False) for amplitude in (fast, tiny)
    )
    np.testing.assert_allclose(rescaled.values, native.values, rtol=1e-12)


def test_coupling_is_higher_when_amplitude_tracks_phase() -> None:
    weak = pac(*_coupled(0.0), windows=[WINDOW], include_global=False)
    strong = pac(*_coupled(0.8), windows=[WINDOW], include_global=False)
    assert strong.values.mean() > weak.values.mean()
    assert weak.values.mean() < 0.05


def test_normalized_coupling_is_invariant_to_overall_amplitude() -> None:
    slow, fast = _coupled(0.8)
    louder = BandSignal.from_arrays(
        analytic=fast.analytic * 1000.0,
        times=fast.times,
        ch_names=fast.ch_names,
        band=fast.band,
        sfreq=fast.sfreq,
        row_ids=fast.row_ids,
    )
    a = pac(slow, fast, windows=[WINDOW], include_global=False)
    b = pac(slow, louder, windows=[WINDOW], include_global=False)
    np.testing.assert_allclose(a.values, b.values, rtol=1e-12)


def test_unnormalized_coupling_scales_with_amplitude() -> None:
    slow, fast = _coupled(0.8)
    louder = BandSignal.from_arrays(
        analytic=fast.analytic * 10.0,
        times=fast.times,
        ch_names=fast.ch_names,
        band=fast.band,
        sfreq=fast.sfreq,
        row_ids=fast.row_ids,
    )
    a = pac(slow, fast, windows=[WINDOW], normalize=False, include_global=False)
    b = pac(slow, louder, windows=[WINDOW], normalize=False, include_global=False)
    np.testing.assert_allclose(b.values, a.values * 10.0, rtol=1e-10)


def test_coupling_has_one_row_per_epoch() -> None:
    slow, fast = _coupled(0.5, n_epochs=6)
    table = pac(slow, fast, windows=[WINDOW], include_global=False)
    assert table.values.shape[0] == 6
    assert table.row_labels is None


def test_the_amplitude_band_is_recorded_on_the_column() -> None:
    slow, fast = _coupled(0.5)
    table = pac(slow, fast, windows=[WINDOW], include_global=False)
    assert table.meta[0].band is GAMMA


def test_both_pac_bands_are_first_class_metadata_and_names() -> None:
    slow, fast = _coupled(0.5)
    table = pac(slow, fast, windows=[WINDOW], include_global=False)
    assert table.meta[0].phase_band is ALPHA
    assert table.meta[0].amplitude_band is GAMMA
    assert "phase-alpha" in table.names[0]
    assert "amp-gamma" in table.names[0]


def test_pac_pairs_with_the_same_amplitude_band_do_not_collide() -> None:
    slow, fast = _coupled(0.5)
    theta = Band("theta", 4.0, 7.0)
    other = BandSignal.from_arrays(
        analytic=slow.analytic,
        times=slow.times,
        ch_names=slow.ch_names,
        band=theta,
        sfreq=slow.sfreq,
        row_ids=slow.row_ids,
    )
    joined = ef.concat(
        [
            pac(slow, fast, windows=[WINDOW], include_global=False),
            pac(other, fast, windows=[WINDOW], include_global=False),
        ]
    )
    assert len(set(joined.names)) == 2


def test_a_phase_band_faster_than_the_amplitude_band_raises() -> None:
    slow, fast = _coupled(0.5)
    with pytest.raises(ValueError, match="must be slower"):
        pac(fast, slow, windows=[WINDOW], include_global=False)


def test_mismatched_channels_or_times_raise() -> None:
    slow, fast = _coupled(0.5)
    renamed = BandSignal.from_arrays(
        analytic=fast.analytic,
        times=fast.times,
        ch_names=("Cz",),
        band=fast.band,
        sfreq=fast.sfreq,
        row_ids=fast.row_ids,
    )
    with pytest.raises(ValueError, match="same channels"):
        pac(slow, renamed, windows=[WINDOW])
    shifted = BandSignal.from_arrays(
        analytic=fast.analytic,
        times=fast.times + 9.0,
        ch_names=fast.ch_names,
        band=fast.band,
        sfreq=fast.sfreq,
        row_ids=fast.row_ids,
    )
    with pytest.raises(ValueError, match="same time axis"):
        pac(slow, shifted, windows=[WINDOW])


def test_pac_rejects_a_sample_shift_regardless_of_time_origin() -> None:
    slow, fast = _coupled(0.5)
    slow = replace(slow, times=slow.times + 1000.0)
    fast = replace(fast, times=fast.times + 1000.0 + 1.0 / SFREQ)

    with pytest.raises(ValueError, match="same time axis"):
        pac(slow, fast, windows=[Window("all", -np.inf, np.inf)])


def test_pac_requires_the_same_sampling_frequency() -> None:
    slow, fast = _coupled(0.5)
    fast = replace(fast, sfreq=fast.sfreq * (1.0 + 5e-8))

    with pytest.raises(ValueError, match="same sampling frequency"):
        pac(slow, fast, windows=[WINDOW])


def test_mismatched_pac_epochs_sampling_or_row_identity_raise() -> None:
    slow, fast = _coupled(0.5)
    fewer = BandSignal.from_arrays(
        analytic=fast.analytic[:-1],
        times=fast.times,
        ch_names=fast.ch_names,
        band=fast.band,
        sfreq=fast.sfreq,
        row_ids=fast.row_ids[:-1],
    )
    with pytest.raises(ValueError, match="same shape"):
        pac(slow, fewer, windows=[WINDOW])
    wrong_rate = BandSignal.from_arrays(
        analytic=fast.analytic,
        times=fast.times[0] + np.arange(fast.times.size) / (fast.sfreq * 2.0),
        ch_names=fast.ch_names,
        band=fast.band,
        sfreq=fast.sfreq * 2.0,
        row_ids=fast.row_ids,
    )
    with pytest.raises(ValueError, match="same time axis"):
        pac(slow, wrong_rate, windows=[WINDOW])
    reordered = BandSignal.from_arrays(
        analytic=fast.analytic,
        times=fast.times,
        ch_names=fast.ch_names,
        band=fast.band,
        sfreq=fast.sfreq,
        row_ids=tuple(reversed(fast.row_ids)),
    )
    with pytest.raises(ValueError, match="row identities"):
        pac(slow, reordered, windows=[WINDOW])


def test_overlapping_pac_bands_are_rejected_unless_explicitly_allowed() -> None:
    slow, _ = _coupled(0.5)
    overlap = Band("overlap", 12.0, 30.0)
    amplitude = BandSignal.from_arrays(
        analytic=slow.analytic,
        times=slow.times,
        ch_names=slow.ch_names,
        band=overlap,
        sfreq=slow.sfreq,
        row_ids=slow.row_ids,
    )
    with pytest.raises(ValueError, match="overlap"):
        pac(slow, amplitude, windows=[WINDOW])
    assert (
        pac(
            slow, amplitude, windows=[WINDOW], allow_overlap=True, include_global=False
        ).values.shape[0]
        == slow.n_epochs
    )


def test_a_silent_amplitude_band_yields_nan_rather_than_zero() -> None:
    slow, fast = _coupled(0.0)
    silent = BandSignal.from_arrays(
        analytic=np.zeros_like(fast.analytic),
        times=fast.times,
        ch_names=fast.ch_names,
        band=fast.band,
        sfreq=fast.sfreq,
        row_ids=fast.row_ids,
    )
    table = pac(slow, silent, windows=[WINDOW], include_global=False)
    assert np.isnan(table.values).all()


def test_the_result_is_a_feature_table() -> None:
    slow, fast = _coupled(0.5)
    assert isinstance(pac(slow, fast, windows=[WINDOW]), FeatureTable)


@pytest.mark.parametrize("measure", [itpc, ppc])
def test_flatlines_do_not_imply_perfect_phase_locking(measure) -> None:
    signal = _from_phase(np.zeros((4, 1, 201)), amplitude=0.0)
    table = measure([signal], windows=[WINDOW], include_global=False)
    assert np.isnan(table.values).all()
    assert table.flags["insufficient_trials"].all()


@pytest.mark.parametrize("measure", [itpc, ppc])
def test_phase_coverage_excludes_samples_with_undefined_phase(measure) -> None:
    amplitude = np.ones((4, 1, 201))
    amplitude[..., :50] = 0.0
    signal = _from_phase(np.zeros_like(amplitude), amplitude=amplitude)

    table = measure([signal], windows=[WINDOW], include_global=False)

    assert table.values.item() == pytest.approx(1.0)
    assert table.coverage.item() == pytest.approx(151 / 201)


@pytest.mark.parametrize("normalize", [True, False])
def test_pac_requires_a_defined_phase(normalize) -> None:
    slow = _from_phase(np.zeros((4, 1, 201)), amplitude=0.0)
    fast = _from_phase(np.zeros((4, 1, 201)), band=GAMMA)
    table = pac(slow, fast, windows=[WINDOW], normalize=normalize, include_global=False)
    assert np.isnan(table.values).all()


def test_pac_coverage_counts_only_jointly_valid_phase_and_amplitude() -> None:
    slow, fast = _coupled(0.5, n_times=201)
    phase = slow.analytic.copy()
    phase[..., :50] = 0.0
    amplitude = fast.analytic.copy()
    amplitude[..., 50:75] = np.nan
    slow = replace(slow, analytic=phase, coverage=np.full_like(slow.coverage, 0.8))
    fast = replace(fast, analytic=amplitude, coverage=np.full_like(fast.coverage, 0.9))

    table = pac(slow, fast, windows=[WINDOW], include_global=False)

    assert np.isfinite(table.values).all()
    np.testing.assert_allclose(table.coverage, 0.8 * (201 - 75) / 201)


def test_pac_records_the_phase_estimator_as_well_as_the_amplitude_estimator() -> None:
    slow, fast = _coupled(0.5)
    other = replace(slow, computation=ComputationSpec.create("hilbert", pad_sec=2.0))

    first = pac(slow, fast, windows=[WINDOW], include_global=False)
    second = pac(other, fast, windows=[WINDOW], include_global=False)

    assert first.names != second.names
    assert (
        second.meta[0].computation.parameters["parameters"]["phase_input_computation"]
        == other.computation.record()
    )


def _irregular_coupling() -> tuple[BandSignal, BandSignal]:
    rng = np.random.default_rng(53)
    n_times = 601
    increments = rng.uniform(0.0, 0.8, (3, 2, n_times))
    phase = np.cumsum(increments, axis=-1)
    amplitude = 1.0 + 0.8 * np.cos(phase)
    times = np.arange(n_times) / SFREQ
    identities = tuple(("test", index, "event") for index in range(3))
    common = dict(times=times, ch_names=("C3", "C4"), sfreq=SFREQ, row_ids=identities)
    slow = BandSignal.from_arrays(analytic=np.exp(1j * phase), band=ALPHA, **common)
    fast = BandSignal.from_arrays(analytic=amplitude.astype(complex), band=GAMMA, **common)
    return slow, fast


def test_pac_surrogates_matches_tensorpac_null_and_preserves_epoch_rows() -> None:
    pytest.importorskip("tensorpac")
    from tensorpac.methods import mean_vector_length, swap_blocks

    import eegtable.phase as phase_methods

    slow, fast = _irregular_coupling()
    table = phase_methods.pac_surrogates(
        slow,
        fast,
        windows=[WINDOW],
        n_surrogates=32,
        surrogate="blocks",
        random_state=7,
        min_shift_seconds=0.0,
        correction="bonferroni",
        include_global=False,
    )
    keep = slow.times <= WINDOW.tmax
    phase = slow.phase[:, :, keep]
    amplitude = fast.envelope[:, :, keep]
    rng = np.random.default_rng(7)
    null = []
    for _ in range(32):
        seed = int(rng.integers(0, 2**32 - 1))
        shifted_phase, shifted_amplitude = swap_blocks(phase, amplitude, random_state=seed)
        normalized = shifted_amplitude / shifted_amplitude.mean(axis=-1, keepdims=True)
        null.append(mean_vector_length(shifted_phase[None], normalized[None])[0, 0])
    null = np.stack(null)
    observed = pac(slow, fast, windows=[WINDOW], include_global=False).values
    np.testing.assert_allclose(table.select(measure="pac").values, observed)
    np.testing.assert_allclose(table.select(measure="pac_null_mean").values, null.mean(axis=0))
    np.testing.assert_allclose(table.select(measure="pac_null_std").values, null.std(axis=0))
    np.testing.assert_allclose(
        table.select(measure="pac_corrected").values,
        observed - null.mean(axis=0),
    )
    expected_p = (1 + np.sum(null >= observed, axis=0)) / 33
    np.testing.assert_allclose(table.select(measure="pac_pvalue").values, expected_p)
    np.testing.assert_allclose(
        table.select(measure="pac_pvalue_adjusted").values,
        np.minimum(1, 2 * expected_p),
    )
    assert table.row_labels is None
    assert table.row_ids == slow.row_ids
    assert all(meta.phase_band == ALPHA and meta.amplitude_band == GAMMA for meta in table.meta)


@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("surrogate", ["blocks", "circular"])
def test_constant_amplitude_surrogates_do_not_report_significant_pac(normalize, surrogate) -> None:
    pytest.importorskip("tensorpac")
    from tensorpac.methods import mean_vector_length

    from eegtable.phase import pac_surrogates

    times = np.arange(100) / SFREQ
    phase = np.random.default_rng(0).uniform(-np.pi, np.pi, (1, 1, times.size))
    common = dict(times=times, ch_names=("C3",), sfreq=SFREQ, row_ids=(("test", 0, "event"),))
    slow = BandSignal.from_arrays(analytic=np.exp(1j * phase), band=ALPHA, **common)
    fast = BandSignal.from_arrays(analytic=np.ones_like(phase, dtype=complex), band=GAMMA, **common)
    table = pac_surrogates(
        slow,
        fast,
        windows=[Window("all", -np.inf, np.inf)],
        n_surrogates=20,
        surrogate=surrogate,
        min_shift_seconds=0.0,
        include_global=False,
        normalize=normalize,
    )
    expected = mean_vector_length(slow.phase[None], fast.envelope[None])[0, 0]
    np.testing.assert_array_equal(table.select(measure="pac").values, expected)
    assert table.select(measure="pac_null_std").values.item() == 0.0
    assert table.select(measure="pac_corrected").values.item() == 0.0
    assert table.select(measure="pac_pvalue").values.item() == 1.0
    assert table.select(measure="pac_pvalue_adjusted").values.item() == 1.0
    zscore = table.select(measure="pac_zscore")
    assert np.isnan(zscore.values).all()
    assert zscore.flags["degenerate_null"].all()


def test_pac_maxstat_counts_floating_point_ties_like_scipy() -> None:
    from eegtable.phase import _adjust_pac_pvalues

    observed = np.ones((1, 2))
    null = np.full((20, 1, 2), np.nextafter(1.0, 0.0))
    adjusted = _adjust_pac_pvalues(observed, null, np.ones_like(observed), "maxstat")
    np.testing.assert_array_equal(adjusted, 1.0)
    null[:] = 1.0 - 1e-10
    adjusted = _adjust_pac_pvalues(observed, null, np.ones_like(observed), "maxstat")
    np.testing.assert_array_equal(adjusted, 1 / 21)


@pytest.mark.parametrize("surrogate", ["blocks", "circular"])
def test_pac_surrogates_is_seeded_and_handles_spatial_null_before_inference(surrogate) -> None:
    pytest.importorskip("tensorpac")
    import eegtable.phase as phase_methods

    slow, fast = _irregular_coupling()
    kwargs = dict(
        windows=[WINDOW],
        n_surrogates=40,
        surrogate=surrogate,
        random_state=11,
        correction="fdr",
        groups={"motor": ["C3", "C4"]},
    )
    first = phase_methods.pac_surrogates(slow, fast, **kwargs)
    second = phase_methods.pac_surrogates(slow, fast, **kwargs)
    np.testing.assert_array_equal(first.values, second.values)
    assert set(meta.space for meta in first.meta) == {"motor", "global"}
    adjusted = first.select(measure="pac_pvalue_adjusted").values
    assert np.all((adjusted >= 1 / 41) & (adjusted <= 1))
    assert first.meta[0].computation.parameters["inference"]["correction_family"] == (
        "nodes_and_windows_within_each_epoch"
    )


@pytest.mark.parametrize(
    "parameters, message",
    [
        ({"n_surrogates": 1}, "n_surrogates"),
        ({"n_surrogates": True}, "n_surrogates"),
        ({"surrogate": "shuffle"}, "surrogate"),
        ({"random_state": -1}, "random_state"),
        ({"min_shift_seconds": 1.1}, "shift"),
        ({"correction": "wrong"}, "correction"),
    ],
)
def test_pac_surrogates_validates_inference_parameters(parameters, message) -> None:
    pytest.importorskip("tensorpac")
    import eegtable.phase as phase_methods

    slow, fast = _irregular_coupling()
    with pytest.raises(ValueError, match=message):
        phase_methods.pac_surrogates(slow, fast, windows=[WINDOW], **parameters)


def test_pac_surrogates_refuses_nonfinite_samples() -> None:
    pytest.importorskip("tensorpac")
    import eegtable.phase as phase_methods

    slow, fast = _irregular_coupling()
    analytic = fast.analytic.copy()
    analytic[0, 0, 10] = np.nan
    with pytest.raises(ValueError, match="finite"):
        phase_methods.pac_surrogates(slow, replace(fast, analytic=analytic), windows=[WINDOW])


@pytest.mark.parametrize("correction", ["none", "fdr", "maxstat"])
def test_pac_adjustment_matches_the_prespecified_family(correction) -> None:
    pytest.importorskip("tensorpac")
    from scipy.stats import false_discovery_control
    from tensorpac.methods import mean_vector_length, swap_blocks

    import eegtable.phase as phase_methods

    slow, fast = _irregular_coupling()
    table = phase_methods.pac_surrogates(
        slow,
        fast,
        windows=[WINDOW],
        n_surrogates=24,
        random_state=13,
        min_shift_seconds=0.0,
        include_global=False,
        correction=correction,
    )
    observed = table.select(measure="pac").values
    pvalues = table.select(measure="pac_pvalue").values
    if correction == "fdr":
        expected = false_discovery_control(pvalues, axis=1)
    elif correction == "none":
        expected = pvalues
    else:
        keep = slow.times <= WINDOW.tmax
        phase, amplitude = slow.phase[:, :, keep], fast.envelope[:, :, keep]
        rng = np.random.default_rng(13)
        maxima = []
        for _ in range(24):
            phase_null, amplitude_null = swap_blocks(
                phase,
                amplitude,
                random_state=int(rng.integers(0, 2**32 - 1)),
            )
            normalized = amplitude_null / amplitude_null.mean(axis=-1, keepdims=True)
            maxima.append(mean_vector_length(phase_null[None], normalized[None])[0, 0].max(axis=1))
        expected = (1 + (np.stack(maxima)[:, :, None] >= observed).sum(axis=0)) / 25
    np.testing.assert_allclose(table.select(measure="pac_pvalue_adjusted").values, expected)


def test_pac_surrogate_inference_computes_roi_null_before_statistics() -> None:
    pytest.importorskip("tensorpac")
    from tensorpac.methods import mean_vector_length, swap_blocks

    import eegtable.phase as phase_methods

    slow, fast = _irregular_coupling()
    windows = [Window("early", 0.0, 2.0), Window("late", 2.1, 4.0)]
    table = phase_methods.pac_surrogates(
        slow,
        fast,
        windows=windows,
        groups={"motor": ["C3", "C4"]},
        include_global=False,
        n_surrogates=20,
        random_state=19,
        min_shift_seconds=0.0,
    )
    rng = np.random.default_rng(19)
    null = []
    for window in windows:
        keep = (slow.times >= window.tmin) & (slow.times <= window.tmax)
        phase, amplitude = slow.phase[:, :, keep], fast.envelope[:, :, keep]
        permutations = []
        for _ in range(20):
            phase_null, amplitude_null = swap_blocks(
                phase,
                amplitude,
                random_state=int(rng.integers(0, 2**32 - 1)),
            )
            normalized = amplitude_null / amplitude_null.mean(axis=-1, keepdims=True)
            permutations.append(
                mean_vector_length(phase_null[None], normalized[None])[0, 0].mean(axis=1)
            )
        null.append(np.stack(permutations))
    null = np.stack(null, axis=-1)
    np.testing.assert_allclose(table.select(measure="pac_null_std").values, null.std(axis=0))
