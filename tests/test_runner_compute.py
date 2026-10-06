"""The runner computes, per recording, what the recipe asks for.

Expected values come from the simulated signal, not from the library: a 10 Hz
sine locked to every epoch has a known peak, a known variance and no event-related
power change.
"""

import json
from dataclasses import replace

import mne
import numpy as np
import pandas as pd
import pytest

import eegtable as ef
from eegtable.runner import load_recipe
from eegtable.runner.compute import compute_features
from tests.synthetic import AMPLITUDE, NOISE, SFREQ, make_epochs

HEAD = """
[inputs]
root = "data"

[output]
root = "out"

"""


@pytest.mark.parametrize("labels", [[1, "1", 2, "2"], [1, "1", 1.0, 1.0], [1, 1.0, 2, 2.0]])
def test_trial_group_labels_cannot_merge_distinct_metadata_values(labels) -> None:
    from eegtable.runner.compute import trial_labels

    epochs = make_epochs(n_epochs=4)
    epochs.metadata = pd.DataFrame({"condition": labels}, dtype=object)
    recipe = load_recipe(
        {
            "trials": {"by": "metadata", "column": "condition"},
            "features": [{"measure": "itpc", "bands": ["alpha"]}],
        }
    )
    with pytest.raises(ValueError, match="distinct.*label"):
        trial_labels(epochs, recipe)


def features(tmp_path, body: str, epochs: mne.EpochsArray | None = None):
    path = tmp_path / "recipe.toml"
    path.write_text(HEAD + body, encoding="utf-8")
    return compute_features(
        make_epochs() if epochs is None else epochs,
        load_recipe(path),
        recording="sub-test_task-test",
    )


def test_per_epoch_measures_give_one_row_per_epoch(tmp_path) -> None:
    result = features(tmp_path, '[[features]]\nmeasure = "integrated_band_power"\n')

    assert result.epochs is not None and result.epochs.n_rows == 12
    assert result.crosstrial is None


def test_peak_frequency_finds_the_simulated_oscillation(tmp_path) -> None:
    # Smoothing is off: its default even-width kernel shifts peaks by half a bin.
    result = features(
        tmp_path,
        '[[features]]\nmeasure = "peak_frequency"\nbands = ["alpha"]\nspatial = ["global"]\n'
        "smoothing_hz = 0.0\n",
    )

    assert result.epochs is not None
    np.testing.assert_allclose(result.epochs.values, 10.0, atol=0.05)


def test_band_power_puts_the_oscillation_in_its_band(tmp_path) -> None:
    result = features(
        tmp_path, '[[features]]\nmeasure = "integrated_band_power"\nspatial = ["global"]\n'
    )

    assert result.epochs is not None
    alpha = result.epochs.select(band=ef.Band("alpha", 8.0, 13.0)).values
    theta = result.epochs.select(band=ef.Band("theta", 4.0, 8.0)).values
    assert np.all(alpha > 100 * theta)


@pytest.mark.parametrize("sfreq", [100.0, 200.0])
def test_multitaper_power_has_physical_units_independent_of_sampling_rate(tmp_path, sfreq):
    times = np.arange(int(4 * sfreq)) / sfreq
    amplitude = 2e-6
    data = np.tile(amplitude * np.sin(2 * np.pi * 10 * times), (3, 1, 1))
    epochs = mne.EpochsArray(data, mne.create_info(["Cz"], sfreq, "eeg"), verbose=False)
    result = features(
        tmp_path,
        '[spectra]\nmethod = "multitaper"\nbandwidth = 1.0\n'
        '[[features]]\nmeasure = "integrated_band_power"\n'
        'bands = ["alpha"]\nspatial = ["global"]\n',
        epochs,
    )
    np.testing.assert_allclose(result.epochs.values, amplitude**2 / 2, rtol=0.02, atol=0)


def test_spatial_levels_choose_the_columns(tmp_path) -> None:
    result = features(
        tmp_path,
        '[rois]\nfrontal = ["Fz", "F3", "F4"]\n\n'
        '[[features]]\nmeasure = "integrated_band_power"\n'
        'bands = ["alpha"]\nspatial = ["rois", "global"]\n',
    )

    assert result.epochs is not None
    assert [m.space for m in result.epochs.meta] == ["frontal", "global"]


def test_broadband_variance_is_that_of_the_simulated_signal(tmp_path) -> None:
    result = features(tmp_path, '[[features]]\nmeasure = "variance"\nspatial = ["global"]\n')

    # A sine of amplitude A has variance A^2 / 2; the noise adds its own variance.
    expected = AMPLITUDE**2 / 2 + NOISE**2
    assert result.epochs is not None
    np.testing.assert_allclose(result.epochs.values, expected, rtol=0.05)


def test_windows_are_measured_separately(tmp_path) -> None:
    result = features(
        tmp_path,
        "[windows]\nearly = [0.0, 0.5]\nlate = [0.5, 1.0]\n\n"
        '[[features]]\nmeasure = "variance"\nspatial = ["global"]\n',
    )

    assert result.epochs is not None
    assert [m.window for m in result.epochs.meta] == ["early", "late"]


def test_stationary_signal_shows_no_event_related_power_change(tmp_path) -> None:
    result = features(
        tmp_path,
        "[windows]\nbase = [-0.5, 0.0]\nstim = [0.25, 1.25]\n\n"
        '[[features]]\nmeasure = "erds_mean"\nbands = ["alpha"]\nbaseline = "base"\n'
        'spatial = ["global"]\n',
    )

    assert result.epochs is not None
    np.testing.assert_allclose(result.epochs.values, 0.0, atol=10.0)


def test_welch_spectra_are_computed_per_window(tmp_path) -> None:
    result = features(
        tmp_path,
        "[windows]\nbase = [-0.5, 0.0]\nstim = [0.0, 1.0]\n\n"
        '[[features]]\nmeasure = "integrated_band_power"\n'
        'bands = ["alpha"]\nspatial = ["global"]\n',
    )

    assert result.epochs is not None
    assert [m.window for m in result.epochs.meta] == ["base", "stim"]
    # Both windows hold the same stationary sine, so their power agrees.
    base, stim = result.epochs.values.T
    np.testing.assert_allclose(base, stim, rtol=0.35)


def test_morlet_spectra_restrict_each_window_to_its_support(tmp_path) -> None:
    # A 1 Hz wavelet outlasts a 2 s epoch, so the grid starts at 4 Hz.
    result = features(
        tmp_path,
        '[spectra]\nmethod = "morlet"\nfmin = 4.0\nn_freqs = 20\n\n'
        "[windows]\nstim = [0.0, 1.0]\n\n"
        '[[features]]\nmeasure = "mean_tfr_power"\nbands = ["theta", "alpha", "beta"]\n'
        'spatial = ["global"]\n',
    )

    assert result.epochs is not None
    assert {m.source for m in result.epochs.meta} == {"morlet"}
    alpha = result.epochs.select(band=ef.Band("alpha", 8.0, 13.0)).values
    beta = result.epochs.select(band=ef.Band("beta", 13.0, 30.0)).values
    assert np.all(alpha > 10 * beta)


def test_periodic_power_measures_the_oscillation_against_the_noise_floor(tmp_path) -> None:
    # White noise is a flat power law, which the fit takes as the floor; the 10 Hz sine
    # stands above it and the beta band does not.
    result = features(
        tmp_path,
        '[spectra]\nmethod = "morlet"\nfmin = 4.0\nn_freqs = 20\n\n'
        "[windows]\nstim = [0.0, 1.0]\n\n"
        '[[features]]\nmeasure = "periodic_power"\nbands = ["alpha", "beta"]\n'
        'fit_range = [4.0, 40.0]\nspatial = ["global"]\n',
    )

    assert result.epochs is not None
    assert {m.measure for m in result.epochs.meta} == {"periodic_power"}
    alpha = result.epochs.select(band=ef.Band("alpha", 8.0, 13.0)).values
    beta = result.epochs.select(band=ef.Band("beta", 13.0, 30.0)).values
    assert np.all(alpha > 2.0 * beta)
    assert 0.5 < np.median(beta) < 2.0
    fit = json.loads(result.epochs.meta[0].computation.parameters_json)["input_computation"]
    assert fit["parameters"]["fit_range"] == [4.0, 40.0]


def test_baseline_normalized_power_consumes_the_baseline(tmp_path) -> None:
    result = features(
        tmp_path,
        "[windows]\nbase = [-0.5, 0.0]\nstim = [0.0, 1.0]\n\n"
        '[[features]]\nmeasure = "integrated_band_power"\nbands = ["alpha"]\nbaseline = "base"\n'
        'normalize = "db"\nspatial = ["global"]\n',
    )

    assert result.epochs is not None
    assert [m.window for m in result.epochs.meta] == ["stim"]
    np.testing.assert_allclose(result.epochs.values, 0.0, atol=1.5)


def test_band_ratios_and_asymmetry_follow_band_power(tmp_path) -> None:
    result = features(
        tmp_path,
        '[[features]]\nmeasure = "integrated_band_power"\nbands = ["theta", "alpha"]\n'
        'ratios = [["alpha", "theta"]]\nasymmetry = [["F3", "F4"]]\n',
    )

    assert result.epochs is not None
    measures = {m.measure for m in result.epochs.meta}
    assert {"band_power", "ratio_alpha_theta", "asymmetry"} <= measures


def test_cross_trial_measures_have_one_row_per_trial_group(tmp_path) -> None:
    result = features(
        tmp_path,
        '[trials]\nby = "event"\n\n'
        '[[features]]\nmeasure = "itpc"\nbands = ["alpha"]\nspatial = ["global"]\n',
    )

    assert result.epochs is None
    assert result.crosstrial is not None
    assert result.crosstrial.row_labels == ("left", "right")
    # The sine has the same phase on every trial.
    assert np.all(result.crosstrial.values > 0.9)


def test_trials_can_be_grouped_by_a_metadata_column(tmp_path) -> None:
    result = features(
        tmp_path,
        '[trials]\nby = "metadata"\ncolumn = "rating"\n\n'
        '[[features]]\nmeasure = "itpc"\nbands = ["alpha"]\nspatial = ["global"]\n',
    )

    assert result.crosstrial is not None
    assert result.crosstrial.row_labels == ("0", "1", "2", "3", "4")


def test_graph_summaries_join_the_connectivity_table(tmp_path) -> None:
    result = features(
        tmp_path,
        '[[features]]\nmeasure = "envelope_correlation"\nbands = ["alpha"]\n'
        'graph = ["global_efficiency", "clustering_coefficient"]\nclustering_threshold = 0.1\n',
    )

    assert result.crosstrial is not None
    measures = {m.measure for m in result.crosstrial.meta}
    assert measures == {"aec", "global_efficiency", "clustering"}


def test_pac_is_computed_for_each_pair(tmp_path) -> None:
    result = features(
        tmp_path,
        '[[features]]\nmeasure = "pac"\npairs = [["theta", "gamma"]]\nspatial = ["global"]\n',
    )

    assert result.epochs is not None
    assert [m.measure for m in result.epochs.meta] == ["pac"]


def test_microstate_coverage_sums_to_one_in_every_window(tmp_path) -> None:
    pytest.importorskip("sklearn")
    result = features(
        tmp_path,
        '[microstates]\nn_states = 3\n\n[[features]]\nmeasure = "microstate_coverage"\n',
    )

    assert result.epochs is not None
    np.testing.assert_allclose(result.epochs.values.sum(axis=1), 1.0)


def test_recipe_parameters_reach_the_library(tmp_path) -> None:
    # Channel columns: the global column averages channels after the log, not before.
    raw = features(
        tmp_path, '[[features]]\nmeasure = "integrated_band_power"\nspatial = ["channels"]\n'
    )
    logged = features(
        tmp_path,
        '[[features]]\nmeasure = "integrated_band_power"\n'
        'spatial = ["channels"]\nnormalize = "log10"\n',
    )

    assert raw.epochs is not None and logged.epochs is not None
    np.testing.assert_allclose(logged.epochs.values, np.log10(raw.epochs.values))


def test_window_outside_the_epoch_is_an_error(tmp_path) -> None:
    with pytest.raises(ValueError, match="late"):
        features(tmp_path, '[windows]\nlate = [1.0, 3.0]\n\n[[features]]\nmeasure = "variance"\n')


def test_roi_naming_an_absent_channel_is_an_error(tmp_path) -> None:
    with pytest.raises(KeyError, match="Oz"):
        features(
            tmp_path,
            '[rois]\nback = ["Pz", "Oz"]\n\n'
            '[[features]]\nmeasure = "integrated_band_power"\nspatial = ["rois"]\n',
        )


def test_each_entry_is_reported_as_a_step(tmp_path) -> None:
    path = tmp_path / "recipe.toml"
    path.write_text(
        HEAD
        + '[[features]]\nmeasure = "integrated_band_power"\n\n[[features]]\nmeasure = "variance"\n',
        encoding="utf-8",
    )
    steps: list[tuple[str, int, int]] = []

    compute_features(
        make_epochs(),
        load_recipe(path),
        recording="sub-test_task-test",
        on_step=lambda *s: steps.append(s),
    )

    assert steps == [("integrated_band_power", 1, 2), ("variance", 2, 2)]


def test_multitaper_spectra_span_the_whole_epoch_by_default(tmp_path) -> None:
    result = features(
        tmp_path,
        '[spectra]\nmethod = "multitaper"\n\n'
        '[[features]]\nmeasure = "integrated_band_power"\nspatial = ["global"]\n',
    )

    assert result.epochs is not None
    assert {m.source for m in result.epochs.meta} == {"multitaper"}
    assert {m.window for m in result.epochs.meta} == {"all"}
    # Multitaper resolves sfreq / n_samples over the whole epoch; welch would give
    # sfreq / n_fft with its 500-sample segments.
    resolutions = [m.freq_resolution_hz for m in result.epochs.meta]
    assert resolutions == pytest.approx([SFREQ / 501] * len(resolutions))


def test_multitaper_windows_measured_together_must_be_equally_long(tmp_path) -> None:
    with pytest.raises(ValueError, match="equally long"):
        features(
            tmp_path,
            '[spectra]\nmethod = "multitaper"\nbandwidth = 3.0\n\n'
            "[windows]\nbase = [-0.5, 0.0]\nstim = [0.0, 1.0]\n\n"
            '[[features]]\nmeasure = "integrated_band_power"\n',
        )


def test_a_multitaper_bandwidth_too_narrow_for_its_window_is_an_error(tmp_path) -> None:
    # 2 Hz is 1.01 frequency bins of a 0.5 s window: MNE would fall back, with only a
    # warning, to a single taper that keeps 79% of its power in the band.
    with pytest.raises(ValueError, match="1.35 frequency bins.*window 'base'"):
        features(
            tmp_path,
            '[spectra]\nmethod = "multitaper"\n\n'
            "[windows]\nbase = [-0.5, 0.0]\n\n"
            '[[features]]\nmeasure = "integrated_band_power"\n',
        )


def test_log_frequency_grid_is_correctly_rounded() -> None:
    # The grid is hashed into every Morlet column name, so it must be the same bits on every
    # platform. np.geomspace goes through libm pow and differs by a few ulp between macOS and
    # Linux; the correctly rounded value is the one answer they can all agree on.
    from decimal import Decimal, localcontext

    from eegtable.runner.compute import _log_grid

    grid = _log_grid(4.0, 45.0, 20)
    with localcontext() as context:
        context.prec = 80
        ratio = Decimal(45) / Decimal(4)
        expected = [float(Decimal(4) * ratio ** (Decimal(k) / Decimal(19))) for k in range(20)]
    assert grid.tolist() == expected
    assert grid[0] == 4.0 and grid[-1] == 45.0
    # One of the values libm rounds wrongly on macOS arm64 (geomspace gives ...505).
    assert grid[3] == 5.861806015358504
    assert _log_grid(8.0, 8.0, 1).tolist() == [8.0]


WELCH_ALPHA = '[[features]]\nmeasure = "integrated_band_power"\nbands = ["alpha"]\n'


def test_welch_columns_ignore_settings_only_other_methods_read(tmp_path) -> None:
    # Morlet's cycles and multitaper's bandwidth never touch a Welch estimate; if they named
    # its columns, changing one of their defaults would rename every Welch feature.
    path = tmp_path / "recipe.toml"
    path.write_text(HEAD + WELCH_ALPHA, encoding="utf-8")
    recipe = load_recipe(path)
    others = replace(
        recipe,
        spectra=replace(recipe.spectra, n_freqs=60, min_cycles=4.0, bandwidth=3.0, decim=2),
    )
    tables = [
        compute_features(make_epochs(), r, recording="sub-test_task-test").epochs
        for r in (recipe, others)
    ]
    assert tables[0] is not None and tables[1] is not None
    assert tables[0].names == tables[1].names


def test_a_setting_welch_reads_still_names_its_columns(tmp_path) -> None:
    default = features(tmp_path, WELCH_ALPHA)
    overlapped = features(tmp_path, "[spectra]\nn_overlap = 10\n\n" + WELCH_ALPHA)
    assert default.epochs is not None and overlapped.epochs is not None
    assert set(default.epochs.names).isdisjoint(overlapped.epochs.names)


def test_default_spectral_recipe_column_names(tmp_path) -> None:
    # Names include the global channels and frequency grid, plus retained Morlet times.
    welch = features(
        tmp_path,
        "[windows]\nbase = [-0.5, 0.0]\nstim = [0.0, 1.0]\n\n"
        '[[features]]\nmeasure = "integrated_band_power"\n'
        'bands = ["alpha"]\nspatial = ["global"]\n',
    )
    morlet = features(
        tmp_path,
        '[spectra]\nmethod = "morlet"\nfmin = 4.0\nn_freqs = 20\n\n'
        "[windows]\nstim = [0.0, 1.0]\n\n"
        '[[features]]\nmeasure = "mean_tfr_power"\nbands = ["alpha"]\nspatial = ["global"]\n',
    )

    assert welch.epochs is not None and morlet.epochs is not None
    assert list(welch.epochs.to_dataframe().columns) == [
        "eeg_band-power_alpha_global_base_raw_p0baba8bb7201",
        "eeg_band-power_alpha_global_stim_raw_p2f519bfa8753",
    ]
    assert list(morlet.epochs.to_dataframe().columns) == [
        "eeg_mean-tfr-power_alpha_global_stim_raw_pb44a1a4fa6b1"
    ]
    for table in (welch.epochs, morlet.epochs):
        assert table.meta[0].computation.parameters["spatial_channels"] == [
            "Cz",
            "F3",
            "F4",
            "Fz",
            "Pz",
        ]


def _burst_epochs() -> mne.EpochsArray:
    # 8 s epochs with a 0.1 s broadband burst at 30 times the noise, 1.0-1.1 s: a small
    # share of a [0, 7] s window.
    epochs = make_epochs(seconds=8.0)
    data = epochs.get_data()
    burst = (epochs.times >= 1.0) & (epochs.times < 1.1)
    data[:, :, burst] += (
        30
        * NOISE
        * np.random.default_rng(1).standard_normal(
            (len(epochs), len(epochs.ch_names), int(burst.sum()))
        )
    )
    return mne.EpochsArray(
        data,
        epochs.info,
        events=epochs.events,
        tmin=epochs.tmin,
        event_id=epochs.event_id,
        verbose="error",
    )


@pytest.mark.parametrize(
    ("spectra", "measure"),
    [('method = "welch"', "integrated_band_power"), ('method = "morlet"', "mean_tfr_power")],
)
def test_median_window_statistic_resists_a_burst(tmp_path, spectra, measure) -> None:
    # A band clear of the 10 Hz sine, whose wavelet leakage reaches into beta: a steady
    # oscillation's median is its mean, so there only noise-like power would be compared.
    body = (
        "[bands]\nhigh = [25.0, 40.0]\n\n"
        "[spectra]\n{spectra}\n{statistic}\n[windows]\nstim = [0.0, 7.0]\n\n"
        '[[features]]\nmeasure = "{measure}"\nbands = ["high"]\nspatial = ["global"]\n'
    )
    mean = features(
        tmp_path, body.format(spectra=spectra, statistic="", measure=measure), _burst_epochs()
    ).epochs
    median = features(
        tmp_path,
        body.format(spectra=spectra, statistic='window_statistic = "median"', measure=measure),
        _burst_epochs(),
    ).epochs

    assert mean is not None and median is not None
    assert np.all(median.values < 0.25 * mean.values)
    assert set(mean.to_dataframe().columns).isdisjoint(median.to_dataframe().columns)


def test_median_welch_needs_three_segments_per_window(tmp_path) -> None:
    # The median of one or two segments is their mean, so the setting would do nothing.
    with pytest.raises(ValueError, match="segments"):
        features(
            tmp_path,
            '[spectra]\nwindow_statistic = "median"\n\n[windows]\nbase = [-0.5, 0.0]\n\n'
            '[[features]]\nmeasure = "integrated_band_power"\n',
        )


def test_welch_segment_longer_than_a_window_is_an_error(tmp_path) -> None:
    with pytest.raises(ValueError, match="n_fft = 400"):
        features(
            tmp_path,
            "[spectra]\nn_fft = 400\n\n[windows]\nbase = [-0.5, 0.0]\n\n"
            '[[features]]\nmeasure = "integrated_band_power"\n',
        )


def test_wpli_is_estimated_per_trial_group(tmp_path) -> None:
    pytest.importorskip("mne_connectivity")
    result = features(
        tmp_path,
        '[trials]\nby = "event"\n\n[[features]]\nmeasure = "wpli"\nbands = ["alpha"]\n',
    )

    assert result.crosstrial is not None
    assert result.crosstrial.row_labels == ("left", "right")
    assert {m.space_kind for m in result.crosstrial.meta} == {"pair"}


def test_morlet_log_spacing_reaches_the_top_band_edge(tmp_path) -> None:
    # fmax defaults to the top band's upper edge, so the grid has to reach it exactly:
    # a last frequency an ULP short leaves that band unintegrable.
    result = features(
        tmp_path,
        "[bands]\ntheta = [4.0, 8.0]\nalpha = [8.0, 13.0]\nbeta = [13.0, 30.0]\n\n"
        '[spectra]\nmethod = "morlet"\nn_freqs = 20\n\n'
        "[windows]\nstim = [0.0, 1.0]\n\n"
        '[[features]]\nmeasure = "mean_tfr_power"\nbands = ["beta"]\nspatial = ["global"]\n',
    )

    assert result.epochs is not None and result.epochs.n_rows == 12


@pytest.mark.parametrize("measure", ["variance", "erds_mean"])
@pytest.mark.parametrize("channel_kind", ["bad_eeg", "eog"])
def test_time_domain_features_preserve_runner_channel_selection(tmp_path, measure, channel_kind):
    epochs = make_epochs()
    if channel_kind == "bad_eeg":
        epochs.info["bads"] = ["Fz"]
    else:
        epochs.set_channel_types({"Fz": "eog"})
    body = f'[[features]]\nmeasure = "{measure}"\nspatial = ["channels"]\n'
    if measure == "erds_mean":
        body = (
            "[windows]\nbase = [-0.5, 0.0]\nstim = [0.25, 1.25]\n"
            + body
            + 'bands = ["alpha"]\nbaseline = "base"\n'
        )

    result = features(tmp_path, body, epochs)

    assert {meta.space for meta in result.epochs.meta} == set(epochs.ch_names)
    assert epochs.info["bads"] == (["Fz"] if channel_kind == "bad_eeg" else [])


def test_an_roi_pattern_is_resolved_against_the_recordings_channels(tmp_path) -> None:
    listed = features(
        tmp_path,
        '[rois]\nfront = ["Fz", "F3", "F4"]\n\n'
        '[[features]]\nmeasure = "integrated_band_power"\nspatial = ["rois"]\n',
    )
    matched = features(
        tmp_path,
        '[rois]\nfront = { match = ["^F[z34]$"] }\n\n'
        '[[features]]\nmeasure = "integrated_band_power"\nspatial = ["rois"]\n',
    )

    assert listed.epochs is not None and matched.epochs is not None
    np.testing.assert_array_equal(matched.epochs.values, listed.epochs.values)


def test_an_roi_pattern_matching_no_channel_is_an_error(tmp_path) -> None:
    with pytest.raises(ValueError, match="'temporal'.*T7"):
        features(
            tmp_path,
            '[rois]\ntemporal = { match = ["^T7$"] }\n\n'
            '[[features]]\nmeasure = "integrated_band_power"\nspatial = ["rois"]\n',
        )


def test_an_error_names_the_file_entry_even_after_a_multi_measure_entry(tmp_path) -> None:
    with pytest.raises(ValueError) as caught:
        features(
            tmp_path,
            "[windows]\nlate = [1.0, 3.0]\n\n"
            '[[features]]\nmeasures = ["variance", "kurtosis"]\nwindows = ["all"]\n\n'
            '[[features]]\nmeasure = "variance"\nwindows = ["late"]\n',
        )

    assert caught.value.__notes__ == ["features[1] (variance)"]


def test_each_entry_is_timed(tmp_path) -> None:
    result = features(
        tmp_path,
        '[[features]]\nmeasures = ["integrated_band_power", "variance"]\n\n'
        '[[features]]\nmeasure = "kurtosis"\n',
    )

    assert [(t.entry, t.measure) for t in result.timings] == [
        (0, "integrated_band_power"),
        (0, "variance"),
        (1, "kurtosis"),
    ]
    assert all(t.seconds >= 0.0 for t in result.timings)


@pytest.mark.parametrize(
    "method", ["coh", "imcoh", "plv", "ciplv", "ppc", "pli", "wpli", "wpli2_debiased"]
)
def test_runner_reaches_all_cross_trial_spectral_methods(tmp_path, method) -> None:
    pytest.importorskip("mne_connectivity")
    result = features(
        tmp_path,
        '[trials]\nby = "event"\n[[features]]\n'
        f'measure = "spectral_connectivity"\nmethod = "{method}"\n'
        'bands = ["alpha"]\nmode = "fourier"\n',
    )
    assert result.epochs is None
    assert result.crosstrial.row_labels == ("left", "right")
    assert {meta.measure for meta in result.crosstrial.meta} == {method}


def test_runner_time_connectivity_concatenates_with_epoch_features_and_graph(tmp_path) -> None:
    pytest.importorskip("mne_connectivity")
    result = features(
        tmp_path,
        '[[features]]\nmeasure = "spectral_connectivity_time"\n'
        'method = "coh"\nbands = ["alpha"]\nfreqs = [8.0, 10.0, 12.0]\n'
        'n_cycles = 3.0\ngraph = ["global_efficiency"]\n'
        '[[features]]\nmeasure = "variance"\nspatial = ["global"]\n',
    )
    assert result.crosstrial is None
    assert result.epochs.n_rows == 12
    assert {meta.measure for meta in result.epochs.meta} == {"coh", "global_efficiency", "variance"}
    assert result.epochs.row_ids[0][0] == "sub-test_task-test"


def test_runner_computes_pac_surrogate_inference(tmp_path) -> None:
    pytest.importorskip("tensorpac")
    result = features(
        tmp_path,
        '[[features]]\nmeasure = "pac_surrogates"\n'
        'pairs = [["theta", "gamma"]]\nn_surrogates = 20\n'
        'spatial = ["global"]\nrandom_state = 42\n',
    )
    assert result.crosstrial is None
    assert result.epochs.n_rows == 12
    assert len(result.epochs.meta) == 7
    assert result.epochs.select(measure="pac_pvalue").values.min() >= 1 / 21


@pytest.mark.parametrize(
    "measure, output_measure, settings",
    [
        ("spectral_parameterization", "specparam_exponent", "fit_range = [4.0, 30.0]\n"),
        ("irasa", "irasa_slope", "fit_range = [4.0, 30.0]\nhset = [1.1, 1.3]\n"),
        ("cycle_features", "cycle_count", ""),
        ("permutation_entropy", "permutation_entropy", ""),
        ("lempel_ziv_complexity", "lempel_ziv_complexity", ""),
        ("detrended_fluctuation", "dfa_exponent", ""),
    ],
)
def test_runner_computes_new_spectral_cycle_and_complexity_methods(
    tmp_path,
    measure,
    output_measure,
    settings,
) -> None:
    from eegtable.runner.measures import REQUIRES

    pytest.importorskip(REQUIRES[measure][0])
    result = features(
        tmp_path,
        f'[[features]]\nmeasure = "{measure}"\n'
        'spatial = ["global"]\n'
        + (
            'bands = ["alpha"]\n'
            if measure in ("spectral_parameterization", "irasa", "cycle_features")
            else ""
        )
        + settings,
        make_epochs(n_epochs=2, seconds=6.0, channels=["Cz"]),
    )
    assert result.crosstrial is None
    assert result.epochs.n_rows == 2
    assert output_measure in {meta.measure for meta in result.epochs.meta}


@pytest.mark.parametrize(
    "grouping, expected",
    [
        ('by = "all"', None),
        ('by = "event"', ("left", "right", "left", "right")),
        ('by = "metadata"\ncolumn = "rating"', ("0", "1", "2", "3")),
    ],
)
def test_trial_labels_exposes_the_runner_grouping(tmp_path, grouping, expected) -> None:
    import eegtable.runner.compute as runner_compute

    path = tmp_path / "recipe.toml"
    path.write_text(
        HEAD + f'[trials]\n{grouping}\n[[features]]\nmeasure = "itpc"\n', encoding="utf-8"
    )
    epochs = make_epochs(n_epochs=4)
    recipe = load_recipe(path)
    assert runner_compute.trial_labels(epochs, recipe) == expected
