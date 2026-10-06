import math
from pathlib import Path

import pytest

from eegtable.bands import BANDS_STANDARD, Band
from eegtable.runner import RecipeError, load_recipe
from eegtable.spectra import Window

HEAD = """
[inputs]
root = "data"

[output]
root = "out"
"""


def _load(tmp_path: Path, body: str, head: str = HEAD):
    path = tmp_path / "recipe.toml"
    path.write_text(head + body, encoding="utf-8")
    return load_recipe(path)


def _problems(tmp_path: Path, body: str, head: str = HEAD) -> str:
    with pytest.raises(RecipeError) as caught:
        _load(tmp_path, body, head)
    return str(caught.value)


# --- defaults -----------------------------------------------------------------


def test_minimal_recipe_takes_the_library_defaults(tmp_path) -> None:
    recipe = _load(tmp_path, '[[features]]\nmeasure = "integrated_band_power"\n')

    assert recipe.bands == BANDS_STANDARD
    assert recipe.windows == ()
    assert recipe.inputs.pattern == "**/*_epo.fif"
    assert recipe.inputs.picks == "eeg"
    assert recipe.inputs.exclude_bads is True
    (spec,) = recipe.features
    assert spec.measure == "integrated_band_power"
    assert spec.bands == BANDS_STANDARD
    assert spec.spatial == ("channels", "global")


def test_roots_resolve_against_the_recipe_directory(tmp_path) -> None:
    recipe = _load(tmp_path, '[[features]]\nmeasure = "integrated_band_power"\n')

    assert recipe.inputs.root == tmp_path / "data"
    assert recipe.output.root == tmp_path / "out"


def test_without_windows_a_measure_spans_the_whole_epoch(tmp_path) -> None:
    recipe = _load(tmp_path, '[[features]]\nmeasure = "variance"\n')

    (window,) = recipe.features[0].windows
    assert window.name == "all"
    assert math.isinf(window.tmin) and math.isinf(window.tmax)


def test_spectra_default_to_the_span_of_the_recipe_bands(tmp_path) -> None:
    recipe = _load(
        tmp_path,
        "[bands]\ntheta = [4.0, 8.0]\nbeta = [13.0, 30.0]\n\n"
        '[[features]]\nmeasure = "integrated_band_power"\n',
    )

    assert recipe.spectra.method == "welch"
    assert (recipe.spectra.fmin, recipe.spectra.fmax) == (4.0, 30.0)


def test_bands_keep_recipe_order(tmp_path) -> None:
    recipe = _load(
        tmp_path,
        "[bands]\nbeta = [13.0, 30.0]\ntheta = [4.0, 8.0]\n\n"
        '[[features]]\nmeasure = "integrated_band_power"\n',
    )

    assert [b.name for b in recipe.bands] == ["beta", "theta"]


def test_default_windows_leave_out_the_entry_baseline(tmp_path) -> None:
    recipe = _load(
        tmp_path,
        "[windows]\nbase = [-0.5, 0.0]\nearly = [0.0, 0.5]\nlate = [0.5, 1.0]\n\n"
        '[[features]]\nmeasure = "burst_rate"\nbaseline = "base"\n',
    )

    (spec,) = recipe.features
    assert spec.baseline == Window("base", -0.5, 0.0)
    assert [w.name for w in spec.windows] == ["early", "late"]


def test_series_default_to_broadband_and_accept_band_names(tmp_path) -> None:
    recipe = _load(
        tmp_path,
        '[[features]]\nmeasure = "variance"\n\n'
        '[[features]]\nmeasure = "sample_entropy"\nseries = ["broadband", "alpha"]\n',
    )

    assert recipe.features[0].series == (None,)
    assert recipe.features[1].series == (None, Band("alpha", 8.0, 13.0))


def test_pac_pairs_resolve_to_bands(tmp_path) -> None:
    recipe = _load(tmp_path, '[[features]]\nmeasure = "pac"\npairs = [["theta", "gamma"]]\n')

    assert recipe.features[0].pairs == ((Band("theta", 4.0, 8.0), Band("gamma", 30.0, 45.0)),)


# --- parameters passed through to the library ---------------------------------


def test_list_parameters_become_the_tuples_the_library_expects(tmp_path) -> None:
    recipe = _load(
        tmp_path,
        '[[features]]\nmeasure = "aperiodic"\nfit_range = [2, 30]\n\n'
        '[[features]]\nmeasure = "multiscale_entropy"\nscales = [1, 2, 3]\n',
    )

    assert recipe.features[0].params == {"fit_range": (2.0, 30.0)}
    assert recipe.features[1].params == {"scales": (1, 2, 3)}


def test_integer_given_for_a_float_parameter_becomes_a_float(tmp_path) -> None:
    recipe = _load(tmp_path, '[[features]]\nmeasure = "burst_rate"\nmin_duration_ms = 150\n')

    value = recipe.features[0].params["min_duration_ms"]
    assert value == 150.0 and isinstance(value, float)


def test_wrongly_typed_parameter_is_rejected(tmp_path) -> None:
    problems = _problems(tmp_path, '[[features]]\nmeasure = "burst_rate"\nthreshold = "high"\n')

    assert "threshold" in problems and "number" in problems


def test_parameter_outside_its_allowed_values_is_rejected(tmp_path) -> None:
    problems = _problems(
        tmp_path, '[[features]]\nmeasure = "integrated_band_power"\nnormalize = "zscore"\n'
    )

    assert "normalize" in problems and "'log10'" in problems


def test_parameter_the_measure_does_not_take_is_rejected(tmp_path) -> None:
    problems = _problems(tmp_path, '[[features]]\nmeasure = "aperiodic"\nbands = ["alpha"]\n')

    assert "aperiodic" in problems and "'bands'" in problems


# --- cross-references ---------------------------------------------------------


def test_unknown_measure_is_rejected(tmp_path) -> None:
    problems = _problems(tmp_path, '[[features]]\nmeasure = "band_powr"\n')

    assert "band_powr" in problems


def test_undefined_band_is_rejected(tmp_path) -> None:
    problems = _problems(
        tmp_path, '[[features]]\nmeasure = "integrated_band_power"\nbands = ["mu"]\n'
    )

    assert "'mu'" in problems


def test_undefined_window_is_rejected(tmp_path) -> None:
    problems = _problems(
        tmp_path,
        '[windows]\nstim = [0.0, 1.0]\n\n[[features]]\nmeasure = "variance"\nwindows = ["late"]\n',
    )

    assert "'late'" in problems


def test_measure_that_needs_a_baseline_requires_one(tmp_path) -> None:
    problems = _problems(
        tmp_path, '[windows]\nstim = [0.0, 1.0]\n\n[[features]]\nmeasure = "erds_mean"\n'
    )

    assert "erds_mean" in problems and "baseline" in problems


def test_baseline_needs_windows_to_name(tmp_path) -> None:
    problems = _problems(tmp_path, '[[features]]\nmeasure = "burst_rate"\nbaseline = "base"\n')

    assert "'base'" in problems


def test_band_power_cannot_both_consume_and_report_its_baseline(tmp_path) -> None:
    problems = _problems(
        tmp_path,
        "[windows]\nbase = [-0.5, 0.0]\nstim = [0.0, 1.0]\n\n"
        '[[features]]\nmeasure = "integrated_band_power"\n'
        'baseline = "base"\nwindows = ["base", "stim"]\n'
        'normalize = "db"\n',
    )

    assert "'base'" in problems


def test_roi_level_requires_rois(tmp_path) -> None:
    problems = _problems(
        tmp_path, '[[features]]\nmeasure = "integrated_band_power"\nspatial = ["rois"]\n'
    )

    assert "[rois]" in problems


def test_graph_measures_need_a_single_node_level(tmp_path) -> None:
    problems = _problems(
        tmp_path,
        '[rois]\nfront = ["Fz", "F3"]\nback = ["Pz", "P3"]\n\n'
        '[[features]]\nmeasure = "envelope_correlation"\nspatial = ["channels", "rois"]\n'
        'graph = ["global_efficiency"]\n',
    )

    assert "graph" in problems


def test_clustering_coefficient_needs_its_threshold(tmp_path) -> None:
    problems = _problems(
        tmp_path,
        '[[features]]\nmeasure = "envelope_correlation"\ngraph = ["clustering_coefficient"]\n',
    )

    assert "clustering_threshold" in problems


def test_pac_requires_pairs(tmp_path) -> None:
    problems = _problems(tmp_path, '[[features]]\nmeasure = "pac"\n')

    assert "pairs" in problems


# --- sections -----------------------------------------------------------------


def test_unknown_section_is_rejected(tmp_path) -> None:
    problems = _problems(
        tmp_path, '[plots]\nsize = 3\n\n[[features]]\nmeasure = "integrated_band_power"\n'
    )

    assert "plots" in problems


def test_unknown_input_key_is_rejected(tmp_path) -> None:
    head = '[inputs]\nroot = "data"\nglob = "*.fif"\n\n[output]\nroot = "out"\n'
    problems = _problems(tmp_path, '[[features]]\nmeasure = "integrated_band_power"\n', head)

    assert "glob" in problems


def test_input_root_is_required(tmp_path) -> None:
    problems = _problems(
        tmp_path, '[[features]]\nmeasure = "integrated_band_power"\n', '[output]\nroot = "out"\n'
    )

    assert "inputs" in problems and "root" in problems


def test_a_recipe_without_features_is_rejected(tmp_path) -> None:
    assert "features" in _problems(tmp_path, "")


def test_invalid_band_bounds_are_rejected(tmp_path) -> None:
    problems = _problems(
        tmp_path,
        '[bands]\nalpha = [13.0, 8.0]\n\n[[features]]\nmeasure = "integrated_band_power"\n',
    )

    assert "alpha" in problems


def test_all_is_reserved_for_the_whole_epoch(tmp_path) -> None:
    problems = _problems(
        tmp_path, '[windows]\nall = [0.0, 1.0]\n\n[[features]]\nmeasure = "variance"\n'
    )

    assert "'all'" in problems


def test_global_is_reserved_among_rois(tmp_path) -> None:
    problems = _problems(
        tmp_path, '[rois]\nglobal = ["Fz"]\n\n[[features]]\nmeasure = "integrated_band_power"\n'
    )

    assert "'global'" in problems


def test_metadata_trials_need_a_column(tmp_path) -> None:
    problems = _problems(tmp_path, '[trials]\nby = "metadata"\n\n[[features]]\nmeasure = "itpc"\n')

    assert "column" in problems


def test_spectral_option_of_another_method_is_rejected(tmp_path) -> None:
    problems = _problems(
        tmp_path,
        '[spectra]\nmethod = "morlet"\nn_fft = 512\n\n[[features]]\nmeasure = "mean_tfr_power"\n',
    )

    assert "n_fft" in problems and "welch" in problems


def test_window_statistic_defaults_to_the_mean(tmp_path) -> None:
    recipe = _load(tmp_path, '[[features]]\nmeasure = "integrated_band_power"\n')

    assert recipe.spectra.window_statistic == "mean"


@pytest.mark.parametrize(
    ("method", "measure"), [("welch", "integrated_band_power"), ("morlet", "mean_tfr_power")]
)
def test_window_statistic_is_read_for_welch_and_morlet(tmp_path, method, measure) -> None:
    recipe = _load(
        tmp_path,
        f'[spectra]\nmethod = "{method}"\nwindow_statistic = "median"\n\n'
        f'[[features]]\nmeasure = "{measure}"\n',
    )

    assert recipe.spectra.window_statistic == "median"


def test_window_statistic_must_be_mean_or_median(tmp_path) -> None:
    problems = _problems(
        tmp_path,
        '[spectra]\nwindow_statistic = "mode"\n\n[[features]]\nmeasure = "integrated_band_power"\n',
    )

    assert "window_statistic" in problems and "'mode'" in problems


def test_window_statistic_does_not_apply_to_multitaper(tmp_path) -> None:
    # A multitaper window is one estimate, with no segments or time points to reduce.
    problems = _problems(
        tmp_path,
        '[spectra]\nmethod = "multitaper"\nwindow_statistic = "median"\n\n'
        '[[features]]\nmeasure = "integrated_band_power"\n',
    )

    assert "window_statistic" in problems and "morlet" in problems


def test_microstate_settings_are_type_checked(tmp_path) -> None:
    problems = _problems(
        tmp_path,
        '[microstates]\nn_states = "four"\n\n[[features]]\nmeasure = "microstate_coverage"\n',
    )

    assert "n_states" in problems


def test_every_problem_is_reported_at_once(tmp_path) -> None:
    problems = _problems(
        tmp_path,
        '[[features]]\nmeasure = "integrated_band_power"\nbands = ["mu"]\n\n'
        '[[features]]\nmeasure = "burst_rate"\nthreshold = "high"\n',
    )

    assert "'mu'" in problems and "threshold" in problems


def test_malformed_toml_names_the_file(tmp_path) -> None:
    problems = _problems(tmp_path, "[[features]\nmeasure = \n")

    assert "recipe.toml" in problems


def test_pac_pairs_in_one_entry_need_distinct_amplitude_bands(tmp_path) -> None:
    problems = _problems(
        tmp_path,
        '[[features]]\nmeasure = "pac"\npairs = [["theta", "gamma"], ["alpha", "gamma"]]\n',
    )

    assert "'gamma'" in problems


def test_measure_whose_optional_dependency_is_missing_is_rejected(tmp_path) -> None:
    import importlib.util

    if importlib.util.find_spec("mne_connectivity") is not None:
        pytest.skip("mne-connectivity is installed")
    problems = _problems(tmp_path, '[[features]]\nmeasure = "wpli"\n')

    assert "eegtable[connectivity]" in problems


def test_a_measure_that_cannot_read_the_chosen_spectrum_is_refused(tmp_path) -> None:
    problems = _problems(
        tmp_path,
        '\n[spectra]\nmethod = "morlet"\n\n'
        '[[features]]\nmeasure = "integrated_band_power"\n\n'
        '[[features]]\nmeasure = "mean_tfr_power"\n',
    )

    assert "integrated_band_power reads a power spectral density" in problems
    assert "morlet" in problems
    assert "mean_tfr_power" not in problems


def test_a_time_frequency_measure_needs_morlet(tmp_path) -> None:
    problems = _problems(tmp_path, '\n[[features]]\nmeasure = "mean_tfr_power"\n')

    assert "mean_tfr_power reads time-frequency power" in problems
    assert "welch" in problems


# --- the whole epoch, shared defaults and multi-measure entries ---------------


def test_the_whole_epoch_can_be_named_beside_defined_windows(tmp_path) -> None:
    recipe = _load(
        tmp_path,
        '[windows]\nstim = [0.0, 1.0]\n\n[[features]]\nmeasure = "variance"\n'
        'windows = ["all", "stim"]\n',
    )

    whole, stim = recipe.features[0].windows
    assert whole.name == "all" and math.isinf(whole.tmin) and math.isinf(whole.tmax)
    assert stim == Window("stim", 0.0, 1.0)


def test_the_whole_epoch_can_be_named_when_no_windows_are_defined(tmp_path) -> None:
    recipe = _load(tmp_path, '[[features]]\nmeasure = "variance"\nwindows = ["all"]\n')

    (window,) = recipe.features[0].windows
    assert window.name == "all" and math.isinf(window.tmin)


def test_defaults_fill_the_keys_an_entry_leaves_out(tmp_path) -> None:
    recipe = _load(
        tmp_path,
        '[rois]\nfront = ["Fz"]\n\n[defaults]\nspatial = ["rois", "global"]\nbands = ["alpha"]\n\n'
        '[[features]]\nmeasure = "integrated_band_power"\n\n'
        '[[features]]\nmeasure = "spectral_entropy"\nspatial = ["channels"]\n',
    )

    power, entropy = recipe.features
    assert power.spatial == ("rois", "global")
    assert power.bands == (Band("alpha", 8.0, 13.0),)
    assert entropy.spatial == ("channels",)


def test_defaults_skip_the_measures_that_do_not_take_them(tmp_path) -> None:
    # Microstate measures have no spatial level and no bands; a shared default must
    # not turn every such entry into an error.
    recipe = _load(
        tmp_path,
        '[defaults]\nspatial = ["global"]\nbands = ["alpha"]\n\n'
        '[[features]]\nmeasure = "microstate_coverage"\n\n'
        '[[features]]\nmeasure = "aperiodic"\n',
    )

    coverage, aperiodic = recipe.features
    assert coverage.spatial == () and coverage.bands == ()
    assert aperiodic.spatial == ("global",)


def test_default_windows_leave_out_the_baseline_of_an_entry_that_has_one(tmp_path) -> None:
    # Spelling out windows = ["base", "stim"] beside baseline = "base" is refused for
    # power, so an inherited default must set the baseline aside the way the implicit
    # default does.
    recipe = _load(
        tmp_path,
        "[windows]\nbase = [-0.5, 0.0]\nstim = [0.0, 1.0]\n\n"
        '[defaults]\nwindows = ["base", "stim"]\n\n'
        '[[features]]\nmeasure = "integrated_band_power"\nnormalize = "db"\nbaseline = "base"\n\n'
        '[[features]]\nmeasure = "variance"\n',
    )

    power, variance = recipe.features
    assert [w.name for w in power.windows] == ["stim"]
    assert [w.name for w in variance.windows] == ["base", "stim"]


def test_defaults_are_checked_against_the_recipe(tmp_path) -> None:
    problems = _problems(
        tmp_path,
        '[defaults]\nwindows = ["late"]\nnormalize = "db"\n\n[[features]]\nmeasure = "variance"\n',
    )

    assert "'late'" in problems
    assert "defaults: unknown key 'normalize'" in problems


def test_one_entry_can_name_several_measures(tmp_path) -> None:
    recipe = _load(
        tmp_path,
        "[windows]\nbase = [-0.5, 0.0]\nstim = [0.0, 1.0]\n\n"
        '[[features]]\nmeasures = ["erds_mean", "erd_magnitude"]\nbands = ["alpha"]\n'
        'baseline = "base"\nnormalize = "percent"\n',
    )

    assert [spec.measure for spec in recipe.features] == ["erds_mean", "erd_magnitude"]
    for spec in recipe.features:
        assert spec.baseline == Window("base", -0.5, 0.0)
        assert spec.params == {"normalize": "percent"}


def test_a_key_one_of_several_measures_does_not_take_names_that_measure(tmp_path) -> None:
    problems = _problems(
        tmp_path, '[[features]]\nmeasures = ["variance", "burst_rate"]\nthreshold = 0.9\n'
    )

    assert "features[0] (variance): variance does not take 'threshold'" in problems
    assert "burst_rate" not in problems


def test_an_entry_names_measure_or_measures_not_both(tmp_path) -> None:
    problems = _problems(tmp_path, '[[features]]\nmeasure = "variance"\nmeasures = ["kurtosis"]\n')

    assert "features[0]" in problems and "not both" in problems


def test_measures_must_be_a_list_of_distinct_names(tmp_path) -> None:
    problems = _problems(tmp_path, '[[features]]\nmeasures = ["variance", "variance"]\n')

    assert "features[0]" in problems and "distinct" in problems


# --- ROI patterns --------------------------------------------------------------


def test_an_roi_can_be_given_as_patterns(tmp_path) -> None:
    recipe = _load(
        tmp_path,
        '[rois]\nfront = { match = ["^F[z34]$"] }\nback = ["Pz"]\n\n'
        '[[features]]\nmeasure = "integrated_band_power"\nspatial = ["rois"]\n',
    )

    assert recipe.rois["back"] == ("Pz",)
    assert recipe.rois["front"].patterns == ("^F[z34]$",)
    assert recipe.rois["front"].resolve(["Fz", "F3", "F4", "Fp1", "Cz"]) == ("Fz", "F3", "F4")


def test_an_invalid_roi_pattern_is_rejected(tmp_path) -> None:
    problems = _problems(
        tmp_path,
        '[rois]\nfront = { match = ["^(F"] }\n\n[[features]]\nmeasure = "integrated_band_power"\n',
    )

    assert "rois: front" in problems and "^(F" in problems


def test_a_bad_default_is_reported_once_not_per_entry(tmp_path) -> None:
    problems = _problems(
        tmp_path,
        '[defaults]\nbands = ["mu"]\n\n'
        '[[features]]\nmeasure = "integrated_band_power"\n\n'
        '[[features]]\nmeasure = "spectral_entropy"\n',
    )

    assert problems.count("'mu'") == 1


@pytest.mark.parametrize(
    "method", ["coh", "imcoh", "plv", "ciplv", "ppc", "pli", "wpli", "wpli2_debiased"]
)
def test_recipes_reach_every_supported_spectral_connectivity_method(tmp_path, method) -> None:
    pytest.importorskip("mne_connectivity")
    recipe = _load(
        tmp_path,
        f'[[features]]\nmeasure = "spectral_connectivity"\n'
        f'method = "{method}"\nmode = "fourier"\nbands = ["alpha"]\n'
        'graph = ["global_efficiency"]\n',
    )
    assert recipe.features[0].params == {"method": method, "mode": "fourier"}
    assert recipe.features[0].graph == ("global_efficiency",)


def test_pac_surrogate_recipe_parameters_are_type_checked(tmp_path) -> None:
    pytest.importorskip("tensorpac")
    recipe = _load(
        tmp_path,
        '[[features]]\nmeasure = "pac_surrogates"\n'
        'pairs = [["theta", "gamma"]]\nn_surrogates = 99\n'
        'surrogate = "circular"\ncorrection = "maxstat"\nrandom_state = 7\n',
    )
    spec = recipe.features[0]
    assert spec.params == {
        "n_surrogates": 99,
        "surrogate": "circular",
        "correction": "maxstat",
        "random_state": 7,
    }
    assert spec.pairs[0][0].name == "theta"


def test_time_connectivity_recipe_requires_method_and_frequencies(tmp_path) -> None:
    problems = _problems(tmp_path, '[[features]]\nmeasure = "spectral_connectivity_time"\n')
    assert "method is required" in problems
    assert "freqs is required" in problems


def test_time_connectivity_recipe_accepts_frequency_grid_and_graph(tmp_path) -> None:
    pytest.importorskip("mne_connectivity")
    recipe = _load(
        tmp_path,
        '[[features]]\nmeasure = "spectral_connectivity_time"\n'
        'method = "plv"\nfreqs = [8.0, 10.0, 12.0]\n'
        'graph = ["global_efficiency"]\n',
    )
    assert recipe.features[0].params["freqs"] == (8.0, 10.0, 12.0)
    assert recipe.features[0].graph == ("global_efficiency",)


@pytest.mark.parametrize(
    "measure, settings",
    [
        ("spectral_parameterization", 'bands = ["alpha"]\naperiodic_mode = "knee"\n'),
        ("irasa", 'bands = ["alpha"]\nhset = [1.1, 1.3, 1.5]\n'),
        (
            "cycle_features",
            'bands = ["alpha"]\n'
            "burst_thresholds = {min_n_cycles = 3, amp_fraction_threshold = 0.2}\n",
        ),
        ("permutation_entropy", 'order = 4\ndelay = 2\nseries = ["broadband", "alpha"]\n'),
        ("lempel_ziv_complexity", 'symbolization = "mean"\n'),
        ("detrended_fluctuation", ""),
    ],
)
def test_recipes_reach_new_spectral_cycle_and_complexity_methods(
    tmp_path, measure, settings
) -> None:
    from eegtable.runner.measures import REQUIRES

    pytest.importorskip(REQUIRES[measure][0])
    recipe = _load(tmp_path, f'[[features]]\nmeasure = "{measure}"\n{settings}')
    assert recipe.features[0].measure == measure
    if measure == "cycle_features":
        assert recipe.features[0].params["burst_thresholds"]["min_n_cycles"] == 3
        assert isinstance(recipe.features[0].params["burst_thresholds"]["min_n_cycles"], int)


@pytest.mark.parametrize("measure", ["irasa", "cycle_features"])
def test_broadband_decomposition_recipes_refuse_band_envelopes(tmp_path, measure) -> None:
    problems = _problems(tmp_path, f'[[features]]\nmeasure = "{measure}"\nseries = ["alpha"]\n')
    assert "requires broadband" in problems


def test_spectral_parameterization_recipe_refuses_tfr_input(tmp_path) -> None:
    problems = _problems(
        tmp_path,
        '[spectra]\nmethod = "morlet"\n[[features]]\nmeasure = "spectral_parameterization"\n',
    )
    assert "power spectral density" in problems
