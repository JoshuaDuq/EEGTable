"""Measures of a user's own, computed with the library's windows, bands, ROIs and identity."""

import importlib.metadata
from types import SimpleNamespace

import numpy as np
import pytest

import eegtable as ef
from eegtable.runner import load_recipe
from eegtable.runner.measures import MEASURES
from tests.synthetic import make_epochs

ALPHA = ef.Band("alpha", 8.0, 13.0)


def _peak_power(power, freqs, weights):
    del freqs, weights
    return np.nanmax(power, axis=-1)


def _largest_deflection(trace, times):
    del times
    return np.nanmax(np.abs(trace), axis=-1)


def test_a_spectral_measure_reduces_each_band_window_and_channel_with_the_kernel() -> None:
    spectra = ef.Spectra.welch(make_epochs(), recording="sub-01", fmin=1.0, fmax=40.0)
    table = ef.spectral_measure(
        spectra, _peak_power, measure="peak_power", unit="V^2/Hz", bands=[ALPHA]
    )
    channel = table.select(space="Cz")
    inside = ALPHA.mask(spectra.freqs)
    cz = spectra.ch_names.index("Cz")
    np.testing.assert_allclose(channel.values[:, 0], spectra.data[:, cz, 0, inside].max(axis=-1))
    assert {meta.measure for meta in table.meta} == {"peak_power"}
    assert {meta.unit for meta in table.meta} == {"V^2/Hz"}


def test_a_signal_measure_gets_windows_and_rois_from_the_library() -> None:
    signal = ef.Signal.from_epochs(make_epochs(), recording="sub-01")
    windows = [ef.Window("base", -0.5, 0.0), ef.Window("stim", 0.0, 1.0)]
    table = ef.signal_measure(
        [signal],
        _largest_deflection,
        measure="largest_deflection",
        unit="V",
        windows=windows,
        groups={"frontal": ["F3", "Fz", "F4"]},
        include_global=False,
    )
    assert [meta.window for meta in table.meta] == ["base", "stim"]
    stim = (signal.times >= 0.0) & (signal.times <= 1.0)
    frontal = [signal.ch_names.index(name) for name in ("F3", "Fz", "F4")]
    expected = np.abs(signal.data[:, frontal][:, :, stim]).max(axis=-1).mean(axis=1)
    np.testing.assert_allclose(table.select(window="stim").values[:, 0], expected)


def test_two_kernels_under_one_label_do_not_share_a_column() -> None:
    # The label is the user's to choose; the kernel's own name keeps two different
    # computations from being stacked as one feature.
    signal = ef.Signal.from_epochs(make_epochs(), recording="sub-01")
    window = [ef.Window("stim", 0.0, 1.0)]

    def smallest(trace, times):
        del times
        return np.nanmin(trace, axis=-1)

    first = ef.signal_measure([signal], _largest_deflection, measure="m", unit="V", windows=window)
    second = ef.signal_measure([signal], smallest, measure="m", unit="V", windows=window)
    assert set(first.names).isdisjoint(second.names)


def channel_max(series, *, windows, groups=None, include_global=True):
    return ef.signal_measure(
        series,
        _largest_deflection,
        measure="channel_max",
        unit="V",
        windows=windows,
        groups=groups,
        include_global=include_global,
    )


@pytest.fixture
def installed_plugin(monkeypatch):
    # A package declaring [project.entry-points."eegtable.measures"] channel_max = "...".
    def entry_points(*, group, name):
        if group == "eegtable.measures" and name == "channel_max":
            return [SimpleNamespace(name=name, load=lambda: channel_max)]
        return []

    monkeypatch.setattr(importlib.metadata, "entry_points", entry_points)
    yield
    MEASURES.pop("channel_max", None)


def test_a_recipe_can_name_a_measure_an_installed_package_registers(installed_plugin) -> None:
    recipe = load_recipe({"features": [{"measure": "channel_max", "spatial": ["channels"]}]})
    features = ef.extract(make_epochs(), recipe, recording="sub-01")
    assert features.epochs is not None
    assert {meta.measure for meta in features.epochs.meta} == {"channel_max"}


def test_an_unregistered_measure_is_still_unknown(installed_plugin) -> None:
    with pytest.raises(ef.runner.RecipeError, match="unknown measure 'channel_min'"):
        load_recipe({"features": [{"measure": "channel_min"}]})
