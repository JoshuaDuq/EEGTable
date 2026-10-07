"""The measures a recipe may name, and how their parameters are checked.

A recipe entry's parameters are the keyword parameters of the library function
it names, checked against that function's own annotations. Adding a parameter to
a measure therefore makes it settable from a recipe with no change here.
"""

from __future__ import annotations

import collections.abc
import importlib.metadata
import inspect
import types
import typing
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

import eegtable as ef
from eegtable.complexity import detrended_fluctuation, lempel_ziv_complexity, permutation_entropy
from eegtable.connectivity import spectral_connectivity, spectral_connectivity_time
from eegtable.cycles import cycle_features
from eegtable.irasa import irasa
from eegtable.phase import pac_surrogates
from eegtable.spectral_model import spectral_parameterization
from eegtable.table import FeatureTable

Kind = Literal["spectra", "series", "signals", "pac", "connectivity", "microstates"]
"""What a measure is computed from, named after the input the runner builds."""

SUPPLIED = frozenset(
    {
        "spectra",
        "series",
        "signals",
        "signal",
        "phase_signal",
        "amplitude_signal",
        "segmentation",
        "bands",
        "band",
        "windows",
        "baseline",
        "groups",
        "include_global",
        "trials",
    }
)
"""Parameters the runner fills in itself; a recipe reaches them through its own keys."""

_KINDS: dict[str, Kind] = {
    "spectra": "spectra",
    "series": "series",
    "signals": "signals",
    "phase_signal": "pac",
    "signal": "connectivity",
    "segmentation": "microstates",
}

_NOUNS: dict[object, str] = {bool: "booleans", int: "integers", float: "numbers", str: "strings"}


@dataclass(frozen=True, eq=False)
class Measure:
    """A library function a recipe can name.

    ``provider`` is the name and version of the installed distribution that registered
    it, None for a built-in measure.
    """

    name: str
    function: Callable[..., Any]
    provider: tuple[str, str] | None = None

    @property
    def parameters(self) -> Mapping[str, inspect.Parameter]:
        """The function's parameters, in order."""
        return inspect.signature(self.function).parameters

    @property
    def kind(self) -> Kind:
        """The input the measure is computed from."""
        return _KINDS[next(iter(self.parameters))]

    def takes(self, parameter: str) -> bool:
        """Whether the function has a parameter of this name."""
        return parameter in self.parameters

    @property
    def baseline_required(self) -> bool:
        """Whether the function has a baseline parameter without a default."""
        baseline = self.parameters.get("baseline")
        return baseline is not None and baseline.default is inspect.Parameter.empty

    @property
    def settable(self) -> dict[str, Any]:
        """Keyword parameters a recipe can set, mapped to their annotations.

        Excludes what the runner supplies and anything a recipe cannot express,
        such as an array.
        """
        hints = typing.get_type_hints(self.function)
        return {
            name: hints[name]
            for name, parameter in self.parameters.items()
            if parameter.kind is inspect.Parameter.KEYWORD_ONLY
            and name not in SUPPLIED
            and describe(hints[name]) is not None
        }

    @property
    def required(self) -> tuple[str, ...]:
        """Required keyword parameters a recipe must supply itself."""
        return tuple(
            name
            for name, parameter in self.parameters.items()
            if parameter.kind is inspect.Parameter.KEYWORD_ONLY
            and parameter.default is inspect.Parameter.empty
            and name not in SUPPLIED
        )


def _measures(*names: str) -> dict[str, Measure]:
    return {name: Measure(name, getattr(ef, name)) for name in names}


MEASURES: dict[str, Measure] = _measures(
    "integrated_band_power",
    "mean_psd",
    "mean_tfr_power",
    "periodic_power",
    "aperiodic",
    "peak_frequency",
    "spectral_centroid",
    "spectral_bandwidth",
    "spectral_entropy",
    "spectral_edge",
    "variance",
    "amplitude_quantile",
    "kurtosis",
    "line_length",
    "root_mean_square",
    "skewness",
    "zero_crossing_rate",
    "higuchi_fractal_dimension",
    "hjorth_mobility",
    "hjorth_complexity",
    "peak_to_peak",
    "mean_amplitude",
    "area_under_curve",
    "peak_amplitude",
    "peak_latency",
    "sample_entropy",
    "multiscale_entropy",
    "burst_count",
    "burst_rate",
    "burst_duration",
    "burst_amplitude",
    "fraction_above_threshold",
    "erds_mean",
    "erds_slope",
    "erd_magnitude",
    "erd_duration",
    "ers_magnitude",
    "ers_duration",
    "erds_peak_latency",
    "erds_onset_latency",
    "erds_rebound_latency",
    "itpc",
    "ppc",
    "pac",
    "envelope_correlation",
    "wpli",
    "microstate_coverage",
    "microstate_duration",
    "microstate_occurrence",
    "microstate_transitions",
)
MEASURES.update(
    {
        function.__name__: Measure(function.__name__, function)
        for function in (
            spectral_connectivity,
            spectral_connectivity_time,
            pac_surrogates,
            spectral_parameterization,
            irasa,
            permutation_entropy,
            lempel_ziv_complexity,
            detrended_fluctuation,
            cycle_features,
        )
    }
)
"""Every measure a recipe entry can name."""

_PLUGINS = "eegtable.measures"


def lookup(name: str) -> Measure | None:
    """The measure a recipe names: built in, or registered by an installed package.

    A package registers a measure under the ``eegtable.measures`` entry-point group,
    naming a function that follows the built-in measures' signatures: its first
    parameter names its input (``spectra``, ``series``, ``signals``, ``signal``,
    ``phase_signal`` or ``segmentation``) and its other parameters are keyword-only.
    """
    measure = MEASURES.get(name)
    if measure is not None:
        return measure
    # Read from installed packages on first use, not at import: a broken plugin then
    # affects only the recipes that name it. Every worker process repeats this lookup.
    for entry in importlib.metadata.entry_points(group=_PLUGINS, name=name):
        dist = getattr(entry, "dist", None)
        found = Measure(name, entry.load(), None if dist is None else (dist.name, dist.version))
        first = next(iter(found.parameters), None)
        if first not in _KINDS:
            raise ValueError(
                f"measure {name!r}, registered by an installed package, must take its input "
                f"first, named one of {sorted(_KINDS)}; it takes {first!r}."
            )
        MEASURES[name] = found
        return found
    return None


def get(name: str) -> Measure:
    """:func:`lookup`, for a name a validated recipe has already resolved."""
    measure = lookup(name)
    if measure is None:
        raise KeyError(f"unknown measure {name!r}")
    return measure


SEGMENTATION = Measure("segment", ef.segment)
"""Microstate segmentation, configured once per recipe and shared by its measures."""

GRAPH: dict[str, Callable[..., FeatureTable]] = {
    "global_efficiency": ef.global_efficiency,
    "clustering_coefficient": ef.clustering_coefficient,
}
"""Network summaries a connectivity entry can add."""

SPECTRAL_INPUT: dict[str, tuple[str, tuple[str, ...]]] = {
    "mean_psd": ("a power spectral density", ("welch", "multitaper")),
    "integrated_band_power": ("a power spectral density", ("welch", "multitaper")),
    "mean_tfr_power": ("time-frequency power", ("morlet",)),
    "spectral_parameterization": ("a power spectral density", ("welch", "multitaper")),
}
"""Measures that read only one kind of spectrum: what they read, and the methods giving it.

A spectral measure not named here reads either kind. The two are dimensionally
different quantities, so no recipe can feed a measure the wrong one.
"""

REQUIRES: dict[str, tuple[str, str]] = {
    "wpli": ("mne_connectivity", "connectivity"),
    "spectral_connectivity": ("mne_connectivity", "connectivity"),
    "spectral_connectivity_time": ("mne_connectivity", "connectivity"),
    "pac_surrogates": ("tensorpac", "pac"),
    "spectral_parameterization": ("specparam", "spectral-model"),
    "irasa": ("neurodsp", "irasa"),
    "cycle_features": ("bycycle", "cycles"),
    "permutation_entropy": ("antropy", "complexity"),
    "lempel_ziv_complexity": ("antropy", "complexity"),
    "detrended_fluctuation": ("antropy", "complexity"),
    "microstate_coverage": ("sklearn", "microstates"),
    "microstate_duration": ("sklearn", "microstates"),
    "microstate_occurrence": ("sklearn", "microstates"),
    "microstate_transitions": ("sklearn", "microstates"),
}
"""Measures that need an optional dependency: the module, and the extra providing it."""


class Mismatch(ValueError):
    """A recipe value does not fit the annotation it was checked against."""


def convert(value: object, annotation: Any) -> object:
    """Turn a TOML value into what an annotation asks for.

    Integers are accepted where a float is expected, and lists become the tuples
    the library takes.

    Raises
    ------
    Mismatch
        When the value cannot be read as the annotation.
    """
    if annotation is bool:
        if isinstance(value, bool):
            return value
        raise Mismatch
    if annotation is int:
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        raise Mismatch
    if annotation is float:
        if isinstance(value, int | float) and not isinstance(value, bool):
            return float(value)
        raise Mismatch
    if annotation is str:
        if isinstance(value, str):
            return value
        raise Mismatch

    origin, args = typing.get_origin(annotation), typing.get_args(annotation)
    if origin is Literal:
        if any(value == option and type(value) is type(option) for option in args):
            return value
        raise Mismatch
    if origin in (typing.Union, types.UnionType):
        if type(value) in args:
            return convert(value, type(value))
        for option in args:
            if option is type(None) or describe(option) is None:
                continue
            try:
                return convert(value, option)
            except Mismatch:
                continue
        raise Mismatch
    if origin is tuple:
        if not isinstance(value, list):
            raise Mismatch
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(convert(item, args[0]) for item in value)
        if len(value) != len(args):
            raise Mismatch
        return tuple(convert(item, arg) for item, arg in zip(value, args, strict=True))
    if origin in (collections.abc.Sequence, list):
        if not isinstance(value, list):
            raise Mismatch
        return tuple(convert(item, args[0]) for item in value)
    if origin in (collections.abc.Mapping, dict):
        if not isinstance(value, dict):
            raise Mismatch
        return {convert(key, args[0]): convert(item, args[1]) for key, item in value.items()}
    raise Mismatch


def describe(annotation: Any) -> str | None:
    """Plain-language description of what an annotation accepts from a recipe.

    None when a recipe cannot express the annotation at all.
    """
    simple = {bool: "true or false", int: "an integer", float: "a number", str: "a string"}
    if annotation in simple:
        return simple[annotation]

    origin, args = typing.get_origin(annotation), typing.get_args(annotation)
    if origin is Literal:
        return "one of " + ", ".join(repr(option) for option in args)
    if origin in (typing.Union, types.UnionType):
        options = [describe(option) for option in args if option is not type(None)]
        present = [option for option in options if option is not None]
        return " or ".join(present) if present else None
    if origin is tuple:
        if len(args) == 2 and args[1] is Ellipsis:
            return _list_of(args[0])
        if len(set(args)) == 1 and args[0] in _NOUNS:
            return f"a list of {len(args)} {_NOUNS[args[0]]}"
        return None
    if origin in (collections.abc.Sequence, list):
        return _list_of(args[0])
    if origin in (collections.abc.Mapping, dict):
        key, value = describe(args[0]), describe(args[1])
        return f"a table mapping {key} to {value}" if key and value else None
    return None


def _list_of(item: Any) -> str | None:
    return f"a list of {_NOUNS[item]}" if item in _NOUNS else None
