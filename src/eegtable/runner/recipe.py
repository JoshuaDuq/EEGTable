"""Recipes: a declarative description of a feature-extraction run.

A recipe is a TOML file naming the input epochs, where results go, the bands,
windows and ROIs, how spectra and band signals are computed, and one entry per
measure. Everything is checked when the recipe loads, and every problem found is
reported at once, so a typo fails before the first recording is read.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import math
import os
import re
import tomllib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, TypeVar

from eegtable.bands import BANDS_STANDARD, Band
from eegtable.runner.measures import (
    GRAPH,
    REQUIRES,
    SEGMENTATION,
    SPECTRAL_INPUT,
    Measure,
    Mismatch,
    convert,
    lookup,
)
from eegtable.runner.measures import describe as describe_annotation
from eegtable.spectra import Window, WindowStatistic

WHOLE_EPOCH = Window("all", -math.inf, math.inf)
"""The window a measure spans when a recipe defines none."""

SpectralMethod = Literal["welch", "multitaper", "morlet"]
TrialsBy = Literal["all", "event", "metadata"]

_SECTIONS = (
    "inputs",
    "output",
    "bands",
    "windows",
    "rois",
    "spectra",
    "band_signal",
    "trials",
    "microstates",
    "defaults",
    "features",
)
# Entry keys a [defaults] section can set for every entry that takes them.
_DEFAULT_KEYS = ("bands", "windows", "spatial", "series")
_SPATIAL_LEVELS = ("channels", "rois", "global")
_SPECTRA_KEYS: dict[str, tuple[str, ...]] = {
    "welch": ("n_fft", "n_overlap", "window_statistic"),
    "multitaper": ("bandwidth",),
    "morlet": (
        "n_freqs",
        "spacing",
        "n_cycles_factor",
        "min_cycles",
        "max_cycles",
        "decim",
        "window_statistic",
    ),
}
_BROADBAND = "broadband"

T = TypeVar("T")


@dataclass(frozen=True)
class Inputs:
    """Where the preprocessed epochs are and which channels to keep."""

    root: Path
    pattern: str = "**/*_epo.fif"
    picks: str | tuple[str, ...] = "eeg"
    exclude_bads: bool = True


@dataclass(frozen=True)
class Output:
    """Where results are written."""

    root: Path
    epoch_metadata: bool = True


@dataclass(frozen=True)
class SpectraSettings:
    """How power spectra are computed for the spectral measures."""

    method: SpectralMethod = "welch"
    fmin: float = 1.0
    fmax: float = 45.0
    n_fft: int | None = None
    n_overlap: int | None = None
    # Fixed in hertz rather than left to MNE, whose default of 8 / window_length Hz
    # smooths a 1 s window over +/-4 Hz, wider than the delta or theta band.
    bandwidth: float = 2.0
    n_freqs: int = 40
    spacing: Literal["log", "linear"] = "log"
    n_cycles_factor: float = 2.0
    min_cycles: float = 3.0
    max_cycles: float = 15.0
    decim: int = 4
    window_statistic: WindowStatistic = "mean"

    def used(self) -> dict[str, object]:
        # Only what this method reads enters its columns' identity: were another method's
        # settings recorded, changing one of their defaults would rename every column. The
        # statistic is left out at the mean, so columns computed before it existed keep names.
        keys = ("method", "fmin", "fmax", *_SPECTRA_KEYS[self.method])
        values = {key: getattr(self, key) for key in keys}
        return {
            key: value
            for key, value in values.items()
            if value is not None and (key, value) != ("window_statistic", "mean")
        }


@dataclass(frozen=True)
class BandSignalSettings:
    """Padding for the bandpass filter behind every band signal."""

    pad_sec: float = 0.5
    pad_cycles: float = 3.0


@dataclass(frozen=True)
class TrialGrouping:
    """How cross-trial measures group epochs into rows."""

    by: TrialsBy = "all"
    column: str | None = None


@dataclass(frozen=True)
class RoiPattern:
    """An ROI given as regular expressions, resolved against each recording's channels."""

    patterns: tuple[str, ...]

    def resolve(self, ch_names: Sequence[str]) -> tuple[str, ...]:
        """The channels any pattern matches, in the recording's order."""
        compiled = [re.compile(pattern) for pattern in self.patterns]
        return tuple(name for name in ch_names if any(c.search(name) for c in compiled))


Roi = tuple[str, ...] | RoiPattern


@dataclass(frozen=True)
class FeatureSpec:
    """One ``[[features]]`` entry, resolved against the rest of the recipe."""

    measure: str
    bands: tuple[Band, ...] = ()
    windows: tuple[Window, ...] = (WHOLE_EPOCH,)
    baseline: Window | None = None
    spatial: tuple[str, ...] = ()
    series: tuple[Band | None, ...] = ()
    pairs: tuple[tuple[Band, Band], ...] = ()
    ratios: tuple[tuple[str, str], ...] = ()
    asymmetry: tuple[tuple[str, str], ...] = ()
    graph: tuple[str, ...] = ()
    clustering_threshold: float | None = None
    params: Mapping[str, object] = field(default_factory=dict)
    # Position of the [[features]] entry in the file; entries naming several
    # measures expand into one spec each, all sharing it.
    entry: int = 0


@dataclass(frozen=True)
class Recipe:
    """A validated recipe."""

    path: Path
    text: str
    inputs: Inputs
    output: Output
    bands: tuple[Band, ...]
    windows: tuple[Window, ...]
    rois: Mapping[str, Roi]
    spectra: SpectraSettings
    band_signal: BandSignalSettings
    trials: TrialGrouping
    microstates: Mapping[str, object]
    features: tuple[FeatureSpec, ...]


class RecipeError(ValueError):
    """A recipe that cannot be run, with every problem found in it."""

    def __init__(self, path: Path, problems: Sequence[str]) -> None:
        self.path = path
        self.problems = tuple(problems)
        noun = "problem" if len(self.problems) == 1 else "problems"
        listing = "\n".join(f"  - {problem}" for problem in self.problems)
        super().__init__(f"{path}: {len(self.problems)} {noun}\n{listing}")


def load_recipe(source: str | os.PathLike[str] | Mapping[str, Any]) -> Recipe:
    """Read and validate a recipe.

    Parameters
    ----------
    source : path-like or mapping
        The TOML file, whose relative paths resolve against its directory; or a
        mapping with the same structure, for a recipe built in code, whose relative
        paths resolve against the working directory. A mapping may leave out
        ``inputs`` and ``output``, which only a batch run reads: applied to epochs
        in memory with :func:`eegtable.extract`, it needs neither.

    Returns
    -------
    Recipe

    Raises
    ------
    RecipeError
        Listing every problem found.
    """
    if isinstance(source, Mapping):
        return _recipe_from_mapping(source)
    source = Path(source)
    text = source.read_text()
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise RecipeError(source, [f"not valid TOML: {exc}"]) from exc
    return _Parser(source).parse(text, data)


def _recipe_from_mapping(source: Mapping[str, Any]) -> Recipe:
    data = copy.deepcopy(dict(source))
    for section in ("inputs", "output"):
        table = data.setdefault(section, {})
        if isinstance(table, dict):
            table.setdefault("root", ".")
    # Stands in for a file's text in the provenance, so the same mapping always hashes
    # the same; settings_sha256 hashes the parsed recipe either way.
    text = json.dumps(source, sort_keys=True, default=str)
    return _Parser(Path.cwd() / "<recipe>").parse(text, data)


class _Parser:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.problems: list[str] = []

    def problem(self, message: str) -> None:
        self.problems.append(message)

    def parse(self, text: str, data: dict[str, Any]) -> Recipe:
        for section in data:
            if section not in _SECTIONS:
                self.problem(f"unknown section [{section}]; expected one of {list(_SECTIONS)}")

        inputs = self.inputs(self.table(data, "inputs", required=True))
        output = self.output(self.table(data, "output", required=True))
        bands = self.bands(data)
        windows = self.windows(self.table(data, "windows"))
        rois = self.rois(self.table(data, "rois"))
        spectra = self.spectra(self.table(data, "spectra"), bands)
        band_signal = self.band_signal(self.table(data, "band_signal"))
        trials = self.trials(self.table(data, "trials"))
        microstates = self.microstates(self.table(data, "microstates"))
        context = _EntryContext(
            bands={b.name: b for b in bands},
            windows={w.name: w for w in windows},
            rois=rois,
            method=spectra.method,
            defaults={},
        )
        context = replace(context, defaults=self.defaults(self.table(data, "defaults"), context))
        features = self.features(data.get("features"), context)

        if self.problems:
            raise RecipeError(self.path, self.problems)
        assert inputs is not None and output is not None
        return Recipe(
            path=self.path,
            text=text,
            inputs=inputs,
            output=output,
            bands=bands,
            windows=windows,
            rois=rois,
            spectra=spectra,
            band_signal=band_signal,
            trials=trials,
            microstates=microstates,
            features=features,
        )

    # --- sections -----------------------------------------------------------

    def inputs(self, table: dict[str, Any]) -> Inputs | None:
        self.only("inputs", table, ("root", "pattern", "picks", "exclude_bads"))
        root = self.value("inputs", table, "root", str, "a string", required=True)
        pattern = self.value("inputs", table, "pattern", str, "a string", default="**/*_epo.fif")
        picks: str | tuple[str, ...] = "eeg"
        if "picks" in table:
            raw = table["picks"]
            if isinstance(raw, str):
                picks = raw
            elif _is_string_list(raw) and raw:
                picks = tuple(raw)
            else:
                self.problem("inputs: picks must be a channel type or a list of channel names")
        exclude_bads = self.value("inputs", table, "exclude_bads", bool, "true or false", True)
        if root is None:
            return None
        return Inputs(self.resolve(root), pattern, picks, exclude_bads)

    def output(self, table: dict[str, Any]) -> Output | None:
        self.only("output", table, ("root", "epoch_metadata"))
        root = self.value("output", table, "root", str, "a string", required=True)
        metadata = self.value("output", table, "epoch_metadata", bool, "true or false", True)
        return None if root is None else Output(self.resolve(root), metadata)

    def bands(self, data: dict[str, Any]) -> tuple[Band, ...]:
        if "bands" not in data:
            return BANDS_STANDARD
        table = self.table(data, "bands")
        if not table:
            self.problem("[bands] is empty; define at least one band or leave the section out")
        return tuple(
            band
            for name, bounds in table.items()
            if (band := self.bounded("bands", name, bounds, Band)) is not None
        )

    def windows(self, table: dict[str, Any]) -> tuple[Window, ...]:
        if WHOLE_EPOCH.name in table:
            self.problem(f"windows: {WHOLE_EPOCH.name!r} is reserved for the whole epoch")
        return tuple(
            window
            for name, bounds in table.items()
            if name != WHOLE_EPOCH.name
            and (window := self.bounded("windows", name, bounds, Window)) is not None
        )

    def rois(self, table: dict[str, Any]) -> dict[str, Roi]:
        rois: dict[str, Roi] = {}
        for name, members in table.items():
            if name == "global":
                self.problem("rois: 'global' is reserved for the mean over all channels")
            elif isinstance(members, dict):
                pattern = self.roi_pattern(name, members)
                if pattern is not None:
                    rois[name] = pattern
            elif not _is_string_list(members) or not members:
                self.problem(
                    f"rois: {name} must be a non-empty list of channel names, "
                    "or { match = [patterns] }"
                )
            else:
                rois[name] = tuple(members)
        return rois

    def roi_pattern(self, name: str, table: dict[str, Any]) -> RoiPattern | None:
        patterns = table.get("match")
        if set(table) != {"match"} or not _is_string_list(patterns) or not patterns:
            self.problem(f"rois: {name} must be {{ match = [patterns] }} with at least one pattern")
            return None
        valid = True
        for pattern in patterns:
            try:
                re.compile(pattern)
            except re.error as exc:
                self.problem(f"rois: {name} has an invalid pattern {pattern!r}: {exc}")
                valid = False
        return RoiPattern(tuple(patterns)) if valid else None

    def spectra(self, table: dict[str, Any], bands: tuple[Band, ...]) -> SpectraSettings:
        method = self.value("spectra", table, "method", str, "a string", default="welch")
        if method not in _SPECTRA_KEYS:
            self.problem(f"spectra: method must be one of {list(_SPECTRA_KEYS)}, got {method!r}")
            method = "welch"
        for key in table:
            owners = [m for m, keys in _SPECTRA_KEYS.items() if key in keys]
            if key in ("method", "fmin", "fmax"):
                continue
            if not owners:
                self.problem(f"spectra: unknown key {key!r}")
            elif method not in owners:
                named = " or ".join(repr(m) for m in owners)
                self.problem(f"spectra: {key} applies only to method = {named}")

        low = min((b.fmin for b in bands), default=1.0)
        high = max((b.fmax for b in bands), default=45.0)
        fmin = self.number("spectra", table, "fmin", low, minimum=0.0)
        fmax = self.number("spectra", table, "fmax", high, minimum=0.0)
        if fmin >= fmax:
            self.problem(f"spectra: fmin ({fmin}) must be below fmax ({fmax})")

        defaults = SpectraSettings()
        statistic = self.value(
            "spectra", table, "window_statistic", str, "a string", defaults.window_statistic
        )
        if statistic not in ("mean", "median"):
            self.problem(f"spectra: window_statistic must be 'mean' or 'median', got {statistic!r}")
            statistic = defaults.window_statistic
        spacing = self.value("spectra", table, "spacing", str, "a string", defaults.spacing)
        if spacing not in ("log", "linear"):
            self.problem(f"spectra: spacing must be 'log' or 'linear', got {spacing!r}")
            spacing = defaults.spacing
        if spacing == "log" and method == "morlet" and fmin <= 0.0:
            self.problem("spectra: log spacing needs fmin above 0")
        min_cycles = self.number("spectra", table, "min_cycles", defaults.min_cycles, above=0.0)
        max_cycles = self.number("spectra", table, "max_cycles", defaults.max_cycles, above=0.0)
        if max_cycles < min_cycles:
            self.problem("spectra: max_cycles must not be below min_cycles")
        return SpectraSettings(
            method=method,
            fmin=fmin,
            fmax=fmax,
            n_fft=self.integer("spectra", table, "n_fft", None, minimum=1),
            n_overlap=self.integer("spectra", table, "n_overlap", None, minimum=0),
            bandwidth=self.number("spectra", table, "bandwidth", defaults.bandwidth, above=0.0),
            n_freqs=self.integer("spectra", table, "n_freqs", defaults.n_freqs, minimum=2)
            or defaults.n_freqs,
            spacing=spacing,
            n_cycles_factor=self.number(
                "spectra", table, "n_cycles_factor", defaults.n_cycles_factor, above=0.0
            ),
            min_cycles=min_cycles,
            max_cycles=max_cycles,
            decim=self.integer("spectra", table, "decim", defaults.decim, minimum=1)
            or defaults.decim,
            window_statistic=statistic,
        )

    def band_signal(self, table: dict[str, Any]) -> BandSignalSettings:
        self.only("band_signal", table, ("pad_sec", "pad_cycles"))
        defaults = BandSignalSettings()
        return BandSignalSettings(
            pad_sec=self.number("band_signal", table, "pad_sec", defaults.pad_sec, minimum=0.0),
            pad_cycles=self.number(
                "band_signal", table, "pad_cycles", defaults.pad_cycles, minimum=0.0
            ),
        )

    def trials(self, table: dict[str, Any]) -> TrialGrouping:
        self.only("trials", table, ("by", "column"))
        by = self.value("trials", table, "by", str, "a string", default="all")
        column = self.value("trials", table, "column", str, "a string", default=None)
        if by not in ("all", "event", "metadata"):
            self.problem(f"trials: by must be 'all', 'event' or 'metadata', got {by!r}")
            return TrialGrouping()
        if by == "metadata" and column is None:
            self.problem("trials: by = 'metadata' needs a column naming the metadata field")
        if by != "metadata" and column is not None:
            self.problem("trials: column applies only when by = 'metadata'")
        return TrialGrouping(by, column)

    def microstates(self, table: dict[str, Any]) -> dict[str, object]:
        return self.parameters("microstates", table, SEGMENTATION.settable)

    def defaults(self, table: dict[str, Any], context: _EntryContext) -> dict[str, Any]:
        self.only("defaults", table, _DEFAULT_KEYS)
        known = {key: value for key, value in table.items() if key in _DEFAULT_KEYS}
        # A default that fails here is dropped, so it is not reported again by every
        # entry that would have inherited it.
        for key, defined in (("bands", context.bands), ("windows", context.whole_and_windows)):
            if key in known:
                before = len(self.problems)
                self.names("defaults", known, key, defined, default=())
                if len(self.problems) > before:
                    known.pop(key)
        if "spatial" in known:
            raw = known["spatial"]
            if not _is_string_list(raw) or not raw or len(set(raw)) != len(raw):
                self.problem(
                    "defaults: spatial must be a list of distinct levels from "
                    f"{list(_SPATIAL_LEVELS)}"
                )
                known.pop("spatial")
            elif unknown := [level for level in raw if level not in _SPATIAL_LEVELS]:
                self.problem(
                    f"defaults: spatial levels {unknown} are not among {list(_SPATIAL_LEVELS)}"
                )
                known.pop("spatial")
        if "series" in known:
            series = known["series"]
            allowed = {_BROADBAND, *context.bands}
            if not _is_string_list(series) or not series:
                self.problem("defaults: series must be a list of 'broadband' and band names")
                known.pop("series")
            elif undefined := [item for item in series if item not in allowed]:
                self.problem(f"defaults: series names undefined {', '.join(map(repr, undefined))}")
                known.pop("series")
        return known

    def features(self, entries: object, context: _EntryContext) -> tuple[FeatureSpec, ...]:
        if not isinstance(entries, list) or not entries:
            self.problem("the recipe needs at least one [[features]] entry")
            return ()
        specs = []
        for index, entry in enumerate(entries):
            for expanded in self.expand(index, entry):
                spec = self.feature(index, expanded, context)
                if spec is not None:
                    specs.append(spec)
        return tuple(specs)

    def expand(self, index: int, entry: object) -> list[object]:
        """One entry per measure an entry names with ``measures``."""
        if not isinstance(entry, dict) or "measures" not in entry:
            return [entry]
        if "measure" in entry:
            self.problem(f"features[{index}]: name one measure or a list of measures, not both")
            return []
        names = entry["measures"]
        if not _is_string_list(names) or not names or len(set(names)) != len(names):
            self.problem(f"features[{index}]: measures must be a non-empty list of distinct names")
            return []
        shared = {key: value for key, value in entry.items() if key != "measures"}
        return [{"measure": name, **shared} for name in names]

    def feature(self, index: int, entry: object, context: _EntryContext) -> FeatureSpec | None:
        if not isinstance(entry, dict):
            self.problem(f"features[{index}] must be a table")
            return None
        name = entry.get("measure")
        if not isinstance(name, str):
            self.problem(f"features[{index}]: 'measure' naming the measure is required")
            return None
        try:
            measure = lookup(name)
        except ValueError as exc:
            self.problem(f"features[{index}]: {exc}")
            return None
        if measure is None:
            self.problem(f"features[{index}]: unknown measure {name!r}")
            return None
        where = f"features[{index}] ({name})"
        if name in REQUIRES and importlib.util.find_spec(REQUIRES[name][0]) is None:
            module, extra = REQUIRES[name]
            self.problem(
                f"{where}: {name} needs {module}, which is not installed; "
                f"install it with: pip install 'eegtable[{extra}]'"
            )
        if name in SPECTRAL_INPUT and context.method not in SPECTRAL_INPUT[name][1]:
            quantity, methods = SPECTRAL_INPUT[name]
            self.problem(
                f"{where}: {name} reads {quantity}, which [spectra] method = "
                f"{context.method!r} does not give; use {' or '.join(map(repr, methods))}"
            )

        settable = measure.settable
        for parameter in measure.required:
            if parameter not in entry:
                self.problem(f"{where}: {parameter} is required")
        accepted = _entry_keys(measure)
        allowed = accepted | set(settable)
        for key in entry:
            if key not in allowed:
                self.problem(f"{where}: {name} does not take {key!r}")
        inherited = {
            key: value
            for key, value in context.defaults.items()
            if key in accepted and key not in entry
        }
        entry = {**entry, **self.inherit(measure, inherited)}

        bands = self.names(
            where, entry, "bands", context.bands, default=tuple(context.bands.values())
        )
        baseline = self.baseline(where, entry, measure, context)
        spatial = self.spatial(where, entry, measure, context)
        graph, threshold = self.graph(where, entry, spatial)
        if "windows" in inherited and baseline is not None:
            # An inherited list sets the baseline aside, as leaving windows out does.
            others = [window for window in entry["windows"] if window != baseline.name]
            entry["windows"] = others or entry["windows"]
        windows = self.entry_windows(where, entry, measure, baseline, context)
        return FeatureSpec(
            measure=name,
            entry=index,
            bands=bands if "bands" in accepted else (),
            windows=windows,
            baseline=baseline,
            spatial=spatial,
            series=self.series(where, entry, measure, context),
            pairs=self.pairs(where, entry, measure, context),
            ratios=self.ratios(where, entry, bands),
            asymmetry=self.asymmetry(where, entry, spatial),
            graph=graph,
            clustering_threshold=threshold,
            params=self.parameters(where, {k: entry[k] for k in settable if k in entry}, settable),
        )

    # --- entry keys ---------------------------------------------------------

    def inherit(self, measure: Measure, inherited: dict[str, Any]) -> dict[str, Any]:
        # A shared spatial default keeps only the levels this measure has, so one
        # [defaults] line serves power and connectivity alike.
        if "spatial" in inherited:
            levels = _spatial_levels(measure)[0]
            kept = [level for level in inherited["spatial"] if level in levels]
            if kept:
                inherited["spatial"] = kept
            else:
                del inherited["spatial"]
        return inherited

    def baseline(
        self, where: str, entry: dict[str, Any], measure: Measure, context: _EntryContext
    ) -> Window | None:
        name = entry.get("baseline")
        if name is None:
            if measure.baseline_required:
                self.problem(f"{where}: {measure.name} requires a baseline window")
            return None
        if not isinstance(name, str) or name not in context.windows:
            self.problem(f"{where}: baseline {name!r} is not a window defined in [windows]")
            return None
        return context.windows[name]

    def entry_windows(
        self,
        where: str,
        entry: dict[str, Any],
        measure: Measure,
        baseline: Window | None,
        context: _EntryContext,
    ) -> tuple[Window, ...]:
        if "windows" not in entry:
            if not context.windows:
                return (WHOLE_EPOCH,)
            remaining = tuple(w for w in context.windows.values() if w != baseline)
            if not remaining:
                self.problem(f"{where}: no windows remain once the baseline is set aside")
            return remaining
        chosen = self.names(where, entry, "windows", context.whole_and_windows, default=())
        if measure.kind == "spectra" and baseline is not None and baseline in chosen:
            self.problem(
                f"{where}: {measure.name} consumes its baseline window {baseline.name!r}, "
                "so it cannot also be an analysis window; remove it from windows"
            )
        return chosen

    def spatial(
        self, where: str, entry: dict[str, Any], measure: Measure, context: _EntryContext
    ) -> tuple[str, ...]:
        if not measure.takes("groups"):
            return ()
        levels, default = _spatial_levels(measure)
        raw = entry.get("spatial", list(default))
        if not _is_string_list(raw) or not raw or len(set(raw)) != len(raw):
            self.problem(f"{where}: spatial must be a list of distinct levels from {list(levels)}")
            return default
        unknown = [level for level in raw if level not in levels]
        if unknown:
            self.problem(f"{where}: spatial levels {unknown} are not among {list(levels)}")
            return default
        if "rois" in raw and not context.rois:
            self.problem(f"{where}: spatial 'rois' needs ROIs defined in [rois]")
        return tuple(raw)

    def series(
        self, where: str, entry: dict[str, Any], measure: Measure, context: _EntryContext
    ) -> tuple[Band | None, ...]:
        if measure.kind != "series":
            return ()
        raw = entry.get("series", [_BROADBAND])
        if measure.name in ("irasa", "cycle_features") and raw != [_BROADBAND]:
            self.problem(f"{where}: {measure.name} requires broadband series = ['broadband']")
            return (None,)
        if not _is_string_list(raw) or not raw:
            self.problem(f"{where}: series must be a list of 'broadband' and band names")
            return (None,)
        resolved: list[Band | None] = []
        for item in raw:
            if item == _BROADBAND:
                resolved.append(None)
            elif item in context.bands:
                resolved.append(context.bands[item])
            else:
                self.problem(f"{where}: series {item!r} is neither 'broadband' nor a defined band")
        return tuple(resolved)

    def pairs(
        self, where: str, entry: dict[str, Any], measure: Measure, context: _EntryContext
    ) -> tuple[tuple[Band, Band], ...]:
        if measure.kind != "pac":
            return ()
        if "pairs" not in entry:
            self.problem(f"{where}: pac needs pairs, e.g. pairs = [['theta', 'gamma']]")
            return ()
        pairs = self.string_pairs(where, "pairs", entry["pairs"])
        amplitudes = [amplitude for _, amplitude in pairs]
        shared = sorted({band for band in amplitudes if amplitudes.count(band) > 1})
        if shared:
            self.problem(
                f"{where}: pairs share the amplitude band {', '.join(map(repr, shared))}; PAC "
                "columns are named by amplitude band, so put such pairs in separate entries"
            )
        resolved = []
        for phase, amplitude in pairs:
            missing = [n for n in (phase, amplitude) if n not in context.bands]
            if missing:
                self.problem(f"{where}: pairs name undefined bands {missing}")
            else:
                resolved.append((context.bands[phase], context.bands[amplitude]))
        return tuple(resolved)

    def ratios(
        self, where: str, entry: dict[str, Any], bands: tuple[Band, ...]
    ) -> tuple[tuple[str, str], ...]:
        if "ratios" not in entry:
            return ()
        pairs = self.string_pairs(where, "ratios", entry["ratios"])
        computed = {b.name for b in bands}
        for pair in pairs:
            missing = [n for n in pair if n not in computed]
            if missing:
                self.problem(f"{where}: ratios need bands {missing} among this entry's bands")
        return pairs

    def asymmetry(
        self, where: str, entry: dict[str, Any], spatial: tuple[str, ...]
    ) -> tuple[tuple[str, str], ...]:
        if "asymmetry" not in entry:
            return ()
        if "channels" not in spatial:
            self.problem(f"{where}: asymmetry pairs channels, so spatial must include 'channels'")
        return self.string_pairs(where, "asymmetry", entry["asymmetry"])

    def graph(
        self, where: str, entry: dict[str, Any], spatial: tuple[str, ...]
    ) -> tuple[tuple[str, ...], float | None]:
        raw = entry.get("graph", [])
        if not _is_string_list(raw) or any(item not in GRAPH for item in raw):
            self.problem(f"{where}: graph must be a list drawn from {list(GRAPH)}")
            return (), None
        if raw and len(spatial) != 1:
            self.problem(
                f"{where}: graph measures summarize one network, so spatial must name a single "
                f"level, got {list(spatial)}"
            )
        threshold = self.optional_number(where, entry, "clustering_threshold")
        if "clustering_coefficient" in raw and threshold is None:
            self.problem(f"{where}: clustering_coefficient needs clustering_threshold")
        if threshold is not None and "clustering_coefficient" not in raw:
            self.problem(f"{where}: clustering_threshold applies only to clustering_coefficient")
        return tuple(raw), threshold

    # --- values -------------------------------------------------------------

    def parameters(
        self, where: str, table: dict[str, Any], settable: Mapping[str, Any]
    ) -> dict[str, object]:
        converted: dict[str, object] = {}
        for key, value in table.items():
            if key not in settable:
                self.problem(f"{where}: unknown key {key!r}; expected one of {sorted(settable)}")
                continue
            try:
                converted[key] = convert(value, settable[key])
            except Mismatch:
                wanted = describe_annotation(settable[key])
                self.problem(f"{where}: {key} must be {wanted}, got {value!r}")
        return converted

    def names(
        self,
        where: str,
        entry: dict[str, Any],
        key: str,
        defined: Mapping[str, T],
        default: tuple[T, ...],
    ) -> tuple[T, ...]:
        if key not in entry:
            return default
        raw = entry[key]
        if not _is_string_list(raw) or not raw:
            self.problem(f"{where}: {key} must be a non-empty list of names")
            return default
        undefined = [name for name in raw if name not in defined]
        if undefined:
            self.problem(f"{where}: {key} names undefined {', '.join(map(repr, undefined))}")
        return tuple(defined[name] for name in raw if name in defined)

    def string_pairs(self, where: str, key: str, raw: object) -> tuple[tuple[str, str], ...]:
        if not isinstance(raw, list) or not all(
            _is_string_list(pair) and len(pair) == 2 for pair in raw
        ):
            self.problem(f"{where}: {key} must be a list of [name, name] pairs")
            return ()
        return tuple((pair[0], pair[1]) for pair in raw)

    def bounded(
        self, section: str, name: str, bounds: object, build: Callable[[str, float, float], T]
    ) -> T | None:
        if not (
            isinstance(bounds, list)
            and len(bounds) == 2
            and all(isinstance(v, int | float) and not isinstance(v, bool) for v in bounds)
        ):
            self.problem(f"{section}: {name} must be [low, high], got {bounds!r}")
            return None
        try:
            return build(name, float(bounds[0]), float(bounds[1]))
        except ValueError as exc:
            self.problem(f"{section}: {exc}")
            return None

    def table(self, data: dict[str, Any], name: str, *, required: bool = False) -> dict[str, Any]:
        value = data.get(name)
        if value is None:
            if required:
                self.problem(f"[{name}] is required, with at least a root")
            return {}
        if not isinstance(value, dict):
            self.problem(f"{name} must be a [{name}] table")
            return {}
        return value

    def only(self, section: str, table: dict[str, Any], keys: Sequence[str]) -> None:
        for key in table:
            if key not in keys:
                self.problem(f"{section}: unknown key {key!r}; expected one of {list(keys)}")

    def value(
        self,
        section: str,
        table: dict[str, Any],
        key: str,
        kind: type[T],
        wanted: str,
        default: Any = None,
        *,
        required: bool = False,
    ) -> Any:
        if key not in table:
            if required:
                self.problem(f"{section}: {key} is required")
            return default
        value = table[key]
        if not isinstance(value, kind) or (kind is not bool and isinstance(value, bool)):
            self.problem(f"{section}: {key} must be {wanted}, got {value!r}")
            return default
        return value

    def number(
        self,
        section: str,
        table: dict[str, Any],
        key: str,
        default: float,
        *,
        minimum: float | None = None,
        above: float | None = None,
    ) -> float:
        value = self.optional_number(section, table, key, minimum=minimum, above=above)
        return default if value is None else value

    def optional_number(
        self,
        section: str,
        table: dict[str, Any],
        key: str,
        *,
        minimum: float | None = None,
        above: float | None = None,
    ) -> float | None:
        if key not in table:
            return None
        value = table[key]
        if not isinstance(value, int | float) or isinstance(value, bool):
            self.problem(f"{section}: {key} must be a number, got {value!r}")
            return None
        if minimum is not None and value < minimum:
            self.problem(f"{section}: {key} must be at least {minimum}, got {value}")
            return None
        if above is not None and value <= above:
            self.problem(f"{section}: {key} must be above {above}, got {value}")
            return None
        return float(value)

    def integer(
        self, section: str, table: dict[str, Any], key: str, default: int | None, *, minimum: int
    ) -> int | None:
        if key not in table:
            return default
        value = table[key]
        if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
            self.problem(f"{section}: {key} must be an integer of at least {minimum}")
            return default
        return value

    def resolve(self, raw: str) -> Path:
        path = Path(raw).expanduser()
        return path if path.is_absolute() else self.path.parent / path


@dataclass(frozen=True)
class _EntryContext:
    bands: Mapping[str, Band]
    windows: Mapping[str, Window]
    rois: Mapping[str, Roi]
    method: SpectralMethod
    defaults: dict[str, Any]

    @property
    def whole_and_windows(self) -> dict[str, Window]:
        """The windows an entry may name: the defined ones and the whole epoch."""
        return {WHOLE_EPOCH.name: WHOLE_EPOCH, **self.windows}


def _spatial_levels(measure: Measure) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """The spatial levels a measure has, and its default ones."""
    if measure.takes("include_global"):
        return _SPATIAL_LEVELS, ("channels", "global")
    return ("channels", "rois"), ("channels",)


def _entry_keys(measure: Measure) -> set[str]:
    keys = {"measure", "windows"}
    if measure.takes("bands") or measure.takes("band") or measure.kind == "signals":
        keys.add("bands")
    if measure.takes("baseline"):
        keys.add("baseline")
    if measure.takes("groups"):
        keys.add("spatial")
    if measure.kind == "series":
        keys.add("series")
    if measure.kind == "pac":
        keys.add("pairs")
    if measure.name in ("integrated_band_power", "mean_psd", "mean_tfr_power"):
        keys |= {"ratios", "asymmetry"}
    if measure.name in (
        "envelope_correlation",
        "wpli",
        "spectral_connectivity",
        "spectral_connectivity_time",
    ):
        keys |= {"graph", "clustering_threshold"}
    return keys


def _is_string_list(value: object) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) for item in value)
