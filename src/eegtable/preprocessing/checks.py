"""Recording-dependent validation before numerical transforms and output writes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ._deps import require
from .config import (
    EventEpochSettings,
    ICASettings,
    PreprocessingConfig,
    ProcessingSettings,
    SSPSettings,
)
from .epochs import analysis_bounds
from .ica import validate_iclabel_passband
from .raw import (
    filter_coefficients,
    good_eeg_names,
    notch_coefficients,
    require_names,
    validate_geometry,
)
from .sampling import validate_sampling

BUNDLE_SUFFIXES = ("_epo.fif", "_events.tsv", "_repairs.tsv", "_preprocessing.json", "_report.html")


def validate_processing(raw: Any, settings: ProcessingSettings) -> None:
    sfreq = raw.info["sfreq"]
    if settings.reference.add_channels and raw.info["custom_ref_applied"]:
        raise ValueError(
            "reference.add_channels: restore acquisition electrodes before re-referencing; "
            "source already has a custom reference"
        )
    if settings.filter.l_freq is not None or settings.filter.h_freq is not None:
        filter_coefficients(sfreq, settings.filter)
    if settings.filter.notch_freqs:
        notch_coefficients(sfreq, settings.filter.notch_freqs)
    if settings.crop is not None and settings.crop.tmax > raw.times[-1]:
        raise ValueError("crop.tmax: exceeds acquisition endpoint")
    if settings.annotations.muscle is not None:
        low, high = settings.annotations.muscle.filter_freq
        if high >= sfreq / 2:
            raise ValueError(
                f"annotations.muscle.filter_freq exceeds the {sfreq / 2:g} Hz Nyquist frequency"
            )
        if low < raw.info["highpass"] or high > raw.info["lowpass"]:
            raise ValueError("annotations.muscle.filter_freq: outside source passband")
    if settings.artifact is not None:
        model = settings.artifact.settings
        channels = list(model.eog_channels)
        if isinstance(model, (ICASettings, SSPSettings)) and model.ecg_channel is not None:
            channels.append(model.ecg_channel)
        require_names(raw, channels, "artifact")
        if isinstance(model, ICASettings):
            validate_geometry(raw)
            if model.method == "picard":
                require("picard", "preprocessing-auto")
            if model.iclabel is not None:
                require("mne_icalabel", "preprocessing-auto")
                validate_iclabel_passband(
                    max(raw.info["highpass"], settings.filter.l_freq or 0.0, model.l_freq),
                    min(raw.info["lowpass"], settings.filter.h_freq or raw.info["lowpass"]),
                )
            if model.n_components is not None and model.n_components > len(good_eeg_names(raw)):
                raise ValueError("artifact.ica.n_components: exceeds good EEG channel count")
        reference = settings.artifact.reference
        if isinstance(reference, tuple):
            restored = set(settings.reference.add_channels) - set(raw.ch_names)
            measured = [name for name in reference if name not in restored]
            require_names(raw, measured, "artifact.reference")
            if not set(reference) <= set(good_eeg_names(raw)) | restored:
                raise ValueError(
                    "artifact.reference: requires good retained EEG channels or "
                    "requested acquisition-reference electrodes"
                )
    if isinstance(settings.epochs, EventEpochSettings) and settings.epochs.events.stim_channel:
        require_names(raw, [settings.epochs.events.stim_channel], "epochs.events.stim_channel")
    tmin, tmax = analysis_bounds(settings.epochs, sfreq)
    if settings.sampling is not None:
        padding = settings.epochs.padding
        # MNE rounds each endpoint independently when constructing the epoch grid.
        first = round((tmin - padding) * sfreq)
        last = round((tmax + padding) * sfreq)
        validate_sampling(
            sfreq, settings.sampling, settings.filter, first / sfreq, last - first + 1
        )


def validate_source_destinations(raw: Any, config: PreprocessingConfig) -> None:
    # Reader-reported companion files count as sources too.
    sources = {config.input.path.resolve()}
    sources.update(Path(path).resolve() for path in raw.filenames if path is not None)
    output = config.output
    for suffix in BUNDLE_SUFFIXES:
        target = output.directory / f"{output.name}{suffix}"
        if target.resolve() in sources:
            raise ValueError(f"output: {target} collides with acquisition input")
