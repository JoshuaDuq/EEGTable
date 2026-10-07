"""Seeded ICA fitting on a separate high-pass training copy."""

from __future__ import annotations

from typing import Any

import mne  # type: ignore[import-untyped]
import numpy as np

from ._deps import require
from .artifacts import ArtifactModel
from .config import ICLABEL_CLASSES, FilterSettings, ICASettings
from .provenance import channel_identity, fingerprint, identity
from .raw import filter_raw, good_eeg_names, require_names, validate_geometry

# ICLabel expects extended infomax; picard reproduces it with ortho off and extended on.
METHODS: dict[str, tuple[str, dict[str, Any]]] = {
    "fastica": ("fastica", {}),
    "infomax": ("infomax", {"extended": True}),
    "picard": ("picard", {"ortho": False, "extended": True}),
}


def validate_iclabel_passband(highpass: float, lowpass: float) -> None:
    if highpass != 1.0 or lowpass != 100.0:
        raise ValueError(
            "artifact.ica.iclabel: requires ICA training data filtered to 1-100 Hz; "
            f"got {highpass:g}-{lowpass:g} Hz."
        )


def _validate_iclabel_reference(raw: Any, picks: list[str]) -> None:
    # MNE's custom-reference flag also marks electrode references. CAR instead
    # makes the fitted EEG channels sum to zero, including with an applied projector.
    # Allow independently rounded samples from single-precision FIF storage.
    tolerance = 8 * np.finfo(np.float32).eps
    for start in range(0, raw.n_times, 100_000):
        data = raw.get_data(picks=picks, start=start, stop=min(start + 100_000, raw.n_times))
        residual = np.abs(data.sum(axis=0))
        amplitude = np.abs(data).sum(axis=0)
        if np.any(residual > tolerance * amplitude):
            raise ValueError(
                "artifact.ica.iclabel: training EEG requires an applied common average reference"
            )


def fit_ica(raw: Any, settings: ICASettings) -> ArtifactModel:
    """Fit seeded ICA on a high-passed training copy and score its components.

    The copy is high-passed at ``settings.l_freq`` only when the recording's
    stored highpass is lower. ICA is fitted on the good EEG channels, skipping BAD
    annotations and segments beyond the thresholds; FastICA and Picard must converge
    within ``max_iter``. MNE's EOG and ECG detectors and, when configured, ICLabel
    (on the copy without its BAD spans) score the components; nothing is excluded.

    Parameters
    ----------
    raw : mne.io.Raw
        Continuous recording with at least two independent good EEG channels.
        ICLabel additionally requires electrode positions, at least four good EEG
        channels, an applied common average reference and training data filtered
        to exactly 1-100 Hz.
    settings : ICASettings
        Algorithm, components (at most the EEG rank; None uses the rank), seed,
        iteration limit, thresholds, segment length, artifact channels and
        ICLabel settings.

    Returns
    -------
    ArtifactModel
        Method ``"ica"``. Its evidence holds the rank, detector scores, ICLabel
        classes, each detector's suggestions and their union
        ``suggested_exclude``.
    """
    require("sklearn", "preprocessing")
    if settings.method == "picard":
        require("picard", "preprocessing-auto")
    if settings.iclabel is not None:
        require("mne_icalabel", "preprocessing-auto")
        validate_iclabel_passband(max(raw.info["highpass"], settings.l_freq), raw.info["lowpass"])
        validate_geometry(raw)
    picks = good_eeg_names(raw)
    if settings.iclabel is not None:
        _validate_iclabel_reference(raw, picks)
    channels = [*settings.eog_channels, *([settings.ecg_channel] if settings.ecg_channel else [])]
    require_names(raw, channels, "artifact.ica")
    # Train on a separate high-pass copy; only add the missing high-pass, never lower one.
    training = (
        raw.copy()
        if raw.info["highpass"] >= settings.l_freq
        else filter_raw(raw, FilterSettings(l_freq=settings.l_freq))
    )
    rank = int(mne.compute_rank(training, proj=False)["eeg"])
    components = rank if settings.n_components is None else settings.n_components
    if rank < 2 or components > rank:
        raise ValueError(f"artifact.ica.n_components: requested {components}, usable rank {rank}")
    method, fit_params = METHODS[settings.method]
    model = mne.preprocessing.ICA(
        n_components=components,
        method=method,
        fit_params=fit_params,
        rng=settings.random_state,
        max_iter=settings.max_iter,
    )
    model.fit(
        training,
        picks=picks,
        reject=settings.reject,
        flat=settings.flat,
        tstep=settings.tstep,
        reject_by_annotation=True,
    )
    # MNE's infomax reports max_iter both when its weight change converges and when it runs
    # out of iterations, so only FastICA and Picard can be held to the limit.
    if settings.method != "infomax" and model.n_iter_ >= settings.max_iter:
        raise ValueError("artifact.ica.max_iter: ICA did not converge")
    # Scores and detector verdicts are evidence for review; nothing is excluded here.
    scores: dict[str, Any] = {}
    suggested: dict[str, Any] = {}
    for channel in settings.eog_channels:
        found, scores[channel] = model.find_bads_eog(training, ch_name=channel)
        suggested[channel] = [int(index) for index in found]
    if settings.ecg_channel is not None:
        found, scores[settings.ecg_channel] = model.find_bads_ecg(
            training, ch_name=settings.ecg_channel, method="correlation"
        )
        suggested[settings.ecg_channel] = [int(index) for index in found]
    evidence: dict[str, Any] = {"rank": rank, "scores": scores}
    if settings.iclabel is not None:
        evidence["iclabel"] = _label_components(training, model, settings)
        suggested["iclabel"] = evidence["iclabel"]["suggested"]
    evidence["suggested"] = suggested
    evidence["suggested_exclude"] = sorted({int(i) for found in suggested.values() for i in found})
    fit_id = identity(
        {
            "data": fingerprint(raw),
            "training": fingerprint(training),
            "settings": settings,
            "method": "ica",
        }
    )
    return ArtifactModel("ica", model, fit_id, channel_identity(raw), evidence)


def _label_components(training: Any, model: Any, settings: ICASettings) -> dict[str, Any]:
    from mne_icalabel.iclabel import (  # type: ignore[import-untyped]
        iclabel_label_components,
    )

    assert settings.iclabel is not None
    # ICLabel reads every sample; BAD spans were left out of the fit and are left unfiltered.
    labelled = mne.io.RawArray(
        training.get_data(reject_by_annotation="omit"), training.info, verbose=False
    )
    # inplace=False keeps the classifier's verdict out of the saved ICA object.
    probabilities = np.asarray(
        iclabel_label_components(labelled, model.copy(), inplace=False, backend="onnx"), dtype=float
    )
    if probabilities.shape != (model.n_components_, len(ICLABEL_CLASSES)):
        raise ValueError("artifact.ica.iclabel: unexpected classifier output shape")
    winners = probabilities.argmax(axis=1)
    labels = [ICLABEL_CLASSES[index] for index in winners]
    confidence = probabilities[np.arange(len(winners)), winners]
    suggested = [
        int(index)
        for index, (label, score) in enumerate(zip(labels, confidence, strict=True))
        if label not in settings.iclabel.keep and score >= settings.iclabel.threshold
    ]
    return {
        "classes": list(ICLABEL_CLASSES),
        "labels": labels,
        "probabilities": probabilities,
        "suggested": suggested,
    }
