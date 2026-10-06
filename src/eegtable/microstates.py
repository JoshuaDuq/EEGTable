from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from heapq import heapify, heappop, heappush
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.optimize import linear_sum_assignment
from scipy.signal import find_peaks

from eegtable._expand import window_mask
from eegtable._validation import minimum_sample_count, validate_names
from eegtable.signal import Signal
from eegtable.spectra import Window
from eegtable.table import ComputationSpec, FeatureMeta, FeatureTable, RowId


@dataclass(frozen=True, eq=False)
class MicrostateSegmentation:
    """Microstate templates and the state each sample was assigned to.

    Produced by :func:`segment`. The measures take this rather than a signal, so
    every measure describes the same segmentation: fitting separately per measure
    would give each one different templates.

    Parameters
    ----------
    templates : ndarray, shape (n_states, n_channels)
        Normalized topographies, one per state.
    states : ndarray of int, shape (n_epochs, n_times)
        Index of the template each sample was assigned to.
    labels : tuple of str
        Name of each state, in template order.
    times : ndarray, shape (n_times,)
        Time axis in seconds.
    sfreq : float
        Sampling frequency in Hz.
    row_ids : tuple of (str, int, str)
        Identity of each segmented epoch.
    global_explained_variance : float
        Squared spatial correlation of each sample with its assigned template,
        averaged with squared global field power as weights.
    computation : ComputationSpec
        How the templates were obtained and the segmentation settings.
    """

    templates: npt.NDArray[np.float64]
    states: npt.NDArray[np.int_]
    labels: tuple[str, ...]
    times: npt.NDArray[np.float64]
    sfreq: float
    row_ids: tuple[RowId, ...]
    global_explained_variance: float
    computation: ComputationSpec

    @property
    def n_states(self) -> int:
        """Number of microstate classes."""
        return int(self.templates.shape[0])


def _validate_topographies(data: npt.NDArray[np.float64]) -> None:
    if not np.isfinite(data).all() or np.any(np.ptp(data, axis=1) == 0.0):
        raise ValueError(
            "Microstate topographies must be finite with nonzero spatial variance "
            "at every sample; reject invalid data before segmentation."
        )


def _training_rows(rows: npt.NDArray[np.intp] | None, n_epochs: int) -> npt.NDArray[np.intp]:
    if rows is None:
        return np.arange(n_epochs, dtype=np.intp)
    indices = np.asarray(rows)
    if indices.ndim != 1 or indices.size == 0:
        raise ValueError("Microstate training rows must be a nonempty 1-D array.")
    if not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("Microstate training rows must be integers.")
    if np.any(indices < 0) or np.any(indices >= n_epochs):
        raise ValueError("Microstate training rows are out of range.")
    if np.unique(indices).size != indices.size:
        raise ValueError("Microstate training rows must be unique.")
    return indices.astype(np.intp)


@dataclass(frozen=True, eq=False)
class MicrostateModel:
    """Frozen spatial templates fitted independently of subsequent recordings.

    Fit on training epochs, then call :meth:`segment` on any recording with the
    same channels in the same order. Use :meth:`from_templates` for an identified
    external reference, and :meth:`match_reference` to assign its state identities
    to a training fit. Unmatched fits have arbitrary ``state1`` onward labels.

    Parameters
    ----------
    templates : ndarray, shape (n_states, n_channels)
        Average-referenced unit topographies, 2 to 12 of them; stored read-only.
    ch_names : tuple of str
        Channel order of the template columns.
    labels : tuple of str
        Name of each state, in template order.
    computation : ComputationSpec
        How the templates were obtained.
    """

    templates: npt.NDArray[np.float64]
    ch_names: tuple[str, ...]
    labels: tuple[str, ...]
    computation: ComputationSpec

    def __post_init__(self) -> None:
        templates = np.array(self.templates, dtype=float, copy=True)
        if templates.ndim != 2 or templates.shape != (len(self.labels), len(self.ch_names)):
            raise ValueError("templates must have shape (n_states, n_channels).")
        if not 2 <= templates.shape[0] <= 12:
            raise ValueError("n_states must be between 2 and 12.")
        validate_names(self.ch_names, "Microstate channels")
        validate_names(self.labels, "Microstate labels")
        if not np.isfinite(templates).all() or np.any(np.ptp(templates, axis=1) == 0.0):
            raise ValueError("Reference templates need finite, nonzero spatial variance.")
        if not np.allclose(templates.mean(axis=1), 0.0) or not np.allclose(
            np.linalg.norm(templates, axis=1), 1.0
        ):
            raise ValueError("Microstate templates must be average-referenced unit vectors.")
        templates.setflags(write=False)
        object.__setattr__(self, "templates", templates)

    def __deepcopy__(self, memo: dict[int, object]) -> MicrostateModel:
        """Retain immutable templates when scikit-learn clones a pipeline."""
        return self

    @classmethod
    def fit(
        cls,
        signal: Signal,
        *,
        rows: npt.NDArray[np.intp] | None = None,
        n_states: int = 4,
        min_peak_distance_ms: float = 10.0,
        max_peaks_per_epoch: int = 400,
        peak_prominence: float | None = None,
        random_state: int = 42,
    ) -> MicrostateModel:
        """Fit modified k-means exclusively on selected training epochs.

        No held-out epoch contributes a GFP peak, a seed, or a template update.
        ``rows=None`` fits all supplied epochs and is descriptive unless the
        supplied signal itself contains only training epochs.

        Parameters
        ----------
        signal : Signal
            Broadband epochs. Topographies of the fitted epochs must be finite and
            vary across channels at every sample.
        rows : ndarray of int, optional
            Epochs that may contribute; unique and in range. None uses all.
        n_states : int, default 4
            Number of templates, from 2 to 12.
        min_peak_distance_ms : float, default 10.0
            Minimum separation between global field power peaks, in milliseconds.
        max_peaks_per_epoch : int, default 400
            Strongest peaks kept per epoch.
        peak_prominence : float, optional
            Minimum prominence for a peak to count, in the signal's units.
        random_state : int, default 42
            Seed of the k-means initialization.

        Returns
        -------
        MicrostateModel
            Labelled ``state1`` onward.
        """
        kmeans = _require_sklearn()
        if isinstance(n_states, bool) or not isinstance(n_states, (int, np.integer)):
            raise ValueError("n_states must be an integer between 2 and 12.")
        if not 2 <= n_states <= 12:
            raise ValueError(f"n_states must be between 2 and 12, got {n_states}.")
        if not np.isfinite(min_peak_distance_ms) or min_peak_distance_ms < 0.0:
            raise ValueError("min_peak_distance_ms must be finite and non-negative.")
        if (
            isinstance(max_peaks_per_epoch, bool)
            or not isinstance(max_peaks_per_epoch, (int, np.integer))
            or max_peaks_per_epoch < 1
        ):
            raise ValueError("max_peaks_per_epoch must be a positive integer.")
        if peak_prominence is not None and (
            not np.isfinite(peak_prominence) or peak_prominence < 0.0
        ):
            raise ValueError("peak_prominence must be finite and non-negative.")
        selected = _training_rows(rows, signal.n_epochs)
        _validate_topographies(signal.data[selected])
        maps = [
            _peak_topographies(
                signal.data[epoch],
                signal.sfreq,
                min_peak_distance_ms,
                max_peaks_per_epoch,
                peak_prominence,
            )
            for epoch in selected
        ]
        stacked = np.concatenate(maps, axis=0)
        if stacked.shape[0] < n_states:
            raise ValueError(
                f"only {stacked.shape[0]} global field power peaks were found across the "
                f"contributing epochs, fewer than the {n_states} states requested."
            )
        kmeans_model = kmeans(n_clusters=n_states, n_init=20, random_state=random_state)
        kmeans_model.fit(_normalize_rows(stacked))
        templates = _modified_kmeans(
            stacked, np.asarray(kmeans_model.cluster_centers_, dtype=float)
        )
        return cls(
            templates=templates,
            ch_names=signal.ch_names,
            labels=tuple(f"state{i + 1}" for i in range(n_states)),
            computation=ComputationSpec.create(
                "modified_kmeans",
                input_computation=signal.computation.record(),
                channels=signal.ch_names,
                templates=templates.tolist(),
                fit_rows=[signal.row_ids[i] for i in selected],
                n_states=n_states,
                random_state=random_state,
                n_init=20,
                template_weighting="gfp_squared",
                min_peak_distance_ms=min_peak_distance_ms,
                max_peaks_per_epoch=max_peaks_per_epoch,
                peak_prominence=peak_prominence,
            ),
        )

    @classmethod
    def from_templates(
        cls,
        templates: npt.NDArray[np.float64],
        *,
        ch_names: tuple[str, ...],
        labels: tuple[str, ...],
        reference_name: str,
    ) -> MicrostateModel:
        """Freeze externally supplied maps with an explicit reference identity.

        Channel means, scales, and polarities are normalized. Label identity is
        supplied by the reference; arbitrary clusters never acquire A-D labels.

        Parameters
        ----------
        templates : ndarray, shape (n_states, n_channels)
            Maps, finite and varying across channels.
        ch_names : tuple of str
            Channel order of the map columns.
        labels : tuple of str
            Name of each state, in map order.
        reference_name : str
            Nonempty identity of the template set, required by
            :meth:`match_reference`.

        Returns
        -------
        MicrostateModel
        """
        if not isinstance(reference_name, str) or not reference_name.strip():
            raise ValueError("reference_name must identify the external template set.")
        maps = np.asarray(templates, dtype=float)
        if maps.ndim != 2 or not np.isfinite(maps).all() or np.any(np.ptp(maps, axis=1) == 0):
            raise ValueError("Reference templates need finite, nonzero spatial variance.")
        maps = _normalize_rows(maps)
        return cls(
            templates=maps,
            ch_names=tuple(ch_names),
            labels=tuple(labels),
            computation=ComputationSpec.create(
                "provided_microstate_templates",
                templates=maps.tolist(),
                channels=ch_names,
                labels=labels,
                reference_name=reference_name,
            ),
        )

    def match_reference(self, reference: MicrostateModel) -> MicrostateModel:
        """Match states one-to-one by maximum absolute spatial correlation.

        The Hungarian assignment maximizes total similarity; templates are
        reordered into reference order. The reference must be fixed independently
        of held-out recordings when state identities are used for prediction.

        Parameters
        ----------
        reference : MicrostateModel
            Identified templates, as from :meth:`from_templates`, with the same
            channels in the same order and the same number of states.

        Returns
        -------
        MicrostateModel
            This fit's templates in reference order, under the reference's labels.
        """
        if self.ch_names != reference.ch_names or self.templates.shape != reference.templates.shape:
            raise ValueError("Reference matching requires the same states and channels in order.")
        reference_name = reference.computation.parameters.get("reference_name")
        if reference_name is None:
            raise ValueError("Reference matching requires an identified external reference.")
        similarity = np.abs(reference.templates @ self.templates.T)
        reference_rows, order = linear_sum_assignment(-similarity)
        templates = self.templates[order]
        return MicrostateModel(
            templates=templates,
            ch_names=self.ch_names,
            labels=reference.labels,
            computation=ComputationSpec.create(
                "matched_microstate_templates",
                template_fit=self.computation.record(),
                reference=reference.computation.record(),
                reference_name=reference_name,
                state_order=order.tolist(),
                spatial_correlations=similarity[reference_rows, order].tolist(),
                templates=templates.tolist(),
                channels=self.ch_names,
                labels=reference.labels,
            ),
        )

    def segment(self, signal: Signal, *, min_duration_ms: float = 20.0) -> MicrostateSegmentation:
        """Assign a new recording using these frozen templates, without fitting.

        Parameters
        ----------
        signal : Signal
            Epochs with the fitted channels in the same order. Topographies must be
            finite and vary across channels at every sample.
        min_duration_ms : float, default 20.0
            Segments shorter than this, in milliseconds, are absorbed into a
            neighbor or split between neighbors on a duration tie.

        Returns
        -------
        MicrostateSegmentation
        """
        if signal.ch_names != self.ch_names:
            raise ValueError("Microstate segmentation requires fitted channels in the same order.")
        if not np.isfinite(min_duration_ms) or min_duration_ms < 0.0:
            raise ValueError("min_duration_ms must be finite and non-negative.")
        _validate_topographies(signal.data)
        min_samples = minimum_sample_count(min_duration_ms / 1000.0, signal.sfreq)
        states = np.stack(
            [_smooth(_assign(epoch, self.templates), min_samples) for epoch in signal.data]
        )
        parameters = dict(self.computation.parameters)
        parameters["min_duration_ms"] = min_duration_ms
        if parameters.get("input_computation") != signal.computation.record():
            parameters["segmentation_input_computation"] = signal.computation.record()
        return MicrostateSegmentation(
            templates=self.templates,
            states=states,
            labels=self.labels,
            times=signal.times,
            sfreq=signal.sfreq,
            row_ids=signal.row_ids,
            global_explained_variance=_global_explained_variance(
                signal.data, self.templates, states
            ),
            computation=ComputationSpec.create(self.computation.method, **parameters),
        )


def segment(
    signal: Signal,
    *,
    n_states: int = 4,
    fit_on: npt.NDArray[np.bool_] | None = None,
    min_duration_ms: float = 20.0,
    min_peak_distance_ms: float = 10.0,
    max_peaks_per_epoch: int = 400,
    peak_prominence: float | None = None,
    random_state: int = 42,
) -> MicrostateSegmentation:
    """Fit microstate templates and assign every sample to one.

    Templates are clustered from the topographies at peaks of the global field
    power, then every sample is assigned by absolute spatial correlation.
    A topography and its inversion represent the same state. Sign normalization
    is a reporting and initialization convention; polarity invariance of the
    modified k-means objective does not require that orientation.

    **Template fitting pools across trials.** ``fit_on`` names which trials may
    contribute. The default uses every trial and is descriptive of that pool.
    For prediction, fit templates within each training partition. Use
    :class:`MicrostateModel` to fit once and assign independent recordings.
    The per-sample assignment and every measure derived from it are per epoch, so
    the measures themselves carry one row per epoch.

    Requires the optional dependency: ``pip install eegtable[microstates]``.

    Parameters
    ----------
    signal : Signal
        Broadband epochs. Topographies are demeaned across channels per sample,
        which is equivalent to an average reference.
    n_states : int, default 4
        Number of microstate classes. Unmatched clusters are named ``state1``
        onward. Canonical A-D labels require explicit matching to an identified
        reference-template set.
    fit_on : ndarray of bool, optional
        Which epochs may contribute topographies to the clustering. None uses all.
    min_duration_ms : float, default 20.0
        Segments shorter than this are absorbed into a neighbor or split between
        neighbors on a duration tie. This smoothing changes temporal summaries.
    min_peak_distance_ms : float, default 10.0
        Minimum separation between global field power peaks.
    max_peaks_per_epoch : int, default 400
        Strongest peaks retained per epoch.
    peak_prominence : float, optional
        Minimum prominence for a peak to count.
    random_state : int, default 42
        Seed for the clustering.

    Returns
    -------
    MicrostateSegmentation
    """
    data = signal.data
    contributing = np.ones(data.shape[0], dtype=bool) if fit_on is None else np.asarray(fit_on)
    if contributing.shape != (data.shape[0],):
        raise ValueError(
            f"fit_on must have one entry per epoch; got {contributing.shape} for "
            f"{data.shape[0]} epochs."
        )
    if contributing.dtype != np.bool_:
        raise ValueError("fit_on must be a boolean mask with one entry per epoch.")
    if not contributing.any():
        raise ValueError("fit_on excludes every epoch, so there is nothing to cluster.")

    model = MicrostateModel.fit(
        signal,
        rows=np.flatnonzero(contributing),
        n_states=n_states,
        random_state=random_state,
        min_peak_distance_ms=min_peak_distance_ms,
        max_peaks_per_epoch=max_peaks_per_epoch,
        peak_prominence=peak_prominence,
    )
    return model.segment(signal, min_duration_ms=min_duration_ms)


def _global_explained_variance(
    data: npt.NDArray[np.float64],
    templates: npt.NDArray[np.float64],
    states: npt.NDArray[np.int_],
) -> float:
    centred = data - np.nanmean(data, axis=1, keepdims=True)
    field_power = np.nanstd(centred, axis=1)
    norm = np.linalg.norm(centred, axis=1)
    unit = np.divide(
        centred,
        norm[:, np.newaxis, :],
        out=np.zeros_like(centred),
        where=norm[:, np.newaxis, :] > 0.0,
    )
    all_correlations = np.abs(np.einsum("ect,kc->ekt", unit, templates))
    assigned = np.take_along_axis(all_correlations, states[:, np.newaxis, :], axis=1)[:, 0, :]
    weights = field_power**2
    denominator = float(np.nansum(weights))
    if denominator <= 0.0:
        return float("nan")
    return float(np.nansum(weights * assigned**2) / denominator)


def microstate_coverage(
    segmentation: MicrostateSegmentation, *, windows: Sequence[Window]
) -> FeatureTable:
    """Fraction of the window spent in each state.

    Sums to one across states, so the values are compositional and not
    independent of one another.

    Parameters
    ----------
    segmentation : MicrostateSegmentation
        From :func:`segment`.
    windows : sequence of Window
        Analysis windows.

    Returns
    -------
    FeatureTable
        One column per state and window, with ``space_kind="state"``.
    """
    return _per_state(segmentation, windows, "coverage", "fraction", _coverage)


def microstate_duration(
    segmentation: MicrostateSegmentation, *, windows: Sequence[Window]
) -> FeatureTable:
    """Mean time spent in a state per visit, in milliseconds.

    NaN for a state the window never enters, rather than zero: a state that did
    not occur has no duration, which is not the same as a very short one.

    Parameters
    ----------
    segmentation : MicrostateSegmentation
        From :func:`segment`.
    windows : sequence of Window
        Analysis windows.

    Returns
    -------
    FeatureTable
        One column per state and window.
    """
    return _per_state(segmentation, windows, "duration", "ms", _duration)


def microstate_occurrence(
    segmentation: MicrostateSegmentation, *, windows: Sequence[Window]
) -> FeatureTable:
    """Number of times a state is entered per second.

    Zero for a state the window never enters, which unlike duration is a real
    measurement: the state occurred zero times.

    Parameters
    ----------
    segmentation : MicrostateSegmentation
        From :func:`segment`.
    windows : sequence of Window
        Analysis windows.

    Returns
    -------
    FeatureTable
        One column per state and window.
    """
    return _per_state(segmentation, windows, "occurrence", "1/s", _occurrence)


def microstate_transitions(
    segmentation: MicrostateSegmentation, *, windows: Sequence[Window]
) -> FeatureTable:
    """Probability of moving from one state to each other state.

    Counted over successive **segments**, not successive samples, so remaining in
    a state is not counted as a transition to itself. Each row of the matrix sums
    to one, or is NaN where the source state was never left.

    Parameters
    ----------
    segmentation : MicrostateSegmentation
        From :func:`segment`.
    windows : sequence of Window
        Analysis windows.

    Returns
    -------
    FeatureTable
        One column per ordered state pair and window, named ``"a-to-b"``, with
        ``space_kind="pair"``.
    """
    columns: list[tuple[FeatureMeta, npt.NDArray[np.float64]]] = []
    for window in windows:
        mask = window_mask(segmentation.times, window)
        matrices = np.stack(
            [_transitions(row[mask], segmentation.n_states) for row in segmentation.states]
        )
        for source in range(segmentation.n_states):
            for target in range(segmentation.n_states):
                if source == target:
                    continue
                columns.append(
                    (
                        _meta(
                            "transition",
                            f"{segmentation.labels[source]}-to-{segmentation.labels[target]}",
                            "pair",
                            window,
                            "probability",
                            segmentation.computation,
                        ),
                        matrices[:, source, target],
                    )
                )
    return _assemble(columns, segmentation.row_ids)


def _require_sklearn() -> Any:
    try:
        from sklearn.cluster import KMeans
    except ImportError as exc:  # pragma: no cover - exercised by the import guard test
        raise ImportError(
            "microstate segmentation needs scikit-learn, which is not installed. "
            "Install it with: pip install eegtable[microstates]"
        ) from exc
    return KMeans


def _normalize_rows(matrix: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    out = np.zeros_like(matrix, dtype=float)
    for index, row in enumerate(matrix):
        vector = row - np.nanmean(row)
        norm = float(np.linalg.norm(vector))
        if not np.isfinite(norm) or norm <= 0.0:
            continue
        vector = vector / norm
        # Fix the sign so a topography and its inversion have one representation.
        if vector[int(np.argmax(np.abs(vector)))] < 0.0:
            vector = -vector
        out[index] = vector
    return out


def _modified_kmeans(
    maps: npt.NDArray[np.float64],
    seeds: npt.NDArray[np.float64],
    *,
    max_iterations: int = 300,
) -> npt.NDArray[np.float64]:
    """Polarity-invariant clustering, after Pascual-Marqui et al. (1995).

    Assignment is by absolute correlation and each template is the principal
    eigenvector of its cluster's raw, average-referenced scatter matrix. Retaining
    map amplitudes weights squared correlations by GFP squared, matching the
    explained-variance objective. The scatter matrix is unchanged when any member
    map is negated. Euclidean k-means over sign-normalized maps is not
    equivalent: orienting a map by the sign of its strongest channel is
    discontinuous, so where a topography has two extrema of similar magnitude,
    noise decides the orientation and one state's maps are canonicalized in
    opposite directions and split across clusters.

    The converged k-means centres (k-means++ initialization, best of the
    restarts) are the starting point, rather than a random draw. The
    refinement below is what decides the templates.
    """
    templates = _unit_rows(seeds)
    previous: npt.NDArray[np.int_] | None = None
    for _ in range(max_iterations):
        labels = np.asarray(np.argmax(np.abs(maps @ templates.T), axis=1), dtype=int)
        if np.unique(labels).size != templates.shape[0]:
            raise ValueError(
                "Fitting produced an empty microstate cluster; use fewer states "
                "or provide more distinct topographies."
            )
        if previous is not None and np.array_equal(labels, previous):
            break
        previous = labels
        for state in range(templates.shape[0]):
            members = maps[labels == state]
            # Largest eigenvector of the scatter matrix: the direction the cluster's
            # maps lie along, irrespective of which way round each one points.
            _, vectors = np.linalg.eigh(members.T @ members)
            templates[state] = vectors[:, -1]
    return _normalize_rows(templates)


def _unit_rows(matrix: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Demeaned unit rows, without the orientation rule that _normalize_rows applies."""
    centred = matrix - np.nanmean(matrix, axis=1, keepdims=True)
    norms = np.linalg.norm(centred, axis=1, keepdims=True)
    return np.asarray(np.divide(centred, norms, out=np.zeros_like(centred), where=norms > 0.0))


def _gfp(epoch: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    demeaned = epoch - np.nanmean(epoch, axis=0, keepdims=True)
    return np.asarray(np.nanstd(demeaned, axis=0), dtype=float)


def _peak_topographies(
    epoch: npt.NDArray[np.float64],
    sfreq: float,
    min_peak_distance_ms: float,
    max_peaks: int,
    prominence: float | None,
) -> npt.NDArray[np.float64]:
    if epoch.ndim != 2 or epoch.shape[1] < 3:
        return np.empty((0, epoch.shape[0]), dtype=float)
    strength = _gfp(epoch)
    if not np.isfinite(strength).any():
        return np.empty((0, epoch.shape[0]), dtype=float)
    distance = minimum_sample_count(min_peak_distance_ms / 1000.0, sfreq)
    peaks, _ = find_peaks(strength, distance=distance, prominence=prominence)
    strongest = peaks[np.argsort(strength[peaks])[::-1][:max_peaks]]
    maps = epoch[:, strongest].T
    return np.asarray(maps - maps.mean(axis=1, keepdims=True), dtype=float)


def _assign(
    epoch: npt.NDArray[np.float64], templates: npt.NDArray[np.float64]
) -> npt.NDArray[np.int_]:
    maps = _normalize_rows(epoch.T)
    # Absolute similarity: a map and its inversion belong to the same state.
    similarity = np.abs(templates @ maps.T)
    return np.asarray(np.argmax(similarity, axis=0), dtype=int)


def _runs(states: npt.NDArray[np.int_]) -> list[tuple[int, int, int]]:
    if states.size == 0:
        return []
    out: list[tuple[int, int, int]] = []
    start = 0
    for index in range(1, states.size + 1):
        if index == states.size or states[index] != states[start]:
            out.append((start, index, int(states[start])))
            start = index
    return out


@dataclass
class _Run:
    start: int
    stop: int
    state: int
    previous: int | None
    following: int | None
    version: int = 0
    active: bool = True

    @property
    def length(self) -> int:
        return self.stop - self.start


def _smooth(states: npt.NDArray[np.int_], min_samples: int) -> npt.NDArray[np.int_]:
    if states.size == 0 or min_samples <= 1:
        return states

    runs = _runs(states)
    nodes = [
        _Run(
            start=start,
            stop=stop,
            state=state,
            previous=index - 1 if index > 0 else None,
            following=index + 1 if index < len(runs) - 1 else None,
        )
        for index, (start, stop, state) in enumerate(runs)
    ]
    queue = [
        (run.length, run.start, index, run.version)
        for index, run in enumerate(nodes)
        if run.length < min_samples
    ]
    heapify(queue)
    out = states.copy()

    def changed(index: int) -> None:
        run = nodes[index]
        run.version += 1
        if run.active and run.length < min_samples:
            heappush(queue, (run.length, run.start, index, run.version))

    while queue:
        _length, _start, index, version = heappop(queue)
        run = nodes[index]
        if not run.active or run.version != version:
            continue

        previous_index = run.previous
        following_index = run.following
        if previous_index is None and following_index is None:
            return out
        previous = nodes[previous_index] if previous_index is not None else None
        following = nodes[following_index] if following_index is not None else None

        if previous is None:
            assert following is not None and following_index is not None
            out[run.start : run.stop] = following.state
            following.start = run.start
            following.previous = None
            run.active = False
            changed(following_index)
            continue
        if following is None:
            assert previous_index is not None
            out[run.start : run.stop] = previous.state
            previous.stop = run.stop
            previous.following = None
            run.active = False
            changed(previous_index)
            continue
        if previous.state == following.state:
            assert previous_index is not None
            out[run.start : run.stop] = previous.state
            previous.stop = following.stop
            previous.following = following.following
            if following.following is not None:
                nodes[following.following].previous = previous_index
            run.active = False
            following.active = False
            changed(previous_index)
            continue
        if previous.length > following.length:
            assert previous_index is not None
            out[run.start : run.stop] = previous.state
            previous.stop = run.stop
            previous.following = following_index
            following.previous = previous_index
            run.active = False
            changed(previous_index)
        elif following.length > previous.length:
            assert following_index is not None
            out[run.start : run.stop] = following.state
            following.start = run.start
            following.previous = previous_index
            previous.following = following_index
            run.active = False
            changed(following_index)
        else:
            assert previous_index is not None and following_index is not None
            middle = run.start + run.length // 2
            out[run.start : middle] = previous.state
            out[middle : run.stop] = following.state
            previous.stop = middle
            previous.following = following_index
            following.start = middle
            following.previous = previous_index
            run.active = False
            changed(previous_index)
            changed(following_index)
    return out


def _coverage(states: npt.NDArray[np.int_], n_states: int, sfreq: float) -> npt.NDArray[np.float64]:
    del sfreq
    if states.size == 0:
        return np.full(n_states, np.nan)
    return np.array([(states == k).mean() for k in range(n_states)], dtype=float)


def _duration(states: npt.NDArray[np.int_], n_states: int, sfreq: float) -> npt.NDArray[np.float64]:
    out = np.full(n_states, np.nan)
    lengths = [(state, stop - start) for start, stop, state in _runs(states)]
    for k in range(n_states):
        visits = [length for state, length in lengths if state == k]
        if visits:
            out[k] = float(np.mean(visits) * 1000.0 / sfreq)
    return out


def _occurrence(
    states: npt.NDArray[np.int_], n_states: int, sfreq: float
) -> npt.NDArray[np.float64]:
    if states.size == 0:
        return np.full(n_states, np.nan)
    seconds = max(states.size / float(sfreq), 1e-12)
    lengths = [state for _start, _stop, state in _runs(states)]
    return np.array([lengths.count(k) / seconds for k in range(n_states)], dtype=float)


def _transitions(states: npt.NDArray[np.int_], n_states: int) -> npt.NDArray[np.float64]:
    counts = np.zeros((n_states, n_states))
    sequence = [state for _start, _stop, state in _runs(states)]
    for source, target in zip(sequence, sequence[1:], strict=False):
        if 0 <= source < n_states and 0 <= target < n_states:
            counts[source, target] += 1.0
    totals = counts.sum(axis=1, keepdims=True)
    out = np.full_like(counts, np.nan)
    nonzero = totals[:, 0] > 0
    out[nonzero] = counts[nonzero] / totals[nonzero]
    return out


def _meta(
    measure: str,
    space: str,
    kind: Any,
    window: Window,
    unit: str,
    computation: ComputationSpec,
) -> FeatureMeta:
    return FeatureMeta(
        measure=measure,
        band=None,
        space=space,
        space_kind=kind,
        window=window.name,
        normalization="raw",
        unit=unit,
        source="microstates",
        window_bounds=(window.tmin, window.tmax),
        computation=ComputationSpec.create(measure, segmentation=computation.record()),
        freq_resolution_hz=None,
    )


def _per_state(
    segmentation: MicrostateSegmentation,
    windows: Sequence[Window],
    measure: str,
    unit: str,
    reduce: Any,
) -> FeatureTable:
    columns: list[tuple[FeatureMeta, npt.NDArray[np.float64]]] = []
    for window in windows:
        mask = window_mask(segmentation.times, window)
        values = np.stack(
            [
                reduce(row[mask], segmentation.n_states, segmentation.sfreq)
                for row in segmentation.states
            ]
        )
        for index, label in enumerate(segmentation.labels):
            columns.append(
                (
                    _meta(measure, label, "state", window, unit, segmentation.computation),
                    values[:, index],
                )
            )
    return _assemble(columns, segmentation.row_ids)


def _assemble(
    columns: list[tuple[FeatureMeta, npt.NDArray[np.float64]]], row_ids: tuple[RowId, ...]
) -> FeatureTable:
    values = np.stack([column for _, column in columns], axis=1)
    return FeatureTable(
        values=values,
        coverage=np.isfinite(values).astype(float),
        meta=tuple(meta for meta, _ in columns),
        row_ids=row_ids,
    )
