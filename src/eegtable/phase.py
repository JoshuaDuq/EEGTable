from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Literal

import numpy as np
import numpy.typing as npt
from scipy.stats import false_discovery_control

from eegtable._expand import expand_signal
from eegtable.groups import aggregate
from eegtable.signal import BandSignal
from eegtable.spectra import Window
from eegtable.table import ComputationSpec, FeatureTable


def _validate_minimum_trials(min_valid_trials: int) -> None:
    if (
        isinstance(min_valid_trials, bool)
        or not isinstance(min_valid_trials, (int, np.integer))
        or min_valid_trials < 2
    ):
        raise ValueError(
            f"min_valid_trials must be an integer of at least 2, got {min_valid_trials!r}."
        )


def itpc(
    signals: Sequence[BandSignal],
    *,
    windows: Sequence[Window],
    trials: Sequence[str] | npt.NDArray[np.str_] | None = None,
    groups: Mapping[str, Sequence[str]] | None = None,
    include_global: bool = True,
    min_valid_trials: int = 2,
) -> FeatureTable:
    """Inter-trial phase coherence.

    The length of the mean unit phase vector across trials, averaged over the
    window:

    .. math::

       \\mathrm{ITPC} = \\frac{1}{T} \\sum_t
       \\left| \\frac{1}{N} \\sum_n e^{i \\phi_n(t)} \\right|

    Identical trial phases give one; exact cancellation of the sampled unit
    vectors gives zero. Independent uniform phases generally give a positive
    finite-sample estimate. Trials are averaged first and time second; reversing
    the order defines a different measure. Latencies with fewer than
    ``min_valid_trials`` defined phases are omitted from the window mean.

    **This is estimated across trials, so the result has one row per trial group,
    not one per epoch.** The returned table carries ``row_labels`` and cannot be
    concatenated with per-epoch features. Broadcasting group-level estimates onto
    single epochs introduces pseudo-replication in downstream statistical models;
    models fitted on broadcasted tables incorrectly treat a single group estimate
    as N independent observations.

    Parameters
    ----------
    signals : sequence of BandSignal
        One per band. The bands axis of the output comes from this sequence.
    windows : sequence of Window
        Analysis windows.
    trials : sequence of str, optional
        A label per epoch, giving the group each trial belongs to. One row is
        returned per distinct label, in sorted order. None estimates from all
        trials together and returns a single row labelled ``"all"``.
    groups : mapping of str to sequence of str, optional
        ROI name to member channels. None gives one column per channel.
    include_global : bool, default True
        Also emit the mean across all channels.
    min_valid_trials : int, default 2
        Minimum number of defined trial phases at each latency. Must be at least
        two. Windows without a qualifying latency return NaN and are flagged
        ``insufficient_trials``.

    Returns
    -------
    FeatureTable
        Coherence in ``[0, 1]``, with one row per trial group.
    """
    _validate_minimum_trials(min_valid_trials)
    n_epochs = signals[0].n_epochs if signals else 0
    row_groups, labels = _resolve_rows(trials, n_epochs)

    def kernel(
        signal: BandSignal,
        trace: npt.NDArray[np.float64],
        times: npt.NDArray[np.float64],
        mask: npt.NDArray[np.bool_],
    ) -> dict[str, npt.NDArray[np.float64]]:
        del signal, times, mask
        return {"itpc": _itpc(trace, row_groups, len(labels), min_valid_trials)}

    table = expand_signal(
        signals,
        # The kernel needs phase, and the expander hands it whatever this returns.
        trace_of=lambda signal: signal.phase,
        kernel=kernel,
        units={"itpc": "a.u."},
        windows=windows,
        groups=groups,
        include_global=include_global,
        mode="raw",
        parameters={
            "trials": None if trials is None else list(trials),
            "min_valid_trials": min_valid_trials,
        },
        row_groups=row_groups,
        row_labels=labels,
    )
    return replace(table, flags={**table.flags, "insufficient_trials": np.isnan(table.values)})


def ppc(
    signals: Sequence[BandSignal],
    *,
    windows: Sequence[Window],
    trials: Sequence[str] | npt.NDArray[np.str_] | None = None,
    groups: Mapping[str, Sequence[str]] | None = None,
    include_global: bool = True,
    min_valid_trials: int = 2,
) -> FeatureTable:
    """Pairwise phase consistency across trials.

    PPC averages the cosine of every unordered trial-phase difference at each
    latency, then averages over the requested window. It estimates squared
    population phase locking without the PLV-squared finite-sample bias under
    independent identically distributed trial phases. Finite-sample values can
    be negative, and trial count still affects uncertainty. Latencies with fewer
    than ``min_valid_trials`` defined phases are omitted. The result has one row
    per trial group and is not numerically interchangeable with ITPC.

    Parameters
    ----------
    signals : sequence of BandSignal
        One per band. The bands axis of the output comes from this sequence.
    windows : sequence of Window
        Analysis windows.
    trials : sequence of str, optional
        A label per epoch, giving the group each trial belongs to. One row is
        returned per distinct label, in sorted order. None estimates from all
        trials together and returns a single row labelled ``"all"``.
    groups : mapping of str to sequence of str, optional
        ROI name to member channels. None gives one column per channel.
    include_global : bool, default True
        Also emit the mean across all channels.
    min_valid_trials : int, default 2
        Minimum number of defined trial phases at each latency; at least two.
        Windows without a qualifying latency return NaN and are flagged
        ``insufficient_trials``.

    Returns
    -------
    FeatureTable
        Consistency with one row per trial group, carrying ``row_labels``.
    """
    _validate_minimum_trials(min_valid_trials)
    n_epochs = signals[0].n_epochs if signals else 0
    row_groups, labels = _resolve_rows(trials, n_epochs)

    def kernel(
        signal: BandSignal,
        trace: npt.NDArray[np.float64],
        times: npt.NDArray[np.float64],
        mask: npt.NDArray[np.bool_],
    ) -> dict[str, npt.NDArray[np.float64]]:
        del signal, times, mask
        return {"ppc": _ppc(trace, row_groups, len(labels), min_valid_trials)}

    table = expand_signal(
        signals,
        trace_of=lambda signal: signal.phase,
        kernel=kernel,
        units={"ppc": "a.u."},
        windows=windows,
        groups=groups,
        include_global=include_global,
        mode="raw",
        parameters={
            "trials": None if trials is None else list(trials),
            "min_valid_trials": min_valid_trials,
        },
        row_groups=row_groups,
        row_labels=labels,
    )
    return replace(table, flags={**table.flags, "insufficient_trials": np.isnan(table.values)})


def pac(
    phase_signal: BandSignal,
    amplitude_signal: BandSignal,
    *,
    windows: Sequence[Window],
    normalize: bool = True,
    groups: Mapping[str, Sequence[str]] | None = None,
    include_global: bool = True,
    allow_overlap: bool = False,
) -> FeatureTable:
    """Phase-amplitude coupling by mean vector length.

    The amplitude-weighted resultant of the slow band's phase:

    .. math::

       \\mathrm{MVL} = \\frac{\\left| \\sum_t A(t) e^{i \\phi(t)} \\right|}{\\sum_t A(t)}

    where :math:`\\phi` is the phase of ``phase_signal`` and :math:`A` the envelope
    of ``amplitude_signal``. With ``normalize=True``, division by summed
    amplitude makes the value invariant to an overall positive amplitude scale
    and bounds it between zero and one. With ``normalize=False``, the modulus
    of the weighted sum is divided by the number of valid samples and retains
    the amplitude unit.

    Computed within each trial, with one row per epoch. Finite windows and
    temporal dependence can produce nonzero estimates without coupling;
    amplitude normalization alone does not remove this bias. No surrogate
    correction is applied. See :func:`pac_surrogates` for inference under an
    explicitly chosen temporal-shift null.

    Parameters
    ----------
    phase_signal : BandSignal
        The slower band, whose phase modulates. Conventionally theta or alpha.
    amplitude_signal : BandSignal
        The faster band, whose envelope is modulated. Conventionally gamma.
        Epoch identities, channel order, time samples and sampling frequency
        must match ``phase_signal`` exactly.
    windows : sequence of Window
        Analysis windows.
    normalize : bool, default True
        Divide by the summed amplitude. False returns the modulus of the mean
        amplitude-weighted phase vector.
    groups : mapping of str to sequence of str, optional
        ROI name to member channels. None gives one column per channel.
    include_global : bool, default True
        Also emit the mean across all channels.
    allow_overlap : bool, default False
        Accept a phase band whose upper edge lies above the amplitude band's lower
        edge, which is otherwise refused. Pass True only for a prespecified
        estimator designed for overlapping bands. The phase band must start below
        the amplitude band either way.

    Returns
    -------
    FeatureTable
        Coupling strength, dimensionless when normalized. Phase and amplitude
        bands are separate metadata fields and both appear in the feature name.
    """
    if phase_signal.ch_names != amplitude_signal.ch_names:
        raise ValueError("phase_signal and amplitude_signal must share the same channels.")
    if phase_signal.analytic.shape != amplitude_signal.analytic.shape:
        raise ValueError(
            "phase_signal and amplitude_signal must have the same shape; got "
            f"{phase_signal.analytic.shape} and {amplitude_signal.analytic.shape}."
        )
    if not np.array_equal(phase_signal.times, amplitude_signal.times):
        raise ValueError("phase_signal and amplitude_signal must share the same time axis.")
    if phase_signal.sfreq != amplitude_signal.sfreq:
        raise ValueError(
            "phase_signal and amplitude_signal must share the same sampling frequency."
        )
    if phase_signal.row_ids != amplitude_signal.row_ids:
        raise ValueError("phase_signal and amplitude_signal must share exact row identities.")
    slow, fast = phase_signal.band, amplitude_signal.band
    if slow is not None and fast is not None:
        if slow.fmin >= fast.fmin:
            raise ValueError(
                f"phase_signal band {slow.name!r} must be slower than amplitude_signal "
                f"band {fast.name!r}; got {slow.fmin} Hz and {fast.fmin} Hz."
            )
        if slow.fmax > fast.fmin and not allow_overlap:
            raise ValueError(
                f"phase band {slow.name!r} and amplitude band {fast.name!r} overlap; "
                "pass allow_overlap=True only for a prespecified specialized estimator."
            )

    unit_phase = np.exp(1j * phase_signal.phase)
    joint_amplitude = replace(
        amplitude_signal,
        coverage=np.where(
            np.isfinite(unit_phase),
            np.minimum(phase_signal.coverage, amplitude_signal.coverage),
            0.0,
        ),
    )

    def kernel(
        signal: BandSignal,
        trace: npt.NDArray[np.float64],
        times: npt.NDArray[np.float64],
        mask: npt.NDArray[np.bool_],
    ) -> dict[str, npt.NDArray[np.float64]]:
        del signal, times
        # The expander's own selector, not one recovered from the time values.
        return {"pac": _mean_vector_length(unit_phase[:, :, mask], trace, normalize)}

    table = expand_signal(
        [joint_amplitude],
        trace_of=lambda signal: signal.envelope,
        kernel=kernel,
        units={"pac": "a.u." if normalize else "V"},
        windows=windows,
        groups=groups,
        include_global=include_global,
        mode="raw",
        parameters={
            "normalize": normalize,
            "allow_overlap": allow_overlap,
            "phase_band": None if slow is None else (slow.name, slow.fmin, slow.fmax),
            "amplitude_band": None if fast is None else (fast.name, fast.fmin, fast.fmax),
            "phase_input_source": phase_signal.source,
            "phase_input_computation": phase_signal.computation.record(),
        },
    )
    return replace(
        table,
        meta=tuple(replace(meta, phase_band=slow, amplitude_band=fast) for meta in table.meta),
    )


def _validate_pac_inference(
    n_surrogates: int,
    surrogate: str,
    random_state: int,
    min_shift_seconds: float,
    correction: str,
) -> None:
    if isinstance(n_surrogates, bool) or not isinstance(n_surrogates, int) or n_surrogates < 2:
        raise ValueError("n_surrogates must be an integer of at least 2.")
    if surrogate not in ("blocks", "circular"):
        raise ValueError("surrogate must be 'blocks' or 'circular'.")
    if isinstance(random_state, bool) or not isinstance(random_state, int) or random_state < 0:
        raise ValueError("random_state must be a non-negative integer.")
    if not np.isfinite(min_shift_seconds) or min_shift_seconds < 0:
        raise ValueError("min_shift_seconds must be finite and non-negative.")
    if correction not in ("none", "fdr", "bonferroni", "maxstat"):
        raise ValueError("correction must be 'none', 'fdr', 'bonferroni', or 'maxstat'.")


def _pac_observed_and_null(
    phase_signal: BandSignal,
    amplitude_signal: BandSignal,
    windows: Sequence[Window],
    *,
    n_surrogates: int,
    surrogate: str,
    random_state: int,
    min_shift_seconds: float,
    normalize: bool,
    groups: Mapping[str, Sequence[str]] | None,
    include_global: bool,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    from eegtable._expand import window_mask

    try:
        from tensorpac.methods import (
            mean_vector_length,
            swap_blocks,
            time_lag,
        )
    except ImportError as exc:
        raise ImportError(
            "pac_surrogates needs tensorpac; install it with: pip install eegtable[pac]"
        ) from exc

    n_epochs, n_channels, _ = phase_signal.analytic.shape
    estimates = np.empty((n_surrogates + 1, n_epochs, n_channels, len(windows)))
    coverage = np.ones((n_epochs, n_channels, len(windows)))
    rng = np.random.default_rng(random_state)
    shift_minimum = max(1, int(np.ceil(min_shift_seconds * phase_signal.sfreq)))
    for index, window in enumerate(windows):
        mask = window_mask(phase_signal.times, window)
        phase = np.ascontiguousarray(phase_signal.phase[:, :, mask])
        amplitude = np.ascontiguousarray(amplitude_signal.envelope[:, :, mask])
        n_times = int(mask.sum())
        if 2 * shift_minimum > n_times:
            raise ValueError(
                f"window {window.name!r} is too short for min_shift_seconds="
                f"{min_shift_seconds}; both surrogate blocks must contain at least "
                f"{shift_minimum} samples."
            )
        if not np.isfinite(phase).all() or not np.isfinite(amplitude).all():
            raise ValueError("PAC surrogate inference requires finite samples in every window.")
        if np.any(amplitude.sum(axis=-1) == 0):
            raise ValueError("PAC surrogate inference requires a positive amplitude envelope.")
        if normalize:
            amplitude = amplitude / amplitude.mean(axis=-1, keepdims=True)
        estimates[0, :, :, index] = mean_vector_length(phase[None], amplitude[None])[0, 0]
        for permutation in range(n_surrogates):
            # Tensorpac permits an identity circular lag. Exclude it, and shifts
            # closer to an edge than the prespecified autocorrelation exclusion.
            while True:
                seed = int(rng.integers(0, 2**32 - 1))
                seed_rng = np.random.RandomState(seed)
                shift = (
                    seed_rng.randint(1, n_times)
                    if surrogate == "blocks"
                    else (seed_rng.randint(n_times))
                )
                if shift_minimum <= shift <= n_times - shift_minimum:
                    break
            shifted_phase, shifted_amplitude = (swap_blocks if surrogate == "blocks" else time_lag)(
                phase, amplitude, random_state=seed
            )
            estimates[permutation + 1, :, :, index] = mean_vector_length(
                shifted_phase[None], shifted_amplitude[None]
            )[0, 0]
    spatial_units = aggregate(
        estimates.reshape((n_surrogates + 1) * n_epochs, n_channels, len(windows)),
        np.broadcast_to(coverage, estimates.shape).reshape(
            (n_surrogates + 1) * n_epochs, n_channels, len(windows)
        ),
        phase_signal.ch_names,
        groups,
        include_global,
    )
    spatial_estimates = np.concatenate([unit.values for unit in spatial_units], axis=1).reshape(
        n_surrogates + 1, n_epochs, -1
    )
    return spatial_estimates[0], spatial_estimates[1:]


def _pac_upper_tail_pvalues(
    observed: npt.NDArray[np.float64], null: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    # Match scipy.stats.permutation_test: numerically equal statistics count as
    # upper-tail ties, avoiding false significance from summation roundoff.
    tolerance = 100 * np.finfo(observed.dtype).eps * np.abs(observed)
    return np.asarray(
        (1.0 + (null >= observed - tolerance).sum(axis=0)) / (null.shape[0] + 1), dtype=float
    )


def _adjust_pac_pvalues(
    observed: npt.NDArray[np.float64],
    null: npt.NDArray[np.float64],
    pvalues: npt.NDArray[np.float64],
    correction: str,
) -> npt.NDArray[np.float64]:
    if correction == "fdr":
        return np.asarray(false_discovery_control(pvalues, axis=1), dtype=float)
    if correction == "bonferroni":
        return np.asarray(np.minimum(1.0, pvalues * pvalues.shape[1]), dtype=float)
    if correction == "maxstat":
        maxima = null.max(axis=2, keepdims=True)
        return _pac_upper_tail_pvalues(observed, maxima)
    return pvalues.copy()


def pac_surrogates(
    phase_signal: BandSignal,
    amplitude_signal: BandSignal,
    *,
    windows: Sequence[Window],
    n_surrogates: int = 200,
    surrogate: Literal["blocks", "circular"] = "blocks",
    random_state: int = 0,
    min_shift_seconds: float = 0.1,
    correction: Literal["none", "fdr", "bonferroni", "maxstat"] = "fdr",
    normalize: bool = True,
    groups: Mapping[str, Sequence[str]] | None = None,
    include_global: bool = True,
    allow_overlap: bool = False,
) -> FeatureTable:
    """Within-epoch PAC with a seeded, temporally shifted surrogate null.

    Returns observed ``pac``, ``pac_null_mean``, ``pac_null_std``,
    ``pac_corrected`` (observed minus null mean), ``pac_zscore``,
    ``pac_pvalue``, and ``pac_pvalue_adjusted``. The standard deviation uses
    Tensorpac's population convention (``ddof=0``). P values are upper-tail
    empirical probabilities ``(1 + count(null >= observed)) / (n_surrogates + 1)``.
    Observed and surrogate estimates share Tensorpac's MVL arithmetic. Ties
    include values within ``100 * eps * abs(observed)``, following SciPy's
    permutation-test convention. A null whose range is within this relative
    roundoff scale has zero deviation and an undefined, flagged z score.
    Corrected estimates within the observed tie tolerance are reported as zero.

    Tensorpac constructs the surrogates: ``blocks`` swaps the two amplitude
    blocks around a random cut, while ``circular`` circularly shifts phase.
    Shifts stay within each analysis window and never combine epochs. Both
    blocks must be at least ``min_shift_seconds`` long, with a minimum of one
    sample. Choose that exclusion from the signal's autocorrelation timescale.
    A stationary sinusoidal modulator can retain the same MVL after a shift;
    these nulls do not guarantee destruction of every kind of coupling.

    ROI/global estimates and their surrogate distributions are averaged before
    inference. Adjustment controls the family of returned spatial units and
    windows separately within each epoch and this phase/amplitude band pair.
    ``fdr`` is Benjamini-Hochberg, ``bonferroni`` multiplies by family size,
    and ``maxstat`` compares each observed estimate to each permutation's
    family maximum. Requires finite samples and the ``pac`` extra.

    Parameters
    ----------
    phase_signal : BandSignal
        The slower band, whose phase modulates.
    amplitude_signal : BandSignal
        The faster band, whose envelope is modulated; the envelope may not be all
        zero in any epoch, channel and window.
    windows : sequence of Window
        Analysis windows, each long enough for two segments of
        ``min_shift_seconds``.
    n_surrogates : int, default 200
        Surrogates per epoch, channel and window; at least 2. The smallest
        attainable p value is ``1 / (n_surrogates + 1)``.
    surrogate : {"blocks", "circular"}, default "blocks"
        Tensorpac surrogate: swap amplitude blocks, or circularly shift phase.
    random_state : int, default 0
        Non-negative seed of the surrogate shifts.
    min_shift_seconds : float, default 0.1
        Minimum length, in seconds and at least one sample, of both segments a
        surrogate shift creates.
    correction : {"none", "fdr", "bonferroni", "maxstat"}, default "fdr"
        Adjustment of ``pac_pvalue_adjusted`` over the spatial units and windows
        of each epoch.
    normalize : bool, default True
        As for :func:`pac`.
    groups : mapping of str to sequence of str, optional
        ROI name to member channels. None gives one column per channel.
    include_global : bool, default True
        Also emit the mean across all channels.
    allow_overlap : bool, default False
        As for :func:`pac`.

    Returns
    -------
    FeatureTable
        One row per epoch and, for each of the seven measures, one column per
        spatial unit and window. ``pac_zscore`` cells with a degenerate null are
        NaN and flagged ``degenerate_null``.
    """
    _validate_pac_inference(n_surrogates, surrogate, random_state, min_shift_seconds, correction)
    observed = pac(
        phase_signal,
        amplitude_signal,
        windows=windows,
        normalize=normalize,
        groups=groups,
        include_global=include_global,
        allow_overlap=allow_overlap,
    )
    observed_values, null = _pac_observed_and_null(
        phase_signal,
        amplitude_signal,
        windows,
        n_surrogates=n_surrogates,
        surrogate=surrogate,
        random_state=random_state,
        min_shift_seconds=min_shift_seconds,
        normalize=normalize,
        groups=groups,
        include_global=include_global,
    )
    observed = replace(observed, values=observed_values)
    tie_relative_tolerance = 100 * np.finfo(null.dtype).eps
    null_scale = np.max(np.abs(null), axis=0)
    degenerate_null = np.ptp(null, axis=0) <= tie_relative_tolerance * null_scale
    null_mean = null.mean(axis=0)
    null_std = np.where(degenerate_null, 0.0, null.std(axis=0))
    corrected = observed.values - null_mean
    corrected = np.where(
        np.abs(corrected) <= tie_relative_tolerance * np.abs(observed.values), 0.0, corrected
    )
    zscore = np.divide(corrected, null_std, out=np.full_like(corrected, np.nan), where=null_std > 0)
    pvalues = _pac_upper_tail_pvalues(observed.values, null)
    results = {
        "pac": observed.values,
        "pac_null_mean": null_mean,
        "pac_null_std": null_std,
        "pac_corrected": corrected,
        "pac_zscore": zscore,
        "pac_pvalue": pvalues,
        "pac_pvalue_adjusted": _adjust_pac_pvalues(observed.values, null, pvalues, correction),
    }
    inference = dict(
        estimator="tensorpac",
        n_surrogates=n_surrogates,
        surrogate=surrogate,
        random_state=random_state,
        min_shift_seconds=min_shift_seconds,
        correction=correction,
        correction_family="nodes_and_windows_within_each_epoch",
        null_standard_deviation_ddof=0,
        empirical_pvalue="plus_one_upper_tail",
        floating_point_tie_relative_tolerance=tie_relative_tolerance,
        constant_null_detection="range_within_relative_roundoff_scale",
        corrected_roundoff="zero_within_observed_relative_tie_tolerance",
    )
    return FeatureTable(
        values=np.concatenate(list(results.values()), axis=1),
        coverage=np.tile(observed.coverage, (1, len(results))),
        meta=tuple(
            replace(
                meta,
                measure=measure,
                unit=(
                    "a.u."
                    if measure in ("pac_zscore", "pac_pvalue", "pac_pvalue_adjusted")
                    else meta.unit
                ),
                computation=ComputationSpec.create(
                    measure,
                    input_computation=meta.computation.record(),
                    inference=inference,
                ),
            )
            for measure in results
            for meta in observed.meta
        ),
        flags={
            "degenerate_null": np.concatenate(
                [
                    (
                        null_std == 0
                        if measure == "pac_zscore"
                        else np.zeros_like(null_std, dtype=bool)
                    )
                    for measure in results
                ],
                axis=1,
            ),
        },
        row_ids=observed.row_ids,
    )


def _resolve_rows(
    trials: Sequence[str] | npt.NDArray[np.str_] | None, n_epochs: int
) -> tuple[npt.NDArray[np.int_], tuple[str, ...]]:
    if trials is None:
        return np.zeros(n_epochs, dtype=int), ("all",)
    labels = np.asarray(trials, dtype=object)
    if labels.shape != (n_epochs,):
        raise ValueError(
            f"trials must have one label per epoch; got {labels.shape} for {n_epochs} epochs."
        )
    if any(not isinstance(value, str) or not value for value in labels):
        raise ValueError("trials must contain non-empty strings.")
    unique = tuple(sorted(set(labels.tolist())))
    index = {name: position for position, name in enumerate(unique)}
    return np.array([index[value] for value in labels], dtype=int), unique


def _itpc(
    phase: npt.NDArray[np.float64],
    row_groups: npt.NDArray[np.int_],
    n_rows: int,
    min_valid_trials: int,
) -> npt.NDArray[np.float64]:
    unit_vectors = np.where(np.isfinite(phase), np.exp(1j * phase), np.nan)
    out = np.full((n_rows, phase.shape[1]), np.nan)
    for row in range(n_rows):
        member = unit_vectors[row_groups == row]
        if member.shape[0] < min_valid_trials:
            continue
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "Mean of empty slice", RuntimeWarning)
            # Trials first, then time: the coherence at each latency, averaged.
            valid = np.isfinite(member)
            count = valid.sum(axis=0)
            coherence = np.where(
                count >= min_valid_trials,
                np.abs(np.nansum(member, axis=0) / np.maximum(count, 1)),
                np.nan,
            )
            out[row] = np.nanmean(np.where(np.isfinite(coherence), coherence, np.nan), axis=1)
    return out


def _ppc(
    phase: npt.NDArray[np.float64],
    row_groups: npt.NDArray[np.int_],
    n_rows: int,
    min_valid_trials: int,
) -> npt.NDArray[np.float64]:
    unit_vectors = np.where(np.isfinite(phase), np.exp(1j * phase), np.nan)
    out = np.full((n_rows, phase.shape[1]), np.nan)
    for row in range(n_rows):
        member = unit_vectors[row_groups == row]
        if member.shape[0] < min_valid_trials:
            continue
        valid = np.isfinite(member)
        count = valid.sum(axis=0)
        resultant_squared = np.abs(np.nansum(member, axis=0)) ** 2
        denominator = count * (count - 1)
        with np.errstate(invalid="ignore", divide="ignore"):
            consistency = np.where(
                count >= min_valid_trials,
                (resultant_squared - count) / denominator,
                np.nan,
            )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "Mean of empty slice", RuntimeWarning)
            out[row] = np.nanmean(consistency, axis=1)
    return out


def _mean_vector_length(
    unit_phase: npt.NDArray[np.complex128],
    amplitude: npt.NDArray[np.float64],
    normalize: bool,
) -> npt.NDArray[np.float64]:
    finite = np.isfinite(amplitude) & np.isfinite(unit_phase)
    weighted = np.where(finite, amplitude * unit_phase, 0.0)
    resultant = np.abs(weighted.sum(axis=2))
    if not normalize:
        total = finite.sum(axis=2)
        return np.where(total > 0, resultant / np.maximum(total, 1), np.nan)
    weight = np.where(finite, amplitude, 0.0).sum(axis=2)
    # Any positive sum is usable: the ratio is dimensionless, so an absolute threshold
    # would withhold a small envelope for its unit alone.
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(weight > 0.0, resultant / weight, np.nan)
