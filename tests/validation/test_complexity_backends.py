"""AntroPy integration on staged Sleep-EDF epochs in two separate windows.

The reference calls use individual contiguous traces, explicit ordinal settings,
and explicitly binarized samples. Agreement checks window/channel selection and
symbolization; it does not independently validate AntroPy's algorithms.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

import eegtable as ef
from tests.validation.loaders import Recording

DATASET = "sleep"
WINDOWS = (ef.Window("early", 0.0, 9.99), ef.Window("middle", 10.0, 19.99))
STAGES = ("W", "N1", "N2", "N3", "R")


@pytest.fixture(scope="module")
def staged_signals(sleep_recordings: list[Recording]) -> list[ef.Signal]:
    signals = []
    for recording in sleep_recordings:
        stages = recording.metadata["stage"].to_numpy()
        rows = np.concatenate([np.flatnonzero(stages == stage)[:2] for stage in STAGES])
        assert all(np.count_nonzero(stages[rows] == stage) == 2 for stage in STAGES)
        epochs = recording.epochs[np.sort(rows)]
        signals.append(ef.Signal.from_epochs(epochs, recording=recording.name))
    return signals


def _check_reference(
    signals: list[ef.Signal],
    table: ef.FeatureTable,
    reference: Callable[[np.ndarray], float],
    record: Callable[[str], None],
) -> None:
    expected_columns = {
        (window.name, channel) for window in WINDOWS for channel in signals[0].ch_names
    }
    assert len(table.meta) == len(expected_columns)
    assert {(meta.window, meta.space) for meta in table.meta} == expected_columns
    expected = np.empty_like(table.values)
    offset = 0
    for signal in signals:
        for column, meta in enumerate(table.meta):
            window = next(window for window in WINDOWS if window.name == meta.window)
            inside = (signal.times >= window.tmin) & (signal.times <= window.tmax)
            channel = signal.ch_names.index(meta.space)
            for epoch, trace in enumerate(signal.data[:, channel, inside]):
                expected[offset + epoch, column] = reference(np.ascontiguousarray(trace))
        offset += len(signal.row_ids)
    error = float(np.max(np.abs(table.values - expected)))
    record(
        f"{expected.size} estimates across two nights and five stages; "
        f"max absolute error {error:.2e}"
    )
    assert table.row_ids == tuple(row for signal in signals for row in signal.row_ids)
    assert np.isfinite(expected).all()
    assert np.isfinite(table.values).all()
    np.testing.assert_allclose(table.values, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(table.coverage, np.ones_like(table.values))


@pytest.mark.validates(
    "permutation_entropy",
    kind="estimator",
    claim="Sleep-EDF ordinal entropy agrees with direct AntroPy calls in each channel/window",
    criterion="rtol=1e-12, atol=1e-12 for order/delay (3, 1) and (4, 2); finite values",
)
@pytest.mark.parametrize("order,delay", [(3, 1), (4, 2)])
def test_permutation_entropy_matches_antropy(
    staged_signals: list[ef.Signal], order: int, delay: int, record: Callable[[str], None]
) -> None:
    antropy = pytest.importorskip("antropy")
    table = ef.stack_rows(
        [
            ef.permutation_entropy(
                [signal], windows=WINDOWS, order=order, delay=delay, include_global=False
            )
            for signal in staged_signals
        ]
    )
    _check_reference(
        staged_signals,
        table,
        lambda trace: antropy.perm_entropy(trace, order=order, delay=delay, normalize=True),
        lambda observed: record(f"order={order}, delay={delay}: {observed}"),
    )
    assert np.all((table.values >= 0.0) & (table.values <= 1.0))


@pytest.mark.validates(
    "lempel_ziv_complexity",
    kind="estimator",
    claim="Sleep-EDF binary complexity agrees with AntroPy after explicit symbolization",
    criterion="rtol=1e-12, atol=1e-12 for median and mean thresholds; finite values",
)
@pytest.mark.parametrize("symbolization", ["median", "mean"])
def test_lempel_ziv_complexity_matches_antropy(
    staged_signals: list[ef.Signal], symbolization: str, record: Callable[[str], None]
) -> None:
    antropy = pytest.importorskip("antropy")
    threshold = {"median": np.median, "mean": np.mean}[symbolization]
    table = ef.stack_rows(
        [
            ef.lempel_ziv_complexity(
                [signal], windows=WINDOWS, symbolization=symbolization, include_global=False
            )
            for signal in staged_signals
        ]
    )
    _check_reference(
        staged_signals,
        table,
        lambda trace: antropy.lziv_complexity(
            (trace >= threshold(trace)).astype(np.uint32), normalize=True
        ),
        lambda observed: record(f"{symbolization}: {observed}"),
    )


@pytest.mark.validates(
    "detrended_fluctuation",
    kind="estimator",
    claim="Sleep-EDF DFA exponents agree with direct AntroPy calls on contiguous windows",
    criterion="rtol=1e-12, atol=1e-12; finite values in both derivations and windows",
)
def test_detrended_fluctuation_matches_antropy(
    staged_signals: list[ef.Signal], record: Callable[[str], None]
) -> None:
    antropy = pytest.importorskip("antropy")
    table = ef.stack_rows(
        [
            ef.detrended_fluctuation([signal], windows=WINDOWS, include_global=False)
            for signal in staged_signals
        ]
    )
    _check_reference(staged_signals, table, antropy.detrended_fluctuation, record)
