from __future__ import annotations

import random as pyrandom
import warnings
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from typing import TypeVar

import numpy as np

from eegtable.model import _deps as _deps
from eegtable.model.splits import Fold

__all__ = [
    "inner_n_jobs",
    "run_folds",
    "seeded",
    "set_random_seeds",
    "should_parallelize",
]

_T = TypeVar("_T")


def set_random_seeds(seed: int, fold: int) -> None:
    """Seed NumPy's legacy global generator and Python's ``random`` with ``seed + fold``.

    The change persists for the rest of the process; :func:`seeded` restores the
    previous state instead.
    """
    combined = seed + fold
    np.random.seed(combined)
    pyrandom.seed(combined)


@contextmanager
def seeded(seed: int, fold: int) -> Iterator[None]:
    """Context manager seeding the global generators with ``seed + fold``.

    Inside the block, NumPy's legacy global generator and Python's ``random`` are
    seeded as by :func:`set_random_seeds`, so an estimator that draws from them is
    reproducible per fold. Their previous states are restored on exit.
    """
    # Seeding the global generators is what makes an estimator that consults them reproducible,
    # but leaving them seeded would reset the caller's own random stream as a side effect of
    # fitting a model, so anything they drew afterwards would depend on how many folds ran.
    np_state = np.random.get_state()
    py_state = pyrandom.getstate()
    try:
        set_random_seeds(seed, fold)
        yield
    finally:
        np.random.set_state(np_state)
        pyrandom.setstate(py_state)


def inner_n_jobs(outer_n_jobs: int, n_jobs: int) -> int:
    """Jobs for work inside a fold: 1 while outer folds run in parallel, else ``n_jobs``.

    Keeps nested parallelism from oversubscribing the processors. Outer folds count
    as parallel when ``outer_n_jobs`` is neither 0 nor 1.
    """
    return 1 if (outer_n_jobs and outer_n_jobs != 1) else n_jobs


def should_parallelize(outer_n_jobs: int, n_folds: int) -> bool:
    """Whether :func:`run_folds` uses worker processes.

    True when ``outer_n_jobs`` is neither 0 nor 1 and there is more than one fold.
    """
    return bool(outer_n_jobs and outer_n_jobs != 1 and n_folds > 1)


def run_folds(
    folds: Sequence[Fold],
    work: Callable[[Fold], _T],
    *,
    outer_n_jobs: int,
) -> list[_T]:
    """Apply ``work`` to every fold, in this process or in joblib worker processes.

    Parameters
    ----------
    folds : sequence of Fold
        Folds to process.
    work : callable
        Called with one fold; its return values are collected.
    outer_n_jobs : int
        joblib ``n_jobs``. 0 or 1, or a single fold, runs serially in this process.

    Returns
    -------
    list
        Results in the order of ``folds``. An exception raised for any fold
        propagates.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            module=r".*sklearn[/.]utils[/.]parallel",
        )
        if should_parallelize(outer_n_jobs, len(folds)):
            from joblib import Parallel, delayed

            results = Parallel(n_jobs=outer_n_jobs, prefer="processes")(
                delayed(work)(fold) for fold in folds
            )
            return list(results)
        return [work(fold) for fold in folds]
