"""Small shared validators for preprocessing contracts."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


def validate_sampling_precision(original: Any, restored: Any) -> None:
    original_rate = float(original.info["sfreq"])
    restored_rate = float(restored.info["sfreq"])
    if restored_rate != original_rate:
        raise ValueError(
            f"sampling frequency {original_rate} Hz cannot be preserved at native FIF "
            f"precision ({restored_rate} Hz); use an exactly representable sampling rate "
            "before serialization."
        )


def number(value: object, path: str, minimum: float | None = None) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{path}: expected a finite number, got {value!r}")
    if not math.isfinite(value) or (minimum is not None and value < minimum):
        raise ValueError(f"{path}: expected finite value >= {minimum}, got {value!r}")


def positive(value: object, path: str) -> None:
    number(value, path)
    if not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"{path}: must be positive, got {value!r}")


def integer(value: object, path: str, minimum: int = 0) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{path}: expected integer >= {minimum}, got {value!r}")


def choice(value: object, choices: Sequence[object], path: str) -> None:
    if value not in choices:
        raise ValueError(f"{path}: expected one of {choices!r}, got {value!r}")


def names(value: object, path: str) -> None:
    if not isinstance(value, (tuple, list)):
        raise TypeError(f"{path}: expected a sequence of names")
    if any(not isinstance(name, str) or not name.strip() for name in value):
        raise ValueError(f"{path}: names must be nonempty strings")
    if len(set(value)) != len(value):
        raise ValueError(f"{path}: duplicate names")


def thresholds(value: Mapping[str, float] | None, path: str) -> None:
    if value is None:
        return
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{path}: expected nonempty threshold mapping")
    for key, threshold in value.items():
        choice(key, ("eeg", "eog", "ecg"), path)
        positive(threshold, f"{path}.{key}")


def mapping(value: object, path: str, allowed: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{path}: expected mapping")
    for key in value:
        if not isinstance(key, str) or key not in allowed:
            raise ValueError(f"{path}.{key}: unknown field")
    return value
