"""Canonical scientific identities without pickle or filesystem timestamps."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np


def serializable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return serializable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, np.ndarray):
        return serializable(value.tolist())
    if isinstance(value, np.generic):
        return serializable(value.item())
    if isinstance(value, dict):
        return {str(key): serializable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [serializable(item) for item in value]
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(serializable(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def identity(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def channel_identity(inst: Any) -> dict[str, Any]:
    # What a fitted operator needs to match; time bounds and passband are not part of it.
    return {
        "channels": inst.ch_names,
        "types": inst.get_channel_types(),
        "bads": sorted(inst.info["bads"]),
        "sfreq": float(inst.info["sfreq"]),
        "reference": int(inst.info["custom_ref_applied"]),
        "projectors": [
            {
                "active": bool(proj["active"]),
                "names": proj["data"]["col_names"],
                "data": proj["data"]["data"],
            }
            for proj in inst.info["projs"]
        ],
    }


def fingerprint(inst: Any) -> str:
    digest = hashlib.sha256()
    digest.update(canonical_json(channel_identity(inst)).encode())
    for channel in inst.info["chs"]:
        digest.update(np.asarray(channel["loc"], dtype="<f8").tobytes())
    digest.update(str(inst.info["meas_date"]).encode())
    digest.update(
        canonical_json(
            {
                "highpass": inst.info["highpass"],
                "lowpass": inst.info["lowpass"],
                "description": inst.info["description"],
                "subject_info": inst.info["subject_info"],
            }
        ).encode()
    )
    if hasattr(inst, "first_samp"):
        digest.update(str(inst.first_samp).encode())
        for start in range(0, inst.n_times, 100_000):
            digest.update(
                np.asarray(
                    inst.get_data(start=start, stop=min(start + 100_000, inst.n_times)), dtype="<f8"
                ).tobytes()
            )
        digest.update(
            canonical_json(
                {
                    "onset": inst.annotations.onset,
                    "duration": inst.annotations.duration,
                    "description": inst.annotations.description,
                    "ch_names": inst.annotations.ch_names.tolist(),
                    "orig_time": str(inst.annotations.orig_time),
                }
            ).encode()
        )
    else:
        digest.update(np.asarray(inst.get_data(), dtype="<f8").tobytes())
        digest.update(
            canonical_json(
                {
                    "events": inst.events,
                    "selection": inst.selection,
                    "drop_log": inst.drop_log,
                    "times": inst.times,
                    "baseline": inst.baseline,
                }
            ).encode()
        )
        if inst.metadata is not None:
            digest.update(inst.metadata.to_json(orient="split", double_precision=15).encode())
    return digest.hexdigest()
