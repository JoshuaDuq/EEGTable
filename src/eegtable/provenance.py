"""Content-based identities shared by extraction and research artifacts."""

from __future__ import annotations

import hashlib
import json
import platform
from collections.abc import Collection
from dataclasses import asdict, is_dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from eegtable.table import _json_value

_PACKAGES = (
    "eegtable",
    "mne",
    "numpy",
    "scipy",
    "pandas",
    "scikit-learn",
    "mne-connectivity",
    "mne-bids",
    "antropy",
    "specparam",
    "neurodsp",
    "bycycle",
    "pyriemann",
    "tensorpac",
    "PyYAML",
    "filelock",
    "h5io",
    "h5py",
    "autoreject",
    "pyprep",
    "mne-icalabel",
    "onnxruntime",
    "python-picard",
)


def serializable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return serializable(asdict(value))
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {str(key): serializable(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [serializable(item) for item in value]
    return _json_value(value)


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


def software_versions() -> dict[str, str]:
    """Record core and installed optional analysis packages without importing them."""
    versions = {"python": platform.python_version()}
    for package in _PACKAGES:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            if package in ("eegtable", "mne", "numpy", "scipy", "pandas"):
                raise
    return versions


def dependency_versions(packages: Collection[str]) -> dict[str, str]:
    """Record required dependency versions; missing distributions raise."""
    return {
        "python": platform.python_version(),
        **{name: version(name) for name in sorted(packages)},
    }


def implementation_hash() -> str:
    """Identify packaged Python source, including uncommitted development edits."""
    root = Path(__file__).parent
    return identity(
        {path.relative_to(root).as_posix(): file_hash(path) for path in sorted(root.rglob("*.py"))}
    )
