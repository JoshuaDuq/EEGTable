"""Validated FIF/TSV/HTML bundles with a last-published completion manifest."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import mne  # type: ignore[import-untyped]
import numpy as np

from ._validation import validate_sampling_precision
from .checkpoints import payload_files, write_json
from .checks import BUNDLE_SUFFIXES
from .config import OutputSettings
from .epochs import validate_metadata_precision
from .pipeline import PreprocessingResult
from .provenance import file_hash, identity
from .report import build_report


def validate_bundle(manifest_path: Path) -> dict[str, Any]:
    manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != 1:
        raise ValueError("export: unsupported manifest schema")
    for name, digest in manifest["files"].items():
        if Path(name).name != name:
            raise ValueError("export: invalid artifact path")
        path = manifest_path.parent / name
        if not path.is_file() or file_hash(path) != digest:
            raise ValueError(f"export: {name} hash mismatch")
    return manifest


def bundle_files(output: OutputSettings) -> list[Path]:
    # MNE names epoch splits <name>_epo-1.fif, <name>_epo-2.fif, ...; a looser pattern
    # would also match another bundle's files, such as <name>_epo-x_epo.fif.
    split = re.compile(rf"{re.escape(output.name)}_epo-\d+\.fif")
    names = {f"{output.name}{suffix}" for suffix in BUNDLE_SUFFIXES}
    if not output.directory.is_dir():
        return []
    return sorted(
        path
        for path in output.directory.iterdir()
        if path.is_file() and (path.name in names or split.fullmatch(path.name))
    )


def check_destinations(output: OutputSettings, *, overwrite: bool = False) -> None:
    existing = bundle_files(output)
    if existing and not overwrite:
        raise FileExistsError(f"output: bundle already exists: {existing[0]}; use --overwrite")


def write_result(
    result: PreprocessingResult, output: OutputSettings, *, overwrite: bool = False
) -> Path:
    """Write a preprocessing result as a verified bundle, publishing its manifest last.

    Writes ``<name>_epo.fif`` (double precision), ``<name>_events.tsv``,
    ``<name>_repairs.tsv`` when autoreject ran, ``<name>_report.html`` and the
    manifest ``<name>_preprocessing.json`` with provenance and file hashes. Files
    are staged and read back first. A directory without the manifest holds no
    finished export.

    Parameters
    ----------
    result : PreprocessingResult
        Epochs, ledgers and provenance to write.
    output : OutputSettings
        Output directory and file name stem.
    overwrite : bool, default False
        Replace an existing bundle, removing its files the new one lacks; otherwise
        an existing bundle raises.

    Returns
    -------
    pathlib.Path
        The manifest.
    """
    check_destinations(output, overwrite=overwrite)
    output.directory.mkdir(parents=True, exist_ok=True)
    manifest_name = f"{output.name}_preprocessing.json"
    manifest_path = output.directory / manifest_name
    # Stage and verify everything, then publish the manifest last as the completion marker.
    with TemporaryDirectory(prefix=f".{output.name}-", dir=output.directory) as temporary:
        staged = Path(temporary)
        epochs = result.epochs.copy()
        # The identity links the FIF back to its manifest without a circular file hash.
        epochs.info["description"] = (
            f"preprocessing={manifest_name}; identity={identity(result.provenance)}"
        )
        epochs.save(staged / f"{output.name}_epo.fif", fmt="double")
        restored = mne.read_epochs(staged / f"{output.name}_epo.fif", preload=True, proj=False)
        validate_sampling_precision(epochs, restored)
        np.testing.assert_allclose(restored.get_data(), epochs.get_data(), rtol=1e-14, atol=0)
        np.testing.assert_array_equal(restored.selection, epochs.selection)
        np.testing.assert_array_equal(restored.events, epochs.events)
        if restored.info["custom_ref_applied"] != epochs.info["custom_ref_applied"]:
            raise ValueError("export: reference changed during native serialization")
        validate_metadata_precision(epochs.metadata, restored.metadata)
        result.events.to_csv(staged / f"{output.name}_events.tsv", sep="\t", index=False)
        if result.repairs is not None:
            result.repairs.to_csv(staged / f"{output.name}_repairs.tsv", sep="\t", index=False)
        build_report(result).save(
            staged / f"{output.name}_report.html", open_browser=False, overwrite=False
        )
        manifest = {
            "schema": 1,
            "provenance": result.provenance,
            "files": {path.name: file_hash(path) for path in payload_files(staged)},
        }
        write_json(staged / manifest_name, manifest)
        validate_bundle(staged / manifest_name)
        # Removing the old completion marker makes a partial overwrite visibly incomplete.
        if manifest_path.exists():
            manifest_path.unlink()
        # Old files the new bundle lacks (repairs, epoch splits) would outlive its manifest.
        published = {path.name for path in payload_files(staged)}
        for path in bundle_files(output):
            if path.name not in published:
                path.unlink()
        for path in payload_files(staged):
            if path.name != manifest_name:
                os.replace(path, output.directory / path.name)
        os.replace(staged / manifest_name, manifest_path)
    validate_bundle(manifest_path)
    return manifest_path
