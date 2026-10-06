"""Put the paradigm's marker names back into files converted before eeg_raw_to_bids kept them.

The BrainVision derivatives (BIDS and every decomb stage) share one marker file per run,
byte for byte, so each is rewritten from a freshly converted BIDS tree whose positions must
match. eegtable bundles get the event name renamed in the recipe, the events ledger, the
epochs' event_id and the manifest's provenance, with the epochs' provenance identity and the
manifest hashes recomputed.

    python paradigm_specific/thermal_pain/repair_markers.py --fixed-bids FIXED \\
        --derivatives BIDS_EEG DECOMB_STAGE... --bundles EEGTABLE --archive ARCHIVE
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import mne
import pandas as pd
import yaml

from eegtable.preprocessing.config import read_yaml
from eegtable.preprocessing.provenance import canonical_json, file_hash, identity

GENERIC = {"Stimulus/S  1": "Trig_therm/T  1", "Stimulus/S  2": "Volume/V  1"}
STAMP = re.compile(r"(preprocessing=[^;]+; identity=)[0-9a-f]{64}")
RUN = re.compile(r"(sub-[A-Za-z0-9]+)_task-thermalactive_run-(\d+)")


def marker_lines(path: Path) -> list[str]:
    return [
        line.split("=", 1)[1]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.startswith("Mk")
    ]


def count_generic_task_markers(markers: list[str]) -> Counter[str]:
    names = {named: generic for generic, named in GENERIC.items()}
    events = []
    for marker in markers:
        kind, description, *geometry = marker.split(",")
        label = f"{kind}/{description}"
        if label in names:
            events.append(",".join([*names[label].split("/", 1), *geometry]))
    return Counter(events)


def splice_markers(fixed: Path, old: Path) -> None:
    # Codes, counts and geometry must agree before restoring the lost task labels.
    named = marker_lines(fixed)
    previous = marker_lines(old)
    old_segments = Counter(marker for marker in previous if marker.startswith("New Segment,"))
    new_segments = Counter(marker for marker in named if marker.startswith("New Segment,"))
    if old_segments != new_segments:
        raise ValueError(
            f"{old}: New Segment counts, positions, durations, channels or timestamps "
            f"differ from {fixed}"
        )
    generic = Counter(marker for marker in previous if not marker.startswith("New Segment,"))
    if generic != count_generic_task_markers(named):
        raise ValueError(
            f"{old}: marker codes, counts, positions, durations or channels differ from {fixed}"
        )
    kept = [
        line for line in old.read_text(encoding="utf-8").splitlines() if not line.startswith("Mk")
    ]
    numbered = [f"Mk{number}={marker}" for number, marker in enumerate(named, start=1)]
    old.write_text("\n".join([*kept, *numbered]) + "\n", encoding="utf-8")


def rename(text: str) -> str:
    return GENERIC.get(text, text)


def rename_labels(value: Any) -> Any:
    if isinstance(value, str):
        return rename(value)
    if isinstance(value, list):
        return [rename_labels(item) for item in value]
    if isinstance(value, dict):
        renamed = {}
        for name, item in value.items():
            key = rename(name) if isinstance(name, str) else name
            if key in renamed:
                raise ValueError(f"Marker labels collide after renaming: {key!r}")
            renamed[key] = rename_labels(item)
        return renamed
    return value


def repair_bundle(manifest_path: Path) -> None:
    manifest = json.loads(manifest_path.read_text())
    manifest["provenance"] = rename_labels(manifest["provenance"])
    stem = manifest_path.name.removesuffix("_preprocessing.json")
    folder = manifest_path.parent
    epochs_file = folder / f"{stem}_epo.fif"
    epochs = mne.read_epochs(epochs_file, preload=True, proj=False, verbose=False)
    epochs.event_id = rename_labels(epochs.event_id)
    # The FIF carries the provenance identity; leaving the old one makes eegtable run refuse it.
    epochs.info["description"] = STAMP.sub(
        rf"\g<1>{identity(manifest['provenance'])}", epochs.info["description"] or ""
    )
    ledger = folder / f"{stem}_events.tsv"
    frame = pd.read_csv(ledger, sep="\t")
    frame["label"] = frame["label"].map(rename)
    recipe = folder / f"{stem}_recipe.yaml"
    settings = rename_labels(read_yaml(recipe))
    epochs.save(epochs_file, overwrite=True, fmt="double", verbose=False)
    frame.to_csv(ledger, sep="\t", index=False)
    recipe.write_text(yaml.safe_dump(settings, sort_keys=False))
    for name in manifest["files"]:
        manifest["files"][name] = file_hash(folder / name)
    manifest_path.write_text(canonical_json(manifest) + "\n")


# Parked generations stay as they were; only bundles still carrying the generic name qualify.
def live_bundles(root: Path) -> list[Path]:
    return [
        manifest
        for manifest in sorted(root.rglob("*_task-thermalactive_run-*_preprocessing.json"))
        if not any(part.startswith("_superseded") for part in manifest.relative_to(root).parts)
        and not manifest.name.startswith("._")  # exFAT AppleDouble sidecars match the glob
        and "Stimulus/S" in manifest.read_text()
    ]


def archive(path: Path, root: Path, archive_root: Path) -> None:
    target = archive_root / path.relative_to(root)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        shutil.copyfile(path, target)  # not copy2: xattrs become ._ sidecars on exFAT


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--fixed-bids", type=Path, required=True, help="BIDS tree written by the fixed converter"
    )
    parser.add_argument(
        "--derivatives",
        type=Path,
        nargs="*",
        default=[],
        help="trees holding *_eeg.vmrk to rewrite",
    )
    parser.add_argument(
        "--bundles", type=Path, default=None, help="EEGTable tree holding *_preprocessing.json"
    )
    parser.add_argument(
        "--archive", type=Path, required=True, help="where originals are copied first"
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    fixed = {
        RUN.search(p.name).groups(): p
        for p in args.fixed_bids.rglob("*_eeg.vmrk")
        if RUN.search(p.name)
    }
    spliced = 0
    for root in args.derivatives:
        for old in sorted(root.rglob("*_task-thermalactive_run-*_eeg.vmrk")):
            key = RUN.search(old.name)
            if key is None or old.name.startswith("._"):
                continue
            source = fixed.get(key.groups())
            if source is None:
                raise FileNotFoundError(f"{old}: no fixed conversion for {key.group(0)}")
            if marker_lines(old) == marker_lines(source):
                continue
            if not args.dry_run:
                archive(old, root, args.archive / root.name)
                splice_markers(source, old)
            spliced += 1
    repaired = 0
    if args.bundles is not None:
        for manifest in live_bundles(args.bundles):
            if not args.dry_run:
                stem = manifest.name.removesuffix("_preprocessing.json")
                for suffix in ("_preprocessing.json", "_epo.fif", "_events.tsv", "_recipe.yaml"):
                    archive(
                        manifest.with_name(stem + suffix),
                        args.bundles,
                        args.archive / args.bundles.name,
                    )
                repair_bundle(manifest)
            repaired += 1
    verb = "would rewrite" if args.dry_run else "rewrote"
    print(f"{verb} {spliced} marker files and {repaired} bundles")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
