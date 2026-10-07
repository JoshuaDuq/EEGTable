import mne
import numpy as np
import pytest

from eegtable.preprocessing.config import FixedEpochSettings, OutputSettings, ProcessingSettings
from eegtable.preprocessing.pipeline import preprocess


@pytest.mark.parametrize(
    "missing", ["_epo.fif", "_events.tsv", "_report.html", "_repairs.tsv", "all"]
)
def test_bundle_requires_its_complete_payload_inventory(tmp_path, missing):
    import json

    from eegtable.preprocessing.io import validate_bundle
    from eegtable.preprocessing.provenance import file_hash

    suffixes = ("_epo.fif", "_events.tsv", "_report.html", "_repairs.tsv")
    files = {}
    for suffix in suffixes:
        if missing not in (suffix, "all"):
            payload = tmp_path / f"subject{suffix}"
            payload.write_bytes(b"payload")
            files[payload.name] = file_hash(payload)
    provenance = {"repair": {}} if missing == "_repairs.tsv" else {}
    manifest = tmp_path / "subject_preprocessing.json"
    manifest.write_text(json.dumps({"schema": 1, "files": files, "provenance": provenance}))
    with pytest.raises(ValueError, match="required payload"):
        validate_bundle(manifest)


def test_export_bundle(raw, tmp_path):
    from eegtable.preprocessing.io import validate_bundle, write_result

    result = preprocess(raw, ProcessingSettings(FixedEpochSettings(2)))
    output = OutputSettings(tmp_path, "subject")
    manifest = write_result(result, output)
    validate_bundle(manifest)
    restored = mne.read_epochs(tmp_path / "subject_epo.fif", preload=True, proj=False)
    np.testing.assert_allclose(restored.get_data(), result.epochs.get_data(), rtol=0, atol=1e-18)
    with pytest.raises(FileExistsError):
        write_result(result, output)


def test_overwrite_removes_files_the_new_bundle_lacks(raw, tmp_path):
    import json
    from dataclasses import replace

    import pandas as pd

    from eegtable.preprocessing.io import write_result

    result = preprocess(raw, ProcessingSettings(FixedEpochSettings(2)))
    output = OutputSettings(tmp_path, "subject")
    write_result(replace(result, repairs=pd.DataFrame({"epoch": [0]})), output)
    # A split left by an earlier, larger export, and files of other bundles or the user.
    for name in ("subject_epo-1.fif", "subject_notes.txt", "subject_epo-x_epo.fif"):
        (tmp_path / name).write_bytes(b"")
    manifest = write_result(result, output, overwrite=True)
    published = set(json.loads(manifest.read_text(encoding="utf-8"))["files"])
    assert published == {"subject_epo.fif", "subject_events.tsv", "subject_report.html"}
    assert {path.name for path in tmp_path.iterdir() if not path.name.startswith(".")} == {
        *published,
        manifest.name,
        "subject_notes.txt",
        "subject_epo-x_epo.fif",
    }


def test_export_rejects_sampling_rate_loss_before_publication(tmp_path):
    from eegtable.preprocessing.io import write_result

    raw = mne.io.RawArray(np.ones((1, 1000)), mne.create_info(["Cz"], 100.1, "eeg"))
    result = preprocess(raw, ProcessingSettings(FixedEpochSettings(200 / 100.1)))

    with pytest.raises(ValueError, match="sampling.*FIF.*precision"):
        write_result(result, OutputSettings(tmp_path, "subject"))

    assert not (tmp_path / "subject_epo.fif").exists()
    assert not (tmp_path / "subject_preprocessing.json").exists()


def test_export_keeps_float_metadata(raw, tmp_path):
    import pandas as pd

    from eegtable.preprocessing.config import EventEpochSettings, EventSettings
    from eegtable.preprocessing.io import write_result

    # MNE stores epoch metadata as JSON at 10 decimal places; this rounding is
    # small relative to ordinary reaction times and must not fail the export.
    trials = tmp_path / "trials.tsv"
    pd.DataFrame({"rt": np.linspace(0.2, 0.9, 5) + 1e-11 * np.pi}).to_csv(
        trials, sep="\t", index=False
    )
    events = EventSettings("stim", {"stimulus": 1}, stim_channel="STI", shortest_event=1)
    result = preprocess(
        raw, ProcessingSettings(EventEpochSettings(events, -0.2, 0.8, metadata=trials))
    )
    write_result(result, OutputSettings(tmp_path / "out", "subject"))
    restored = mne.read_epochs(tmp_path / "out" / "subject_epo.fif", preload=True, proj=False)
    np.testing.assert_allclose(restored.metadata["rt"], result.epochs.metadata["rt"], rtol=1e-9)


def test_export_names_metadata_that_native_fif_cannot_preserve(raw, tmp_path):
    import pandas as pd

    from eegtable.preprocessing.io import write_result

    result = preprocess(raw, ProcessingSettings(FixedEpochSettings(2)))
    result.epochs.metadata = pd.DataFrame(
        {"amplitude": np.linspace(1.1e-12, 2.2e-12, len(result.epochs))}
    )
    output = OutputSettings(tmp_path, "subject")
    with pytest.raises(ValueError, match="metadata.*precision.*rescale"):
        write_result(result, output)
    assert not (tmp_path / "subject_preprocessing.json").exists()
