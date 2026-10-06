import numpy as np

from eegtable.preprocessing import preprocess
from eegtable.preprocessing.config import FixedEpochSettings, ICASettings, ProcessingSettings
from eegtable.preprocessing.ica import fit_ica
from eegtable.preprocessing.report import build_artifact_report, build_report


def test_report_is_headless_and_leaves_result_unchanged(raw, tmp_path):
    result = preprocess(raw, ProcessingSettings(FixedEpochSettings(2)))
    before = result.epochs.get_data().copy()
    path = tmp_path / "report.html"
    build_report(result).save(path, open_browser=False)
    html = path.read_text(encoding="utf-8")
    assert "Provenance" in html and "Event and rejection ledger" in html and "original_row" in html
    np.testing.assert_array_equal(result.epochs.get_data(), before)


def test_artifact_report_shows_every_component(mixture, tmp_path):
    model = fit_ica(mixture, ICASettings(n_components=4))
    path = tmp_path / "ica.html"
    build_artifact_report(model, mixture).save(path, open_browser=False)
    html = path.read_text(encoding="utf-8")
    assert all(f"component {index}." in html for index in range(4))
    assert "rank" in html
