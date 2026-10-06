import mne
import numpy as np
import pytest

from eegtable.preprocessing.config import AmplitudeSettings, AnnotationSettings, StimulationSettings


def test_candidates_do_not_change_raw(raw):
    from eegtable.preprocessing.quality import detect_annotations

    settings = AnnotationSettings(amplitude=AmplitudeSettings(peak={"eeg": 1e-6}))
    result = detect_annotations(raw, settings)
    assert len(raw.annotations) == 0
    assert raw.info["bads"] == []
    assert result.bads


def test_muscle_scores_preserve_excluded_samples_in_checkpoint(raw, tmp_path):
    from eegtable.preprocessing.checkpoints import publish_checkpoint
    from eegtable.preprocessing.config import (
        BadSpan,
        FixedEpochSettings,
        MuscleSettings,
        ProcessingSettings,
    )
    from eegtable.preprocessing.pipeline import StageData, execute_numeric
    from eegtable.preprocessing.provenance import canonical_json

    raw.set_annotations(mne.Annotations([2.0], [1.0], ["BAD_recorded"]))
    settings = ProcessingSettings(
        FixedEpochSettings(2),
        annotations=AnnotationSettings(
            bad_spans=(BadSpan(4.0, 1.0, "BAD_manual"),),
            muscle=MuscleSettings((90.0, 110.0), 4.0, 0.1),
        ),
    )
    state = execute_numeric("load", StageData(raw), settings)
    state = execute_numeric("annotate", state, settings)
    evidence = state.candidates["evidence"]
    canonical_json(evidence)
    scores = evidence["muscle_scores"]
    expected = mne.preprocessing.annotate_muscle_zscore(
        state.raw, ch_type="eeg", filter_freq=(90.0, 110.0)
    )[1]
    missing = np.isnan(expected)
    np.testing.assert_array_equal([score is None for score in scores], missing)
    np.testing.assert_array_equal(np.asarray(scores, dtype=float)[~missing], expected[~missing])
    assert missing.sum() == 2 * raw.info["sfreq"]

    checkpoint = publish_checkpoint(tmp_path, "annotate", "a" * 64, state, {}, {})
    assert checkpoint.state.candidates["evidence"] == evidence
    np.testing.assert_array_equal(checkpoint.state.raw.get_data(), raw.get_data())


def test_stimulation_agrees(raw):
    from eegtable.preprocessing.quality import repair_stimulation

    events = np.array([[2250, 0, 1], [3500, 0, 1]])
    settings = StimulationSettings((1,), ("C3",), -0.004, 0.008, "linear")
    actual = repair_stimulation(raw, events, settings)
    expected = mne.preprocessing.fix_stim_artifact(
        raw.copy(), events=events, event_id=1, tmin=-0.004, tmax=0.008, mode="linear", picks=["C3"]
    )
    np.testing.assert_array_equal(actual.get_data(), expected.get_data())


@pytest.mark.parametrize("position", ["before", "first", "last", "after"])
def test_public_stimulation_repair_rejects_invalid_windows(raw, position):
    from eegtable.preprocessing.quality import repair_stimulation

    samples = {
        "before": raw.first_samp - 100,
        "first": raw.first_samp,
        "last": raw.last_samp,
        "after": raw.last_samp + 100,
    }
    events = np.array([[samples[position], 0, 1]])
    settings = StimulationSettings((1,), ("C3",), -0.004, 0.008, "linear")

    with pytest.raises(ValueError, match="neighboring samples"):
        repair_stimulation(raw, events, settings)


def test_review_unknown_candidate_fails(raw):
    from eegtable.preprocessing.quality import apply_raw_review

    with pytest.raises(ValueError, match="missing"):
        apply_raw_review(raw, ("unknown",), ())


def test_pyprep_minimum(raw):
    from eegtable.preprocessing.config import BadChannelSettings
    from eegtable.preprocessing.quality import detect_bad_channels

    result = detect_bad_channels(raw, BadChannelSettings(methods=("flat",)))
    assert isinstance(result.bads, tuple)


def test_pyprep_repeats_vote_and_notch_copy(raw):
    from eegtable.preprocessing.config import BadChannelSettings
    from eegtable.preprocessing.quality import detect_bad_channels

    pytest.importorskip("pyprep")
    with pytest.raises(ValueError, match="repeats need ransac"):
        BadChannelSettings(repeats=3)
    before = raw.get_data().copy()
    settings = BadChannelSettings(methods=("flat", "deviation"), notch_freqs=(60.0,))
    result = detect_bad_channels(raw, settings)
    # Only the diagnostic copy is notched; the recording is untouched.
    np.testing.assert_array_equal(raw.get_data(), before)
    assert len(result.evidence["repeats"]) == 1
    assert isinstance(result.bads, tuple)


def test_bridge_evidence_names_pairs_and_fits_json(raw):
    from eegtable.preprocessing.provenance import canonical_json
    from eegtable.preprocessing.quality import detect_bridges

    # A near-identical pair behind a leading EOG channel: MNE indexes its EEG picks, not
    # ch_names, and its electrical-distance matrix is NaN off the upper triangle.
    raw.reorder_channels(["VEOG", *[name for name in raw.ch_names if name != "VEOG"]])
    raw.apply_function(lambda values: raw.get_data(picks=["C3"])[0] + 1e-9, picks=["C4"])
    evidence = detect_bridges(raw).evidence
    canonical_json(evidence)
    assert [item["pair"] for item in evidence["bridged"]] == [["C3", "C4"]]
    assert evidence["bridged"][0]["electrical_distance"] < 1e-6


def test_stimulation_intervals_use_mne_rounding(raw):
    from eegtable.preprocessing.quality import stimulation_intervals

    # fix_stim_artifact ceils tmin*sfreq and tmax*sfreq; -1.25 -> -1, 1.25 -> 2.
    events = np.array([[2250, 0, 1]])
    settings = StimulationSettings((1,), ("C3",), -0.005, 0.005, "linear")
    assert stimulation_intervals(raw, events, settings) == [(2249, 2252)]
