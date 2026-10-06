from datetime import UTC, datetime

import numpy as np
import pytest

from eegtable.preprocessing.config import (
    AnnotationSettings,
    BadSpan,
    ChannelSettings,
    FilterSettings,
)


def test_channel_copy(raw):
    from eegtable.preprocessing.raw import prepare_channels

    result = prepare_channels(raw, ChannelSettings(bads=("C3",)))
    assert result.info["bads"] == ["C3"]
    assert raw.info["bads"] == []
    np.testing.assert_array_equal(result.get_data(), raw.get_data())


@pytest.mark.parametrize("dated", [False, True])
def test_annotation_time(raw, dated):
    from eegtable.preprocessing.raw import annotate_raw

    if dated:
        raw.set_meas_date(datetime(2020, 1, 1, tzinfo=UTC))
    result = annotate_raw(raw, AnnotationSettings(bad_spans=(BadSpan(5, 1, "BAD_manual"),)))
    assert result.annotations.onset[0] == 5 + raw.first_time
    assert len(raw.annotations) == 0


def test_filter_agrees_and_leaves_stim(raw):
    from eegtable.preprocessing.raw import filter_raw

    settings = FilterSettings(l_freq=1, h_freq=40)
    actual = filter_raw(raw, settings)
    expected = raw.copy().filter(1, 40, picks=raw.ch_names[:-1], skip_by_annotation=("edge", "bad"))
    np.testing.assert_allclose(actual.get_data(), expected.get_data(), atol=1e-15)
    np.testing.assert_array_equal(actual.get_data(picks=["STI"]), raw.get_data(picks=["STI"]))


def test_short_filter_segment_fails(raw):
    from eegtable.preprocessing.raw import filter_raw

    with pytest.raises(ValueError, match="support"):
        filter_raw(raw.copy().crop(0, 1), FilterSettings(l_freq=1))


def test_notch_attenuates_mains_and_keeps_alpha(raw):
    from eegtable.preprocessing.raw import notch_coefficients, notch_raw

    times = raw.times
    raw.apply_function(lambda values: values + 20e-6 * np.sin(2 * np.pi * 60 * times), picks="eeg")
    notched = notch_raw(raw, FilterSettings(notch_freqs=(60.0,)))
    # Only judge samples beyond the filter's own support at both ends.
    margin = len(notch_coefficients(raw.info["sfreq"], (60.0,)))
    inner = slice(margin, raw.n_times - margin)

    def amplitude(inst, frequency):
        signal = inst.get_data(picks=["C3"])[0, inner]
        basis = np.exp(-2j * np.pi * frequency * times[inner])
        return 2 * abs(np.mean(signal * basis))

    assert amplitude(notched, 60) < amplitude(raw, 60) / 10
    assert abs(amplitude(notched, 10) - amplitude(raw, 10)) < 0.05 * amplitude(raw, 10)


def test_montage_from_a_custom_file(raw, tmp_path):
    from eegtable.preprocessing.raw import prepare_channels

    # Any format MNE's read_custom_montage accepts, not only digitized FIF.
    lines = [
        "\t".join([name, *map(str, ch["loc"][:3])])
        for name, ch in zip(raw.ch_names[:8], raw.info["chs"][:8], strict=True)
    ]
    # A digitizer's file carries the fiducials MNE derives head coordinates from.
    fiducials = raw.get_montage().get_positions()
    lines += [
        "\t".join([name, *map(str, fiducials[point])])
        for name, point in (("FidNz", "nasion"), ("FidT9", "lpa"), ("FidT10", "rpa"))
    ]
    path = tmp_path / "electrodes.sfp"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    bare = raw.copy()
    bare.set_montage(None)
    result = prepare_channels(bare, ChannelSettings(montage=path))
    np.testing.assert_allclose(
        [ch["loc"][:3] for ch in result.info["chs"][:8]],
        [ch["loc"][:3] for ch in raw.info["chs"][:8]],
        atol=1e-6,
    )
