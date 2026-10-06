import pytest


def write_config(tmp_path, extra=""):
    path = tmp_path / "preprocessing.yaml"
    path.write_text(
        "input: {path: raw.fif}\noutput: {directory: out, name: subject}\n"
        "epochs: {kind: fixed, duration: 2.0}\n" + extra,
        encoding="utf-8",
    )
    return path


def test_paths_and_defaults(tmp_path):
    from eegtable.preprocessing import load_config

    config = load_config(write_config(tmp_path))
    assert config.input.path == tmp_path / "raw.fif"
    assert config.output.directory == tmp_path / "out"
    assert config.processing.filter.l_freq is None
    assert config.processing.artifact is None


@pytest.mark.parametrize(
    "extra,match",
    [
        ("filter: {highpass: 1}\n", "filter.highpass"),
        ("filter: {l_freq: true}\n", "filter.l_freq"),
        ("filter: {l_freq: .nan}\n", "filter.l_freq"),
        ("filter: {l_freq: -1}\n", "filter.l_freq"),
        ("filter: {l_freq: 2, l_freq: 3}\n", "duplicate"),
        ("channels: {bads: [C3, C3]}\n", "channels.bads"),
        ("reference: {channels: guessed}\n", "reference.channels"),
        ("unexpected: {}\n", "unexpected"),
    ],
)
def test_invalid_config(tmp_path, extra, match):
    from eegtable.preprocessing import load_config

    with pytest.raises((ValueError, TypeError), match=match):
        load_config(write_config(tmp_path, extra))


@pytest.mark.parametrize("value", [True, -1, float("inf"), "1"])
def test_direct_settings_validate(value):
    from eegtable.preprocessing.config import FilterSettings

    with pytest.raises((TypeError, ValueError), match="filter.l_freq"):
        FilterSettings(l_freq=value)
