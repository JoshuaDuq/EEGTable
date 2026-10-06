import numpy as np
import pytest

from eegtable.bands import Band
from eegtable.derived import asymmetry, band_ratio
from eegtable.table import ComputationSpec, FeatureMeta, FeatureTable, Normalization

THETA, BETA = Band("theta", 4.0, 8.0), Band("beta", 13.0, 30.0)


def _table(values: np.ndarray, bands, spaces, norm: Normalization = "raw") -> FeatureTable:
    meta = tuple(
        FeatureMeta(
            measure="power",
            band=b,
            space=s,
            space_kind="channel",
            window="stim",
            normalization=norm,
            unit="V^2/Hz",
            source="test",
            window_bounds=(0.0, 1.0),
            computation=ComputationSpec.create("test"),
        )
        for b, s in zip(bands, spaces, strict=True)
    )
    return FeatureTable(
        values=values,
        coverage=np.ones(values.shape),
        meta=meta,
        row_ids=tuple(("test", index, "event") for index in range(values.shape[0])),
    )


def test_raw_power_ratio_divides() -> None:
    table = _table(np.array([[8.0, 2.0]]), [THETA, BETA], ["C3", "C3"])
    out = band_ratio(table, "theta", "beta")
    assert out.values.item() == pytest.approx(4.0)
    assert out.meta[0].measure == "ratio_theta_beta"
    assert out.meta[0].band is None


def test_log_power_ratio_subtracts() -> None:
    table = _table(np.array([[3.0, 1.0]]), [THETA, BETA], ["C3", "C3"], norm="log10")
    out = band_ratio(table, "theta", "beta")
    assert out.values.item() == pytest.approx(2.0)


def test_ratio_is_computed_per_space_and_window() -> None:
    table = _table(
        np.array([[8.0, 2.0, 9.0, 3.0]]),
        [THETA, BETA, THETA, BETA],
        ["C3", "C3", "C4", "C4"],
    )
    out = band_ratio(table, "theta", "beta")
    assert out.values.shape == (1, 2)
    np.testing.assert_allclose(out.values, [[4.0, 3.0]])


def test_a_missing_band_raises() -> None:
    table = _table(np.array([[8.0]]), [THETA], ["C3"])
    with pytest.raises(ValueError, match="beta"):
        band_ratio(table, "theta", "beta")


def test_raw_asymmetry_is_normalized_difference() -> None:
    table = _table(np.array([[1.0, 3.0]]), [THETA, THETA], ["F3", "F4"])
    out = asymmetry(table, pairs=[("F3", "F4")])
    assert out.values.item() == pytest.approx((3.0 - 1.0) / (3.0 + 1.0))
    assert out.meta[0].space == "F3-F4"
    assert out.meta[0].space_kind == "pair"


def test_log_asymmetry_is_a_plain_difference() -> None:
    table = _table(np.array([[1.0, 3.0]]), [THETA, THETA], ["F3", "F4"], norm="log_ratio")
    out = asymmetry(table, pairs=[("F3", "F4")])
    assert out.values.item() == pytest.approx(2.0)


def test_an_unknown_channel_in_a_pair_raises() -> None:
    table = _table(np.array([[1.0, 3.0]]), [THETA, THETA], ["F3", "F4"])
    with pytest.raises(KeyError, match="Fz"):
        asymmetry(table, pairs=[("Fz", "F4")])


def _flagged(values: np.ndarray, bands, spaces, norm: Normalization = "raw") -> FeatureTable:
    table = _table(values, bands, spaces, norm)
    flags = np.zeros(values.shape, dtype=bool)
    flags[:, 1] = True  # the denominator / right-hand operand
    return FeatureTable(
        values=table.values,
        coverage=table.coverage,
        meta=table.meta,
        flags={"artifact": flags},
        row_ids=table.row_ids,
    )


def test_a_ratio_describes_its_own_derivation_not_the_numerator() -> None:
    table = _table(np.array([[8.0, 2.0]]), [THETA, BETA], ["C3", "C3"])
    computation = band_ratio(table, "theta", "beta").meta[0].computation
    assert computation.method == "ratio_theta_beta"
    assert computation.parameters["operation"] == "quotient"
    assert computation.parameters["numerator"]["band"] == {
        "name": "theta",
        "fmin": 4.0,
        "fmax": 8.0,
    }
    assert computation.parameters["denominator"]["band"] == {
        "name": "beta",
        "fmin": 13.0,
        "fmax": 30.0,
    }


def test_swapping_the_operands_changes_the_parameter_hash() -> None:
    table = _table(np.array([[8.0, 2.0]]), [THETA, BETA], ["C3", "C3"])
    forward = band_ratio(table, "theta", "beta").meta[0]
    reverse = band_ratio(table, "beta", "theta").meta[0]
    assert forward.computation.parameter_hash != reverse.computation.parameter_hash


def test_ratios_of_different_denominators_do_not_share_a_hash() -> None:
    alpha = Band("alpha", 8.0, 13.0)
    table = _table(np.array([[8.0, 2.0, 4.0]]), [THETA, BETA, alpha], ["C3", "C3", "C3"])
    beta = band_ratio(table, "theta", "beta").meta[0]
    alpha_ratio = band_ratio(table, "theta", "alpha").meta[0]
    assert beta.computation.parameter_hash != alpha_ratio.computation.parameter_hash


@pytest.mark.parametrize(
    ("norm", "unit"),
    [("raw", "ratio"), ("log10", "log10 ratio"), ("db", "dB")],
)
def test_the_ratio_unit_names_the_scale_it_is_on(norm: Normalization, unit: str) -> None:
    # A difference of two dB values is a dB difference, not a bare log ratio.
    table = _table(np.array([[8.0, 2.0]]), [THETA, BETA], ["C3", "C3"], norm=norm)
    assert band_ratio(table, "theta", "beta").meta[0].unit == unit


@pytest.mark.parametrize(
    ("norm", "unit"), [("raw", "a.u."), ("log10", "log10 ratio"), ("db", "dB")]
)
def test_the_asymmetry_unit_names_the_scale_it_is_on(norm: Normalization, unit: str) -> None:
    table = _table(np.array([[1.0, 3.0]]), [THETA, THETA], ["F3", "F4"], norm=norm)
    assert asymmetry(table, pairs=[("F3", "F4")]).meta[0].unit == unit


def test_an_asymmetry_records_both_channels_and_which_side_is_subtracted() -> None:
    table = _table(np.array([[1.0, 3.0]]), [THETA, THETA], ["F3", "F4"])
    computation = asymmetry(table, pairs=[("F3", "F4")]).meta[0].computation
    assert computation.method == "asymmetry"
    assert computation.parameters["operation"] == "normalized_difference"
    assert computation.parameters["left"]["space"] == "F3"
    assert computation.parameters["right"]["space"] == "F4"


def test_a_ratio_keeps_a_flag_raised_on_either_operand() -> None:
    table = _flagged(np.array([[8.0, 2.0]]), [THETA, BETA], ["C3", "C3"])
    out = band_ratio(table, "theta", "beta")
    assert out.flags["artifact"].tolist() == [[True]]


def test_an_asymmetry_keeps_a_flag_raised_on_either_operand() -> None:
    table = _flagged(np.array([[1.0, 3.0]]), [THETA, THETA], ["F3", "F4"])
    out = asymmetry(table, pairs=[("F3", "F4")])
    assert out.flags["artifact"].tolist() == [[True]]


def test_asymmetry_rejects_mismatched_channel_specifications() -> None:
    meta_left = FeatureMeta(
        measure="power",
        band=THETA,
        space="F3",
        space_kind="channel",
        window="stim",
        normalization="raw",
        unit="V^2/Hz",
        source="test",
        window_bounds=(0.0, 1.0),
        computation=ComputationSpec.create("test"),
    )
    meta_right = FeatureMeta(
        measure="power",
        band=THETA,
        space="F4",
        space_kind="channel",
        window="stim",
        normalization="log10",
        unit="log10",
        source="test",
        window_bounds=(0.0, 1.0),
        computation=ComputationSpec.create("test"),
    )
    table = FeatureTable(
        values=np.array([[1.0, 3.0]]),
        coverage=np.ones((1, 2)),
        meta=(meta_left, meta_right),
        row_ids=(("test", 0, "event"),),
    )
    with pytest.raises(ValueError, match="matching measurement specifications"):
        asymmetry(table, pairs=[("F3", "F4")])


def test_band_ratio_rejects_ambiguous_features() -> None:
    meta_theta1 = FeatureMeta(
        measure="power",
        band=THETA,
        space="C3",
        space_kind="channel",
        window="stim",
        normalization="raw",
        unit="V^2/Hz",
        source="welch",
        window_bounds=(0.0, 1.0),
        computation=ComputationSpec.create("welch", n_fft=256),
        freq_resolution_hz=0.5,
    )
    meta_theta2 = FeatureMeta(
        measure="power",
        band=THETA,
        space="C3",
        space_kind="channel",
        window="stim",
        normalization="raw",
        unit="V^2/Hz",
        source="welch",
        window_bounds=(0.0, 1.0),
        computation=ComputationSpec.create("welch", n_fft=256),
        freq_resolution_hz=1.0,
    )
    meta_beta = FeatureMeta(
        measure="power",
        band=BETA,
        space="C3",
        space_kind="channel",
        window="stim",
        normalization="raw",
        unit="V^2/Hz",
        source="welch",
        window_bounds=(0.0, 1.0),
        computation=ComputationSpec.create("welch", n_fft=256),
        freq_resolution_hz=0.5,
    )
    table = FeatureTable(
        values=np.array([[1.0, 2.0, 3.0]]),
        coverage=np.ones((1, 3)),
        meta=(meta_theta1, meta_theta2, meta_beta),
        row_ids=(("test", 0, "event"),),
    )
    with pytest.raises(ValueError, match="Ambiguous"):
        band_ratio(table, "theta", "beta")


@pytest.mark.parametrize("norm", ["percent"])
def test_band_ratio_refuses_signed_change_scales(norm: Normalization) -> None:
    # A percent change crosses zero, so a quotient of two of them is not a band
    # ratio: it is unbounded and can be negative, which no power ratio can be.
    table = _table(np.array([[30.0, -20.0]]), [THETA, BETA], ["C3", "C3"], norm=norm)
    with pytest.raises(ValueError, match="percent"):
        band_ratio(table, "theta", "beta")


@pytest.mark.parametrize("norm", ["percent"])
def test_asymmetry_refuses_signed_change_scales(norm: Normalization) -> None:
    # (right - left) / (right + left) is bounded to [-1, 1] only for non-negative
    # inputs; on percent changes it diverges wherever the two nearly cancel.
    table = _table(np.array([[30.0, -20.0]]), [THETA, THETA], ["C3", "C4"], norm=norm)
    with pytest.raises(ValueError, match="percent"):
        asymmetry(table, [("C3", "C4")])


def test_a_ratio_rests_on_the_less_supported_of_its_two_bands() -> None:
    # Like coverage: the ratio cannot rest on more of its window than either operand did.
    base = _table(np.array([[8.0, 2.0]]), [THETA, BETA], ["C3", "C3"])
    table = FeatureTable(
        values=base.values,
        coverage=base.coverage,
        meta=base.meta,
        row_ids=base.row_ids,
        support=np.array([[0.3, 0.9]]),
    )
    ratio = band_ratio(table, "theta", "beta")
    assert ratio.support is not None and ratio.support.tolist() == [[0.3]]


def test_an_asymmetry_rests_on_the_less_supported_of_its_two_channels() -> None:
    base = _table(np.array([[2.0, 1.0]]), [THETA, THETA], ["C3", "C4"])
    table = FeatureTable(
        values=base.values,
        coverage=base.coverage,
        meta=base.meta,
        row_ids=base.row_ids,
        support=np.array([[0.8, 0.4]]),
    )
    result = asymmetry(table, [("C3", "C4")])
    assert result.support is not None and result.support.tolist() == [[0.4]]
