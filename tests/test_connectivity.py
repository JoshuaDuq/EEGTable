import sys
from dataclasses import replace

import numpy as np
import pytest
from scipy.signal import hilbert

import eegtable as ef
from eegtable.bands import Band
from eegtable.connectivity import (
    _clustering,
    _global_efficiency,
    clustering_coefficient,
    envelope_correlation,
    global_efficiency,
)
from eegtable.signal import BandSignal, Signal
from eegtable.spectra import Window

SFREQ = 100.0
ALPHA = Band("alpha", 8.0, 13.0)
WINDOW = Window("all", 0.0, 4.0)
CHANNELS = ("C3", "C4", "P3", "P4")


def _signal(envelope: np.ndarray) -> BandSignal:
    return BandSignal.from_arrays(
        analytic=envelope.astype(complex),
        times=np.arange(envelope.shape[-1]) / SFREQ,
        ch_names=CHANNELS,
        band=ALPHA,
        sfreq=SFREQ,
        row_ids=tuple(("test", index, "event") for index in range(envelope.shape[0])),
    )


def _analytic(real: np.ndarray) -> BandSignal:
    """Hilbert a real band-limited signal, as ``BandSignal.from_epochs`` does.

    Casting a bare envelope to complex, as an earlier version of these fixtures
    did, leaves every channel at zero phase. Nothing that reads phase can be
    tested against such a signal.
    """
    return _signal(hilbert(np.asarray(real, dtype=float), axis=-1))


def _slow_drive(rng: np.random.RandomState, n_epochs: int, n_times: int) -> np.ndarray:
    # Slow relative to the 10 Hz carrier, or the Hilbert envelope does not recover it.
    t = np.arange(n_times) / SFREQ
    phase = rng.rand(n_epochs, 1) * 2 * np.pi
    return 1.0 + 0.8 * np.sin(2 * np.pi * 0.7 * t[np.newaxis, :] + phase)


def _shared_driver(strength: float, n_epochs: int = 8, n_times: int = 401) -> BandSignal:
    """C3 and C4 share a slow amplitude envelope but sit a quarter cycle apart.

    Two interacting sources: the amplitude coupling is real and survives
    orthogonalization, because the phase lag is not zero.
    """
    rng = np.random.RandomState(0)
    t = np.arange(n_times) / SFREQ
    carrier = 2 * np.pi * 10.0 * t
    drive = _slow_drive(rng, n_epochs, n_times)
    noise = rng.randn(n_epochs, 4, n_times) / max(strength, 1e-9)
    real = np.stack(
        [
            drive * np.cos(carrier),
            drive * np.cos(carrier + np.pi / 2),
            rng.randn(n_epochs, n_times),
            rng.randn(n_epochs, n_times),
        ],
        axis=1,
    )
    return _analytic(real + 0.1 * noise)


def _zero_lag_leakage(strength: float, n_epochs: int = 8, n_times: int = 401) -> BandSignal:
    """One source reaching C3 and C4 at different gains, plus independent sensor noise.

    That is what volume conduction looks like: not two coupled sources, but one
    source counted twice. It is the case orthogonalization exists to remove.
    """
    rng = np.random.RandomState(0)
    t = np.arange(n_times) / SFREQ
    drive = _slow_drive(rng, n_epochs, n_times)
    source = drive * np.cos(2 * np.pi * 10.0 * t)
    sensor_noise = 0.15 * rng.randn(n_epochs, 2, n_times) / max(strength, 1e-9)
    real = np.stack(
        [
            source + sensor_noise[:, 0],
            0.8 * source + sensor_noise[:, 1],
            rng.randn(n_epochs, n_times),
            rng.randn(n_epochs, n_times),
        ],
        axis=1,
    )
    return _analytic(real)


def _by_pair(table) -> dict[str, float]:
    return dict(zip([m.space for m in table.meta], table.values[0], strict=True))


def test_channels_sharing_a_driver_correlate_and_others_do_not() -> None:
    values = _by_pair(envelope_correlation([_shared_driver(5.0)], windows=[WINDOW]))
    assert values["C3-C4"] > 0.8
    assert abs(values["P3-P4"]) < 0.3


@pytest.mark.parametrize("members", [("C3", "C4"), ("C3",)])
def test_pairwise_nodes_refuse_overlapping_rois(members) -> None:
    from eegtable.connectivity import _nodes

    with pytest.raises(ValueError, match="overlap"):
        _nodes(CHANNELS, {"a": ("C3",), "b": members})


def test_zero_lag_coupling_is_suppressed_but_lagged_coupling_survives() -> None:
    # The two fixtures carry identical amplitude coupling on C3-C4 and differ only
    # in phase. Volume conduction is the zero-lag one, so it is the one that has to
    # go; a measure that cannot tell them apart is measuring the head, not the brain.
    lagged = _by_pair(envelope_correlation([_shared_driver(5.0)], windows=[WINDOW]))
    zero_lag = _by_pair(envelope_correlation([_zero_lag_leakage(5.0)], windows=[WINDOW]))
    assert lagged["C3-C4"] > 0.8
    # The residual is not zero, because the leakage fixture carries independent
    # sensor noise. Across seeds it lands anywhere in 0.03-0.08, so what is
    # asserted is the suppression ratio rather than a bound tuned to one draw.
    assert zero_lag["C3-C4"] < 0.15
    assert lagged["C3-C4"] / zero_lag["C3-C4"] > 10.0


def test_without_orthogonalization_leakage_is_indistinguishable_from_coupling() -> None:
    # Why the default is "pairwise": the raw correlation reports the same strong
    # edge for both, which is exactly the failure mode.
    lagged = _by_pair(
        envelope_correlation([_shared_driver(5.0)], windows=[WINDOW], orthogonalize=None)
    )
    zero_lag = _by_pair(
        envelope_correlation([_zero_lag_leakage(5.0)], windows=[WINDOW], orthogonalize=None)
    )
    assert lagged["C3-C4"] > 0.8
    assert zero_lag["C3-C4"] > 0.8


def test_an_unknown_orthogonalization_raises() -> None:
    with pytest.raises(ValueError, match="orthogonalize"):
        envelope_correlation([_shared_driver(1.0)], windows=[WINDOW], orthogonalize="symmetric")


def test_the_orthogonalization_setting_is_recorded_and_changes_the_feature_name() -> None:
    orthogonalized = envelope_correlation([_shared_driver(1.0)], windows=[WINDOW])
    raw = envelope_correlation([_shared_driver(1.0)], windows=[WINDOW], orthogonalize=None)
    parameters = orthogonalized.meta[0].computation.parameters["estimator_parameters"]
    assert parameters == {"orthogonalize": "pairwise", "absolute": True}
    assert orthogonalized.names[0] != raw.names[0]


def test_the_coupled_pair_is_the_one_labelled_as_coupled() -> None:
    # Pair placement, not pair values: a matrix reconstructed with the wrong index
    # order still produces the right set of numbers against the wrong labels.
    table = envelope_correlation([_shared_driver(5.0)], windows=[WINDOW])
    values = dict(zip([m.space for m in table.meta], table.values[0], strict=True))
    assert max(values, key=lambda name: values[name]) == "C3-C4"


def test_every_unordered_pair_appears_once() -> None:
    table = envelope_correlation([_shared_driver(1.0)], windows=[WINDOW])
    spaces = [m.space for m in table.meta]
    assert len(spaces) == 6  # 4 nodes -> 4*3/2
    assert len(set(spaces)) == 6
    assert all(m.space_kind == "pair" for m in table.meta)


def test_the_result_has_one_row_per_trial_group() -> None:
    signal = _shared_driver(5.0)
    table = envelope_correlation([signal], windows=[WINDOW])
    assert table.row_labels == ("all",)
    grouped = envelope_correlation([signal], windows=[WINDOW], trials=["a"] * 4 + ["b"] * 4)
    assert grouped.row_labels == ("a", "b")
    assert grouped.values.shape[0] == 2


def test_it_cannot_be_joined_to_per_epoch_features() -> None:
    signal = _shared_driver(1.0)
    pairs = envelope_correlation([signal], windows=[WINDOW])
    per_epoch = ef.variance([signal], windows=[WINDOW], include_global=False)
    with pytest.raises(ValueError, match="row semantics"):
        ef.concat([pairs, per_epoch])


def test_groups_make_rois_the_nodes() -> None:
    table = envelope_correlation(
        [_shared_driver(1.0)],
        windows=[WINDOW],
        groups={"front": ["C3", "C4"], "back": ["P3", "P4"]},
    )
    assert [m.space for m in table.meta] == ["front-back"]


@pytest.mark.parametrize("amplitude", [1.0, 0.1, 1e-6])
@pytest.mark.parametrize("orthogonalize", ["pairwise", None])
def test_aec_withholds_a_constant_envelope(amplitude, orthogonalize) -> None:
    times = np.arange(401) / SFREQ
    varying = (1.0 + times) * 1j
    signal = BandSignal.from_arrays(
        analytic=np.stack([np.full(times.size, amplitude, dtype=complex), varying])[None],
        times=times,
        ch_names=("C3", "C4"),
        band=ALPHA,
        sfreq=SFREQ,
        row_ids=(("test", 0, "event"),),
    )
    table = envelope_correlation([signal], windows=[WINDOW], orthogonalize=orthogonalize)
    assert np.isnan(table.values).all()
    assert (table.coverage == 0.0).all()


@pytest.mark.parametrize("method", ["aec", "coh"])
def test_connectivity_rejects_duplicated_roi_members(method, monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "mne_connectivity", None)
    groups = {"front": ["C3", "C3", "C4"], "back": ["P3", "P4"]}
    with pytest.raises(ValueError, match="duplicate"):
        if method == "aec":
            envelope_correlation([_shared_driver(1.0)], windows=[WINDOW], groups=groups)
        else:
            ef.spectral_connectivity(
                _coupled_broadband(),
                method=method,
                bands=[ALPHA],
                windows=[WINDOW],
                groups=groups,
            )


@pytest.mark.parametrize("method", ["aec", "coh"])
def test_roi_identity_records_members_independent_of_order(method) -> None:
    signal = _shared_driver(1.0)

    def extract(members):
        groups = {"front": members, "back": ["P3", "P4"]}
        if method == "aec":
            return envelope_correlation([signal], windows=[WINDOW], groups=groups)
        pytest.importorskip("mne_connectivity")
        broadband = Signal.from_arrays(
            data=signal.analytic.real,
            times=signal.times,
            ch_names=signal.ch_names,
            sfreq=signal.sfreq,
            row_ids=signal.row_ids,
        )
        return ef.spectral_connectivity(
            broadband, method=method, bands=[ALPHA], windows=[WINDOW], groups=groups
        )

    assert extract(["C3"]).names != extract(["C4"]).names
    first, reversed_members = extract(["C3", "C4"]), extract(["C4", "C3"])
    assert first.names == reversed_members.names
    np.testing.assert_allclose(first.values, reversed_members.values)


def test_an_unknown_channel_in_a_group_raises() -> None:
    with pytest.raises(KeyError, match="Fz"):
        envelope_correlation([_shared_driver(1.0)], windows=[WINDOW], groups={"front": ["Fz"]})


def test_correlation_is_bounded() -> None:
    table = envelope_correlation([_shared_driver(3.0)], windows=[WINDOW])
    assert np.all(np.abs(table.values) <= 1.0 + 1e-12)


def test_trial_correlations_are_averaged_instead_of_pooling_samples() -> None:
    x = np.linspace(0.0, 1.0, 401)
    envelope = np.ones((2, 4, x.size))
    envelope[0, 0] = 10.0 + x
    envelope[0, 1] = 10.0 + x
    envelope[1, 0] = 1.0 + x
    envelope[1, 1] = 2.0 - x

    # orthogonalize=None because this fixture is a bare envelope with no phase;
    # the subject here is how trials are combined, not how leakage is removed.
    table = envelope_correlation([_signal(envelope)], windows=[WINDOW], orthogonalize=None)
    values = dict(zip([meta.space for meta in table.meta], table.values[0], strict=True))

    assert values["C3-C4"] == pytest.approx(0.0, abs=1e-12)


# --- graph measures -------------------------------------------------------------------


def test_global_efficiency_of_a_fully_connected_unit_graph() -> None:
    # Every edge has weight 1, so every shortest path is one step of length ~1.
    matrix = np.ones((4, 4))
    np.fill_diagonal(matrix, 0.0)
    assert _global_efficiency(matrix) == pytest.approx(1.0, rel=1e-6)


def test_global_efficiency_rises_with_connection_strength() -> None:
    weak = np.full((4, 4), 0.1)
    strong = np.full((4, 4), 0.9)
    np.fill_diagonal(weak, 0.0)
    np.fill_diagonal(strong, 0.0)
    assert _global_efficiency(strong) > _global_efficiency(weak)


def test_global_efficiency_uses_paths_not_direct_edges() -> None:
    # A chain A-B-C: A to C has no direct edge but a two-step path exists.
    chain = np.zeros((3, 3))
    chain[0, 1] = chain[1, 0] = 1.0
    chain[1, 2] = chain[2, 1] = 1.0
    complete = np.ones((3, 3))
    np.fill_diagonal(complete, 0.0)
    assert 0.0 < _global_efficiency(chain) < _global_efficiency(complete)


def test_disconnected_node_pairs_contribute_exactly_zero_efficiency() -> None:
    disconnected = np.zeros((2, 2))
    assert _global_efficiency(disconnected) == 0.0


def test_clustering_of_a_triangle_is_one_and_of_a_chain_is_zero() -> None:
    triangle = np.ones((3, 3))
    np.fill_diagonal(triangle, 0.0)
    assert _clustering(triangle, 0.5) == pytest.approx(1.0)
    chain = np.zeros((3, 3))
    chain[0, 1] = chain[1, 0] = chain[1, 2] = chain[2, 1] = 1.0
    assert _clustering(chain, 0.5) == pytest.approx(0.0)


def test_clustering_is_zero_when_no_node_has_two_neighbours() -> None:
    sparse = np.zeros((4, 4))
    sparse[0, 1] = sparse[1, 0] = 1.0
    assert _clustering(sparse, 0.5) == 0.0


def test_clustering_average_includes_isolated_nodes_as_zero() -> None:
    triangle_and_isolate = np.zeros((4, 4))
    triangle_and_isolate[:3, :3] = 1.0
    np.fill_diagonal(triangle_and_isolate, 0.0)
    assert _clustering(triangle_and_isolate, 0.5) == pytest.approx(0.75)


def test_clustering_average_includes_leaf_nodes_as_zero() -> None:
    triangle_and_leaf = np.zeros((4, 4))
    triangle_and_leaf[:3, :3] = 1.0
    triangle_and_leaf[0, 3] = triangle_and_leaf[3, 0] = 1.0
    np.fill_diagonal(triangle_and_leaf, 0.0)

    assert _clustering(triangle_and_leaf, 0.5) == pytest.approx(7.0 / 12.0)


def test_hyphenated_node_names_are_not_parsed_from_display_strings() -> None:
    envelope = np.random.RandomState(21).rand(4, 2, 401)
    signal = BandSignal.from_arrays(
        analytic=envelope.astype(complex),
        times=np.arange(401) / SFREQ,
        ch_names=("EEG-C3", "EEG-C4"),
        band=ALPHA,
        sfreq=SFREQ,
        row_ids=tuple(("test", i, "event") for i in range(4)),
    )
    pairs = envelope_correlation([signal], windows=[WINDOW])
    assert pairs.meta[0].nodes == ("EEG-C3", "EEG-C4")
    assert np.isfinite(global_efficiency(pairs).values).all()


def test_graph_measures_reduce_a_pair_table_to_one_global_column() -> None:
    pairs = envelope_correlation([_shared_driver(5.0)], windows=[WINDOW])
    for table in (global_efficiency(pairs), clustering_coefficient(pairs, threshold=0.2)):
        assert table.values.shape == (1, 1)
        assert table.meta[0].space == "global"
        assert table.meta[0].space_kind == "global"
        assert table.row_labels == pairs.row_labels


def test_the_clustering_threshold_is_recorded() -> None:
    pairs = envelope_correlation([_shared_driver(5.0)], windows=[WINDOW])
    assert "0.2" in clustering_coefficient(pairs, threshold=0.2).meta[0].unit


def test_negative_clustering_threshold_raises() -> None:
    pairs = envelope_correlation([_shared_driver(5.0)], windows=[WINDOW])
    with pytest.raises(ValueError, match="non-negative"):
        clustering_coefficient(pairs, threshold=-0.1)


def test_graph_measures_refuse_a_non_pairwise_table() -> None:
    signal = _shared_driver(1.0)
    per_epoch = ef.variance([signal], windows=[WINDOW], include_global=False)
    with pytest.raises(ValueError, match="pairwise"):
        global_efficiency(per_epoch)


@pytest.mark.parametrize("method", ["coh", "wpli"])
def test_connectivity_reports_its_missing_dependency_clearly(method, monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "mne_connectivity", None)
    signal = _coupled_broadband()
    with pytest.raises(ImportError, match=r"eegtable\[connectivity\]"):
        if method == "wpli":
            ef.wpli(signal, bands=[ALPHA], windows=[WINDOW])
        else:
            ef.spectral_connectivity(signal, method=method, bands=[ALPHA], windows=[WINDOW])


@pytest.mark.skipif(
    __import__("importlib.util", fromlist=["util"]).find_spec("mne_connectivity") is None,
    reason="mne-connectivity is not installed in this environment",
)
def test_wpli_places_a_planted_coupling_on_the_right_pair() -> None:
    from eegtable.connectivity import wpli
    from eegtable.signal import Signal

    n, n_epochs = 800, 12
    times = np.arange(n) / 200.0
    rng = np.random.RandomState(0)
    data = rng.randn(n_epochs, 4, n) * 0.5
    data[:, 0, :] += np.sin(2 * np.pi * 10 * times)
    data[:, 1, :] += np.sin(2 * np.pi * 10 * times + 0.8)  # constant lag with C3
    signal = Signal.from_arrays(
        data=data,
        times=times,
        ch_names=CHANNELS,
        sfreq=200.0,
        row_ids=tuple(("test", index, "event") for index in range(n_epochs)),
    )
    table = wpli(signal, bands=[ALPHA], windows=[Window("all", 0.0, 3.995)])
    values = dict(zip([m.space for m in table.meta], table.values[0], strict=True))
    assert max(values, key=lambda name: values[name]) == "C3-C4"


def _pair_table(
    measure: str, value: float, nodes: tuple[str, ...] = CHANNELS[:3]
) -> ef.FeatureTable:
    """A pairwise table shaped like a connectivity result, with a constant edge weight."""
    metas, columns = [], []
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            metas.append(
                ef.FeatureMeta(
                    measure=measure,
                    band=ALPHA,
                    space=f"{nodes[i]}-{nodes[j]}",
                    space_kind="pair",
                    window=WINDOW.name,
                    normalization="raw",
                    unit="r" if measure == "aec" else "a.u.",
                    source="hilbert",
                    window_bounds=(WINDOW.tmin, WINDOW.tmax),
                    computation=ef.ComputationSpec.create(
                        measure, estimator=measure, nodes=list(nodes)
                    ),
                    nodes=(nodes[i], nodes[j]),
                )
            )
            columns.append(np.full(1, value))
    return ef.FeatureTable(
        values=np.stack(columns, axis=1),
        coverage=np.ones((1, len(columns))),
        meta=tuple(metas),
        row_labels=("all",),
    )


@pytest.mark.parametrize("order", [("aec", "wpli"), ("wpli", "aec")])
def test_concatenated_estimators_yield_separate_graphs(order: tuple[str, str]) -> None:
    # Grouping on band and window alone collapsed these into one graph whose edges
    # came from whichever estimator was concatenated last.
    weights = {"aec": 0.8, "wpli": 0.2}
    table = ef.concat([_pair_table(name, weights[name]) for name in order])
    out = global_efficiency(table)
    assert out.values.shape == (1, 2)
    by_estimator = {
        m.computation.parameters["input_measure"]: value
        for m, value in zip(out.meta, out.values[0], strict=True)
    }
    assert by_estimator == {"aec": pytest.approx(0.8), "wpli": pytest.approx(0.2)}


@pytest.mark.parametrize("order", [("aec", "wpli"), ("wpli", "aec")])
def test_each_graph_keeps_the_provenance_of_its_own_estimator(order: tuple[str, str]) -> None:
    table = ef.concat([_pair_table(name, 0.5) for name in order])
    for meta in global_efficiency(table).meta:
        parameters = meta.computation.parameters
        assert parameters["input_computation"]["method"] == parameters["input_measure"]


def test_a_graph_measure_matches_the_one_built_from_that_estimator_alone() -> None:
    aec = _pair_table("aec", 0.8)
    mixed = global_efficiency(ef.concat([aec, _pair_table("wpli", 0.2)]))
    alone = global_efficiency(aec)
    assert alone.values.shape == (1, 1)
    np.testing.assert_allclose(mixed.values[:, 0], alone.values[:, 0])
    assert mixed.meta[0].name == alone.meta[0].name


def test_the_same_estimator_over_different_node_sets_stays_separate() -> None:
    # Channel-level and ROI-level AEC differ only in their nodes; merged, they
    # would form one graph over the union, which is a graph of neither.
    signal = _shared_driver(5.0)
    channels = envelope_correlation([signal], windows=[WINDOW])
    rois = envelope_correlation(
        [signal], windows=[WINDOW], groups={"left": ["C3", "P3"], "right": ["C4", "P4"]}
    )
    out = global_efficiency(ef.concat([channels, rois]))
    assert out.values.shape == (1, 2)
    np.testing.assert_allclose(
        sorted(out.values[0]),
        sorted([global_efficiency(rois).values[0, 0], global_efficiency(channels).values[0, 0]]),
    )


def test_one_edge_measured_twice_within_a_group_raises() -> None:
    table = _pair_table("aec", 0.8, nodes=("C3", "C4"))
    mirrored = ef.FeatureTable(
        values=table.values,
        coverage=table.coverage,
        meta=(replace(table.meta[0], space="C4-C3", nodes=("C4", "C3")),),
        row_labels=table.row_labels,
    )
    with pytest.raises(ValueError, match="measured twice"):
        global_efficiency(ef.concat([table, mirrored]))


def test_permuting_declared_nodes_does_not_create_a_second_estimator() -> None:
    first = _pair_table("aec", 0.8, nodes=("C3", "C4", "P3"))
    permuted = _pair_table("aec", 0.8, nodes=("P3", "C4", "C3"))

    with pytest.raises(ValueError, match="measured twice"):
        global_efficiency(ef.concat([first, permuted]))


def test_graph_measures_reject_an_incomplete_edge_set() -> None:
    complete = _pair_table("aec", 0.8)
    incomplete = ef.FeatureTable(
        values=complete.values[:, :-1],
        coverage=complete.coverage[:, :-1],
        meta=complete.meta[:-1],
        row_labels=complete.row_labels,
    )

    with pytest.raises(ValueError, match=r"missing.*C4-P3"):
        global_efficiency(incomplete)


def test_graph_measures_reject_self_edges() -> None:
    complete = _pair_table("aec", 0.8)
    malformed = ef.FeatureTable(
        values=complete.values,
        coverage=complete.coverage,
        meta=(replace(complete.meta[0], nodes=("C3", "C3")), *complete.meta[1:]),
        row_labels=complete.row_labels,
    )

    with pytest.raises(ValueError, match="self-edge"):
        global_efficiency(malformed)


def test_the_clustering_threshold_is_recorded_as_a_number() -> None:
    out = clustering_coefficient(_pair_table("aec", 0.8), threshold=0.5)
    assert out.meta[0].computation.parameters["threshold"] == 0.5


def test_a_single_node_set_has_no_pairs_to_report() -> None:
    # One ROI yields zero node pairs; without this the empty column list reaches numpy.
    with pytest.raises(ValueError, match="needs at least two nodes"):
        envelope_correlation(
            [_shared_driver(5.0)], windows=[WINDOW], groups={"central": ["C3", "C4"]}
        )


def test_two_node_sets_give_the_one_pair_between_them() -> None:
    table = envelope_correlation(
        [_shared_driver(5.0)],
        windows=[WINDOW],
        groups={"central": ["C3", "C4"], "parietal": ["P3", "P4"]},
    )

    assert [m.nodes for m in table.meta] == [("central", "parietal")]


def test_wpli_rejects_single_epoch_trial_groups() -> None:
    signal = ef.Signal.from_arrays(
        data=np.random.default_rng(0).normal(size=(3, 4, 401)),
        times=np.arange(401) / SFREQ,
        ch_names=CHANNELS,
        sfreq=SFREQ,
        row_ids=tuple(("test", i, "event") for i in range(3)),
    )
    with pytest.raises(ValueError, match="at least two epochs"):
        ef.wpli(signal, bands=[ALPHA], windows=[WINDOW], trials=["a", "a", "b"])


@pytest.mark.parametrize("invalid", [np.nan, np.inf])
def test_missing_edges_are_not_treated_as_observed_disconnections(invalid) -> None:
    pairs = _pair_table("aec", 0.8)
    values = pairs.values.copy()
    values[0, 0] = invalid
    incomplete = replace(pairs, values=values, coverage=np.isfinite(values).astype(float))
    assert np.isnan(global_efficiency(incomplete).values).all()
    assert np.isnan(clustering_coefficient(incomplete, threshold=0.5).values).all()


def test_wpli_preserves_estimator_reliability_warnings(monkeypatch) -> None:
    import warnings

    import eegtable.connectivity as connectivity

    class Result:
        freqs = np.array([8.0, 10.0, 12.0])

        def get_data(self, output):
            return np.zeros((4, 4, 3))

    def estimate(*args, **kwargs):
        warnings.warn("too few cycles", UserWarning, stacklevel=2)
        return Result()

    monkeypatch.setattr(connectivity, "_require_mne_connectivity", lambda: estimate)
    signal = ef.Signal.from_arrays(
        data=np.ones((2, 4, 401)),
        times=np.arange(401) / SFREQ,
        ch_names=CHANNELS,
        sfreq=SFREQ,
        row_ids=tuple(("test", i, "event") for i in range(2)),
    )
    with pytest.warns(UserWarning, match="too few cycles"):
        ef.wpli(signal, bands=[ALPHA], windows=[WINDOW])


# --- agreement with the reference implementation --------------------------------------


@pytest.mark.skipif(
    __import__("importlib.util", fromlist=["util"]).find_spec("mne_connectivity") is None,
    reason="mne-connectivity is not installed",
)
@pytest.mark.parametrize("fixture", [_shared_driver, _zero_lag_leakage])
def test_orthogonalized_aec_agrees_with_mne_connectivity(fixture) -> None:
    # The defaults exist to match MNE, so they are worth nothing unless the numbers
    # match too. Compared per trial, before the Fisher-z averaging that MNE has no
    # equivalent of.
    from mne_connectivity import envelope_correlation as reference

    from eegtable.connectivity import _trial_correlation

    signal = fixture(5.0)
    for trial in signal.analytic:
        ours = _trial_correlation(trial, orthogonalize="pairwise", absolute=True)
        theirs = np.squeeze(
            np.asarray(
                reference([trial], orthogonalize="pairwise", absolute=True).get_data("dense")
            )
        )
        np.testing.assert_allclose(ours, theirs, rtol=0, atol=0)


# --- wPLI band edges ------------------------------------------------------------------


def _estimator_returning_frequency_as_value(monkeypatch, freqs):
    """Stub whose connectivity value at each frequency is that frequency."""
    import eegtable.connectivity as connectivity

    class Result:
        def __init__(self):
            self.freqs = np.asarray(freqs, dtype=float)

        def get_data(self, output):
            lower = np.tril(np.ones((4, 4)), -1)[:, :, np.newaxis]
            return lower * np.asarray(freqs, dtype=float)[np.newaxis, np.newaxis, :]

    monkeypatch.setattr(connectivity, "_require_mne_connectivity", lambda: lambda *a, **k: Result())


def _broadband(n_epochs: int = 4) -> ef.Signal:
    return ef.Signal.from_arrays(
        data=np.ones((n_epochs, 4, 401)),
        times=np.arange(401) / SFREQ,
        ch_names=CHANNELS,
        sfreq=SFREQ,
        row_ids=tuple(("test", i, "event") for i in range(n_epochs)),
    )


def test_wpli_reduces_bands_half_open_so_an_edge_bin_is_not_counted_twice(monkeypatch) -> None:
    # mne-connectivity treats [fmin, fmax] as closed, so 13 Hz would otherwise land in
    # both alpha and beta. Band.mask is half-open, and this is where that is enforced.
    _estimator_returning_frequency_as_value(monkeypatch, [8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0])
    table = ef.wpli(
        _broadband(),
        bands=[Band("alpha", 8.0, 13.0), Band("beta", 13.0, 15.0)],
        windows=[WINDOW],
    )
    by_band = {(m.band.name, m.space): v for m, v in zip(table.meta, table.values[0], strict=True)}
    assert by_band[("alpha", "C3-C4")] == pytest.approx(np.mean([8, 9, 10, 11, 12]))
    assert by_band[("beta", "C3-C4")] == pytest.approx(np.mean([13, 14]))


def test_wpli_refuses_a_band_the_estimator_returned_no_frequencies_for(monkeypatch) -> None:
    _estimator_returning_frequency_as_value(monkeypatch, [20.0, 21.0])
    with pytest.raises(ValueError, match="contains none of the frequencies"):
        ef.wpli(_broadband(), bands=[Band("alpha", 8.0, 13.0)], windows=[WINDOW])


@pytest.mark.parametrize("mode", ["fourier", "multitaper"])
def test_connectivity_does_not_truncate_a_band_above_nyquist(mode, monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "mne_connectivity", None)
    with pytest.raises(ValueError, match="Nyquist"):
        ef.spectral_connectivity(
            _coupled_broadband(),
            method="coh",
            bands=[Band("unavailable", 40.0, 60.0)],
            windows=[WINDOW],
            mode=mode,
        )


# --- spectral_connectivity --------------------------------------------------------------

_HAS_MNE_CONNECTIVITY = (
    __import__("importlib.util", fromlist=["util"]).find_spec("mne_connectivity") is not None
)
_SUPPORTED = ["coh", "imcoh", "plv", "ciplv", "ppc", "pli", "wpli", "wpli2_debiased"]


def _coupled_broadband(n_epochs: int = 24, lag: int = 6) -> ef.Signal:
    rng = np.random.RandomState(1)
    n = 401
    t = np.arange(n) / SFREQ
    source = (1.0 + 0.8 * np.sin(2 * np.pi * 0.7 * t)) * np.cos(2 * np.pi * 10.0 * t)
    data = np.stack(
        [
            np.stack(
                [
                    source + 0.3 * rng.randn(n),
                    np.roll(source, lag) + 0.3 * rng.randn(n),
                    rng.randn(n),
                    rng.randn(n),
                ]
            )
            for _ in range(n_epochs)
        ]
    )
    return ef.Signal.from_arrays(
        data=data,
        times=t,
        ch_names=CHANNELS,
        sfreq=SFREQ,
        row_ids=tuple(("test", i, "event") for i in range(n_epochs)),
    )


@pytest.mark.skipif(not _HAS_MNE_CONNECTIVITY, reason="mne-connectivity is not installed")
@pytest.mark.parametrize("method", _SUPPORTED)
def test_every_supported_method_produces_a_finite_pair_table(method: str) -> None:
    table = ef.spectral_connectivity(
        _coupled_broadband(), method=method, bands=[ALPHA], windows=[WINDOW]
    )
    assert table.values.shape == (1, 6)  # 4 nodes -> 4*3/2 pairs
    assert np.isfinite(table.values).all()
    assert all(m.space_kind == "pair" for m in table.meta)
    assert all(m.measure == method for m in table.meta)
    values = _by_pair(table)
    assert values["C3-C4"] > max(values["P3-P4"], values["C3-P3"])


@pytest.mark.skipif(not _HAS_MNE_CONNECTIVITY, reason="mne-connectivity is not installed")
@pytest.mark.parametrize("method", _SUPPORTED)
def test_node_order_does_not_change_any_reported_value(method: str) -> None:
    # These pairs are unordered, so exchanging two channels must not move a number.
    # imcoh is antisymmetric and only survives this because it is reported rectified.
    signal = _coupled_broadband()
    swapped = ef.Signal.from_arrays(
        data=signal.data[:, [1, 0, 2, 3], :],
        times=signal.times,
        ch_names=("C4", "C3", "P3", "P4"),
        sfreq=signal.sfreq,
        row_ids=signal.row_ids,
    )
    direct = _by_pair(
        ef.spectral_connectivity(signal, method=method, bands=[ALPHA], windows=[WINDOW])
    )
    other = _by_pair(
        ef.spectral_connectivity(swapped, method=method, bands=[ALPHA], windows=[WINDOW])
    )
    assert direct["C3-C4"] == pytest.approx(other["C4-C3"], rel=1e-9)


@pytest.mark.skipif(not _HAS_MNE_CONNECTIVITY, reason="mne-connectivity is not installed")
def test_wpli_is_exactly_spectral_connectivity_with_that_method() -> None:
    signal = _coupled_broadband()
    np.testing.assert_array_equal(
        ef.wpli(signal, bands=[ALPHA], windows=[WINDOW]).values,
        ef.spectral_connectivity(signal, method="wpli", bands=[ALPHA], windows=[WINDOW]).values,
    )


@pytest.mark.parametrize(
    ("method", "match"),
    [("dpli", "directional"), ("cohy", "complex"), ("nonsense", "unknown connectivity method")],
)
def test_methods_that_cannot_be_an_unordered_pair_are_refused(method: str, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        ef.spectral_connectivity(_broadband(), method=method, bands=[ALPHA], windows=[WINDOW])


def test_an_unknown_spectral_mode_is_refused() -> None:
    with pytest.raises(ValueError, match="mode must be"):
        ef.spectral_connectivity(
            _broadband(), method="wpli", bands=[ALPHA], windows=[WINDOW], mode="wavelet"
        )


def test_the_method_and_mode_are_recorded_and_separate_the_columns(monkeypatch) -> None:
    _estimator_returning_frequency_as_value(monkeypatch, [8.0, 9.0, 10.0])
    coh = ef.spectral_connectivity(_broadband(), method="coh", bands=[ALPHA], windows=[WINDOW])
    pli = ef.spectral_connectivity(_broadband(), method="pli", bands=[ALPHA], windows=[WINDOW])
    assert coh.meta[0].computation.parameters["estimator_parameters"]["mode"] == "multitaper"
    assert coh.meta[0].unit != pli.meta[0].unit
    # Same band, same window, same values: only the estimator differs, and the names
    # still have to, or a concat would collide and a graph would mix the two.
    assert coh.names[0] != pli.names[0]
    joined = ef.concat([coh, pli])
    assert len(set(joined.names)) == len(joined.names)


@pytest.mark.skipif(not _HAS_MNE_CONNECTIVITY, reason="mne-connectivity is not installed")
@pytest.mark.parametrize(("orthogonalize", "absolute"), [("pairwise", True), (None, False)])
def test_the_per_trial_matrix_is_the_one_mne_connectivity_computes(
    orthogonalize: str | None, absolute: bool
) -> None:
    # envelope_correlation is the one connectivity measure eegtable computes itself
    # rather than delegating, and its docstring claims Hipp et al. (2012) as
    # mne-connectivity implements it. Nothing else pins that claim.
    from mne_connectivity import envelope_correlation as mne_envelope_correlation

    from eegtable.connectivity import _trial_correlation

    rng = np.random.RandomState(3)
    trial = rng.randn(5, 800) + 1j * rng.randn(5, 800)
    mine = _trial_correlation(trial, orthogonalize=orthogonalize, absolute=absolute)
    theirs = np.squeeze(
        np.asarray(
            mne_envelope_correlation(
                trial[np.newaxis], orthogonalize=orthogonalize or False
            ).get_data("dense")
        )
    )
    upper = np.triu_indices(5, 1)
    np.testing.assert_allclose(mine[upper], theirs[upper], atol=1e-12)


def test_multitaper_bandwidth_is_fixed_in_hertz_and_recorded() -> None:
    pytest.importorskip("mne_connectivity")
    from eegtable.connectivity import spectral_connectivity

    rng = np.random.RandomState(0)
    times = np.arange(0.0, 2.0, 1.0 / 200.0)
    data = rng.normal(size=(4, 2, times.size)) * 1e-6
    signal = Signal.from_arrays(
        data=data,
        times=times,
        ch_names=("C3", "C4"),
        sfreq=200.0,
        row_ids=tuple(("test", index, "event") for index in range(4)),
    )
    window = Window("all", 0.0, 1.995)
    table = spectral_connectivity(
        signal, method="coh", bands=[ALPHA], windows=[window], bandwidth=3.0
    )
    recorded = table.meta[0].computation.parameters["estimator_parameters"]
    assert recorded["bandwidth_hz"] == 3.0
    # 1.2 frequency bins: MNE would fall back to one leaky taper; the window is named.
    with pytest.raises(ValueError, match="window 'all'"):
        spectral_connectivity(signal, method="coh", bands=[ALPHA], windows=[window], bandwidth=0.6)


@pytest.mark.parametrize("method", ["coh", "imcoh", "plv", "ciplv", "pli", "wpli"])
def test_time_connectivity_matches_mne_per_epoch_and_preserves_row_identity(method) -> None:
    pytest.importorskip("mne_connectivity")
    from mne.time_frequency import morlet
    from mne_connectivity import spectral_connectivity_time as reference

    import eegtable.connectivity as connectivity_methods

    signal = _coupled_broadband(n_epochs=3)
    freqs = np.array([8.0, 9.0, 10.0, 11.0, 12.0, 13.0])
    n_cycles = 3.0
    padding = np.nextafter(
        max(
            len(wavelet) // 2
            for wavelet in morlet(
                SFREQ,
                freqs,
                n_cycles=n_cycles,
                zero_mean=True,
            )
        )
        / SFREQ,
        np.inf,
    )
    expected = reference(
        signal.data,
        freqs=freqs,
        method=method,
        sfreq=SFREQ,
        n_cycles=n_cycles,
        padding=padding,
        average=False,
        verbose=False,
    )
    dense = expected.get_data(output="dense")
    selected = dense[..., freqs < 13]
    if method == "imcoh":
        selected = np.abs(selected)
    matrices = selected.mean(axis=-1)
    matrices = matrices + matrices.transpose(0, 2, 1)
    table = connectivity_methods.spectral_connectivity_time(
        signal,
        method=method,
        freqs=freqs,
        n_cycles=n_cycles,
        bands=[ALPHA],
        windows=[WINDOW],
    )
    indices = np.triu_indices(len(CHANNELS), 1)
    np.testing.assert_allclose(table.values, matrices[:, indices[0], indices[1]])
    assert table.row_labels is None
    assert table.row_ids == signal.row_ids
    assert table.meta[0].computation.parameters["estimator_parameters"]["averaging"] == "time"
    graph = global_efficiency(table)
    assert graph.row_ids == signal.row_ids
    assert graph.row_labels is None
    assert graph.n_rows == signal.data.shape[0]


def test_time_connectivity_rejects_a_wavelet_longer_than_the_window() -> None:
    pytest.importorskip("mne_connectivity")
    import eegtable.connectivity as connectivity_methods

    with pytest.raises(ValueError, match="wavelet|support"):
        connectivity_methods.spectral_connectivity_time(
            _coupled_broadband(),
            method="coh",
            freqs=[8.0, 10.0, 12.0],
            bands=[ALPHA],
            windows=[Window("short", 0.0, 0.2)],
        )


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"freqs": [10.0, 8.0]}, "increasing"),
        ({"freqs": [0.0, 8.0]}, "positive"),
        ({"freqs": [8.0, 60.0]}, "Nyquist"),
        ({"n_cycles": 0.0}, "n_cycles"),
        ({"method": "ppc"}, "method"),
        ({"smoothing_seconds": -1.0}, "smoothing"),
    ],
)
def test_time_connectivity_validates_estimator_inputs(kwargs, message) -> None:
    import eegtable.connectivity as connectivity_methods

    parameters = {"method": "coh", "freqs": [8.0, 10.0, 12.0], "n_cycles": 3.0, **kwargs}
    with pytest.raises(ValueError, match=message):
        connectivity_methods.spectral_connectivity_time(
            _coupled_broadband(),
            bands=[ALPHA],
            windows=[WINDOW],
            **parameters,
        )


@pytest.mark.parametrize("smoothing_seconds", [0.015, 0.02, 0.024])
def test_time_connectivity_rejects_an_undefined_two_sample_hann_smoother(
    smoothing_seconds,
) -> None:
    import eegtable.connectivity as connectivity_methods

    with pytest.raises(ValueError, match="Hanning.*two samples"):
        connectivity_methods.spectral_connectivity_time(
            _coupled_broadband(),
            method="coh",
            freqs=[8.0, 10.0, 12.0],
            n_cycles=3.0,
            smoothing_seconds=smoothing_seconds,
            bands=[ALPHA],
            windows=[WINDOW],
        )


@pytest.mark.parametrize("method", ["coh", "imcoh", "plv", "ciplv", "wpli"])
def test_time_connectivity_surfaces_undefined_backend_estimates(method) -> None:
    pytest.importorskip("mne_connectivity")
    import eegtable.connectivity as connectivity_methods

    signal = _coupled_broadband()
    data = signal.data.copy()
    data *= 1e-200
    signal = replace(signal, data=data)
    with (
        np.errstate(divide="ignore", invalid="ignore"),
        pytest.raises(ValueError, match="finite.*connectivity"),
    ):
        connectivity_methods.spectral_connectivity_time(
            signal,
            method=method,
            freqs=[8.0, 10.0, 12.0],
            n_cycles=3.0,
            bands=[ALPHA],
            windows=[WINDOW],
            groups={"left": ["C3", "P3"], "right": ["C4", "P4"]},
        )


@pytest.mark.parametrize("groups", [None, {"left": ["C3", "P3"], "right": ["C4", "P4"]}])
def test_time_connectivity_retains_pairwise_input_coverage_for_quality(groups) -> None:
    pytest.importorskip("mne_connectivity")
    from eegtable.connectivity import spectral_connectivity_time
    from eegtable.quality import QualityPolicy, apply_quality

    signal = _coupled_broadband()
    coverage = np.ones_like(signal.coverage)
    coverage[:, 0, ::2] = 0.2
    coverage[:, 1, 1::2] = 0.4
    signal = replace(signal, coverage=coverage)
    table = spectral_connectivity_time(
        signal,
        method="coh",
        freqs=[8.0, 10.0, 12.0],
        n_cycles=3.0,
        bands=[ALPHA],
        windows=[WINDOW],
        groups=groups,
    )
    mask = (signal.times >= WINDOW.tmin) & (signal.times <= WINDOW.tmax)
    memberships = {name: [name] for name in signal.ch_names} if groups is None else groups
    expected = np.column_stack(
        [
            np.mean(
                [
                    np.minimum(
                        coverage[:, signal.ch_names.index(left), mask],
                        coverage[:, signal.ch_names.index(right), mask],
                    ).mean(axis=-1)
                    for left in memberships[meta.nodes[0]]
                    for right in memberships[meta.nodes[1]]
                ],
                axis=0,
            )
            for meta in table.meta
        ]
    )
    np.testing.assert_allclose(table.coverage, expected)
    retained = apply_quality(table, QualityPolicy(min_coverage=0.9)).table
    np.testing.assert_array_equal(np.isnan(retained.values), expected < 0.9)


def test_time_connectivity_padding_discards_the_full_integer_wavelet_support() -> None:
    pytest.importorskip("mne_connectivity")
    from mne.time_frequency import morlet
    from mne_connectivity import spectral_connectivity_time as reference

    from eegtable.connectivity import _time_connectivity_padding, spectral_connectivity_time

    signal = _coupled_broadband(n_epochs=2)
    freqs = np.array([8.0, 10.0, 12.0])
    half_support = max(
        len(wavelet) // 2 for wavelet in morlet(SFREQ, freqs, n_cycles=3.0, zero_mean=False)
    )
    padding = _time_connectivity_padding(signal, freqs, 3.0, 0.0)
    assert int(np.floor(padding * SFREQ)) == half_support
    expected = reference(
        signal.data,
        freqs=freqs,
        method="coh",
        sfreq=SFREQ,
        n_cycles=3.0,
        padding=np.nextafter(half_support / SFREQ, np.inf),
        average=False,
        verbose=False,
    ).get_data(output="dense")
    matrix = expected.mean(axis=-1)
    matrix += matrix.transpose(0, 2, 1)
    table = spectral_connectivity_time(
        signal, method="coh", freqs=freqs, n_cycles=3.0, bands=[ALPHA], windows=[WINDOW]
    )
    indices = np.triu_indices(len(CHANNELS), 1)
    np.testing.assert_allclose(table.values, matrix[:, indices[0], indices[1]])


@pytest.mark.parametrize("method", ["coh", "imcoh", "plv", "ciplv", "pli", "wpli"])
@pytest.mark.parametrize("constant", [0.0, 1.0])
def test_time_connectivity_rejects_constant_channels_inside_each_window(method, constant) -> None:
    pytest.importorskip("mne_connectivity")
    from eegtable.connectivity import spectral_connectivity_time

    signal = _coupled_broadband(n_epochs=2)
    data = signal.data.copy()
    data[:, 0, :301] = constant
    signal = replace(signal, data=data)
    assert np.all(np.ptp(signal.data[:, 0], axis=-1) > 0)
    with pytest.raises(ValueError, match="nonconstant.*channel.*window"):
        spectral_connectivity_time(
            signal,
            method=method,
            freqs=[8.0, 10.0, 12.0],
            n_cycles=3.0,
            bands=[ALPHA],
            windows=[Window("flat", 0.0, 3.0)],
        )
