"""
Single-Trial ERD During Hand Movement
=====================================

Mu and beta power over the sensorimotor cortex drops while a hand moves: an
event-related desynchronization (ERD). This tutorial measures it in every trial
of a public recording, from raw data to a table a statistics package reads.

Install ``python -m pip install -e ".[docs]"`` to run this example. The public
recordings download on first use; later runs reuse MNE's data cache.
"""

# %%
# A public recording
# ------------------
# Subject 1 of the EEG Motor Movement/Imagery dataset (Schalk et al., 2004)
# opens and closes the left or the right fist in runs 3, 7 and 11. MNE
# downloads the three runs on first use, about 7 MB.
# Keep the acquisition reference and EEG in volts. Filtering and rejecting
# annotated boundaries prepare this demonstration; reviewed bad channels and
# artifact handling are also needed for research, as in
# :doc:`/guides/preprocessing`. Baseline and movement windows are chosen before
# inspecting the results, with margins inside each epoch for band filtering.

import matplotlib.pyplot as plt
import mne
import numpy as np
from mne.datasets import eegbci

import eegtable as ef

paths = eegbci.load_data(1, [3, 7, 11], update_path=False, verbose=False)
raw = mne.concatenate_raws(
    [mne.io.read_raw_edf(path, preload=True, verbose=False) for path in paths], verbose=False
)
eegbci.standardize(raw)
raw.filter(1.0, 40.0, verbose=False)
events, _ = mne.events_from_annotations(raw, event_id={"T1": 1, "T2": 2}, verbose=False)
epochs = mne.Epochs(
    raw,
    events,
    {"left": 1, "right": 2},
    tmin=-2.0,
    tmax=4.0,
    baseline=None,
    preload=True,
    reject_by_annotation=True,
    verbose=False,
)
print(epochs)

# %%
# The analysis, stated once
# -------------------------
# A recipe names the channels, bands, windows and measures. ``erds_mean`` is each
# trial's mean power change in the movement window, in decibels relative to that
# trial's own baseline. Negative values are ERD. The same recipe, saved as TOML,
# runs over a whole cohort with ``eegtable run``.

recipe = {
    "inputs": {"picks": ["C3", "Cz", "C4"]},
    "bands": {"mu": [8.0, 13.0], "beta": [13.0, 30.0]},
    "windows": {"baseline": [-1.5, -0.5], "movement": [0.5, 2.5]},
    "features": [
        {
            "measure": "erds_mean",
            "bands": ["mu", "beta"],
            "baseline": "baseline",
            "windows": ["movement"],
            "spatial": ["channels"],
        }
    ],
}
table = ef.extract(epochs, recipe, recording="S001").epochs
assert table is not None
assert np.isfinite(table.values).all()
np.testing.assert_array_equal([row[1] for row in table.row_ids], epochs.selection)
print(table)

# %%
# Verify the reduction, including when decibels are taken
# ---------------------------------------------------------
# Hilbert amplitude squared provides band power. Each trial/channel's arithmetic
# mean baseline power normalizes every movement sample before conversion to dB.
# ``erds_mean`` then averages that dB trace. It is not the dB of the ratio of
# two mean powers, because logarithms and averaging do not commute.
# This NumPy calculation checks the reduction on the same filtered signals;
# it does not independently validate the bandpass/Hilbert estimator.

channels = ["C3", "Cz", "C4"]
selected_epochs = epochs.copy().pick(channels)
for band_name, limits in recipe["bands"].items():
    signal = ef.BandSignal.from_epochs(
        selected_epochs, ef.Band(band_name, *limits), recording="S001"
    )
    base_mask = (signal.times >= -1.5) & (signal.times <= -0.5)
    movement_mask = (signal.times >= 0.5) & (signal.times <= 2.5)
    reference_power = signal.power[:, :, base_mask].mean(axis=-1)
    decibels = 10.0 * np.log10(signal.power[:, :, movement_mask] / reference_power[:, :, None])
    expected = decibels.mean(axis=-1)
    for channel, name in enumerate(channels):
        column = next(
            index
            for index, feature in enumerate(table.meta)
            if feature.band.name == band_name and feature.space == name
        )
        np.testing.assert_allclose(table.values[:, column], expected[:, channel])

# %%
# One row per value
# -----------------
# Every value comes with its trial, its column's definition, and how much valid
# input it rests on. This long layout is what mixed models and plotting libraries
# read; nothing has to be parsed out of a column name.

long = table.to_long()
print(long[["epoch", "event", "band", "space", "window", "unit", "value", "coverage"]].head())

# %%
# The ERD, trial by trial
# -----------------------
# Almost every trial shows a power decrease in both bands at all three central
# channels, by different amounts from trial to trial.

fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.2), sharey=True, layout="constrained")
for ax, band in zip(axes, ["mu", "beta"], strict=True):
    rows = long[long["band"] == band]
    ax.boxplot(
        [rows.loc[rows["space"] == channel, "value"] for channel in channels],
        tick_labels=channels,
    )
    ax.axhline(0.0, color="0.5", linewidth=0.8)
    ax.set_title(f"{band}, movement vs baseline")
axes[0].set_ylabel("power change (dB)")
plt.show()

# %%
# Left and right hand
# -------------------
# Each trial keeps its event, so conditions compare directly. Across 20
# subjects of this dataset, EEGTable's validation found no consistent
# contralateral dominance in real movement, so no particular difference is
# expected here.

summary = (
    long[long["band"] == "mu"].groupby(["event", "space"])["value"].agg(["mean", "count"]).round(2)
)
print(summary)

# %%
# Describe this participant's trials
# ----------------------------------
# Report the mean, count and fraction of trials with negative ERD at C3. These
# are descriptive summaries of one participant, not population estimates.
# Trials pooled across three runs can share temporal and run-level influences;
# a one-sample t-test would additionally assume independent observations.
# A study needs participant replication and a model of those dependencies,
# with participant/run identities retained alongside the feature table.

for band in ("mu", "beta"):
    values = long.query("band == @band and space == 'C3'")["value"]
    print(
        f"{band} at C3: mean {values.mean():.2f} dB over {len(values)} trials, "
        f"{(values < 0.0).mean():.1%} below baseline"
    )

# %%
# References
# ----------
# Schalk, G., McFarland, D. J., Hinterberger, T., Birbaumer, N., & Wolpaw, J. R.
# (2004). BCI2000: a general-purpose brain-computer interface (BCI) system. IEEE
# Transactions on Biomedical Engineering, 51(6), 1034-1043.
