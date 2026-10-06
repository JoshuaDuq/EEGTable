"""
Single-Trial ERD During Hand Movement
=====================================

Mu and beta power over the sensorimotor cortex drops while a hand moves: an
event-related desynchronization (ERD). This tutorial measures it in every trial
of a public recording, from raw data to a table a statistics package reads.
"""

# %%
# A public recording
# ------------------
# Subject 1 of the EEG Motor Movement/Imagery dataset (Schalk et al., 2004)
# opens and closes the left or the right fist in runs 3, 7 and 11. MNE
# downloads the three runs on first use, about 7 MB.

import matplotlib.pyplot as plt
import mne
from mne.datasets import eegbci
from scipy import stats

import eegtable as ef

mne.set_log_level("error")
paths = eegbci.load_data(1, [3, 7, 11], update_path=False)
raw = mne.concatenate_raws([mne.io.read_raw_edf(path, preload=True) for path in paths])
eegbci.standardize(raw)
raw.set_montage("standard_1005")
raw.filter(1.0, 40.0)
events, _ = mne.events_from_annotations(raw, event_id={"T1": 1, "T2": 2})
epochs = mne.Epochs(
    raw, events, {"left": 1, "right": 2}, tmin=-2.0, tmax=4.0, baseline=None, preload=True
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
print(table)

# %%
# One row per value
# -----------------
# Every value comes with its trial, its column's definition, and how much valid
# input it rests on. This long layout is what mixed models and plotting libraries
# read; nothing has to be parsed out of a column name.

long = table.to_long()
long[["epoch", "event", "band", "space", "window", "unit", "value", "coverage"]].head()

# %%
# The ERD, trial by trial
# -----------------------
# Almost every trial shows a power decrease in both bands at all three central
# channels, by different amounts from trial to trial.

fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.2), sharey=True, layout="constrained")
channels = ["C3", "Cz", "C4"]
for ax, band in zip(axes, ["mu", "beta"], strict=True):
    rows = long[long["band"] == band]
    ax.boxplot(
        [rows.loc[rows["space"] == channel, "value"].dropna() for channel in channels],
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
    long[long["band"] == "mu"]
    .groupby(["event", "space"])["value"]
    .agg(["mean", "count"])
    .round(2)
)
summary

# %%
# A first test
# ------------
# Within one recording, trials are the units of this test: it asks whether this
# participant's ERD at C3 differs from zero, not whether the effect holds across
# people. A cohort analysis would model trials within participants, for example a
# mixed model in R or statsmodels, reading the long table written with
# ``long.to_csv(...)``.

for band in ("mu", "beta"):
    values = long.query("band == @band and space == 'C3'")["value"].dropna()
    result = stats.ttest_1samp(values, 0.0)
    print(
        f"{band} at C3: mean {values.mean():.2f} dB over {len(values)} trials, "
        f"t = {result.statistic:.1f}, p = {result.pvalue:.1g}"
    )

# %%
# References
# ----------
# Schalk, G., McFarland, D. J., Hinterberger, T., Birbaumer, N., & Wolpaw, J. R.
# (2004). BCI2000: a general-purpose brain-computer interface (BCI) system. IEEE
# Transactions on Biomedical Engineering, 51(6), 1034-1043.
