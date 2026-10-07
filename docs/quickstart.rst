Quick Start
===========

.. rst-class:: hero-lede

   Extract PSD power, Morlet power, and burst and ERD/ERS measures from
   preprocessed MNE epochs.

Input requirements
------------------

The examples share one EEG epochs file. Replace ``sub-01_epo.fif`` with your
preprocessed recording. For all three workflows, use epochs spanning at least
-2 to 2 seconds with sampling frequency above 90 Hz. EEG amplitudes must be in
volts, as in MNE. The file must contain the central channels listed below;
choose an ROI that matches your montage and scientific question.

.. code-block:: python

   import mne
   import eegtable as ef

   epochs = mne.read_epochs("sub-01_epo.fif", preload=True)
   epochs.pick("eeg", exclude="bads")
   recording = "sub-01"
   alpha = ef.Band("alpha", 8.0, 13.0)
   rois = {"central": ["C3", "Cz", "C4"]}

``recording`` must identify the recording uniquely within the cohort. The
wrappers preserve MNE's original epoch selection and event labels, which lets
:func:`~eegtable.concat` verify row alignment. See :doc:`concepts` for the data
model and :doc:`guides/preprocessing` for raw-to-epochs preparation.

PSD band power
--------------

:meth:`~eegtable.Spectra.welch` computes the spectra and records the settings
that produced them. One-second segments give 1 Hz frequency spacing; the
default overlap is half a segment. Shorter segments or a different estimator
change the estimate and should be chosen explicitly for the study.

.. code-block:: python

   spectra = ef.Spectra.welch(
       epochs,
       recording=recording,
       fmin=1.0,
       fmax=45.0,
       n_fft=round(epochs.info["sfreq"]),
   )
   power_table = ef.integrated_band_power(spectra, bands=[alpha], groups=rois)
   peak_table = ef.peak_frequency(spectra, band=alpha, groups=rois)
   spectral_features = ef.concat([power_table, peak_table])

Raw integrated PSD power is in V²; PSD density is in V²/Hz. Setting
``normalize="log10"`` produces log-scaled power. Peak frequency is in Hz and
returns ``NaN`` with a ``no_peak`` flag when no qualifying peak is found.

It is the estimator the batch runner uses, so the same settings name the same
columns in a notebook and in ``eegtable run``.
:meth:`~eegtable.Spectra.multitaper` is its multitaper counterpart. To wrap a
spectrum MNE has already computed, use :meth:`~eegtable.Spectra.from_spectrum`
and declare the settings MNE does not keep on ``Spectrum``, such as ``n_fft``. The
declaration is checked against the spectrum's frequency axis.

Morlet time-frequency power
---------------------------

:meth:`~eegtable.Spectra.morlet` computes unaveraged Morlet power and reduces
it to each window. EEGTable divides MNE's power by the original sampling
frequency and retains only coefficients whose complete wavelet support lies
inside each analysis window.

.. code-block:: python

   import numpy as np

   freqs = np.linspace(8.0, 30.0, num=23)
   windows = [
       ef.Window("baseline", -1.8, -0.2),
       ef.Window("stimulus", 0.2, 1.8),
   ]
   spectra_tfr = ef.Spectra.morlet(
       epochs, windows, recording=recording, freqs=freqs, n_cycles=4.0
   )
   tfr_power = ef.mean_tfr_power(
       spectra_tfr,
       bands=[alpha],
       groups=rois,
       baseline="baseline",
       normalize="db",
   )

These windows are long enough to retain alpha coefficients with four cycles.
The half-support is ``5 * n_cycles / (2 * pi * frequency)`` seconds: at 8 Hz,
about 0.40 seconds are excluded at each edge. Short windows may retain no
coefficients and raise an error. The fraction of each window a value rests on
is the table's ``support``, separate from numerical ``coverage``.

For a TFR computed in MNE, :meth:`~eegtable.Spectra.from_tfr` takes it with the
cycle count, the explicit ``zero_mean`` setting, and the sampling rate before
decimation, which MNE does not retain. The baseline is a window in the wrapped
representation; do not call MNE's ``apply_baseline`` first. These estimator
conventions follow MNE's
`time-frequency tutorial
<https://mne.tools/stable/auto_tutorials/time-freq/20_sensors_time_frequency.html>`_.
See :doc:`methods/spectral` for scaling and temporal attribution.

Bursts and ERD/ERS
------------------

``BandSignal.from_epochs`` filters each epoch and computes its analytic signal.
Padding reduces edge effects; it does not recover continuous data outside the
epoch. The baseline and analysis windows below are interior to the recording.

.. code-block:: python

   beta_signal = ef.BandSignal.from_epochs(
       epochs,
       band=ef.Band("beta", 13.0, 30.0),
       recording=recording,
       pad_sec=0.5,
   )
   baseline = ef.Window("baseline", -1.5, -0.2)
   stimulus = ef.Window("stimulus", 0.2, 1.5)
   burst_table = ef.burst_rate(
       [beta_signal],
       windows=[stimulus],
       baseline=baseline,
       groups=rois,
       threshold=0.75,
       min_duration_ms=100.0,
   )
   erds_table = ef.erds_mean(
       [beta_signal], baseline=baseline, windows=[stimulus], groups=rois
   )
   features = ef.concat([spectral_features, tfr_power, burst_table, erds_table])

``threshold=0.75`` is the baseline envelope quantile, not an absolute voltage
threshold. Burst rate is in bursts per second. ``erds_mean`` defaults to
decibels and averages the per-sample dB trace. This differs from taking dB after
averaging power, as in ``mean_tfr_power``. See :doc:`methods/dynamics` before
comparing those quantities or interpreting single-trial latencies.

Inspect and save
----------------

.. code-block:: python

   from eegtable.io import read_table, write_table

   frame = features.to_dataframe()
   if epochs.metadata is not None:
       frame = frame.join(epochs.metadata, on="epoch")
   long = features.to_long()
   paths = write_table(features, "sub-01_features.tsv", rows=epochs.metadata)
   restored = read_table("sub-01_features.tsv")

``to_dataframe`` indexes rows by recording, epoch and event. The epoch is the
epoch's original number, which ``epochs.metadata`` keeps too, so the join pairs
each row with its own trial even after MNE dropped epochs. ``to_long`` gives one
row per value with the column's measure, band, space, window, unit, coverage
and support beside it, the layout mixed models and plotting libraries read.

The writer creates values and coverage TSVs and a JSON sidecar with feature
metadata, row identities, flags, descriptor types, and payload checksums.
Coverage is finite-input availability, not an artifact score.

The same with a recipe
----------------------

A recipe states the bands, windows and measures once, and
:func:`~eegtable.extract` applies it to epochs in memory, computing exactly
what ``eegtable run`` would write for that recording:

.. code-block:: python

   recipe = {
       "windows": {"baseline": [-1.5, -0.2], "stimulus": [0.2, 1.5]},
       "features": [
           {"measure": "integrated_band_power", "bands": ["alpha"]},
           {"measure": "erds_mean", "bands": ["beta"], "baseline": "baseline"},
       ],
   }
   result = ef.extract(epochs, recipe, recording=recording)
   recipe_features = result.epochs

The keys are those of a TOML recipe, described in :doc:`guides/runner`.

Next steps
----------

- :doc:`auto_tutorials/index`: runnable analyses on public EEG and seeded
  simulations, with generated figures and interpretation.
- :doc:`guides/tables`: select features, preserve metadata, and stack recordings.
- :doc:`guides/runner`: apply a recipe consistently across a cohort.
- :doc:`guides/modeling`: choose grouping, tuning, and inference for a prediction task.
- :doc:`guides/validation`: inspect numerical and public-dataset evidence.
