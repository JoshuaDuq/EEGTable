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

Declare the Welch parameters once and pass them to both MNE and the wrapper.
One-second segments give approximately 1 Hz frequency spacing without zero
padding. Shorter segments or a different estimator change the estimate and
should be chosen explicitly for the study.

.. code-block:: python

   segment_samples = round(epochs.info["sfreq"])
   welch_parameters = {
       "fmin": 1.0,
       "fmax": 45.0,
       "n_fft": segment_samples,
       "n_per_seg": segment_samples,
       "n_overlap": segment_samples // 2,
       "window": "hamming",
       "average": "mean",
       "remove_dc": True,
   }
   spectrum = epochs.compute_psd(method="welch", **welch_parameters)
   spectra = ef.Spectra.from_spectrum(
       spectrum,
       recording=recording,
       estimator_parameters={"method": "welch", **welch_parameters},
   )
   power_table = ef.integrated_band_power(spectra, bands=[alpha], groups=rois)
   peak_table = ef.peak_frequency(spectra, band=alpha, groups=rois)
   spectral_features = ef.concat([power_table, peak_table])

Raw integrated PSD power is in V²; PSD density is in V²/Hz. Setting
``normalize="log10"`` produces log-scaled power. Peak frequency is in Hz and
returns ``NaN`` with a ``no_peak`` flag when no qualifying peak is found.

MNE does not retain every estimator argument on ``Spectrum``. The declaration
above preserves those settings in the column specification. For multitaper,
compute with ``normalization="full"`` and declare it in
``estimator_parameters``; the default length normalization is rejected.

Morlet time-frequency power
---------------------------

Use unaveraged, real-valued, uncorrected Morlet power. EEGTable divides MNE's
power by the original sampling frequency and retains only coefficients whose
complete wavelet support lies inside each analysis window.

.. code-block:: python

   import numpy as np

   freqs = np.linspace(8.0, 30.0, num=23)
   n_cycles = 4.0
   windows = [
       ef.Window("baseline", -1.8, -0.2),
       ef.Window("stimulus", 0.2, 1.8),
   ]
   tfr = epochs.compute_tfr(
       method="morlet",
       freqs=freqs,
       n_cycles=n_cycles,
       output="power",
       average=False,
       return_itc=False,
       zero_mean=True,
   )
   spectra_tfr = ef.Spectra.from_tfr(
       tfr,
       windows=windows,
       recording=recording,
       n_cycles=n_cycles,
       sfreq=epochs.info["sfreq"],
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
coefficients and raise an error. Inspect ``spectra_tfr.support`` separately
from numerical ``coverage``.

``sfreq`` is the sampling rate before any TFR decimation. The baseline is a
window in the wrapped representation; do not call MNE's ``apply_baseline``
first. These estimator conventions follow MNE's
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
   features = ef.concat([spectral_features, burst_table, erds_table])

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
   coverage = features.coverage
   flags = features.flags
   paths = write_table(features, "sub-01_features.tsv", rows=epochs.metadata)
   restored = read_table("sub-01_features.tsv")

The writer creates values and coverage TSVs and a JSON sidecar with feature
metadata, row identities, flags, descriptor types, and payload checksums.
``to_dataframe`` contains values; retain the table or saved bundle to keep the
associated evidence. Coverage is finite-input availability, not an artifact
score.

Next steps
----------

- :doc:`guides/tables`: select features, preserve metadata, and stack recordings.
- :doc:`guides/runner`: apply a recipe consistently across a cohort.
- :doc:`guides/modeling`: choose grouping, tuning, and inference for a prediction task.
- :doc:`guides/validation`: inspect numerical and public-dataset evidence.
