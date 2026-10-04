Dynamics Methods
================

This page defines baseline-relative band-power dynamics, envelope bursts, and
time-domain waveform summaries. Inputs are :class:`~eegtable.Signal` and
:class:`~eegtable.BandSignal`. Each result retains one row per epoch.

Signatures are in :doc:`/api/dynamics`.

Event-Related Desynchronization and Synchronization (ERDS)
----------------------------------------------------------

ERDS is the change in band-limited power relative to a baseline window
:math:`B`. It is reported per sample as a trace, then summarized inside the
analysis window.

.. math::

   \begin{aligned}
   B_\epsilon &= \max(B, \epsilon), \qquad P_\epsilon(t) = \max(P(t), \epsilon) \\[6pt]
   \text{ERDS}_{\%}(t) &= \frac{P(t) - B_\epsilon}{B_\epsilon} \cdot 100 \\[6pt]
   \text{ERDS}_{\text{dB}}(t) &= 10 \log_{10}\left(\frac{P_\epsilon(t)}{B_\epsilon}\right)
   \end{aligned}

Terms
~~~~~

- :math:`P(t)` is instantaneous power :math:`|z(t)|^2` from the Hilbert
  envelope.
- :math:`B` is mean baseline power.
- :math:`\epsilon` is :math:`10^{-12}` of the largest finite power of that
  epoch and channel. The floor is relative, so it does not depend on the
  recording units.
- The percent numerator is unfloored, so zero power is an exact
  :math:`-100\%` change.

Baseline checks
~~~~~~~~~~~~~~~

A baseline is rejected when any of these holds:

- it is non-finite;
- it is non-positive;
- it is no greater than :math:`10^{-6}` of that channel's mean power over the
  epoch.

The :math:`10^{-6}` guard is relative to the channel, so it does not depend on
the recording units.

**Missing values.** Rejected cells are NaN and carry ``baseline_degenerate``.

**Flags.** A cell whose peak power exceeds :math:`10^4` times its baseline is
reported and flagged ``baseline_extreme_ratio``.

Trace
~~~~~

.. code-block:: python

   power = np.where(np.isfinite(signal.power), signal.power, np.nan)
   baseline_power = np.nanmean(power[..., baseline_mask], axis=-1)
   baseline_power = np.where(degenerate, np.nan, baseline_power)
   floor = 1e-12 * np.nanmax(power, axis=-1)
   baseline_power = np.maximum(baseline_power, floor)
   percent = (power - baseline_power[..., None]) / baseline_power[..., None] * 100.0
   decibels = 10.0 * np.log10(np.maximum(power, floor[..., None]) / baseline_power[..., None])

Summaries
~~~~~~~~~

Summaries use the finite samples :math:`\{t_k\}` inside the analysis window.
Latency is reported in seconds on the input time axis. The slope unit is dB/s
or percent/s. Channel summaries are computed before ROI/global averaging, so
an ROI latency is a mean of channel latencies rather than a peak of an averaged
trace.

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Summary
     - Definition
   * - ``mean``
     - Mean of the trace, in percent or in decibels.
   * - ``slope``
     - Ordinary least-squares slope of the trace against time. Needs at least
       three finite samples.
   * - ``erd_magnitude``
     - Mean of :math:`|\text{ERDS}|` over samples with :math:`\text{ERDS} < 0`.
       The value is 0 when no sample is negative.
   * - ``erd_duration``
     - Number of negative samples divided by the sampling rate, in seconds.
   * - ``ers_magnitude``
     - Mean of :math:`\text{ERDS}` over samples with :math:`\text{ERDS} > 0`.
       The value is 0 when no sample is positive.
   * - ``ers_duration``
     - Number of positive samples divided by the sampling rate, in seconds.
   * - ``peak_latency``
     - Time of the maximum of :math:`|\text{ERDS}|`.
   * - ``onset_latency``
     - Start of the first run of samples on which
       :math:`|P(t) - B| > \sigma_B`. See below.
   * - ``rebound_latency``
     - Time of the maximum of :math:`\text{ERDS}` strictly after
       ``peak_latency``. That maximum need not be positive.

Onset rule
~~~~~~~~~~

- The run must last at least ``min_duration_cycles`` cycles of the band's low
  edge (default 6), or ``min_duration_ms`` when that is given.
  The required duration is rounded up to whole samples.
- :math:`\sigma_B` is the population standard deviation (divisor :math:`n`) of
  baseline power.
- A non-finite sample breaks the run.
- The test uses raw power, before percent or decibel conversion.
- A band beginning at zero requires an explicit ``min_duration_ms`` because
  its low edge cannot define a cycle duration.
- No qualifying run returns NaN. The persistence rule is a detection setting;
  it does not set a significance level or guarantee a false-onset rate.

Interpretation
~~~~~~~~~~~~~~

Negative and positive excursions occur during stationary activity. The
conditional ERD and ERS magnitudes average only one side of zero and therefore
can both be positive under a no-change condition. Durations need not divide
the window equally. For example, if instantaneous power is exponential and
the baseline equals its population mean, the expected fraction below baseline
is :math:`1-e^{-1}\approx0.632`. An estimated baseline, non-Gaussian activity,
and temporal dependence change this reference calculation.

The six-cycle onset default expresses persistence relative to the band's low
edge. Filter bandwidth and envelope autocorrelation affect its behavior;
the rule is not an inferential test. Compare conditions using an appropriate
experimental or statistical reference, and report the baseline, window
duration, scale, and onset criterion.

Scale
~~~~~

Every ``erds_*`` function reports decibels by default. Percent is available
with ``normalize="percent"``.

Percent change gives equal increments for equal additive power changes
relative to baseline. Decibels give equal increments for equal multiplicative
changes. Either scale remains sensitive to baseline estimation; the choice
changes the summary and its interpretation.

**Order of averaging.** Two decibel quantities in the library average in a
different order.

- The ``erds_*`` functions average the per-sample dB trace.
- :func:`~eegtable.mean_tfr_power` with a baseline and ``normalize="db"`` takes
  the decibel of the window-mean power.
- For the same positive power samples and baseline, Jensen's inequality makes
  the mean log power no greater than the log of mean power. An exponential
  population gives a gap of about 2.51 dB, before finite-sample effects.
- ``erds_*`` uses Hilbert band power, whereas ``mean_tfr_power`` uses Morlet
  power, so their difference also reflects the spectral estimator and support.

**Reference.** The ERD/ERS terms and the baseline-referenced interpretation
follow Pfurtscheller and Lopes da Silva (1999). The relative baseline guard,
the onset rule, and the rebound rule are EEGTable choices defined above. The
`MNE ERDS example
<https://mne.tools/stable/auto_examples/time_frequency/time_frequency_erds.html>`__
demonstrates baseline-relative time-frequency power and a separate statistical
analysis.

Oscillatory Bursts
------------------

A burst is a contiguous run of the band-limited amplitude envelope :math:`E(t)`
above a threshold :math:`\theta`.

Detection
~~~~~~~~~

1. **Threshold.** :math:`\theta` is an envelope quantile (default
   ``threshold=0.75``) or an array supplied by the caller.

   - The quantile is computed separately for each epoch and channel.
   - It is calibrated on the baseline when one is given, and on the union of
     analysis-window samples otherwise. Overlapping samples are counted once.
   - A sample is above threshold when :math:`E(t) > \theta`.
   - A non-finite sample is not above threshold, so it ends a run.

2. **Runs.** Contiguous runs are found by differencing. A run that touches the
   window edge is kept.
3. **Minimum duration.** Runs shorter than ``min_duration_ms`` (default
   100 ms, rounded up to whole samples, minimum one) are dropped.

.. code-block:: python

   threshold = np.nanquantile(envelope[..., calibration_mask], q, axis=-1)
   above = envelope > threshold[..., None]
   min_samples = max(1, int(np.ceil(min_duration_ms * sfreq / 1000.0)))
   runs = contiguous_true_runs(above)
   retained = [run for run in runs if len(run) >= min_samples]
   count = len(retained)
   rate = count / (n_times / sfreq)
   fraction_above = np.count_nonzero(above) / np.count_nonzero(np.isfinite(envelope))

The sample conversion allows one floating-point ULP of roundoff at exact
boundaries, so a 140 ms minimum at 100 Hz requires 14 samples, not 15.

An externally supplied threshold array is broadcast to
``(n_epochs, n_channels)`` and used in place of the quantile.

Summaries
~~~~~~~~~

Five summary outputs describe the retained runs and threshold exceedance.

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Summary
     - Definition
   * - ``count``
     - Number of retained bursts.
   * - ``rate``
     - ``count`` divided by the window length in seconds.
   * - ``duration_mean``
     - Mean duration of retained bursts, in seconds.
   * - ``amp_mean``
     - Mean of the peak envelope inside each retained burst.
   * - ``fraction_above``
     - Fraction of finite samples above threshold, computed before the
       duration filter. A missing sample is omitted from this fraction.

**Missing values.** With no retained burst:

- ``count`` and ``rate`` are 0.
- ``duration_mean`` and ``amp_mean`` are NaN.
- ``fraction_above`` is 0 when no sample exceeds :math:`\theta`.

A window with no finite sample, or a non-finite threshold, returns NaN for all
five summaries. Rates use the full sampled window duration, including gaps;
the exceedance fraction uses only finite samples. When thresholds are estimated
from the analysis windows, changes in amplitude also change the threshold.
Use a calibration interval appropriate to the intended comparison.

**Reference.** Whitten et al. (2011) and Hughes et al. (2012) detect
oscillations by fitting a 1/f background, applying a chi-square threshold to
power, and requiring a minimum number of cycles (BOSC). The detector here uses
the envelope quantile or the supplied threshold, and ``min_duration_ms``.

Time-Domain Measures
--------------------

These measures summarize one window: ``variance``, ``mean_amplitude``,
``peak_to_peak``, ``area_under_curve``, ``amplitude_quantile``,
``root_mean_square``, ``skewness``, ``kurtosis``, ``line_length``, and
``zero_crossing_rate``. ``hjorth_mobility`` and ``hjorth_complexity`` are
also included.

**Input.**

- On a :class:`~eegtable.Signal` the input is the waveform.
- On a :class:`~eegtable.BandSignal` the input is the envelope.
- Dimensional unit labels assume EEG amplitudes in volts. Variance uses V²;
  amplitude summaries use V; area uses V·s; line length uses V/s. Skewness,
  excess kurtosis, and Hjorth complexity are dimensionless.

**Missing values and coverage.**

- Non-finite samples are omitted and counted in ``coverage``.
- ``coverage`` is the window mean of the signal's per-sample coverage, which by
  default is the finite fraction.
- A large finite artifact remains in the summary unless it was rejected before
  extraction.

Simple reductions
~~~~~~~~~~~~~~~~~

The other reductions are the corresponding NumPy reductions with non-finite
samples omitted: ``nanvar``, ``nanmean``, ``nanmax - nanmin``,
``sqrt(nanmean(x**2))``, and ``nanquantile``. Hjorth activity is ``variance``.
Subha, Joseph, Acharya, and Lim (2010) review this family as EEG features.

Area under the curve
~~~~~~~~~~~~~~~~~~~~

``area_under_curve`` applies the trapezoid rule on each contiguous run of
finite samples and sums the runs. A gap contributes nothing. It is not bridged
by a straight line.

Skewness and kurtosis
~~~~~~~~~~~~~~~~~~~~~

- **Skewness** is SciPy's biased Fisher–Pearson :math:`g_1 = m_3 / m_2^{3/2}`.
- **Kurtosis** is Fisher excess :math:`g_2 = m_4 / m_2^2 - 3`.
- NaNs are omitted, and :math:`m_r` uses divisor :math:`n`.

**Missing values.**

- Skewness needs at least three finite samples.
- Kurtosis needs at least four.

**Reference.** Joanes and Gill (1998). See
`scipy.stats.skew <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.skew.html>`__
and
`scipy.stats.kurtosis <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kurtosis.html>`__.

Line length
~~~~~~~~~~~

``line_length`` is the mean absolute first difference times the sampling rate,
in the lineage of Esteller et al. (2001). It is a mean, not a cumulative sum.

.. code-block:: python

   clean = np.where(np.isfinite(signal), signal, np.nan)
   finite_differences = np.abs(np.diff(clean))
   finite_differences = np.where(np.isfinite(finite_differences),
                                 finite_differences, np.nan)
   line_length = np.nanmean(finite_differences, axis=-1) * sfreq

Zero-crossing rate
~~~~~~~~~~~~~~~~~~

``zero_crossing_rate`` counts sign changes between successive nonzero finite
samples within each contiguous finite segment and divides by the window length
in seconds (Rice, 1944, 1945). Zeros keep the previous sign. Missing samples
reset it, so a sign difference across a gap is not counted as a crossing.

Hjorth parameters
~~~~~~~~~~~~~~~~~

``hjorth_mobility`` and ``hjorth_complexity`` follow Hjorth (1970).

- Mobility uses the sample difference scaled by the sampling rate, so the
  derivative is per second.
- Mobility is divided by :math:`2\pi` and reported in hertz.
- This division changes conventional Hjorth mobility, whose unit is inverse
  seconds, to a frequency-equivalent scale. For a well-sampled sinusoid it
  approaches the sinusoid's frequency.
- Finite differences introduce a gain
  :math:`\sin(\pi f/f_s)/(\pi f/f_s)` relative to a continuous derivative.
  Sampling rate and signal bandwidth therefore still affect the estimate.
- Complexity is a ratio of mobilities, so unscaled differences suffice.
- Mobility needs two finite differences and positive signal variance;
  complexity also needs two finite second differences and positive
  first-difference variance. Constant or insufficient windows return NaN.

.. code-block:: python

   derivative = np.diff(signal) * sfreq
   mobility = np.sqrt(np.nanvar(derivative) / np.nanvar(signal)) / (2.0 * np.pi)
   first = np.diff(signal)
   second = np.diff(signal, n=2)
   complexity = np.sqrt(np.nanvar(second) * np.nanvar(signal)) / np.nanvar(first)

Peak Amplitude and Latency
--------------------------

``peak_amplitude`` is the signed extremum in the window, and ``peak_latency``
is its time.

**Polarity.**

- ``polarity="positive"`` searches the signal.
- ``"negative"`` searches its negation.
- ``"absolute"`` searches its magnitude.
- The returned amplitude is the original signed sample in every case.

**Naming.** The window name is not read as a component label. There is no
inference of N2, P300, or any other named component.

Without a prominence criterion, an edge extremum is allowed. Ties return the
first qualifying sample. Latency is quantized to the input time grid; no
temporal interpolation is performed.

**Prominence.** When ``prominence`` is set:

- Candidates are the local maxima returned by
  `scipy.signal.find_peaks <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html>`__
  at that prominence, and the most prominent is kept.
- An edge sample has no interior prominence, so it is never a candidate.
- Non-finite samples split the search into contiguous finite stretches; gaps
  cannot establish a peak's baseline.
- When no local maximum reaches the prominence, amplitude and latency are NaN.
  A monotonic trend has no qualifying interior peak.
- A tall interior sample with finite neighbours may qualify, however narrow.

**Reference.** Duncan et al. (2009) review windowed peak measures for ERP
components. Report the window, the polarity, and the prominence.

References
----------

* Pfurtscheller, G., & Lopes da Silva, F. H. (1999). *Event-related EEG/MEG
  synchronization and desynchronization: Basic principles*. Clinical
  Neurophysiology, 110(11), 1842--1857.
  `doi:10.1016/S1388-2457(99)00141-8
  <https://doi.org/10.1016/S1388-2457(99)00141-8>`__.
* Whitten, T. A., Hughes, A. M., Dickson, C. T., & Caplan, J. B. (2011). *A
  better oscillation detection method robustly extracts EEG rhythms across
  brain state changes: The human alpha rhythm as a test case*. NeuroImage, 54,
  860--874. `doi:10.1016/j.neuroimage.2010.08.064
  <https://doi.org/10.1016/j.neuroimage.2010.08.064>`__.
* Hughes, A. M., Whitten, T. A., Caplan, J. B., & Dickson, C. T. (2012). *BOSC:
  A better oscillation detection method, extracts both sustained and transient
  rhythms from rat hippocampal recordings*. Hippocampus, 22, 1417--1428.
  `doi:10.1002/hipo.20979 <https://doi.org/10.1002/hipo.20979>`__.
* Subha, D. P., Joseph, P. K., Acharya, U. R., & Lim, C. M. (2010). *EEG signal
  analysis: A survey*. Journal of Medical Systems, 34, 195--212.
  `doi:10.1007/s10916-008-9231-z
  <https://doi.org/10.1007/s10916-008-9231-z>`__.
* Joanes, D. N., & Gill, C. A. (1998). *Comparing measures of sample skewness
  and kurtosis*. The Statistician, 47, 183--189.
  `doi:10.1111/1467-9884.00122 <https://doi.org/10.1111/1467-9884.00122>`__.
* Esteller, R., Echauz, J., Tcheng, T., Litt, B., & Pless, B. (2001). *Line
  length: An efficient feature for seizure onset detection*. Proceedings of
  the 23rd Annual International Conference of the IEEE Engineering in Medicine
  and Biology Society, 1707--1710.
  `doi:10.1109/IEMBS.2001.1020545
  <https://doi.org/10.1109/IEMBS.2001.1020545>`__.
* Rice, S. O. (1944). *Mathematical analysis of random noise*. Bell System
  Technical Journal, 23(3), 282--332.
  `doi:10.1002/j.1538-7305.1944.tb00874.x
  <https://doi.org/10.1002/j.1538-7305.1944.tb00874.x>`__.
* Rice, S. O. (1945). *Mathematical analysis of random noise* (conclusion).
  Bell System Technical Journal, 24(1), 46--156.
  `doi:10.1002/j.1538-7305.1945.tb00453.x
  <https://doi.org/10.1002/j.1538-7305.1945.tb00453.x>`__.
* Hjorth, B. (1970). *EEG analysis based on time domain properties*.
  Electroencephalography and Clinical Neurophysiology, 29(3), 306--310.
  `doi:10.1016/0013-4694(70)90143-4
  <https://doi.org/10.1016/0013-4694(70)90143-4>`__.
* Duncan, C. C., Barry, R. J., Connolly, J. F., Fischer, C., Michie, P. T.,
  Näätänen, R., Polich, J., Reinvang, I., & Van Petten, C. (2009).
  *Event-related potentials in clinical research: Guidelines for eliciting,
  recording, and quantifying mismatch negativity, P300, and N400*. Clinical
  Neurophysiology, 120(11), 1883--1908.
  `doi:10.1016/j.clinph.2009.07.045
  <https://doi.org/10.1016/j.clinph.2009.07.045>`__.
