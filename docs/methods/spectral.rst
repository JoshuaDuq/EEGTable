Spectral Methods
================

This page defines power reductions, spectral descriptors, and periodic and
aperiodic estimates. Most functions read a :class:`~eegtable.Spectra` container;
IRASA reads broadband :class:`~eegtable.Signal` inputs. PSD and Morlet power
represent different spectral estimates, even when their reported units agree.

Signatures are in :doc:`/api/spectral`.

Spectral Power
--------------

Band power is the PSD integrated over the band limits. The mean forms divide
that integral by the bandwidth, or average Morlet power over the band.

Integrated Band Power
~~~~~~~~~~~~~~~~~~~~~

``integrated_band_power`` integrates a PSD over the band limits.

.. math::

   P_B = \int_{f_{\min}}^{f_{\max}} S(f)\,df

- **Quadrature**: piecewise linear, with interpolated contributions at both
  boundaries. Both linear and logarithmic grids integrate the requested
  interval; their numerical estimates can differ for curved spectra.
- **Units**: for an EEG PSD in V²/Hz the integral is in V².
- **Missing values**: NaN if any weighted bin is non-finite.

Mean PSD and Mean TFR Power
~~~~~~~~~~~~~~~~~~~~~~~~~~~

With complete data, ``mean_psd`` divides the integral above by the bandwidth
and stays in V²/Hz. ``mean_tfr_power`` applies the same frequency weights to
the window-reduced Morlet power, also in V²/Hz.

- **Units**: V²/Hz for both. After the sampling-rate division described below,
  each ``mean_tfr_power`` value is a density in V²/Hz, smoothed over the wavelet
  bandwidth.
- **Averaging**: ``mean_tfr_power`` averages within the band rather than
  integrating, so the result is comparable to ``mean_psd``. Integrating it would
  give wavelet-smoothed band power in V².
- **Missing values**: both omit non-finite bins and renormalize the remaining
  weights. With gaps, the result is a mean over the retained contributions,
  rather than the full-band integral divided by its nominal width.
- **Representation**: :class:`~eegtable.Spectra` records which representation it
  holds. Each function rejects the other representation.

Morlet Scaling
~~~~~~~~~~~~~~

MNE scales its Morlet wavelets to energy 2 (norm :math:`\sqrt{2}`). For
stationary signals and predominantly positive-frequency wavelets, dividing
coefficient power by the original sampling rate gives a wavelet-smoothed
estimate on the scale of a one-sided density.

- **Correction**: :meth:`~eegtable.Spectra.from_tfr` divides that factor out and
  therefore needs the sampling rate the TFR was computed at.
- **Decimated TFR**: it reports only the decimated rate.
- **Scaling**: raw coefficient power scales approximately with the sampling
  rate for fixed physical wavelet parameters. Discrete sampling, low cycle
  counts, and frequency smoothing affect agreement with a PSD.
- **Backend**: the wavelet definition and normalization are documented in the
  `MNE Morlet API
  <https://mne.tools/stable/generated/mne.time_frequency.morlet.html>`__.

The default time reduction is the arithmetic mean. ``statistic="median"``
instead divides the temporal median by :math:`\ln 2`, which calibrates it to
the mean of an exponential power distribution. This calibration does not hold
for every signal: a constant-power oscillation is multiplied by
:math:`1/\ln 2`. Report the reduction when comparing power estimates.

Integration Weights
~~~~~~~~~~~~~~~~~~~

Weights are the piecewise-linear weights on the closed interval
:math:`[f_{\min}, f_{\max}]`. The grid points on either side of each bound carry
interpolated weight.

Sources
~~~~~~~

These functions summarize an estimate MNE has already computed.

- **Welch**: the short-segment modified periodogram of Welch (1967).
- **Multitaper**: Thomson (1982).
- **Morlet**: Morlet, Arens, Fourgeau, and Giard (1982).

Normalization
-------------

Normalization is applied per channel, then channels are aggregated. An ROI
value is the mean of the per-channel normalized values.

Here :math:`P` is the power in the analysis window and :math:`B` is the same
measure computed in the named baseline window.

.. list-table::
   :header-rows: 1

   * - Option
     - Definition
   * - ``log10``
     - :math:`\log_{10}(\max(P, \epsilon))`
   * - ``log_ratio``
     - :math:`\log_{10}\left(\frac{\max(P, \epsilon)}{\max(B, \epsilon)}\right)`
   * - ``db``
     - :math:`10 \log_{10}\left(\frac{\max(P, \epsilon)}{\max(B, \epsilon)}\right)`
   * - ``percent``
     - :math:`\frac{P - \max(B, \epsilon)}{\max(B, \epsilon)} \cdot 100`

Power Floor
~~~~~~~~~~~

:math:`\epsilon` is :math:`10^{-12}` of the largest finite power of that epoch
and channel, over every window and the baseline.

- **Amplitude scaling**: multiplying input power by a positive constant scales
  the floor by the same constant. Baseline ratios are therefore unchanged;
  absolute ``log10`` power still depends on the amplitude unit.
- **Missing values**: an epoch and channel with no positive power has no floor,
  and every value is NaN.
- **Logarithmic forms**: the numerator and the denominator are floored.
- **Percent**: only the denominator is floored, so zero power is an exact
  :math:`-100\%` change.

Interpretation
~~~~~~~~~~~~~~

``percent`` is ERD/ERS% in the sense of Pfurtscheller and Lopes da Silva
(1999), where a negative value is desynchronization. ``log_ratio`` and ``db``
are logarithmic forms of the same baseline ratio.

Coverage and Metadata
~~~~~~~~~~~~~~~~~~~~~

For baseline-normalized power:

- **Coverage**: the minimum of the analysis-window and baseline coverage on that
  channel.
- **Flags**: flags from either window are copied to the result.
- **Metadata**: the baseline name and its time bounds are stored in the column
  metadata.

Wavelet Support Restriction
---------------------------

Only Morlet coefficients whose full wavelet support lies inside the analysis
window and the available TFR time range enter the window mean. Requested bounds
outside that range, including infinite whole-segment bounds, are intersected
with the available range before restricting wavelet support. Keep any padding
in the TFR passed to ``Spectra.from_tfr``; cropped-away padding cannot establish
support for the remaining coefficients.

A Morlet wavelet at frequency :math:`f` with :math:`n_{\text{cycles}}` cycles
has temporal half-support

.. math::

   \tau(f) = \frac{5 n_{\text{cycles}}}{2 \pi f}

The coefficient at time :math:`t` uses samples in
:math:`[t - \tau(f), t + \tau(f)]`. For the intersected window
:math:`[t_{\min}, t_{\max}]`, its support must satisfy

.. math::

   t_{\min} + \tau(f) \le t \le t_{\max} - \tau(f)

Coefficients outside this interval are excluded before the window mean.

**Missing values**

- If :math:`\tau(f) > (t_{\max} - t_{\min}) / 2`, the frequency has no usable
  coefficient in the window and the result is NaN.
- If no frequency on the TFR axis retains a coefficient for a requested
  window, ``Spectra.from_tfr`` raises ``ValueError``. A later band reduction
  can return NaN when only that band's frequencies have no usable coefficients.

``Spectra.support`` records the fraction of requested time points that meet
the support restriction. ``coverage`` records numerical finiteness among the
supported coefficients. These quantities describe different limitations.

Peak Frequency
--------------

``peak_frequency`` returns the strongest qualifying local maximum within a
band, after aperiodic adjustment and smoothing. Setting ``min_prominence=0``
selects the bare in-band argmax instead.

Aperiodic Adjustment
~~~~~~~~~~~~~~~~~~~~

The search runs on :math:`P(f) / P_{\text{ap}}(f)`, where a pure power law is
flat at one. On a steep spectrum an unadjusted maximum can be driven by the
low-frequency background instead of an oscillatory peak.

- **Fit range**: ``fit_range``, default
  :math:`(\min(2, f_{\min}), \max(40, f_{\max}))`, which extends outside the
  analysis band. A narrow fit range can poorly constrain the background.
- **Measure name**: with this adjustment the measure name is
  ``peak_freq_adjusted``.
- **Fit**: the robust log-log line in `Aperiodic Fit`_. It is a straight line.
  FOOOF also estimates a knee and uses a different peak model.
- **Reference**: the split between aperiodic background and oscillatory peaks
  follows Donoghue et al. (2020).

Smoothing
~~~~~~~~~

Each frequency is replaced by the piecewise-linear mean over a centred window of
total width ``smoothing_hz`` (:math:`\pm` ``smoothing_hz``/2), truncated at the
band edges.

- **Short bands**: bands of three bins or fewer are not smoothed.
- **Grid**: distances are in hertz on both linear and logarithmic grids.
- **Missing values**: non-finite bins are left out of the mean. The gap is not
  filled.

Prominence Guard
~~~~~~~~~~~~~~~~

SciPy's ``find_peaks`` selects local maxima whose topographic prominence in
:math:`\log_{10}` power reaches ``min_prominence``. Prominence is the height
above the higher of the two surrounding bases, rather than height above the
band median. The strongest qualifying peak is retained.

- **Power used**: prominence is evaluated in :math:`\log_{10}` of the
  smoothed, optionally aperiodic-adjusted power. Among qualifying peaks, the
  largest linear-power value is retained. Interpolation uses that linear
  power, rather than its logarithm.
- **Missing values**: the search runs separately on contiguous finite spans;
  missing bins cannot provide a peak's bases. Smoothing preserves missing bins.
- **No peak**: when none qualifies, the frequency is NaN and ``no_peak`` is set.
  A zero-power spectrum has no peak even with the guard disabled. A centroid is
  available separately from ``spectral_centroid``.
- **Reference**: `SciPy peak prominence
  <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.peak_prominences.html>`_
  defines this estimator. Reporting NaN for an absent oscillation also follows
  `FOOOF band-peak extraction
  <https://fooof-tools.github.io/fooof/generated/fooof.analysis.get_band_peak_fg.html>`_.

Parabolic Interpolation
~~~~~~~~~~~~~~~~~~~~~~~

The retained maximum is refined by parabolic interpolation through the peak bin
and its two neighbours, using their actual frequency coordinates. Let
:math:`h_L=f_k-f_{k-1}` and :math:`h_R=f_{k+1}-f_k`.

.. math::

   \begin{aligned}
   s_L &= \frac{P(f_k)-P(f_{k-1})}{h_L}, \qquad
   s_R = \frac{P(f_{k+1})-P(f_k)}{h_R} \\[6pt]
   a &= \frac{s_R-s_L}{h_L+h_R}, \qquad b = s_L + a h_L \\[6pt]
   \delta &= -\frac{b}{2a}, \qquad f_{\text{peak}} = f_k + \delta
   \end{aligned}

:math:`k` is the retained peak bin.

- **Minimum bins**: interpolation needs at least three bins, which is also the
  requirement for an interior maximum. Narrower bands raise.
- **Disabling**: ``interpolate=False`` returns the bin frequency.
- **Grid**: the parabola uses actual bin coordinates on uniform or non-uniform
  grids. Its vertex refines the grid estimate; it does not establish the
  accuracy of a physiological peak frequency.
- **Degenerate cases**: a zero denominator, or a non-finite :math:`\delta`, is
  treated as :math:`\delta = 0`.
- **Clipping**: :math:`\delta` is clipped to :math:`[-h_L/2, h_R/2]` in Hz.
- **Flags**: with the prominence guard disabled, a maximum on the first or last
  bin is not interpolated and ``edge_hit`` is set.
- **Stored resolution**: every column also stores ``freq_resolution_hz``, the
  median in-band bin spacing.

Spectral Centroid and Bandwidth
-------------------------------

The spectral centroid is the centre of mass of power in the band.
These descriptors select bins with :math:`f_{\min}\le f<f_{\max}`; they do
not interpolate contributions at the numerical band boundaries. At least
three in-band bins are required.

.. math::

   f_c = \frac{\sum_i f_i P(f_i) \Delta f_i}{\sum_i P(f_i) \Delta f_i}

Bandwidth is the mass-weighted standard deviation about that centroid.

.. math::

   \text{BW} = \sqrt{\frac{\sum_i (f_i - f_c)^2 P(f_i) \Delta f_i}{\sum_i P(f_i) \Delta f_i}}

- **Bin width**: :math:`\Delta f_i` is from :func:`numpy.gradient`, a central
  difference in the interior and one-sided at the band edges.
- **Why weight**: weighting by :math:`\Delta f_i` integrates a density. Unequal
  bins are not treated as equally probable samples.
- **Reference**: the two quantities are the spectral centroid and spread (the
  first moment and the square root of the second central moment) in Peeters
  (2004), adapted here to a PSD.
- **Missing values**: non-finite bins contribute no mass. A zero total mass
  returns NaN. Comparisons with different patterns of missing bins can describe
  different retained spectral shapes.

Spectral Edge Frequency
-----------------------

Spectral edge frequency is the lowest grid frequency at which the cumulative
band power reaches a fraction :math:`\alpha`. The default is 0.95.

.. math::

   \frac{\sum_{i=0}^{k} P(f_i) \Delta f_i}{\sum_{i} P(f_i) \Delta f_i} \ge \alpha

- **Interpolation**: none between bins.
- **Rounding**: if floating-point rounding means no bin reaches :math:`\alpha`,
  the last bin is returned.
- **Argument**: the quantile is the ``percentile`` argument.
- **Reference**: Szeto (1990) used this quantile for electrocortical maturation,
  with :math:`\alpha = 0.90`.

Spectral Entropy
----------------

Spectral entropy is the normalized Shannon entropy of the in-band power
distribution, in :math:`[0, 1]`.

.. math::

   \begin{aligned}
   p_i &= \frac{P(f_i)}{\sum_j P(f_j)} \\[6pt]
   H &= -\frac{\sum_i p_i \ln(p_i)}{\ln(N)}
   \end{aligned}

:math:`N` is the number of bins.

- **Range**: :math:`H = 1` is uniform power across the band. :math:`H = 0` is
  power in a single bin.
- **Zero terms**: the undefined term :math:`0 \ln 0` is omitted.
- **Missing bins**: non-finite bins contribute zero mass, but :math:`N` still
  counts all selected bins. Missingness can therefore change the normalized
  entropy. No positive total mass returns NaN.
- **Grid**: the definition uses one count per bin and requires an approximately
  uniform frequency grid. A non-uniform grid is rejected. Interpolate onto a
  uniform grid in hertz first.
- **Comparability**: because the denominator is :math:`\ln(N)`, values from bands
  with different bin counts are not comparable.
- **Reference**: the entropy of the power-spectrum proportions in Inouye et al.
  (1991).

Aperiodic Fit
-------------

The aperiodic background is a line in log-log coordinates.

.. math::

   \log_{10} P(f) = \text{offset} + \text{slope} \cdot \log_{10} f

Robust Fitting
~~~~~~~~~~~~~~

An initial least-squares line is fit on ``fit_range``. Residuals are
:math:`r(f) = \log_{10} P(f) - (\text{offset} + \text{slope} \log_{10} f)`.
Points with :math:`r(f) > z \cdot \mathrm{MAD}(r)` are removed and the line is
refit, up to ``max_iterations`` times.

- **Threshold**: the default is :math:`z = 2.5`. The threshold is
  :math:`z \cdot 1.4826\,\mathrm{MAD}(r)`, the normal-consistent MAD.
- **Median**: the residual median enters the MAD but is not subtracted from
  :math:`r` in the comparison.
- **One-sided**: only positive residuals are removed. Peaks above the
  background can bias the fitted slope; the direction depends on their
  position and shape.
- **Input**: only positive finite power enters the fit.
- **Minimum points**: fewer than five usable positive-frequency bins returns
  NaN parameters. A fit range selecting fewer than five axis bins raises.
- **Weighting**: each retained frequency bin has equal regression weight.
  Changing grid density can therefore change the fitted line.
- **Passband**: ``fit_range`` must stay within the recording's reported filter
  bounds. Including suppressed frequencies biases the fitted background, as
  illustrated in `FOOOF's filtering example
  <https://fooof-tools.github.io/fooof/auto_examples/processing/plot_line_noise.html>`_.
- **Stopping**: the loop stops when any of these holds.

  - MAD is non-finite or below :math:`10^{-12}`.
  - Fewer than 5 points remain.
  - The mask stops changing.

Goodness of Fit
~~~~~~~~~~~~~~~

The column ``r_squared`` is the coefficient of determination on the points that
survived rejection. Rejected peaks are left out of it. It describes the retained
line fit, rather than the whole spectrum or the validity of the decomposition.
Constant retained log power has undefined ``r_squared`` and returns NaN even
when slope and offset can be estimated.

- **Why exclude peaks**: an alpha peak is not a failure of the line, and scoring
  the line against that peak is low on ordinary spectra.
- **Sensitivity**: the statistic does respond to a knee inside ``fit_range``, and
  a straight line through a knee biases the exponent.
- **Reference**: Donoghue et al. (2020) also report goodness of fit, though for
  their full model including peaks.

Adjusted Power
~~~~~~~~~~~~~~

On :math:`f > 0` the fitted power is
:math:`10^{\text{offset} + \text{slope} \log_{10} f}`, and aperiodic-adjusted
power is the ratio of the observed power to that curve.

- **Missing values**: if the fit fails, ``aperiodic_ratio`` returns NaN for that
  cell and sets ``aperiodic_fit_failed``. The unadjusted spectrum is not
  relabelled as adjusted.
- **Units**: the ratio is dimensionless and records
  ``representation="aperiodic_ratio"``. Physical PSD and TFR power reducers
  reject this representation. The DC bin is undefined and has zero coverage.

Periodic Power
~~~~~~~~~~~~~~

``periodic_power`` is the frequency-weighted mean of that ratio over a band,
weighted as ``mean_psd`` weighs power.

- **Reference value**: a pure power law gives 1 in every band.
- **Broadband changes**: multiplication by a power-law gain that the line can
  follow is absorbed by the fit. Real artifacts need not follow this model,
  so the ratio does not provide artifact correction.
- **Contrast with band power**: band power sums the two, so a broadband rise can
  hide an oscillatory fall.
- **Units**: the ratio is dimensionless. Identical spectral shapes differing
  only by a positive scale give the same ratio; PSD and time-frequency
  estimates of one recording can differ because of smoothing and time reduction.
- **Normalization**: the normalizations are those above, with :math:`B` the
  periodic power of the baseline window, itself divided by that window's own fit.
- **Missing values**: a cell whose fit fails is NaN and carries
  ``aperiodic_fit_failed``.

Relation to FOOOF
~~~~~~~~~~~~~~~~~

The line is the model above. FOOOF (Donoghue et al., 2020) adds a knee parameter
and a different peak parameterization. Matching those settings is a separate
analysis.

Full Spectral Parameterization
------------------------------

``spectral_parameterization`` delegates Gaussian peak fitting and aperiodic
parameterization to the explicit ``specparam==2.0.0rc7`` API. It leaves the
existing robust straight-line ``aperiodic`` estimator unchanged.

The fixed model has :math:`A(f)=b-\chi\log_{10}(f)`; the knee model has
:math:`A(f)=b-\log_{10}(k+f^\chi)`. Offset :math:`b`, positive exponent
:math:`\chi`, and, for the knee model, denominator parameter :math:`k` are
reported separately. The knee parameter is not a frequency in Hz.

Each requested half-open band selects the fitted peak with greatest height
whose center falls in that band. Outputs are center frequency in Hz, height
above the aperiodic component in log10 power, and bandwidth equal to twice the
Gaussian standard deviation. Missing peaks produce NaN and
``spectral_no_peak``; a band with no peak is distinct from a failed model fit.

Fit diagnostics report peak count, squared Pearson correlation between fitted
and observed log spectra, and mean absolute error in log10 power. Every output,
including band peaks, inherits coverage and input flags from the entire fitted
range. Groups average channel parameters after individual fits.

Input must be finite, strictly positive, linear PSD on a uniformly spaced
frequency axis spanning the full requested fit range. Fitting uses the existing
half-open frequency convention. The fit range must be inside the recording
passband. Logged data, aperiodic ratios, and time-frequency power are unsuitable.
Smooth Welch or multitaper PSD estimates are preferable to raw FFT power.
``debug=True`` ensures backend fitting errors surface.

The number and width of fitted peaks depend on spectral resolution, the fit
range, and peak thresholds. Inspect fits before interpreting individual
parameters. A high fit score alone does not establish that every Gaussian
represents a distinct physiological oscillation.

See the `official SpectralModel API
<https://specparam-tools.github.io/generated/specparam.SpectralModel.html>`__
for the supported backend and its fitting settings. Install
``eegtable[spectral-model]``.

IRASA
------

``irasa`` delegates decomposition to `NeuroDSP compute_irasa
<https://neurodsp-tools.github.io/neurodsp/generated/neurodsp.aperiodic.irasa.compute_irasa.html>`__.
Irregular resampling by paired factors :math:`h` and :math:`1/h`, geometric
means, and a median across factors estimate the aperiodic PSD. The periodic
component is the signed difference between the original and aperiodic PSD.
Negative residuals are retained; no clipping or periodic threshold is applied.

The default resampling factors are 1.10--1.90 in steps of 0.05. Factors are
rounded to four decimals as in NeuroDSP and recorded in provenance. The
factors must remain strictly between 1 and 2 after rounding. Welch
spectra use a Hann window, arithmetic averaging, and 50 percent overlap, with
``segment_seconds`` converted to the nearest integer sample count. The fit
range is half-open. NeuroDSP's ``fit_irasa`` fits offset and slope in log-log
space; a typical 1/f slope is negative. These are separate estimators from
specparam's positive exponent.

Both component band powers integrate the piecewise-linear PSD between exact
band boundaries and therefore sum to the original band power. The adjacent
PSD bins bracketing non-grid-aligned boundaries are retained for integration;
the slope fit still uses only bins in the requested half-open fit range. The original
amplitude must be in volts for the reported V-squared unit.

Finite broadband ``Signal`` inputs are required. Each analysis window must
retain a complete Welch segment after the largest downsampling factor.
For the outward bracketing-bin bounds, the expanded frequency support
:math:`[f_\mathrm{min}/h_\mathrm{max}, f_\mathrm{max}h_\mathrm{max}]`
must fit inside the recording passband and below Nyquist. Constant traces and
non-finite spectra raise errors. Coverage describes input samples in the
window; it is not a measure of decomposition quality.

IRASA assumes a component that remains approximately scale-free under the
chosen resampling factors. Filter transitions, broad overlapping peaks, and
limited frequency support can affect the separation. Its signed periodic
integral can be negative and is not the dimensionless ratio returned by
``periodic_power``.

The `official IRASA tutorial
<https://neurodsp-tools.github.io/neurodsp/auto_tutorials/aperiodic/plot_IRASA.html>`__
explains decomposition and subsequent fits. Install ``eegtable[irasa]``.

Band Ratio and Asymmetry
------------------------

Ratios and asymmetry are computed from a band-power table. The arithmetic
follows the normalization already stored on that table.

Band Ratio
~~~~~~~~~~

For raw power the ratio of band :math:`A` to band :math:`B` is

.. math::

   R_{A/B} = \frac{P_A}{P_B}

Raw input is divided as stored. Percent input is refused, because a percent
change is a signed deviation and a quotient of two of them is not a power ratio.

For ``log10``, ``log_ratio``, and ``db`` the stored values are subtracted.

.. math::

   R_{A/B} = P_A^{(\mathrm{log})} - P_B^{(\mathrm{log})}

- ``log10`` input gives :math:`\log_{10}(P_A / P_B)`.
- ``log_ratio`` input gives :math:`\log_{10}((P_A / B_A) / (P_B / B_B))`.
- ``db`` input is ten times that baseline-relative difference.

Asymmetry
~~~~~~~~~

For a homologous pair :math:`(L, R)`, raw asymmetry is

.. math::

   A_{L, R} = \frac{P_R - P_L}{P_R + P_L}

Percent input is refused. Logarithmic input subtracts the stored values.

.. math::

   A_{L, R} = P_R^{(\mathrm{log})} - P_L^{(\mathrm{log})}

- ``log10`` input gives :math:`\log_{10}(P_R / P_L)`.
- ``log_ratio`` input gives :math:`\log_{10}((P_R / B_R) / (P_L / B_L))`.
- ``db`` input is ten times that difference, in dB.
- **Missing values**: a zero denominator or a zero sum returns NaN.
- **Reference**: right-minus-left asymmetry follows the frontal-asymmetry
  convention of Davidson, Chapman, Chapman, and Henriques (1990).

Output Units
~~~~~~~~~~~~

The output unit names the scale of that result. Log-ratio asymmetry is reported
in the unit of its input, from the table below.

.. list-table::
   :header-rows: 1

   * - Input normalization
     - Band ratio
     - Asymmetry
   * - ``"raw"``
     - ``ratio``
     - ``a.u.``
   * - ``"log10"``
     - ``log10 ratio``
     - ``log10 ratio``
   * - ``"log_ratio"``
     - ``log10 ratio``
     - ``log10 ratio``
   * - ``"db"``
     - ``dB``
     - ``dB``

References
----------

* Welch, P. D. (1967). *The use of fast Fourier transform for the estimation of
  power spectra: A method based on time averaging over short, modified
  periodograms*. IEEE Transactions on Audio and Electroacoustics, 15(2),
  70--73. `doi:10.1109/TAU.1967.1161901
  <https://doi.org/10.1109/TAU.1967.1161901>`__.
* Thomson, D. J. (1982). *Spectrum estimation and harmonic analysis*.
  Proceedings of the IEEE, 70(9), 1055--1096.
  `doi:10.1109/PROC.1982.12433 <https://doi.org/10.1109/PROC.1982.12433>`__.
* Morlet, J., Arens, G., Fourgeau, E., & Giard, D. (1982). *Wave propagation
  and sampling theory; Part I, Complex signal and scattering in multilayered
  media*. Geophysics, 47(2), 203--221. `doi:10.1190/1.1441328
  <https://doi.org/10.1190/1.1441328>`__.
* Donoghue, T., Haller, M., Peterson, E. J., Varma, P., Sebastian, P., Gao, R.,
  Noto, T., Lara, A. H., Wallis, J. D., Knight, R. T., Shestyuk, A., & Voytek,
  B. (2020). *Parameterizing neural power spectra into periodic and aperiodic
  components*. Nature Neuroscience, 23, 1655--1665.
  `doi:10.1038/s41593-020-00744-x
  <https://doi.org/10.1038/s41593-020-00744-x>`__.
* Wen, H., & Liu, Z. (2016). *Separating fractal and oscillatory components in
  the power spectrum of neurophysiological signal*. Brain Topography, 29,
  13--26. `doi:10.1007/s10548-015-0448-0
  <https://doi.org/10.1007/s10548-015-0448-0>`__.
* Peeters, G. (2004). *A large set of audio features for sound description
  (similarity and classification) in the CUIDADO project*. IRCAM technical
  report. `Report PDF
  <https://recherche.ircam.fr/anasyn/peeters/ARTICLES/Peeters_2003_cuidadoaudiofeatures.pdf>`__.
* Szeto, H. H. (1990). *Spectral edge frequency as a simple quantitative
  measure of the maturation of electrocortical activity*. Pediatric Research,
  27, 289--292. `doi:10.1203/00006450-199003000-00018
  <https://doi.org/10.1203/00006450-199003000-00018>`__.
* Inouye, T., Shinosaki, K., Sakamoto, H., Toi, S., Ukai, S., Iyama, A.,
  Katsuda, Y., & Hirano, M. (1991). *Quantification of EEG irregularity by use
  of the entropy of the power spectrum*. Electroencephalography and Clinical
  Neurophysiology, 79(3), 204--210.
  `doi:10.1016/0013-4694(91)90138-T
  <https://doi.org/10.1016/0013-4694(91)90138-T>`__.
* Pfurtscheller, G., & Lopes da Silva, F. H. (1999). *Event-related EEG/MEG
  synchronization and desynchronization: Basic principles*. Clinical
  Neurophysiology, 110(11), 1842--1857.
  `doi:10.1016/S1388-2457(99)00141-8
  <https://doi.org/10.1016/S1388-2457(99)00141-8>`__.
* Davidson, R. J., Chapman, J. P., Chapman, L. J., & Henriques, J. B. (1990).
  *Asymmetrical brain electrical activity discriminates between
  psychometrically-matched verbal and spatial cognitive tasks*. Psychophysiology,
  27, 528--543. `doi:10.1111/j.1469-8986.1990.tb01970.x
  <https://doi.org/10.1111/j.1469-8986.1990.tb01970.x>`__.
