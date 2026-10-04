Complexity and Microstate Methods
=================================

This page defines univariate regularity and scaling measures, followed by
scalp-map segmentation into recurring microstates. The univariate estimators
use waveform samples from :class:`~eegtable.Signal` or envelope samples from
:class:`~eegtable.BandSignal`, computed independently per channel and window.
ROI/global outputs average channel estimates; they do not estimate multivariate
complexity. Microstates instead use the spatial map across channels.

Signatures are in :doc:`/api/complexity`.

Sample and Multiscale Entropy
-----------------------------

Sample entropy is the negative log probability that two templates matching over
``order`` samples (:math:`m`) still match over :math:`m + 1` (Richman and
Moorman, 2000).

.. math::

   H(x, m, r) = -\log \frac{A}{B}

Definition
~~~~~~~~~~

- :math:`B` and :math:`A` count the pairs of templates, among the first
  :math:`N - m`, that match over :math:`m` and over :math:`m + 1` samples.
- Two templates match when their Chebyshev distance is strictly below
  :math:`r \sigma_x`.
- :math:`\sigma_x` is the population standard deviation of the finite samples.
- Each unordered pair is counted once, and self-matches are excluded.
- The log is natural.
- Cost grows with the square of the window length.

Only complete finite embeddings of length :math:`m+1` enter either count.
Changing the embedding dimension, tolerance, sampling rate, preprocessing, or
window length can change the estimate. Sample entropy describes recurrence
at those settings; it is not by itself a measure of physiological health or
information content. The `AntroPy sample-entropy documentation
<https://raphaelvallat.com/antropy/generated/antropy.sample_entropy.html>`__
provides a reference implementation; EEGTable implements its own pair counting
and gap handling.

Missing values
~~~~~~~~~~~~~~

- No match at length :math:`m` returns NaN.
- Matches at length :math:`m` and none at length :math:`m + 1` return infinity.
- Zero is the value for a perfectly regular series, so those two outcomes are
  not replaced by zero.

Zero tolerance
~~~~~~~~~~~~~~

A constant series has :math:`r\sigma_x = 0`. Under the strict distance
criterion, no pair matches, so entropy is undefined and returns NaN. The
tolerance is never replaced with an epsilon. A non-finite tolerance also
returns NaN. The log is not evaluated when either match count is zero.

Multiscale entropy
~~~~~~~~~~~~~~~~~~

``multiscale_entropy`` repeats the measure after coarse-graining by
non-overlapping block means, one column per scale (Costa, Goldberger, and Peng,
2002).

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - ``tolerance_mode``
     - Tolerance
   * - ``"original_sd"``
     - Keeps :math:`r \sigma_x` from the original series at every scale.
   * - ``"scale_sd"``
     - Recomputes :math:`\sigma` after coarse-graining. Stored as a separate
       estimator.

Missing samples:

- A block that contains a non-finite sample stays non-finite.
- A trailing incomplete block is dropped.
- An embedding template that crosses a gap is excluded.
- Missing samples are not deleted in a way that would make new neighbours.

A scale of :math:`s` averages blocks of :math:`s` original samples, so its
duration is :math:`s/f_s` seconds. Increasing scale reduces the number of
available templates; computationally valid coarse-grained estimates may still
be imprecise. Report the original sampling rate, scale range, and tolerance mode.

.. code-block:: python

   n_blocks = signal.size // scale
   blocks = signal[:n_blocks * scale].reshape(n_blocks, scale)
   coarse = np.where(np.isfinite(blocks).all(axis=1), blocks.mean(axis=1), np.nan)
   source = signal if tolerance_mode == "original_sd" else coarse
   tolerance = r * np.std(source[np.isfinite(source)])

Higuchi Fractal Dimension
-------------------------

``higuchi_fractal_dimension`` retraces the series at integer strides
:math:`k`, estimates the mean curve length :math:`L(k)`, and fits
:math:`L(k) \propto k^{-D}` (Higuchi, 1988). The quantity is geometric scale
dependence, a different estimator from sample entropy.

For :math:`N` samples and stride :math:`k`,

.. math::

   L_m(k) = \frac{N-1}{q_{\max} k^2} \sum_{q=0}^{q_{\max}-1} |x_{m+(q+1)k} - x_{m+qk}|,
   \qquad L(k) = \frac{1}{k}\sum_{m=0}^{k-1} L_m(k),

with :math:`x` indexed from 0 and :math:`q_{\max} = \lfloor (N - m - 1) / k \rfloor`.

**Fit**
   :math:`D` is the slope of :math:`\log L(k)` against :math:`-\log k`, computed
   as ``np.polyfit(-log(k), log(L), 1)`` over the strides with a positive
   :math:`L(k)`.

**Interpretation**
   Values near 1 are smooth curves. Larger values keep structure at finer
   scales. This finite-stride regression is not proof of fractal scaling and
   is not clipped to the ideal graph-dimension interval :math:`[1,2]`.
   Compare estimates at consistent sampling rates, window lengths, and
   ``k_max`` settings.

**Cost**
   For a fixed ``k_max`` the cost is linear in the number of samples.

**Missing values**
   A window returns NaN if:

   - any sample is non-finite;
   - it has fewer than ``2 * k_max`` samples, so not every starting offset has
     a difference at the largest stride;
   - fewer than two strides give a positive :math:`L(k)` (a flat window, for
     example).

Additional AntroPy Complexity Estimators
----------------------------------------

The following estimators use AntroPy rather than independent reimplementations.
Install ``eegtable[complexity]``. They accept broadband signals or band envelopes,
compute each channel/window independently, and preserve row identity, coverage,
and computation metadata. ROI and global columns average the channel estimates.
Non-finite samples raise errors; deleting gaps would change the sequence.

``permutation_entropy`` measures ordinal-pattern diversity for embeddings of
``order`` samples separated by ``delay`` samples. AntroPy's base-two entropy is
divided by :math:`\log_2(\mathrm{order}!)`, producing values from zero to one.
Tie ordering follows the backend. The window must support at least one complete
embedding. See `AntroPy permutation entropy
<https://raphaelvallat.com/antropy/generated/antropy.perm_entropy.html>`__.

``lempel_ziv_complexity`` explicitly binarizes each channel/window. Samples at
or above its selected ``median`` or ``mean`` become 1, and others become 0.
AntroPy counts newly encountered substrings and divides by
:math:`N/\log_2(N)`. EEGTable requires both symbols to occur; a one-symbol
sequence raises an error rather than entering AntroPy's observed-alphabet
normalization with an alphabet of size one.
Finite-length normalized values can exceed one. Threshold choice is stored
with the feature. See `AntroPy Lempel--Ziv complexity
<https://raphaelvallat.com/antropy/generated/antropy.lziv_complexity.html>`__.

``detrended_fluctuation`` reports ``dfa_exponent``. AntroPy cumulatively sums
the demeaned signal, linearly detrends nonoverlapping blocks, and estimates
scaling from log fluctuation versus log block size. Its block sizes start at
four samples, grow by 1.2, and reach ten percent of the window. At least 50
samples are required to support two distinct block sizes; constant windows
raise an error. Fifty samples is a computational minimum, not evidence of
reliable long-range scaling. The exponent is interpretable only across a
suitable scaling range. See `AntroPy DFA
<https://raphaelvallat.com/antropy/generated/antropy.detrended_fluctuation.html>`__.

Microstates
-----------

Templates are clustered from the scalp maps at local maxima of the global field
power (Lehmann, Ozaki, and Pal, 1987; Pascual-Marqui, Michel, and Lehmann,
1995).

.. math::

   \text{GFP}(t) = \sqrt{\frac{1}{N_{\text{channels}}} \sum_{c=1}^{N_{\text{channels}}} \left( V_c(t) - \bar{V}(t) \right)^2 }

:math:`\bar{V}(t)` is the average reference at that sample.

Clustering and assignment
~~~~~~~~~~~~~~~~~~~~~~~~~

Clustering is the polarity-invariant modified :math:`k`-means of Pascual-Marqui
et al. (1995). Peak maps retain their average-referenced amplitudes during
template fitting. Assignment uses absolute spatial correlation.

.. math::

   s(t) = \arg\max_k \frac{|V(t)^T \mu_k|}{\|V(t)\| \|\mu_k\|}

:math:`V(t)` is the average-referenced map, so the ratio is the spatial
correlation.

**Templates**
   Each template is the principal eigenvector of its members' scatter matrix,
   so negating a member does not change the template. The raw peak amplitudes
   give each map a weight proportional to GFP squared, matching the global
   explained-variance objective. Unit-normalizing members before this update
   would instead give every peak equal weight.
   An empty cluster raises an error because the data do not support that fit's
   requested number of states.

**Start**
   The start is scikit-learn :math:`k`-means run to convergence on the
   sign-normalized peak maps (k-means++ initialisation, best of ``n_init=20``
   restarts by inertia).

**Notes**
   Ordinary :math:`k`-means on sign-normalized maps is a different procedure.
   Orientation by the strongest channel jumps when two extrema have similar
   magnitude, and noise can then split one state's maps across clusters.

**Short segments**
   Segments shorter than ``min_duration_ms`` are absorbed into the longer
   neighbouring state, or split between the two neighbours on a tie.
   Minimum segment duration and peak separation are rounded up to whole samples.

Row normalization subtracts the channel mean, divides by the Euclidean norm,
and flips the sign so the largest-magnitude channel is positive. The sign flip
is a reporting convention. The modified :math:`k`-means objective does not use
it, but the Euclidean :math:`k`-means start is computed on sign-flipped maps and
so depends on it.

Fitting and labelling
~~~~~~~~~~~~~~~~~~~~~

**fit_on**
   ``microstates.segment`` accepts a boolean mask naming the trials whose
   maps may enter the fit. The default uses every trial. Templates learned
   from all rows describe that pool and include any rows subsequently scored
   as prediction targets.

**Frozen templates**
   :class:`~eegtable.MicrostateModel` separates fitting from assignment.
   ``MicrostateModel.fit(signal, rows=training_rows)`` accepts integer epoch
   indices and uses only those epochs for peak selection, initialization, and
   template updates. ``model.segment(other_signal)`` assigns new epochs with
   the frozen templates. The channel names and order must match exactly.
   Prediction requires fitting inside each training partition, including
   inner tuning partitions when applicable. See :doc:`/guides/learned_features`.

**Output shape**
   Assignment and the measures below are computed per epoch, so each returned
   table has one row per epoch.

**Labels**
   Cluster indices have no anatomical meaning. Templates that are not matched
   to a reference are labelled ``state1`` onward. Labels A--D require a
   one-to-one match to an identified reference set. ``MicrostateModel.from_templates``
   accepts an external template set, its channel order, state labels, and
   ``reference_name``. ``model.match_reference(reference)`` uses a Hungarian
   assignment to maximize total absolute spatial correlation and reorders
   the fitted states into reference order. It requires the same number of
   states and the same channels in the same order. A match establishes the
   chosen labeling correspondence; inspect the recorded correlations before
   interpreting the match. For prediction, the reference must be fixed
   independently of the held-out data.

**Comparability**
   Temporal features from two fits are comparable after that topographic match,
   and not before. The segmentation returns its templates and the global
   explained variance. Column identity includes the templates, the channel
   order, the rows that contributed, and the segmentation settings, so
   ``state1`` from two fits is two features.

**Missing values**
   If any sample is non-finite or spatially constant, ``segment`` raises
   ``ValueError`` for the whole input, so reject such data first.

**Relation to Pycrostates**
   The objective is the polarity-invariant modified :math:`k`-means described
   in the `Pycrostates ModKMeans API
   <https://pycrostates.readthedocs.io/en/stable/api/generated/pycrostates.cluster.ModKMeans.html>`__.
   The template update (the exact principal eigenvector), the stopping rule
   (labels unchanged, at most 300
   iterations), initialisation, GFP-peak selection, and short-segment smoothing
   are EEGTable choices, so templates need not match a Pycrostates fit.
   Fitting and assignment settings are recorded with the templates.

**Reference**
   Michel and Koenig (2018) review why GFP peaks, topographic correlation,
   polarity, and the temporal summaries are separate choices.

Temporal measures
~~~~~~~~~~~~~~~~~

From the sequence :math:`s(t)`:

.. list-table::
   :widths: 18 52 30
   :header-rows: 1

   * - Measure
     - Definition
     - Never-entered state
   * - **coverage**
     - Occupancy, :math:`n_T^{-1} \sum_t \mathbb{I}[s(t) = k]` over the
       :math:`n_T` samples in the window. Across states the values sum to 1.
     - 0
   * - **duration**
     - Mean dwell of a visit, in milliseconds.
     - NaN
   * - **occurrence**
     - Number of visits per second.
     - 0
   * - **transitions**
     - Probabilities between successive segments (below).
     - Row is NaN

.. math::

   P_{i \to j} = \frac{N_{i \to j}}{\sum_{l \ne i} N_{i \to l}} \quad (i \ne j)

Self-transitions are omitted. Each row sums to 1, or is NaN when the source
state is never left inside the window.

Global explained variance
~~~~~~~~~~~~~~~~~~~~~~~~~

Global explained variance is the GFP-weighted squared correlation with the
assigned template after smoothing, pooled over every epoch, including those
outside ``fit_on``. After smoothing, the assigned template is not always the
best-correlated one.

For ``MicrostateModel.segment``, this statistic describes the recording being
assigned. It is a descriptive fit statistic and does not select or validate
the number of states. State occupancies are compositional, and durations and
transition probabilities depend on the smoothing and window boundaries.

.. code-block:: python

   gfp = np.nanstd(epoch - np.nanmean(epoch, axis=0, keepdims=True), axis=0)
   correlations = np.abs(normalized_maps @ templates.T)
   assigned = correlations[np.arange(n_times), states]
   gev = np.sum(gfp ** 2 * assigned ** 2) / np.sum(gfp ** 2)

Implementation
~~~~~~~~~~~~~~

Coverage, duration, occurrence, and transitions are then computed as follows:

- **coverage**: ``mean(state == k)``.
- **duration**: mean run length divided by the sampling rate and times 1000. A
  run cut by the window edge counts as a visit.
- **occurrence**: run count divided by the window length in seconds.
- **transitions**: row-normalized counts of successive runs.

**Requirement**
   Template fitting requires scikit-learn (``pip install eegtable[microstates]``).
   Assignment with externally supplied or already fitted templates does not
   run clustering.

References
----------

* Higuchi, T. (1988). *Approach to an irregular time series on the basis of
  the fractal theory*. Physica D: Nonlinear Phenomena, 31(2), 277--283.
  `doi:10.1016/0167-2789(88)90081-4
  <https://doi.org/10.1016/0167-2789(88)90081-4>`__.
* Richman, J. S., & Moorman, J. R. (2000). *Physiological time-series analysis
  using approximate entropy and sample entropy*. American Journal of
  Physiology--Heart and Circulatory Physiology, 278(6), H2039--H2049.
  `doi:10.1152/ajpheart.2000.278.6.H2039
  <https://doi.org/10.1152/ajpheart.2000.278.6.H2039>`__.
* Costa, M., Goldberger, A. L., & Peng, C.-K. (2002). *Multiscale entropy
  analysis of complex physiologic time series*. Physical Review Letters, 89(6),
  068102. `doi:10.1103/PhysRevLett.89.068102
  <https://doi.org/10.1103/PhysRevLett.89.068102>`__.
* Lehmann, D., Ozaki, H., & Pal, I. (1987). *EEG alpha map series: Brain
  micro-states by space-oriented adaptive segmentation*. Electroencephalography
  and Clinical Neurophysiology, 67(3), 271--288.
  `doi:10.1016/0013-4694(87)90025-3
  <https://doi.org/10.1016/0013-4694(87)90025-3>`__.
* Pascual-Marqui, R. D., Michel, C. M., & Lehmann, D. (1995). *Segmentation of
  brain electrical activity into microstates: Model estimation and
  validation*. IEEE Transactions on Biomedical Engineering, 42(7), 658--665.
  `doi:10.1109/10.391164 <https://doi.org/10.1109/10.391164>`__.
* Michel, C. M., & Koenig, T. (2018). *EEG microstates as a tool for studying
  the temporal dynamics of whole-brain neuronal networks: A review*.
  NeuroImage, 180(B), 577--593.
  `doi:10.1016/j.neuroimage.2017.11.062
  <https://doi.org/10.1016/j.neuroimage.2017.11.062>`__.
