Phase and Connectivity Methods
==============================

This page defines phase consistency, phase-amplitude coupling, sensor/ROI
connectivity, common spatial patterns, and graph summaries. ITPC, PPC, envelope
correlation, cross-trial spectral connectivity, and wPLI return one row per
trial group. PAC and time-averaged spectral connectivity retain one row per
epoch. See :ref:`concepts-row-kinds`.

Signatures are in :doc:`/api/connectivity`.

The :doc:`sensor-connectivity tutorial </auto_tutorials/plot_sensor_connectivity>`
compares PLV and wPLI for random-phase, shared, and phase-lagged signals, with
matching MNE-Connectivity calculations and explicit trial-group rows.

Inter-Trial Phase Coherence
---------------------------

Inter-trial phase coherence (ITPC) measures how consistent the phase at one
sensor is across trials. It is the length of the across-trial mean unit-phase
vector, averaged over time.

.. math::

   \mathrm{ITPC} = \frac{1}{T} \sum_t
   \left| \frac{1}{N} \sum_n e^{i \phi_n(t)} \right|

**Range**
   Values lie in :math:`[0,1]`. Identical phases give 1; exact cancellation of
   the finite set of unit vectors gives 0. Random uniform phases generally
   give a positive finite-sample estimate.

**Order of averaging**
   The mean across trials is taken at each time, and those values are then
   averaged over time. Averaging over time first is a different quantity.
   A phase that drifts within the trial but matches across trials is 1 in the
   order above and near 0 if time is averaged first.

**Bias**
   At one latency, let :math:`R_t=|N^{-1}\sum_n e^{i\phi_n(t)}|`.
   For independent uniform trial phases, :math:`E[R_t^2]=1/N`, and for
   sufficiently large :math:`N`, :math:`E[R_t]\approx\sqrt{\pi/(4N)}`.
   The first identity does not apply to the square of the time-averaged
   ITPC. Trial count, dependence, and phase availability affect the bias;
   comparisons should account for these differences.

**Missing values and flags**
   - The default requires at least two valid trials.
   - Cells below ``min_valid_trials`` are flagged.
   - Time points with fewer finite phases than ``min_valid_trials`` are
     omitted from the mean.
   - Zero-amplitude analytic samples have undefined phase and contribute zero
     coverage, even when their real and imaginary parts are finite.

**Reference**
   ITPC is used by Tallon-Baudry, Bertrand, Delpuech, and Pernier (1996) and
   described in Delorme and Makeig (2004).

Pairwise Phase Consistency
~~~~~~~~~~~~~~~~~~~~~~~~~~

Pairwise phase consistency (PPC; Vinck et al., 2010) estimates squared
population phase locking without the PLV-squared finite-sample bias under
independent identically distributed trial phases. Its expectation is 0 under
uniform phase. Finite-sample PPC can be negative; with :math:`N` valid trials
at one latency its range is :math:`[-1/(N-1),1]`. Different trial counts still
produce different uncertainty.

.. math::

   \mathrm{PPC} = \frac{1}{T} \sum_t \left( \frac{2}{N(N - 1)} \sum_{j < k} \cos(\phi_j(t) - \phi_k(t)) \right)

**Implementation**
   The equivalent form :math:`(|\sum_n e^{i\phi_n}|^2 - N) / (N(N - 1))` is
   used at each time, then averaged over time.
   When phases are missing, :math:`N` is the valid count at that latency.
   Latencies below ``min_valid_trials`` are omitted, as for ITPC.

Row layout of ITPC and PPC
~~~~~~~~~~~~~~~~~~~~~~~~~~

ITPC and PPC have one row per trial group. The same summary copied onto each
member epoch would repeat one number across rows.

- :func:`~eegtable.itpc` and :func:`~eegtable.ppc` return ``row_labels``.
- :func:`~eegtable.concat` rejects a join of group rows with per-epoch rows.
- Pass ``trials`` to compute the measure inside groups such as experimental
  condition.

Phase-Amplitude Coupling
------------------------

Mean vector length (MVL; Canolty et al., 2006) is the modulus of the mean of
:math:`A(t)e^{i\phi(t)}`, where :math:`\phi` is the phase of the slower band.
That raw form is ``normalize=False``. The default divides by the summed
amplitude instead of the sample count, which bounds the value in
:math:`[0, 1]`.

.. math::

   \mathrm{MVL} = \frac{\left| \sum_t A(t) e^{i \phi(t)} \right|}{\sum_t A(t)}

**Normalization**
   Division by the summed amplitude removes the scale of the envelope.
   Canolty et al. instead z-scored the raw value against surrogates.

**Flags**
   - ``normalize=True`` uses the summed-amplitude denominator. It returns NaN
     when that denominator is zero or no sample has both a defined phase and
     a finite amplitude. Every positive sum is usable, independent of units.
   - ``normalize=False`` divides the modulus of the weighted sum by the number
     of finite samples. It returns that real mean, not the unscaled complex
     sum.

**Rows**
   The measure is computed inside each trial, so the table has one row per
   epoch.

**Coverage and provenance**
   Coverage counts samples with both a defined slow-band phase and a finite
   fast-band amplitude, using the minimum of the two input coverages. Both
   input estimators are recorded in the computation metadata, so changing the
   phase estimator changes the feature identity.

**Notes**
   - ``pac`` returns the observed estimate with the selected amplitude
     normalization; ``pac_surrogates`` adds null inference.
   - Finite windows and temporal dependence can produce nonzero MVL without
     coupling. Amplitude normalization alone does not remove this bias.
   - A null for this value can be built from circular time shifts inside the
     trial, which keep the single-trial spectrum.

**Reference**
   Tort, Komorowski, Eichenbaum, and Kopell (2010) compare coupling estimators
   and note that raw mean vector length depends on the amplitude of the
   modulated band. It should therefore not be read as a coupling strength
   without a specified statistical reference. Surrogate inference depends on
   the null construction and its exchangeability assumptions.

Surrogate inference
~~~~~~~~~~~~~~~~~~~

``pac_surrogates`` retains the input epoch identities and returns seven measures:
observed ``pac``, ``pac_null_mean``, ``pac_null_std``, ``pac_corrected``,
``pac_zscore``, ``pac_pvalue``, and ``pac_pvalue_adjusted``. The corrected estimate
subtracts the null mean; the z score divides that difference by the population
standard deviation (``ddof=0``, matching Tensorpac). A surrogate distribution
whose range is at most ``100 * eps * max(abs(null))`` is treated as constant
within floating-point precision. Its deviation is zero, its z score is undefined,
and ``degenerate_null`` is set. This includes exact constant nulls and variation
caused solely by summation roundoff.

Surrogates are constructed by
`Tensorpac <https://etiennecmb.github.io/tensorpac/generated/tensorpac.Pac.html>`__:
``surrogate="blocks"`` exchanges the two amplitude blocks around a random cut;
``surrogate="circular"`` shifts phase circularly. Every shift occurs inside its
analysis window, without exchanging epochs. Both resulting blocks must be at
least ``min_shift_seconds`` long; identity shifts are excluded. Set that exclusion
from the autocorrelation timescale, and record ``random_state`` for reproducibility.
Perfectly stationary sinusoidal coupling can retain the same MVL after shifting,
so temporal shifts are not a universal coupling null.

ROI and global channel averages are applied to each permutation before computing
its null statistics. For each epoch, one-sided empirical p values are
:math:`(1 + \#\{\mathrm{null} \geq \mathrm{observed}\})/(N_\mathrm{surrogates}+1)`.
Observed and surrogate values share the same Tensorpac MVL arithmetic. Comparisons
count numerical ties within ``100 * eps * abs(observed)``, including for maxstat,
following `SciPy's permutation-test convention
<https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html>`__.
Corrected estimates within the same observed tie tolerance are reported as zero.
``correction`` accepts ``"none"``, Benjamini-Hochberg ``"fdr"``, ``"bonferroni"``,
or permutation ``"maxstat"``. The correction family consists of all spatial units
and windows returned by that call, within one epoch and one band pair. Separate
calls and recipe entries form separate families. The ``pac`` extra is required;
nonfinite samples and windows too short for the shift exclusion raise errors.

.. code-block:: python

   inferred = ef.pac_surrogates(
       theta_signal, gamma_signal, windows=[response],
       n_surrogates=999, surrogate="blocks", min_shift_seconds=0.5,
       random_state=42, correction="maxstat",
   )

Connectivity
------------

Two families of coupling between nodes are provided: amplitude-envelope
correlation and spectral connectivity. Nodes are channels, or ROIs when
``groups`` is given.

Functions
~~~~~~~~~

``envelope_correlation``
   Correlates band envelopes between every pair of nodes. By default each
   envelope is first orthogonalized against the other node
   (``orthogonalize="pairwise"``) and the magnitude is taken
   (``absolute=True``). ``orthogonalize=None`` gives the plain Pearson
   correlation; ``absolute`` only affects the orthogonalized branch.

``spectral_connectivity``
   Calls ``mne_connectivity.spectral_connectivity_epochs`` for coherence,
   imaginary coherency, the phase-locking value, pairwise phase consistency,
   the phase-lag index, and wPLI.

``wpli``
   The weighted phase-lag index. It weights phase-lag signs by the magnitude
   of the imaginary cross-spectrum (Vinck et al., 2011). This suppresses
   sensitivity to instantaneous mixing under the method's assumptions;
   it does not establish direct or causal interaction between sources.

``spectral_connectivity_time``
   Calls ``mne_connectivity.spectral_connectivity_time`` with ``average=False``
   to estimate coherence, magnitude imaginary coherency, PLV, ciPLV, PLI or wPLI
   over time separately within each epoch. Rows retain their epoch identities and
   can be joined to other per-epoch features and graph summaries.

Requirements and options
~~~~~~~~~~~~~~~~~~~~~~~~

- Spectral connectivity requires the ``connectivity`` extra
  (``pip install eegtable[connectivity]``).
- Every trial group needs at least two epochs for ``spectral_connectivity`` and
  ``wpli``. ``spectral_connectivity_time`` can estimate a single epoch.
- A spectral band must lie at or below the signal's Nyquist frequency;
  unavailable high frequencies raise an error rather than truncating the band.
- ``method="wpli2_debiased"`` applies the sample-size correction in Vinck et
  al. (2011) to squared wPLI. It can return negative finite-sample values and
  is a different scale from ``method="wpli"``.
- Warnings raised by the MNE estimator are left visible.
- Cross-trial spectral estimates assume comparable spectral statistics across
  the trials being combined. Two epochs is a computational minimum, not a
  guarantee of a reliable estimate.
- The default multitaper ``bandwidth=2.0`` fixes smoothing in hertz. Each
  window must support a bandwidth of at least
  :math:`1.35 f_s/N_\mathrm{samples}` to retain a low-bias taper. Fourier mode
  uses the delegated Fourier estimator.

Nodes and ROIs
~~~~~~~~~~~~~~

- For ``spectral_connectivity`` and ``wpli``, the channel-level matrix is
  averaged inside each ROI block, and a node's own block excludes the
  diagonal.
- For ``envelope_correlation``, a node's series is the mean complex analytic
  signal of its member channels, and envelopes are taken from that mean.
- Repeated channels within an ROI raise ``ValueError``; members have equal
  weight and must be unique.
- ``envelope_correlation``, ``spectral_connectivity`` and ``wpli`` have one row
  per trial group, as :func:`~eegtable.itpc` does.

Cross-trial pair tables use output finiteness as coverage: 1 for a finite
estimate and 0 otherwise. This does not report trial counts or the fraction
of usable source samples. Per-epoch time connectivity has the input-coverage
definition below.

Per-epoch time averaging
~~~~~~~~~~~~~~~~~~~~~~~~

``spectral_connectivity_time`` uses an explicit increasing ``freqs`` grid and
Morlet coefficients controlled by ``n_cycles``. Each analysis window is decomposed
independently. The complete half-support of the longest wavelet is pruned at both
edges. A window must then retain at least two samples and one cycle at the lowest
grid frequency. Short windows, nonfinite samples, and channels that are constant
inside an analysis window raise errors. These checks reject undefined inputs
and estimates; they do not remove all spectral leakage. Optional
``smoothing_seconds`` applies MNE's temporal Hanning smoother; zero leaves the
coefficients unsmoothed. A duration that rounds to two samples raises because the
Hanning kernel then has zero total weight. Undefined backend estimates raise
before ROI averaging. Frequency reduction uses the same half-open bands and
imaginary-coherency rectification as cross-trial connectivity.

Edge coverage is the mean input coverage across the requested window, taking the
minimum of the two channels at each sample. ROI coverage averages those
channel-pair coverages across the same block used for the connectivity estimate.

Time averaging and trial averaging define different estimators. In particular,
PPC and debiased squared wPLI remain available through the cross-trial API.
See the official
`comparison of time and trial connectivity <https://mne.tools/mne-connectivity/stable/auto_examples/compare_connectivity_over_time_over_trial.html>`__
and
`time connectivity API <https://mne.tools/mne-connectivity/stable/generated/mne_connectivity.spectral_connectivity_time.html>`__.

Envelope correlation
~~~~~~~~~~~~~~~~~~~~

Envelope correlation of analytic amplitudes is the approach used by Brookes et
al. (2011). The default orthogonalization is the pairwise projection of Hipp,
Hawellek, Corbetta, Siegel, and Engel (2012). It is not the symmetric
multivariate leakage correction of Colclough, Brookes, Smith, and Woolrich
(2015), which orthogonalizes all nodes jointly.

For analytic signals :math:`z_i(t)`,

.. math::

   a_{i\perp j}(t) = \left|\operatorname{Im}\left(z_i(t)
   \frac{\overline{z_j(t)}}{|z_j(t)|}\right)\right|,
   \qquad r_{i\perp j} = \operatorname{corr}(a_{i\perp j}, |z_j|)

- The orthogonalized branch replaces one envelope with the projected envelope
  and then symmetrizes.
- The implementation averages :math:`r_{i\perp j}` with its transpose. It can
  take the absolute value before that average.
- Trials are combined by a Fisher transform. Correlations are clipped to
  :math:`[-0.999999, 0.999999]` before :math:`\operatorname{arctanh}`.
- A constant original envelope has undefined Pearson correlation and returns
  NaN, including with orthogonalization. Its output coverage is zero.

For per-trial correlations :math:`r_e`, the returned group estimate is
:math:`\tanh(\operatorname{mean}_e\operatorname{arctanh}(r_e))`, after
clipping and omitting undefined trial correlations. It is an equally weighted
mean on the Fisher scale, rather than a correlation of concatenated trials.
The pairwise projection and symmetrization follow the
`MNE-Connectivity envelope-correlation API
<https://mne.tools/mne-connectivity/stable/generated/mne_connectivity.envelope_correlation.html>`__;
the across-trial Fisher aggregation is an EEGTable choice.

Spectral estimator formulas
~~~~~~~~~~~~~~~~~~~~~~~~~~~

:math:`S_{xy}^{(e)}` is the epoch cross-spectrum and :math:`\langle\cdot\rangle_e`
is the mean over the group's epochs. Before pair symmetrization the delegated
estimators are

.. math::

   \begin{aligned}
   \mathrm{coh} &= \frac{|\langle S_{xy}\rangle_e|}
      {\sqrt{\langle S_{xx}\rangle_e\langle S_{yy}\rangle_e}} \\
   \mathrm{imcoh} &= \frac{\operatorname{Im}(\langle S_{xy}\rangle_e)}
      {\sqrt{\langle S_{xx}\rangle_e\langle S_{yy}\rangle_e}} \\
   \mathrm{PLV} &= \left|\left\langle S_{xy}/|S_{xy}|\right\rangle_e\right| \\
   \mathrm{ciPLV} &= \frac{|\langle\operatorname{Im}(S_{xy}/|S_{xy}|)\rangle_e|}
      {\sqrt{1-|\langle\operatorname{Re}(S_{xy}/|S_{xy}|)\rangle_e|^2}} \\
   \mathrm{PLI} &= |\langle\operatorname{sign}(\operatorname{Im}S_{xy})\rangle_e| \\
   \mathrm{WPLI} &= \frac{|\langle\operatorname{Im}S_{xy}\rangle_e|}
      {\langle|\operatorname{Im}S_{xy}|\rangle_e}.
   \end{aligned}

**Notes**
   - ``coh`` is magnitude coherency, not magnitude-squared coherence.
   - PPC in this list is the unbiased estimator of squared PLV given in the
     ITPC section.
   - ``wpli2_debiased`` is MNE-Connectivity's debiased estimator of squared
     WPLI.
   - Cross-spectral accumulation is computed by MNE-Connectivity.
   - MNE returns a signed imaginary coherency. This package takes the absolute
     value before the frequency average, because the published pairs are
     unordered and the sign would flip if the node order were swapped.

The band reduction applied afterwards is

.. code-block:: python

   keep = band.mask(result.freqs)  # fmin <= f < fmax
   dense = result.get_data(output="dense")
   selected = np.abs(dense[..., keep]) if method == "imcoh" else dense[..., keep]
   band_matrix = selected.mean(axis=-1)
   band_matrix = band_matrix + band_matrix.T

**References**
   - The phase-locking value is Lachaux, Rodriguez, Martinerie, and Varela
     (1999).
   - Imaginary coherency is Nolte et al. (2004).
   - The corrected imaginary phase-locking value is Bruña, Maestú, and Pereda
     (2018).
   - The phase-lag index is Stam, Nolte, and Daffertshofer (2007).
   - ``wpli`` is the Vinck et al. (2011) estimator.
   - Parameter names and cross-spectral accumulation are those of
     `MNE-Connectivity <https://mne.tools/mne-connectivity/stable/generated/mne_connectivity.spectral_connectivity_epochs.html>`__.
   - ``spectral_connectivity`` is the common wrapper around the estimators
     above.

Common Spatial Patterns
-----------------------

:class:`~eegtable.CommonSpatialPattern` finds spatial filters that maximize the
variance ratio between two classes (Ramoser, Müller-Gerking, and Pfurtscheller,
2000). The labels determine the filters.

Requirements
~~~~~~~~~~~~

- Input should be band-limited.
- The number of components must be even.
- There must be two classes.

Cross-fitting and leakage
~~~~~~~~~~~~~~~~~~~~~~~~~

- :func:`~eegtable.csp_features` fits on each training fold and transforms the
  held-out rows of that fold. The resulting table is a description of those
  held-out rows.
- A supplied ``window`` restricts both fitting and held-out projected variance;
  samples outside that window do not enter the features.
- Reusing the same folds as a classifier's outer split still leaks. A fold's
  training rows include features generated by other CSP fits that used the
  classifier's held-out labels. A preassembled table therefore cannot replace
  fitting CSP inside that classifier's training partition.
- :func:`eegtable.model.build_design` rejects these columns.
- The split that produced a column is stored in its computation metadata.

To use CSP as a predictor:

- Fit it inside each training fold, and transform the training rows and the
  test rows with that fit.
- Repeat the fit inside inner tuning.
- In a scikit-learn pipeline, put ``mne.decoding.CSP`` in the pipeline.

EEGTable also provides a fold-fitted ``eegtable.model.CSPTransformer``; see
:doc:`/guides/learned_features` for predictive use with the package's CSP
normalization.

Covariance and filters
~~~~~~~~~~~~~~~~~~~~~~

For class :math:`c`, each selected training epoch is centered in time and its
covariance is normalized by the trace. A non-finite or temporally constant
training epoch raises an error; selected rows are never silently discarded.
Missing samples in held-out transformations are omitted from projected variance.

.. math::

   C_c = \frac{1}{|E_c|} \sum_{e \in E_c}
   \frac{(X_e-\bar X_e)(X_e-\bar X_e)^\mathsf{T}}
   {\operatorname{tr}\left((X_e-\bar X_e)(X_e-\bar X_e)^\mathsf{T}\right)}.

The data rank is determined before regularization. Covariances are projected
into the non-null subspace of :math:`C_0+C_1`. With ``regularization``
:math:`\rho > 0`, each projected covariance :math:`\widetilde C_c` is shrunk to
:math:`(1-\rho)\widetilde C_c + \rho\,\operatorname{tr}(\widetilde C_c)I/r`
for data rank :math:`r`. Shrinkage cannot restore removed dimensions.

The filters solve :math:`C_0 w = \lambda(C_0+C_1)w`.

- An average reference or removed ICA components make :math:`C_0+C_1`
  singular. The problem is then solved in the span of its non-null
  eigenvectors, as MNE's CSP does, and at most that many components exist.
- Components are taken alternately from the largest and the smallest
  eigenvalues.
- The feature is the natural log of projected variance divided by the sum
  over the retained components. Changing the component count changes this
  denominator.
- Trace normalization, component order, and the cross-fitting split are stored
  with the column.

.. code-block:: python

   projected = filters @ epoch
   variance = np.nanvar(projected, axis=-1)
   csp_log_power = np.log(variance / np.sum(variance))

Comparison with MNE
~~~~~~~~~~~~~~~~~~~

`mne.decoding.CSP <https://mne.tools/stable/generated/mne.decoding.CSP.html>`__
with ``cov_est="epoch"``, ``norm_trace=True``, and
``component_order="alternate"`` follows the same generalized eigenvalue
problem and component ordering. MNE normalizes each class covariance after
averaging epochs; this implementation normalizes each epoch before averaging.
Filters can therefore differ when epoch amplitudes vary. MNE's features are
log mean power; these features are log relative variance.

Graph Measures
--------------

``global_efficiency`` and ``clustering_coefficient`` reduce a pairwise table to
one value per band and window, in the same way :func:`~eegtable.band_ratio`
reduces a power table. Both are computed in this package.

Global efficiency
~~~~~~~~~~~~~~~~~

Global efficiency (Latora and Marchiori, 2001) is the mean inverse shortest-path
distance over node pairs.

- A nonzero weight :math:`w_{ij}` becomes a distance
  :math:`L_{ij} = 1/|w_{ij}|`.
- A zero weight is treated as a missing edge.
- Shortest paths are found with Floyd–Warshall.
- A disconnected pair contributes 0.

Weights enter through their absolute values. Negative correlations or negative
debiased estimates therefore contribute their magnitudes as positive weights;
this graph is not a signed-network analysis. Efficiency also depends on the
connectivity scale, node set, and edge estimator.

.. math::

   E_{\text{global}} = \frac{2}{N (N - 1)} \sum_{i < j} \frac{1}{d_{ij}}

:math:`N` is the number of nodes. The sum is over unordered pairs.

Clustering coefficient
~~~~~~~~~~~~~~~~~~~~~~

The clustering coefficient averages the local clustering of Watts and
Strogatz (1998) over all :math:`N` nodes. The graph is
binarized at ``threshold`` :math:`\theta` (:math:`A_{ij} = 1` when
:math:`|w_{ij}| > \theta`).

.. math::

   \begin{aligned}
   C_i &= \begin{cases}
      \dfrac{(A^3)_{ii}}{k_i (k_i - 1)} = \dfrac{2 T_i}{k_i (k_i - 1)}, & k_i \ge 2, \\
      0, & k_i < 2,
   \end{cases} \\[6pt]
   C &= \frac{1}{N} \sum_{i=1}^{N} C_i
   \end{aligned}

:math:`(A^3)_{ii}` is twice the number of triangles at node :math:`i`, and
:math:`k_i = \sum_j A_{ij}`.

- Nodes with :math:`k_i < 2` contribute 0.
- An eligible node with no triangles contributes 0.

**Comparability**
   This follows `NetworkX average_clustering
   <https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.cluster.average_clustering.html>`__
   with ``count_zeros=True`` and the Brain Connectivity Toolbox convention.
   Isolates and leaves remain in the denominator, so adding disconnected
   nodes cannot preserve the average of a densely connected subnetwork.
   The node-averaging convention is stored in the computation metadata.

Missing values
~~~~~~~~~~~~~~

These rules apply to both graph measures unless stated otherwise.

- A non-finite edge makes either summary NaN, with output coverage 0. It is
  not entered as a zero-weight disconnection.
- A pair absent from the table raises ``ValueError``, because the edge set must
  be complete.
- A finite graph with no triangles has clustering coefficient 0, including
  when no node has two neighbours.

References
----------

* Tallon-Baudry, C., Bertrand, O., Delpuech, C., & Pernier, J. (1996).
  *Stimulus specificity of phase-locked and non-phase-locked 40 Hz visual
  responses in human*. The Journal of Neuroscience, 16(13), 4240--4249.
  `doi:10.1523/JNEUROSCI.16-13-04240.1996
  <https://doi.org/10.1523/JNEUROSCI.16-13-04240.1996>`__.
* Delorme, A., & Makeig, S. (2004). *EEGLAB: An open source toolbox for
  analysis of single-trial EEG dynamics including independent component
  analysis*. Journal of Neuroscience Methods, 134(1), 9--21.
  `doi:10.1016/j.jneumeth.2003.10.009
  <https://doi.org/10.1016/j.jneumeth.2003.10.009>`__.
* Lachaux, J.-P., Rodriguez, E., Martinerie, J., & Varela, F. J. (1999). *Measuring
  phase synchrony in brain signals*. Human Brain Mapping, 8(4), 194--208.
  `doi:10.1002/(SICI)1097-0193(1999)8:4%3C194::AID-HBM4%3E3.0.CO;2-C
  <https://doi.org/10.1002/(SICI)1097-0193(1999)8:4%3C194::AID-HBM4%3E3.0.CO;2-C>`__.
* Nolte, G., Bai, O., Wheaton, L., Mari, Z., Vorbach, S., & Hallett, M. (2004).
  *Identifying true brain interaction from EEG data using the imaginary part of
  coherency*. Clinical Neurophysiology, 115(10), 2292--2307.
  `doi:10.1016/j.clinph.2004.04.029
  <https://doi.org/10.1016/j.clinph.2004.04.029>`__.
* Bruña, R., Maestú, F., & Pereda, E. (2018). *Phase locking value revisited:
  Teaching new tricks to an old dog*. Journal of Neural Engineering, 15(5),
  056011. `doi:10.1088/1741-2552/aacfe4
  <https://doi.org/10.1088/1741-2552/aacfe4>`__.
* Vinck, M., van Wingerden, M., Womelsdorf, T., Fries, P., & Pennartz, C. M.
  A. (2010). *The pairwise phase consistency: A bias-free measure of rhythmic
  neuronal synchronization*. NeuroImage, 51(1), 112--122.
  `doi:10.1016/j.neuroimage.2010.01.073
  <https://doi.org/10.1016/j.neuroimage.2010.01.073>`__.
* Canolty, R. T., Edwards, E., Dalal, S. S., Soltani, M., Nagarajan, S. S.,
  Kirsch, H. E., Berger, M. S., Barbaro, N. M., & Knight, R. T. (2006). *High
  gamma power is phase-locked to theta oscillations in human neocortex*.
  Science, 313(5793), 1626--1628.
  `doi:10.1126/science.1128115 <https://doi.org/10.1126/science.1128115>`__.
* Tort, A. B. L., Komorowski, R., Eichenbaum, H., & Kopell, N. (2010). *Measuring
  phase-amplitude coupling between neuronal oscillations of different
  frequencies*. Journal of Neurophysiology, 104(2), 1195--1210.
  `doi:10.1152/jn.00106.2010 <https://doi.org/10.1152/jn.00106.2010>`__.
* Brookes, M. J., Hale, J. R., Zumer, J. M., Stevenson, C. M., Francis, S. T.,
  Barnes, G. R., Owen, J. P., Morris, P. G., & Nagarajan, S. S. (2011).
  *Measuring functional connectivity using MEG: Methodology and comparison
  with fcMRI*. NeuroImage, 56(3), 1082--1104.
  `doi:10.1016/j.neuroimage.2011.02.054
  <https://doi.org/10.1016/j.neuroimage.2011.02.054>`__.
* Colclough, G. L., Brookes, M. J., Smith, S. M., & Woolrich, M. W. (2015). *A
  symmetric multivariate leakage correction for MEG connectomes*. NeuroImage,
  117, 439--448. `doi:10.1016/j.neuroimage.2015.03.071
  <https://doi.org/10.1016/j.neuroimage.2015.03.071>`__.
* Hipp, J. F., Hawellek, D. J., Corbetta, M., Siegel, M., & Engel, A. K. (2012).
  *Large-scale cortical correlation structure of spontaneous oscillatory
  activity*. Nature Neuroscience, 15(6), 884--890.
  `doi:10.1038/nn.3101 <https://doi.org/10.1038/nn.3101>`__.
* Stam, C. J., Nolte, G., & Daffertshofer, A. (2007). *Phase lag index:
  Assessment of functional connectivity from multi channel EEG and MEG with
  diminished bias from common sources*. Human Brain Mapping, 28, 1178--1193.
  `doi:10.1002/hbm.20346 <https://doi.org/10.1002/hbm.20346>`__.
* Vinck, M., Oostenveld, R., van Wingerden, M., Battaglia, F., & Pennartz, C.
  M. A. (2011). *An improved index of phase-synchronization for
  electrophysiological data in the presence of volume-conduction, noise and
  sample-size bias*. NeuroImage, 55(4), 1548--1565.
  `doi:10.1016/j.neuroimage.2011.01.055
  <https://doi.org/10.1016/j.neuroimage.2011.01.055>`__.
* Ramoser, H., Müller-Gerking, J., & Pfurtscheller, G. (2000). *Optimal spatial
  filtering of single trial EEG during imagined hand movement*. IEEE
  Transactions on Rehabilitation Engineering, 8(4), 441--446.
  `doi:10.1109/86.895946 <https://doi.org/10.1109/86.895946>`__.
* Latora, V., & Marchiori, M. (2001). *Efficient behavior of small-world
  networks*. Physical Review Letters, 87(19), 198701.
  `doi:10.1103/PhysRevLett.87.198701
  <https://doi.org/10.1103/PhysRevLett.87.198701>`__.
* Watts, D. J., & Strogatz, S. H. (1998). *Collective dynamics of small-world
  networks*. Nature, 393, 440--442. `doi:10.1038/30918
  <https://doi.org/10.1038/30918>`__.
