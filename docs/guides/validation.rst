Validation on Public Datasets
=============================

.. rst-class:: hero-lede

   Numerical comparisons, physiological contrasts, and held-out prediction
   checks on public EEG recordings, with saved evidence for each run.

.. include:: ../validation/summary.inc

Evidence scope
--------------

The generated summary, scorecard, and check table describe the saved validation
run. ``docs/validation/results.json`` records its timestamp, source hash,
software versions, criteria, observations, and pass/fail outcomes. The same
run is archived under ``docs/validation/runs/<run-id>/results.json``.
These artifacts establish what was tested in that environment; they do not
establish the validity of every public API or cover later source changes.

Synthetic unit tests separately check formulas and boundary conditions.
Public-dataset tests examine selected estimators and effects in real
recordings. The narrative below summarizes those checks and their limits.
A partial run replaces the current summary with only that run's evidence; it
does not merge results from older runs.

Some implementation choices were informed by these datasets. Repeated success
on them is regression evidence, rather than independent prospective validation.
Study-specific windows, channel choices, targets, and exclusion rules still
need scientific justification.

.. _validation-scorecard:

Reading the scorecard
---------------------

One row per tested public function, one column per evidence category. Counts
refer to distinct claims; multiple parameterized cases of a claim count once.
A dash means no recorded check in that category. Read the individual criteria
and outcomes below before interpreting a count as coverage of a method.

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Category
     - What the check establishes
   * - Formula
     - Agreement with an independent NumPy, SciPy, or closed-form calculation
       on the same input, within the stated numerical tolerance.
   * - Other estimator
     - Agreement with a named alternative estimator or package under the
       tested settings and tolerance.
   * - Known physiology
     - A specified physiological contrast in the selected recordings,
       channels, windows, and analysis units.
   * - Decoding
     - Held-out prediction performance under the recorded split and fitting
       procedure. This does not establish causal interpretation.
   * - Behaviour
     - A documented error, identity check, null behavior, or command-line
       operation under the tested condition.

.. include:: ../validation/scorecard.inc

Observed patterns
-----------------

The saved check table gives the quantitative criteria and observations.
The main tested contrasts are:

- **Motor recordings:** movement-related mu/beta desynchronization, reduced
  beta bursts, and movement prediction with sensorimotor power or cross-fitted
  CSP. These findings concern movement versus rest; they do not establish
  robust left-versus-right-hand discrimination.
- **Visual entrainment:** spectral peaks and harmonics at the driven SSVEP
  frequencies, trial phase consistency, and occipital coherence.
- **Evoked responses:** N1, P3, and error-related responses in the selected ERP
  CORE participant, together with direct numerical comparisons to MNE's peak
  methods.
- **Sleep staging:** spectral and complexity differences across stages in the
  selected Sleep-EDF recordings, held-out staging, and depth regression.
  Conclusions depend on the derivation and target definition.
- **Connectivity and microstates:** numerical comparisons to MNE-Connectivity,
  effects of pairwise orthogonalization, and descriptive microstate
  segmentation on the selected resting data.

Interpretation limits
---------------------

**Statistical inference.** Cross-validated predictions have overlapping
training sets, and EEG epochs can be temporally dependent. Held-out balanced
accuracy alone does not justify a binomial p-value. Use a null procedure whose
exchangeability units match the study design and repeat preprocessing, tuning,
and fitting within that procedure. See :doc:`modeling` and the analysis by
`Noirhomme et al. (2014)
<https://doi.org/10.1016/j.nicl.2014.04.004>`_.

**Baseline and normalization.** ``erds_mean`` averages a per-sample dB trace;
``mean_tfr_power`` takes dB after window reduction. The values therefore differ
even when both use the same baseline. The default onset persistence is six
cycles of the band's low edge; this is a detection rule, not a universal
false-positive guarantee. See :doc:`/methods/dynamics`.

**Unresolved effects.** Exploratory analyses accompanying development did not
establish robust contralateral motor dominance, a group-level post-movement
beta rebound, or reliable single-trial onset discrimination in these motor
recordings. These observations are not all rerun as suite assertions and should
not be treated as general absence-of-effect findings.

**PAC.** Nonzero raw phase-amplitude coupling is not evidence of physiological
coupling by itself. The suite checks selected formulas and surrogate behavior;
use a study-appropriate null and inspect waveform and filtering confounds.
See :doc:`/methods/connectivity`.

**Prediction intervals.** The recorded conformal checks describe empirical
coverage in the tested recordings. Group-disjoint fitting alone does not
establish exchangeability or guarantee coverage for a new participant or
population. Discrete targets can also produce conservative intervals.

**Sensor dependence.** Eye and movement activity in frontal derivations can
increase low-frequency waking power. Several sleep spectral and Hjorth checks
therefore use Pz-Oz. A finding from that derivation does not establish the same
contrast at Fpz-Cz.

Datasets
--------

.. list-table::
   :header-rows: 1
   :widths: 30 36 34

   * - Dataset
     - Tested sample
     - Main checks
   * - PhysioNet EEG Motor Movement/Imagery
     - Twenty subjects; 64 channels at 160 Hz; three cued hand-movement runs
       with intervening rest.
     - ERD/ERS, bursts, movement prediction, CSP, connectivity, and microstates.
   * - MNE SSVEP example
     - One subject; 32 channels; ten 20-second trials at each of 12 and 15 Hz.
     - Driven spectral peaks, phase consistency, and connectivity.
   * - ERP CORE, Flankers task
     - One subject; 30 channels at 1024 Hz; stimulus and response events.
     - Time-domain responses and comparisons to MNE peak extraction.
   * - Sleep-EDF (PhysioNet)
     - Three subjects; two EEG derivations at 100 Hz; expert-scored
       30-second epochs.
     - Stage-dependent features, sleep prediction, regression, and intervals.

Sample selection and preprocessing are defined in
``tests/validation/loaders.py``. References below identify the dataset methods;
MNE's `dataset documentation <https://mne.tools/stable/datasets.html>`_ describes
the download interfaces.

Run the suite
-------------

From the repository root:

.. code-block:: bash

   python -m pip install -e ".[dev,model,connectivity,microstates]"
   EEGTABLE_DATASETS=1 python -m pytest tests/validation -ra

On Windows PowerShell, set ``$env:EEGTABLE_DATASETS = "1"`` before running
``python -m pytest tests/validation -ra``.

The dataset suite is skipped unless explicitly enabled. It downloads several
hundred megabytes on first use into MNE's configured data directory
(``MNE_DATA``, normally ``~/mne_data``). Additional tests of optional backends
need their corresponding extras; skipped checks do not establish evidence for
those integrations.

Each run rewrites the current validation JSON and included fragments, and saves
a separate snapshot with the run identity. A documentation build renders the
saved evidence without running the suite. Keep a snapshot with the analysis
when it supports a scientific claim.

The ``validation`` GitHub workflow runs weekly and on demand, checks MNE 1.8
and the installed current release, caches data, and uploads each job's evidence.
The current-release job is not a fixed dependency environment; use the saved
versions to identify what it tested.

.. _validation-results:

Individual checks
-----------------

Each row identifies the claim, criterion, observed result, and outcome, grouped
by dataset. Consult the corresponding test when the observation is empty or
when more detail about sample selection is needed.

.. include:: ../validation/results.inc

Reporting an analysis
---------------------

Record the EEGTable version and source revision, dependency versions,
preprocessing and reference, channels/ROIs, bands/windows, estimator parameters,
coverage/flag exclusions, sample and split units, tuning, null scheme, and seeds.
Retain recipes, feature sidecars, and the relevant evidence snapshot. Cite the
underlying methods and packages used, following the references in the method
pages. Validation counts alone are insufficient to support a study's
scientific interpretation.

References
----------

* Schalk, G., McFarland, D. J., Hinterberger, T., Birbaumer, N., & Wolpaw, J. R.
  (2004). *BCI2000: A general-purpose brain-computer interface (BCI) system*. IEEE
  Transactions on Biomedical Engineering, 51(6), 1034--1043.
  `doi:10.1109/TBME.2004.827072 <https://doi.org/10.1109/TBME.2004.827072>`__.
* Goldberger, A. L., Amaral, L. A. N., Glass, L., Hausdorff, J. M., Ivanov, P. Ch.,
  Mark, R. G., Mietus, J. E., Moody, G. B., Peng, C.-K., & Stanley, H. E. (2000).
  *PhysioBank, PhysioToolkit, and PhysioNet*. Circulation, 101(23), e215--e220.
  `doi:10.1161/01.CIR.101.23.e215 <https://doi.org/10.1161/01.CIR.101.23.e215>`__.
* Kemp, B., Zwinderman, A. H., Tuk, B., Kamphuisen, H. A. C., & Oberyé, J. J. L.
  (2000). *Analysis of a sleep-dependent neuronal feedback loop: The slow-wave
  microcontinuity of the EEG*. IEEE Transactions on Biomedical Engineering, 47(9),
  1185--1194. `doi:10.1109/10.867928 <https://doi.org/10.1109/10.867928>`__.
* Kappenman, E. S., Farrens, J. L., Zhang, W., Stewart, A. X., & Luck, S. J. (2021).
  *ERP CORE: An open resource for human event-related potential research*.
  NeuroImage, 225, 117465. `doi:10.1016/j.neuroimage.2020.117465
  <https://doi.org/10.1016/j.neuroimage.2020.117465>`__.
