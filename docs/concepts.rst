Data Concepts
=============

.. raw:: html

   <p class="hero-lede">
     Containers, <code>FeatureTable</code> fields, column names, cross-trial
     tables, and missing values.
   </p>

.. _concepts-mental-model:

From MNE objects to feature tables
----------------------------------

``eegtable`` accepts MNE ``Spectrum``, ``EpochsTFR``, and ``Epochs`` objects.
PSD and Morlet time-frequency representations are computed in MNE, then wrapped.
:class:`~eegtable.BandSignal` is the exception.
:meth:`~eegtable.BandSignal.from_epochs` applies its documented band-pass and
Hilbert transform.

.. grid:: 1 1 3 3
   :gutter: 3
   :class-container: nav-cards

   .. grid-item-card:: 1. Wrap MNE Objects

      Convert MNE outputs into :class:`~eegtable.Spectra`, :class:`~eegtable.Signal`,
      or :class:`~eegtable.BandSignal`.

   .. grid-item-card:: 2. Extract Features

      Call an extractor with explicit bands, windows, and regions of interest.

   .. grid-item-card:: 3. Query and Export

      Filter columns with :meth:`~eegtable.FeatureTable.select`, read
      ``coverage``, or export with
      :meth:`~eegtable.FeatureTable.to_dataframe`.

The spectral estimator, the filter, and the trial grouping are arguments of the
wrapper, or keys in a runner recipe. See :doc:`/guides/runner`.

.. _concepts-containers:

The containers
--------------

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Container
     - Contents
   * - :class:`~eegtable.Spectra`
     - PSD density, support-restricted Morlet power, or a dimensionless ratio
       to an aperiodic fit, with estimator parameters. The representation is
       explicit; incompatible operations raise errors.
   * - :class:`~eegtable.Signal`
     - Broadband time-domain data.
   * - :class:`~eegtable.BandSignal`
     - Band-limited analytic signal. ``from_epochs`` filters and takes the
       Hilbert transform.
   * - :class:`~eegtable.Band`
     - A named half-open frequency interval.
   * - :class:`~eegtable.Window`
     - A named time interval in seconds relative to the epoch origin. Sample
       selection includes both bounds; Morlet reduction further restricts
       coefficients by wavelet support.

Bands and windows are named, not positional. A column records that it is
``alpha`` in ``baseline``, and :meth:`~eegtable.FeatureTable.select` matches
those fields.

.. _concepts-feature-table:

Anatomy of a feature table
--------------------------

A :class:`~eegtable.FeatureTable` has five aligned parts.

``values``
   Shape ``(n_rows, n_features)``.

``coverage``
   Same shape. Fraction of finite input behind each value, in ``[0, 1]``.
   This is the fraction of the input that was present. It is not an artifact
   score. For Morlet input, ``support`` on :class:`~eegtable.Spectra` is the fraction of
   the requested window with complete wavelet support.

``meta``
   One :class:`~eegtable.FeatureMeta` per column. Fields are the measure, band,
   space and its kind (``space_kind``), window and its bounds, normalization,
   unit, source, the frequency resolution, the phase and amplitude bands of a
   coupling measure, the node pair of a pairwise measure, and the
   :class:`~eegtable.ComputationSpec` that produced the column. Each field is
   constant across rows.

``flags``
   Per-cell boolean annotations, for example ``no_peak``, ``edge_hit``,
   and ``aperiodic_fit_failed``. A fact that varies by row belongs here.

``row_ids`` / ``row_labels``
   Row identity. See below.

Slicing, concatenation, and a round trip through disk keep this metadata.
:meth:`~eegtable.FeatureTable.select` matches the metadata. It does not parse
the column name.

.. _concepts-naming:

Column names
------------

Column names are generated. Each name contains six readable underscore-separated fields and a
seventh field containing the hash suffix.

.. code-block:: text

   eeg_band-power_alpha_cz_all_raw_p8f3c1d5e9a02
   \_/ \________/ \___/ \/ \_/ \_/ \____________/
    |       |       |   |   |   |         |
    |    measure   band |   |   |    first 12 hex of the SHA-256
  domain            space   |  normalization   of the complete column spec
                          window

Underscores and spaces inside a field are replaced by hyphens, so the split
into six fields is unambiguous. A missing band is written ``broadband``. A
missing window is written ``all``.

The hash is the SHA-256 of the full column specification, not only the
arguments passed to the extractor. Two columns that differ only in a parameter
such as ``fit_range`` or a burst threshold share the six readable fields and
differ in the hash. Both can sit in one table.

Spatial aggregation records the sorted member channels in the computation
specification. Changing an ROI's membership, or the channels entering a global
mean, changes its feature identity even when the spatial label stays the same.
Connectivity records the member channels of every node as well. Reordering
members within an ROI does not change its identity.

.. _concepts-row-kinds:

Epoch rows and group rows
-------------------------

There are two kinds of table. They are not merged.

**Per-epoch tables** have one row per epoch. ``row_ids`` is a
``(recording, epoch index, event)`` triple per row. Joining feature columns checks those identities; stacking recordings
retains them and rejects duplicates. :func:`eegtable.model.build_design` accepts only
per-epoch tables.

**Group-row tables** come from measures that are undefined on one trial.
Those measures are inter-trial phase coherence, pairwise phase consistency,
envelope correlation (per-trial correlations averaged in Fisher :math:`z`),
epoch-averaged ``spectral_connectivity`` methods including wPLI, and graph
summaries of those group estimates. ``spectral_connectivity_time`` instead
returns per-epoch estimates. Graph summaries preserve the row kind of their
input. The value describes a
set of trials. The table carries ``row_labels`` for those groups and cannot
carry ``row_ids``.

Copying a group value onto its member epochs repeats one number across rows.
This can overstate the effective sample size and leak shared information
across training and test rows. The runner writes
the two kinds of table to separate files. ``eegtable.group`` provides native
group-sample loading and design construction. See :doc:`/guides/cohorts` and
:doc:`/guides/runner`.

.. _concepts-missing:

Missing values
--------------

``NaN`` in ``values`` denotes a missing or withheld estimate. It can reflect
insufficient input, an undefined statistic, a failed fit, a quality exclusion,
or a column absent from a recording in a union schema.
Non-finite samples are treated as missing. Data-dependent failures use the measure's documented ``coverage`` and
``flags``. Invalid arguments, incompatible representations, and invalid file
manifests raise errors instead. See the method definition for the exact rule.

A finite value can still come from a small fraction of its input. Read
``coverage`` with the value. ``max_feature_missingness`` in
:class:`~eegtable.model.PreprocessingConfig` is a separate check. It drops a
column when the fraction of ``NaN`` rows exceeds the threshold. It does not
read ``coverage``.

Units and normalization
-----------------------

EEG input amplitudes follow MNE's convention of volts; frequencies are in Hz
and times in seconds unless a parameter explicitly names milliseconds.
Inspect each column's ``FeatureMeta.unit`` rather than inferring its unit from
its name.

.. list-table::
   :header-rows: 1
   :widths: 36 64

   * - Quantity
     - Interpretation
   * - Raw PSD density
     - V²/Hz for EEG, with full multitaper normalization when applicable.
   * - Integrated PSD band power
     - V², using interpolation at the exact band bounds.
   * - Morlet container values
     - MNE power divided by the original sampling frequency: a smoothed
       density representation. It is not numerically identical to a PSD.
   * - ``log10``
     - Base-10 logarithm of the value in its declared input unit.
   * - ``log_ratio``, ``db``, ``percent``
     - Baseline-relative scales. The order of averaging and normalization is
       measure-specific; read :doc:`/methods/spectral` and
       :doc:`/methods/dynamics` before comparing them.

Column identity records the scientific definition, including spatial members
and computation settings. It does not identify the complete dataset or
software environment; those belong to recording and run provenance. Preserve
both when reproducing an analysis.
