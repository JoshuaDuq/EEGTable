Feature Tables and Files
========================

.. raw:: html

   <p class="hero-lede">
     Select columns by metadata, write TSV and JSON, and stack recordings.
   </p>

Use this guide after :doc:`/quickstart`; the examples reuse ``ef``,
``epochs``, and ``spectral_features`` defined there. Field definitions and
column identities are in :doc:`/concepts`.

Querying a table
----------------

.. code-block:: python

   # An empty table when nothing matches.
   alpha_channels = spectral_features.select(
       band=ef.Band("alpha", 8.0, 13.0), space_kind="channel"
   )

   # Coverage measures finite input, not artifact quality.
   valid_fractions = spectral_features.coverage

   if "no_peak" in spectral_features.flags:
       absent_peaks = spectral_features.flags["no_peak"]

   df = spectral_features.to_dataframe()
   print(df.head())
   # eeg_band-power_alpha_cz_all_raw_p<hash>
   # eeg_peak-freq-adjusted_alpha_cz_all_raw_p<hash>

Three methods narrow a table further:

- ``select`` matches :class:`~eegtable.FeatureMeta` fields.
- ``take`` keeps rows by position or mask, in the order given, with their
  identities and flags.
- ``drop_missing`` keeps the columns missing in at most a given fraction of
  rows.

.. code-block:: python

   first_ten = spectral_features.take(range(10))
   mostly_measured = spectral_features.drop_missing(0.2)

Writing and reading
-------------------

``eegtable.io`` writes a table to TSV and JSON, reads it back, and stacks
per-epoch tables into a dataset.

.. code-block:: python

   from eegtable.io import read_dataset, read_table, write_table

   paths = write_table(spectral_features, "sub-01_features.tsv", rows=epochs.metadata)
   restored = read_table("sub-01_features.tsv")
   dataset = read_dataset(
       ["sub-01_features.tsv", "sub-02_features.tsv", "sub-03_features.tsv"]
   )

Descriptor column names must be unique, nonempty strings. Invalid names raise
before any files are written, preventing ambiguous targets when the bundle is
read back.
Schema 2 sidecars record which descriptor columns are text, preserving distinct
labels such as ``"01"`` and ``"1"`` through epoch and group dataset loading.
Numeric descriptors retain their numerical values.

The functions:

- :func:`eegtable.io.write_table` writes a TSV of values, a ``_coverage.tsv`` of
  aligned feature matrix and row keys, and a JSON sidecar. The coverage file
  omits the descriptive columns included in the values file.
- :func:`eegtable.io.read_table` restores metadata, flags, and row identity.
- :func:`eegtable.io.read_dataset` stacks per-epoch tables and returns descriptor
  columns separately from features.

**File layout.** The filenames follow BIDS TSV and JSON sidecar conventions. The
layout is not a validated BIDS derivative.

Schema 2 sidecars include descriptor types and content checksums of the values
and coverage files.
For schema 2, altered values or coverage payloads, incomplete manifests, and
unsupported schema versions raise errors. Regenerate bundles for a current,
checksummed analysis. The existing reader also recognizes historical
``eegfeat_version`` sidecars without a schema field; those files have neither
payload checksums nor a complete descriptor type manifest and do not provide
the same integrity guarantees. Extraction sidecars
also retain resolved defaults, software/source identities and checked upstream
preprocessing evidence. See :doc:`cohorts` for quality policies and group samples.

:doc:`/examples` shows the files the runner writes for a five-subject simulated
cohort.

Building a cohort
-----------------

:func:`eegtable.stack_rows` concatenates per-epoch tables in input order.

**Refusals.**

- Duplicate row identities.
- Cross-trial group tables.

**Column handling** (``columns``):

- ``columns="union"`` keeps every column any recording measured. A recording
  that did not measure a column gets ``NaN`` and zero coverage there.
- ``columns="identical"`` (the ``stack_rows`` default) requires one schema. A mismatch names
  the columns that differ.

.. code-block:: python

   cohort_features = ef.stack_rows(
       [sub_01_features, sub_02_features, sub_03_features], columns="union"
   )

``read_dataset`` defaults to ``columns="union"``. Pass
``columns="identical"`` explicitly when every recording must have the same
feature schema. Union schemas can contain distinct columns for the same
readable feature when computation settings or ROI membership differ; inspect
metadata before modeling.

**Modeling input.** The frame passed to :func:`eegtable.model.build_design` needs
matching ``recording``, ``epoch``, and ``event`` keys, plus the target and
grouping columns. The cross-fitting workflow is in :doc:`/guides/modeling`.
