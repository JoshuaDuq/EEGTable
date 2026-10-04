Containers and I/O
==================

:class:`~eegtable.FeatureTable` stores feature values and their aligned
metadata. This page also documents input containers, file interfaces, quality
policies, and reliability summaries. Field definitions are in :doc:`/concepts`.

Feature tables
--------------

.. autoclass:: eegtable.FeatureTable
   :members:
   :show-inheritance:

.. autoclass:: eegtable.FeatureMeta
   :members:
   :show-inheritance:

.. autoclass:: eegtable.ComputationSpec
   :members:
   :show-inheritance:

.. autofunction:: eegtable.concat

.. autofunction:: eegtable.stack_rows

Reading and writing
-------------------

.. autoclass:: eegtable.io.FeatureDataset
   :members:

.. autofunction:: eegtable.io.read_dataset

.. autofunction:: eegtable.io.read_table

.. autofunction:: eegtable.io.write_table

Group samples
-------------

.. autoclass:: eegtable.group.GroupDataset
   :members:

.. autoclass:: eegtable.group.GroupDesign
   :members:

Native BIDS input
-----------------

See :doc:`preprocessing` for :class:`~eegtable.bids.BIDSQuery`,
:class:`~eegtable.bids.BIDSRecording`, discovery, loading, and preprocessing.
The workflow and supported input scope are in :doc:`/guides/bids`.

Bands and windows
-----------------

.. autoclass:: eegtable.Band
   :members:
   :show-inheritance:

.. autoclass:: eegtable.Window
   :members:
   :show-inheritance:

.. autodata:: eegtable.BANDS_STANDARD
   :no-value:

.. autofunction:: eegtable.passband_fraction

.. autofunction:: eegtable.check_passband

Signal containers
-----------------

.. autoclass:: eegtable.Spectra
   :members:
   :show-inheritance:

.. autoclass:: eegtable.Signal
   :members:
   :show-inheritance:

.. autoclass:: eegtable.BandSignal
   :members:
   :show-inheritance:

Quality policies and summaries
------------------------------

Choose exclusion rules before evaluating outcomes. Coverage measures numerical
availability; it is not an artifact score. See :doc:`/guides/cohorts`.

.. autoclass:: eegtable.QualityPolicy
   :members:

.. autoclass:: eegtable.QualityResult
   :members:

.. autofunction:: eegtable.apply_quality

.. autofunction:: eegtable.feature_quality

.. autofunction:: eegtable.cohort_quality

Quality reports
~~~~~~~~~~~~~~~

.. autofunction:: eegtable.report.write_quality_report

.. autofunction:: eegtable.report.recording_quality

Repeated-session reliability
----------------------------

The input must contain one estimate per subject/session in a complete balanced
design with at least three subjects and two sessions. Aggregate repeated
epochs explicitly before computing ICC. The function
reports single-measure absolute agreement ICC(2,1) and consistency ICC(3,1);
it rejects missing values, duplicate samples, and degenerate features. Negative
ICC estimates can occur and should not be clipped when reporting results.

.. autofunction:: eegtable.intraclass_reliability
