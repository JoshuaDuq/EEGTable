Phase and Connectivity
======================

Phase consistency, coupling, spatial filtering, and graph summaries. Definitions,
row semantics, and estimator limitations are in :doc:`/methods/connectivity`.

Across-Trial Phase Consistency
------------------------------

These functions return one row per trial group.

.. autofunction:: eegtable.itpc

.. autofunction:: eegtable.ppc

Phase-Amplitude Coupling
------------------------

These functions return one row per epoch. Surrogate inference requires
``eegtable[pac]``.

.. autofunction:: eegtable.pac

.. autofunction:: eegtable.pac_surrogates

Sensor and ROI Connectivity
---------------------------

Envelope correlation and cross-trial spectral connectivity return one row per
trial group. ``spectral_connectivity_time`` returns one row per epoch. Both
spectral functions and ``wpli`` require ``eegtable[connectivity]``.

.. autofunction:: eegtable.envelope_correlation

.. autofunction:: eegtable.spectral_connectivity

.. autofunction:: eegtable.spectral_connectivity_time

.. autofunction:: eegtable.wpli

Common Spatial Patterns
-----------------------

For predictive use, spatial filters must be fitted inside the training
partitions. See :doc:`/guides/learned_features`.

.. autoclass:: eegtable.CommonSpatialPattern
   :members:

.. autofunction:: eegtable.csp_features

Graph Summaries
---------------

These functions reduce complete pairwise ``FeatureTable`` edge sets and retain
their input row identities.

.. autofunction:: eegtable.global_efficiency

.. autofunction:: eegtable.clustering_coefficient
