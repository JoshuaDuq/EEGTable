Spectral Features
=================

Power reductions, spectral descriptors, and spectral parameterization. Most
functions accept :class:`~eegtable.Spectra`; ``irasa`` accepts broadband
:class:`~eegtable.Signal` inputs. Definitions, representation requirements, and
units are in :doc:`/methods/spectral`.

Power Reductions
----------------

.. autofunction:: eegtable.integrated_band_power

.. autofunction:: eegtable.mean_psd

.. autofunction:: eegtable.mean_tfr_power

.. autofunction:: eegtable.periodic_power

Spectral Descriptors
--------------------

.. autofunction:: eegtable.peak_frequency

.. autofunction:: eegtable.spectral_centroid

.. autofunction:: eegtable.spectral_bandwidth

.. autofunction:: eegtable.spectral_edge

.. autofunction:: eegtable.spectral_entropy

Aperiodic and Periodic Estimates
--------------------------------

.. autofunction:: eegtable.aperiodic

.. autofunction:: eegtable.aperiodic_ratio

.. autofunction:: eegtable.spectral_parameterization

.. autofunction:: eegtable.irasa

Derived Power Features
----------------------

These functions accept a power :class:`~eegtable.FeatureTable` and preserve
the meaning of its stored normalization.

.. autofunction:: eegtable.band_ratio

.. autofunction:: eegtable.asymmetry
