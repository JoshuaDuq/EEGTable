Dynamics
========

Measures estimated from :class:`~eegtable.Signal` and :class:`~eegtable.BandSignal`
containers, with one result row per epoch. Definitions and units are in
:doc:`/methods/dynamics`; cycle-by-cycle methods are in :doc:`/methods/cycles`.

Time-Domain Measures
--------------------

.. autofunction:: eegtable.variance

.. autofunction:: eegtable.amplitude_quantile

.. autofunction:: eegtable.kurtosis

.. autofunction:: eegtable.line_length

.. autofunction:: eegtable.root_mean_square

.. autofunction:: eegtable.skewness

.. autofunction:: eegtable.zero_crossing_rate

.. autofunction:: eegtable.hjorth_mobility

.. autofunction:: eegtable.hjorth_complexity

.. autofunction:: eegtable.mean_amplitude

.. autofunction:: eegtable.peak_to_peak

.. autofunction:: eegtable.area_under_curve

.. autofunction:: eegtable.peak_amplitude

.. autofunction:: eegtable.peak_latency

Oscillatory Bursts
------------------

These functions detect envelope-threshold runs on ``BandSignal`` inputs.

.. autofunction:: eegtable.burst_count

.. autofunction:: eegtable.burst_rate

.. autofunction:: eegtable.burst_duration

.. autofunction:: eegtable.burst_amplitude

.. autofunction:: eegtable.fraction_above_threshold

Cycle-by-Cycle Features
-----------------------

This function uses broadband ``Signal`` inputs and requires ``eegtable[cycles]``.

.. autofunction:: eegtable.cycle_features

ERDS Dynamics
-------------

These functions use ``BandSignal`` power relative to an explicit baseline
window. The default scale is decibels.

.. autofunction:: eegtable.erds_mean

.. autofunction:: eegtable.erds_slope

.. autofunction:: eegtable.erd_magnitude

.. autofunction:: eegtable.erd_duration

.. autofunction:: eegtable.ers_magnitude

.. autofunction:: eegtable.ers_duration

.. autofunction:: eegtable.erds_peak_latency

.. autofunction:: eegtable.erds_onset_latency

.. autofunction:: eegtable.erds_rebound_latency
