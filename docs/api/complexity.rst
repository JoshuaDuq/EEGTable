Complexity and Microstates
==========================

Definitions, input semantics, and estimator limitations are in
:doc:`/methods/complexity`. Univariate measures use waveforms from ``Signal``
or envelopes from ``BandSignal`` and retain one row per epoch.

Univariate Complexity and Scaling
---------------------------------

.. autofunction:: eegtable.higuchi_fractal_dimension

.. autofunction:: eegtable.sample_entropy

.. autofunction:: eegtable.multiscale_entropy

The following functions require ``eegtable[complexity]`` for AntroPy.

.. autofunction:: eegtable.permutation_entropy

.. autofunction:: eegtable.lempel_ziv_complexity

.. autofunction:: eegtable.detrended_fluctuation

Microstates
-----------

Template fitting requires ``eegtable[microstates]``. ``MicrostateModel`` keeps
the fitted or externally supplied templates separate from subsequent
assignment; ``microstates.segment`` fits and assigns in one call. Temporal
summaries consume the resulting ``MicrostateSegmentation``.

.. autoclass:: eegtable.MicrostateModel
   :members:

.. autofunction:: eegtable.microstates.segment

.. autofunction:: eegtable.microstates.microstate_coverage

.. autofunction:: eegtable.microstates.microstate_duration

.. autofunction:: eegtable.microstates.microstate_occurrence

.. autofunction:: eegtable.microstates.microstate_transitions

.. autoclass:: eegtable.microstates.MicrostateSegmentation
   :members:
   :show-inheritance:
