Methods
=======

.. raw:: html

   <p class="hero-lede">
     Definitions, frequency and time grids, and the conditions that return
     NaN or set a flag.
   </p>

Method pages describe estimators, assumptions, units, and undefined results.
The :doc:`/api/index` documents callable signatures. Computational validity
and coverage do not establish physiological specificity or statistical
significance; interpretation depends on preprocessing and the study design.

.. grid:: 1 1 2 2
   :gutter: 3
   :class-container: nav-cards

   .. grid-item-card:: Spectral
      :link: spectral
      :link-type: doc

      Band power and normalization, wavelet support, peak frequency,
      spectral descriptors, aperiodic fitting, ratios, and asymmetry.

   .. grid-item-card:: Dynamics
      :link: dynamics
      :link-type: doc

      ERDS, oscillatory bursts, time-domain descriptors, peak amplitude, and
      latency.

   .. grid-item-card:: Phase & Connectivity
      :link: connectivity
      :link-type: doc

      ITPC, phase-amplitude coupling, envelope correlation, wPLI, common
      spatial patterns, and graph summaries.

   .. grid-item-card:: Complexity & Microstates
      :link: complexity
      :link-type: doc

      Entropy, fractal dimension, Lempel--Ziv complexity, DFA, and frozen or
      fitted microstate templates.

   .. grid-item-card:: Cycle features
      :link: cycles
      :link-type: doc

      ByCycle waveform shape, cycle counts, and consistency-based burst labels.

Cross-fitting, permutation nulls, and conformal intervals are in
:doc:`/guides/modeling`.

.. toctree::
   :hidden:
   :maxdepth: 2

   Spectral <spectral>
   Dynamics <dynamics>
   Phase & connectivity <connectivity>
   Complexity & microstates <complexity>
   Cycle features <cycles>
