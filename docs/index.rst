:hide-toc:

EEGTable
========

.. image:: ../assets/branding/eegtable-logo.svg
   :alt: EEGTable
   :width: 268
   :height: 64
   :class: hero-logo only-light

.. image:: ../assets/branding/eegtable-logo-dark.svg
   :alt: EEGTable
   :width: 268
   :height: 64
   :class: hero-logo only-dark

.. rst-class:: hero-lede

   Labelled EEG feature extraction and modeling for MNE objects.

Extract spectral, temporal, connectivity, complexity, cycle, and microstate
measures while retaining their units and provenance. Evaluate regression and
binary classification models with group-disjoint validation. Method definitions describe each measure's assumptions and missing-value behavior.
This is development version ``0.1.0.dev0``; retain the source revision and
dependency versions with each analysis.

.. container:: overview-links

   - :doc:`Quick start <quickstart>`
   - :doc:`API reference <api/index>`
   - :doc:`Validation evidence <guides/validation>`

Start here
----------

.. grid:: 1 1 3 3
   :gutter: 3
   :class-container: document-directory

   .. grid-item::

      .. rubric:: :doc:`Installation <install>`

      Install from source, prepare your environment, and choose optional extras.

   .. grid-item::

      .. rubric:: :doc:`Quick start <quickstart>`

      Work through PSD, time-frequency, and burst examples with MNE objects.

   .. grid-item::

      .. rubric:: :doc:`Data concepts <concepts>`

      Inspect containers, feature names, units, provenance, and missing values.

Explore the methods
-------------------

Every feature has a definition. Start with the signal property you want to measure.

.. container:: method-directory

   :doc:`Spectral <methods/spectral>`
      Band power, spectral shape, aperiodic fits, IRASA, peak detection,
      Morlet support masking, and spectral parameterization.

      ``integrated_band_power`` · ``peak_frequency`` · ``aperiodic_ratio``

   :doc:`Dynamics <methods/dynamics>`
      Envelope bursts, ERD/ERS magnitudes and latencies, and time-domain
      waveform descriptors.

      ``burst_rate`` · ``erds_mean`` · ``hjorth_complexity``

   :doc:`Phase & connectivity <methods/connectivity>`
      Phase consistency, phase-amplitude coupling, envelope correlation,
      wPLI, spatial patterns, and graph summaries.

      ``itpc`` · ``pac`` · ``wpli``

   :doc:`Complexity & microstates <methods/complexity>`
      Sample and multiscale entropy, Higuchi fractal dimension, and
      GFP-peak clustered microstate segmentation.

      ``sample_entropy`` · ``MicrostateModel`` · ``segment``

   :doc:`Cycle waveforms <methods/cycles>`
      Cycle timing, amplitude, rise/decay symmetry, and consistency-based burst
      labels from broadband signals.

      ``cycle_features``

Build an analysis workflow
--------------------------

.. grid:: 1 1 3 3
   :gutter: 3
   :class-container: document-directory

   .. grid-item::

      .. rubric:: :doc:`Preprocess recordings <guides/preprocessing>`

      Raw-to-epochs cleaning with reviewed artifacts and validated checkpoints.

   .. grid-item::

      .. rubric:: :doc:`Process a cohort <guides/runner>`

      Apply one TOML recipe to a folder of preprocessed epochs files.

   .. grid-item::

      .. rubric:: :doc:`Evaluate a model <guides/modeling>`

      Grouped regression and binary classification, nested tuning, and nulls.

.. container:: research-note

   .. rubric:: Inspect the evidence

   Review the :doc:`method definitions <methods/index>` and
   :doc:`public-dataset validation <guides/validation>` before interpreting
   a feature. Explore :doc:`example output <examples>` from a simulated
   five-subject cohort, or learn to :doc:`query and export tables <guides/tables>`.

.. toctree::
   :hidden:
   :caption: Getting started

   install
   Quick start <quickstart>
   Data concepts <concepts>

.. toctree::
   :hidden:
   :caption: Guides

   guides/preprocessing
   Native BIDS input <guides/bids>
   Cohort runner <guides/runner>
   Tables & files <guides/tables>
   Reproducible cohorts <guides/cohorts>
   Predictive modeling <guides/modeling>
   Learned features <guides/learned_features>
   Modeling recipes <guides/model_recipes>

.. toctree::
   :hidden:
   :caption: Methods
   :maxdepth: 2

   Method definitions <methods/index>

.. toctree::
   :hidden:
   :caption: API reference
   :maxdepth: 2

   API overview <api/index>

.. toctree::
   :hidden:
   :caption: Research resources

   Validation <guides/validation>
   Example outputs <examples>
