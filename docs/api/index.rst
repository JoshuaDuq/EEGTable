API Reference
=============

.. raw:: html

   <p class="hero-lede">
     Signatures, parameters, and return types.
   </p>

Core containers, feature extractors, microstate functions, quality policies,
and reliability are exported from ``eegtable``. Modeling, preprocessing, native
BIDS input, file I/O, and group-sample design interfaces are documented under
``eegtable.model``, ``eegtable.preprocessing``, ``eegtable.bids``,
``eegtable.io``, and ``eegtable.group``. Each entry shows its import path.
Method definitions and assumptions are in :doc:`/methods/index`.

Browse the complete :ref:`symbol index <genindex>` to find a function or class by name.

Containers and files
--------------------

.. grid:: 1
   :gutter: 3
   :class-container: nav-cards

   .. grid-item-card:: Containers & I/O
      :link: containers
      :link-type: doc

      ``FeatureTable``, ``Spectra``, ``Signal``, ``BandSignal``, ``Band``,
      ``Window``, table reading and writing, group samples, quality policies,
      and repeated-session reliability.

Feature extraction
------------------

.. grid:: 1 1 2 2
   :gutter: 3
   :class-container: nav-cards

   .. grid-item-card:: Spectral Features
      :link: spectral
      :link-type: doc

      Band power, peak frequency, spectral descriptors, aperiodic fits, ratios.

   .. grid-item-card:: Dynamics
      :link: dynamics
      :link-type: doc

      Time-domain descriptors, oscillatory bursts, ERDS magnitudes and latencies.

   .. grid-item-card:: Phase & Connectivity
      :link: connectivity
      :link-type: doc

      ITPC, PPC, PAC, envelope correlation, wPLI, CSP, graph summaries.

   .. grid-item-card:: Complexity & Microstates
      :link: complexity
      :link-type: doc

      Higuchi fractal dimension, sample and multiscale entropy, microstates.

Analysis workflows
------------------

.. grid:: 1 1 2 2
   :gutter: 3
   :class-container: nav-cards

   .. grid-item-card:: Predictive Modeling
      :link: model
      :link-type: doc

      Design construction, cross-fitting, metrics, nulls, uncertainty, importance.

   .. grid-item-card:: Preprocessing
      :link: preprocessing
      :link-type: doc

      Raw-to-epochs workflow, checkpoints, review, and the numerical operations.

.. toctree::
   :hidden:
   :maxdepth: 2

   Containers & I/O <containers>
   Spectral <spectral>
   Dynamics <dynamics>
   Phase & connectivity <connectivity>
   Complexity & microstates <complexity>
   Predictive modeling <model>
   preprocessing
