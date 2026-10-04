Example Output
==============

.. raw:: html

   <p class="hero-lede">
     Files written by <code>examples/make_examples.py</code> from simulated
     recordings. Five subjects, two runs each, sixteen trials per run.
   </p>

The files are in the `examples/ directory
<https://github.com/JoshuaDuq/EEGTable/tree/main/examples>`_.
The simulation uses eight EEG channels and a fixed random seed (11).
The complete cohort is generated in a temporary directory; only the showcase
recording and model outputs are copied here. Existing output files are
replaced when you regenerate them with

.. code-block:: bash

   python -m pip install -e ".[model]"
   python examples/make_examples.py

What feature extraction writes
------------------------------

`recipe.toml <https://github.com/JoshuaDuq/EEGTable/blob/main/examples/recipe.toml>`_
sets bands, windows, regions, and one entry per measure. ``eegtable run`` applies
it to every epochs file and writes, per recording,

.. list-table::
   :header-rows: 1
   :widths: 42 58

   * - File
     - Contents
   * - ``*_features.tsv``
     - One row per epoch. Leading columns are the epoch identity and the
       metadata attached to it (``subject``, ``run``, ``trial``,
       ``intensity``, ``rating``, ``painful``). The remaining columns are
       features.
   * - ``*_features_coverage.tsv``
     - Aligned feature columns and row identities; descriptive columns are
       omitted. Fraction of finite input behind each feature cell.
   * - ``*_features.json``
     - Sidecar. For each column, the measure, band, space, window, unit,
       normalization, computation parameters, and hash. Also flags and run
       provenance.
   * - ``*_crosstrial.tsv``
     - Measures defined on a set of trials. In this recipe, inter-trial phase
       coherence. One row per trial group. These rows are not written into the
       per-epoch table (:ref:`concepts-row-kinds`).
   * - ``*_crosstrial_coverage.tsv``
     - Finite-input coverage for the cross-trial table.
   * - ``*_crosstrial.json``
     - Cross-trial metadata and provenance.

``sub-01_task-pain_run-01_features.json`` next to its TSV is a concrete example
of :ref:`concepts-naming`.

What modeling writes
--------------------

Leave-one-subject-out ridge regression of ``rating`` on the band-power and ERDS
columns. Scores are per subject. The cohort correlation is tested against 200
within-subject permutations.

.. list-table::
   :header-rows: 1
   :widths: 42 58

   * - File
     - Contents
   * - ``example_model_scores.tsv``
     - Cohort correlation and the permutation *p*, then each subject's
       correlation. There is no interval: every held-out score comes from a
       model trained on the other subjects, so the scores are not independent.
   * - ``example_model_predictions.tsv``
     - Held-out prediction for every trial, with the fold index.

.. note::

   These numbers describe the simulation. Alpha power falls with stimulus
   intensity. ``rating`` includes noise that these EEG features do not explain.

The generator uses the explicit ``ridge_pipeline`` and ``ridge_grid`` API.
The current modeling recipe starter instead uses training-scaled Ridge; the
saved example is not a benchmark of that starter. Its calls are in
``examples/make_examples.py``; see :doc:`/guides/modeling` for current workflows.
