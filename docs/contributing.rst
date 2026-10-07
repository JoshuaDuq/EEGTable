Contributing
============

.. rst-class:: hero-lede

   Bug reports, fixes and new measures are welcome.

Report a bug, or ask for a measure, on the
`issue tracker <https://github.com/JoshuaDuq/EEGTable/issues>`__. A bug report
needs the shortest code that shows the problem, ideally on simulated data or a
public MNE dataset, and the versions in use. A measure request needs its formula
and references, so the result can be validated against them.

`CONTRIBUTING <https://github.com/JoshuaDuq/EEGTable/blob/main/CONTRIBUTING.md>`__
covers setting up, the checks a pull request must pass, the conventions code
follows here, and the steps for adding a measure. To use a measure of your own
without changing EEGTable, see :doc:`guides/extending`.

Everyone taking part follows the
`code of conduct <https://github.com/JoshuaDuq/EEGTable/blob/main/CODE_OF_CONDUCT.md>`__.

Writing tutorials
-----------------

Tutorial sources live in ``tutorials/plot_*.py``. Sphinx-Gallery builds the
rendered pages, downloadable scripts, and notebooks in ``docs/auto_tutorials``;
edit the source scripts rather than generated files.

Follow MNE's `narrative tutorial format
<https://mne.tools/stable/auto_tutorials/index.html>`__: start with a scientific
question and prerequisites, then use ``# %%`` blocks for data preparation,
extraction, figures, and interpretation. Keep every tutorial independently
runnable. Use public datasets with a citation and download requirements, or
label simulations clearly and fix their random seeds. Explain units, estimator
settings, boundary handling, and the sample unit. Include numerical checks
against a reference implementation or known ground truth when available.

Add the source filename to the learning order in ``docs/conf.py`` and link its
generated page in ``docs/index.rst``. Verify both direct execution and the gallery:

.. code-block:: bash

   python -i tutorials/plot_resting_alpha.py
   python -m ruff check tutorials docs/conf.py
   python -m black --check tutorials docs/conf.py
   MPLBACKEND=Agg sphinx-build -b html -W docs docs/_build/html

Every tutorial executes during the build, including previously cached scripts.
An execution error stops the build. Inspect the rendered figures and text as
well as the numerical output before submitting a tutorial.
