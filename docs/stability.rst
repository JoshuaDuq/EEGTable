Versions and Stability
======================

.. rst-class:: hero-lede

   What a version number promises, what renames a column, and how settled each
   part of EEGTable is.

Versions
--------

EEGTable follows semantic versioning. Before 1.0, a minor version may change
behaviour, and every such change is listed in the :doc:`changelog`. Record the
EEGTable version with each analysis; each output's provenance records it, with
the versions of its dependencies and a digest of the code that ran.

Column identity
---------------

A column's name ends in a digest of everything that defines it: the measure,
band, space, window and its bounds, normalization, unit, ROI membership and the
parameters of its computation. Only what can change a value enters it. Settings
that change how a computation runs, such as ``n_jobs``, never do.

A release that changes what enters an identity renames the affected columns.
Tables computed before it then do not stack with new ones under
``columns="identical"``. Such a change is always listed under **Changed** in
the changelog, and existing results already read as stale after any code change,
so ``eegtable run --overwrite`` brings a cohort back to one set of columns.

Files
-----

Feature bundles carry a schema number. Readers keep reading earlier schemas,
including bundles written under the package's former name, ``eegfeat``. A
bundle written by a newer release may not read in an older one.

How settled each part is
------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Part
     - Status
     - What that means
   * - Feature functions, containers, tables, :mod:`eegtable.io`
     - Stable
     - Changed only with a changelog entry; a renamed or removed name is
       deprecated for at least one minor release first.
   * - Runner and recipes (``eegtable run``, :func:`~eegtable.extract`)
     - Stable
     - Recipe keys are kept, or refused with a message that names the
       replacement.
   * - :mod:`eegtable.model` and model recipes
     - Evolving
     - Interfaces may change between minor releases, with changelog entries.
   * - :mod:`eegtable.preprocessing` and the ``tui/`` front end
     - Experimental
     - May change without deprecation; pin the version you use.

Supported versions
------------------

EEGTable needs Python 3.11 or later. CI tests the newest release of every
dependency, and the oldest releases ``pyproject.toml`` admits: NumPy 1.26, SciPy
1.11, pandas 2.0 and MNE-Python 1.10. A minimum rises when an older release no
longer works with the current scientific stack. MNE 1.8 and 1.9, for example,
cannot be imported with SciPy 1.15 or later.
