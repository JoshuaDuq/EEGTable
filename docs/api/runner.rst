Feature runner
==============

The runner reads a TOML recipe, finds the epochs files it selects, and writes
one feature table per recording; ``eegtable run`` executes it. The functions
below are the same workflow without the command line and are imported from
``eegtable.runner``. Recipe keys, outputs, and status semantics are in
:doc:`/guides/runner`.

.. code-block:: python

   from eegtable.runner import check, load_recipe, run, status

   recipe = load_recipe("recipe.toml")
   check(recipe, quick=True)
   outcome = run(recipe, resume=True, workers=4)
   states = status(recipe)

Configuration and discovery problems raise before any recording is computed.
A recording that fails during a run does not stop the others; its error stays
in the returned :class:`~eegtable.runner.RunResult`.

Recipes
-------

.. autofunction:: eegtable.runner.load_recipe

.. autoclass:: eegtable.runner.Recipe

.. autoexception:: eegtable.runner.RecipeError

Recipes on epochs in memory
---------------------------

:func:`eegtable.extract` applies a recipe to one recording's epochs without reading
or writing files, and returns what :func:`~eegtable.runner.run` would write for it.

.. autoclass:: eegtable.runner.RecordingFeatures
   :members:

Recordings
----------

.. autofunction:: eegtable.runner.discover

.. autoclass:: eegtable.runner.Recording
   :members:

Checking a recipe
-----------------

``check`` computes the first recording without writing anything and reads
every recording's header for the channels the recipe's ROIs and asymmetry
pairs name.

.. autofunction:: eegtable.runner.check

.. autoclass:: eegtable.runner.CheckReport

.. autoclass:: eegtable.runner.Trial

.. autoexception:: eegtable.runner.TrialError

Running
-------

.. autofunction:: eegtable.runner.run

.. autoclass:: eegtable.runner.RunResult
   :members:

.. autoclass:: eegtable.runner.RecordingResult
   :members:

.. autoexception:: eegtable.runner.RunError

Status
------

.. autofunction:: eegtable.runner.status

.. autoclass:: eegtable.runner.RecordingStatus
   :members:
