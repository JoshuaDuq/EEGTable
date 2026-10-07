Tutorials
=========

.. rst-class:: hero-lede

   Worked analyses on public EEG and reproducible simulations. Each one runs
   whenever the documentation is built, so its numbers and figures come from
   the current code.

Start with **Resting Alpha Power** to learn how MNE epochs become labelled
feature tables. **Morlet Power and Support** verifies time-frequency window
reduction and distinguishes temporal support from numerical coverage.
**ERP Measurements** adds baseline correction, trial rejection,
and temporal measurements without losing original trial identities. **Beta
Bursts** and **Phase Consistency** use known simulated signals to explain what
their measures recover. **Sensor Connectivity** compares shared instantaneous
signals with phase lags and checks the estimates against MNE-Connectivity.
**Single-Trial ERD** applies a recipe to public movement data. **Session
Reliability** distinguishes agreement from consistency after explicit trial
aggregation. **Held-Out Participants** combines extraction with nested
participant-disjoint model evaluation.

Each tutorial is a standalone Python script, with a downloadable notebook and
the figures and output from its execution. Familiarity with Python, NumPy, and
MNE epochs is assumed; :doc:`/concepts` introduces EEGTable's data containers.

Run locally
-----------

From the repository root, install the tutorial dependencies and run a script:

.. code-block:: bash

   python -m pip install -e ".[docs,model,connectivity]"
   python -i tutorials/plot_resting_alpha.py

The connectivity tutorial needs the ``connectivity`` extra, and the prediction
tutorial needs ``model``. All other tutorials run with ``docs`` alone. Optional
dependencies are required explicitly; a missing package stops the example.

``python -i`` keeps the session open to inspect variables and figures, as in
MNE's tutorials. The two public-data analyses download EEGBCI files on first
use and reuse MNE's cache thereafter. The seven simulation tutorials require
no data download and set explicit random seeds. Figures display convenient
units such as µV², while feature tables retain the units in their metadata.

Worked analyses
---------------
