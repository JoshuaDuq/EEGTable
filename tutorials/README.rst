Tutorials
=========

.. rst-class:: hero-lede

   Worked analyses on public EEG and reproducible simulations. Each one runs
   whenever the documentation is built, so its numbers and figures come from
   the current code.

Begin with :doc:`resting alpha power <plot_resting_alpha>` to turn MNE epochs
into labelled tables. :doc:`Morlet support <plot_morlet_support>`,
:doc:`aperiodic backgrounds <plot_aperiodic_background>`, and
:doc:`IRASA <plot_irasa>` explain different spectral estimands and their limits.

For waveform measurements, follow :doc:`ERPs <plot_erp_measurements>`,
:doc:`beta bursts <plot_beta_bursts>`, :doc:`cycle shapes <plot_cycle_waveforms>`,
:doc:`signal complexity <plot_signal_complexity>`, and the public-data
:doc:`movement ERD analysis <plot_motor_erds>`. Compare
:doc:`phase consistency <plot_phase_consistency>` with
:doc:`sensor connectivity <plot_sensor_connectivity>` before interpreting
synchronization estimates.

For repeated measurements and learned features, use
:doc:`session reliability <plot_session_reliability>`,
:doc:`frozen microstate templates <plot_microstate_templates>`,
:doc:`CSP decoding <plot_csp_decoding>`, and
:doc:`held-out participant modeling <plot_grouped_modeling>`. These tutorials
make aggregation, training partitions, and the sample unit explicit.

Each tutorial is a standalone Python script, with a downloadable notebook and
the figures and output from its execution. Familiarity with Python, NumPy, and
MNE epochs is assumed; :doc:`/concepts` introduces EEGTable's data containers.

Run locally
-----------

From the repository root, install the tutorial dependencies and run a script:

.. code-block:: bash

   python -m pip install -e ".[docs,model,connectivity,microstates,irasa,cycles]"
   python -i tutorials/plot_resting_alpha.py

The connectivity tutorial needs ``connectivity``, the IRASA tutorial needs
``irasa``, cycle waveforms need ``cycles``, and microstate templates need
``microstates``. CSP and prediction use ``model``. The other tutorials run with
``docs`` alone. Optional dependencies are required explicitly; a missing
package stops the example.

``python -i`` keeps the session open to inspect variables and figures, as in
MNE's tutorials. The two public-data analyses download EEGBCI files on first
use and reuse MNE's cache thereafter. The other tutorials use simulations or
analytic spectra, require no data download, and fix randomness where used.
Figures display convenient units such as µV², while feature tables retain the
units in their metadata.

Worked analyses
---------------
