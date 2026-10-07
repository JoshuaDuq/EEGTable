Preprocessing
=============

.. raw:: html

   <p class="hero-lede">
     Turn raw recordings into epochs from one YAML recipe. Each stage of each
     recording is a checkpoint. The feature runner reads the exported FIF and
     does not call this package.
   </p>

Install the ``preprocessing`` extra.

- **Automatic methods**: PyPREP, autoreject, ICLabel, and Picard need
  ``preprocessing-auto``.
- **Viewers**: ``review`` and ``inspect`` open MNE viewers when
  ``preprocessing-gui`` is installed.
- **Suffixes**: ``.fif``, ``.fif.gz``, ``.edf``, ``.bdf``, ``.vhdr``, and ``.set``.

Optional stages run only when enabled by their settings: for example,
``artifact: null`` skips artifact correction, and ``filter.notch_freqs: []``
skips the notch. ``eegtable preprocess steps`` lists the complete stage graph
and the stages this recipe enables. Amplitudes are volts, times are seconds,
and frequencies are hertz. BIDS input is described in :doc:`bids`.

Quick Start
-----------

``python -m eegtable preprocess`` is the same command as ``eegtable preprocess``.

Create the Recipe
~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python -m pip install -e ".[preprocessing]"
   eegtable preprocess init preprocessing.yaml --mode events

- ``--mode resting`` writes ``epochs.kind: fixed`` and ``duration: 2.0`` instead of
  an event block.
- ``init`` will not replace an existing file.

Edit the recipe before ``check``. ``init`` writes placeholders.

- **input.path**: ``recording_raw.fif``. Point it at the recording, or
  replace it with ``input.root`` and ``input.pattern`` to select a cohort and
  delete ``output.name``.
- **epochs.events**: ``source: annotations`` and ``event_id: {stimulus: 1}``.
  The names and codes have to be annotations on the recordings. Use
  ``source: stim`` or ``source: file`` when they are not.
- **workflow**: ``raw_review: required``. Keep it to review each recording
  yourself, or set ``suggested`` to save the detectors' verdict and continue.
- **channels.projections**: ``error``. A FIF with inactive projectors stops
  ``check`` until this is ``apply`` or ``discard-inactive``.

Check and Run
~~~~~~~~~~~~~

.. code-block:: bash

   eegtable preprocess check preprocessing.yaml
   eegtable preprocess run preprocessing.yaml

- ``check`` reads every recording and runs ``load``, ``prepare``, and ``events``.
  It writes nothing.
- ``run`` processes the recordings in order. A failure in one is reported and
  the next one starts.
- With ``raw_review: required``, each recording stops at the raw gate. The run
  prints the command that continues it and exits 3:

.. code-block:: text

   [1/2] sub-01
         ✓ awaiting review-raw · eegtable preprocess review preprocessing.yaml --recording sub-01 raw

Review the Raw Gate
~~~~~~~~~~~~~~~~~~~

- **With a display**: install ``eegtable[preprocessing-gui]`` and run that
  printed command. It opens the Qt browser. The decision is saved when the
  dialog is accepted.
- **Without a display**: fill in the pending file the run wrote,
  ``<bundle directory>/.preprocessing/<name>/decisions/review-raw.pending.yaml``.
  Each field is explained in a comment above it; replace every ``null`` and
  leave ``parent_id`` as written. Then run the same ``review`` command: it reads
  the filled file.
- **Cohorts**: without ``--recording``, ``review`` walks every recording
  that awaits that gate.

.. code-block:: bash

   eegtable preprocess review preprocessing.yaml raw
   eegtable preprocess run preprocessing.yaml

That second ``run`` exports.

- The init recipe has ``artifact: null`` and ``epoch_review: optional``, so
  there is no second gate.
- ``optional`` and ``disabled`` both skip epoch review. Only
  ``epoch_review: required`` stops at ``review-epochs``.
- An ``artifact`` block adds ``review-artifact``, governed by
  ``artifact_review``, with the same exit code and the same pending-file
  pattern.
- :file:`examples/preprocessing.yaml` is a cohort recipe with ICA and
  both gates set to ``suggested``, so it runs to export unattended.

Status and Resuming
~~~~~~~~~~~~~~~~~~~

When every recording is exported, ``run`` prints the ``eegtable init`` command
and the ``inputs.root`` to set.

- **status on a cohort**: one line per recording (``exported``,
  ``awaiting <stage>``, ``stale at <stage>``, or how many stages are done) and
  the next command to run.
- **status on one recording**: every stage with its state.
- **--recording LABEL**: limits a command to one recording. On a cohort it is
  required for ``step``, ``next``, ``inspect``, and ``reset``.
  Labels are the derived export names when unique, otherwise the relative
  bundle paths shown by ``status``.

A later ``run`` reuses a checkpoint whose recipe and parents still match.

- **Reuse check**: only the checkpoint's identity is checked. A payload is read
  and hashed when a stage consumes it.
- **status --verify**: re-reads and verifies every completed checkpoint for
  all selected recordings, in either text or JSON mode. ``--recording`` limits it.
- **Source recording**: read and fingerprinted once per recording per ``run``.
  ``status`` uses the saved source fingerprint; even ``--verify`` verifies
  checkpoint payloads rather than rereading the acquisition recording. A later
  ``run`` detects changes to that recording and refuses stale checkpoints.
  The fingerprint includes ``info["description"]``, whose original value is
  preserved in provenance, so corrected acquisition notes also require a reset.
- **External inputs**: montage, event, and metadata files are hashed when
  checking checkpoint identities. Editing a file in place makes its stage
  (``prepare`` for montages, ``events`` for events or metadata) and dependent
  checkpoints stale; reset from that stage before running again. Missing
  files raise an error. Rewriting identical contents preserves reuse.
- **Scientific packages**: PyPREP, autoreject, scikit-learn, Picard, ICLabel, and ONNX Runtime
  versions enter the identities of the enabled stages that use them. Upgrading
  one of these packages makes that stage and its dependent checkpoints stale.
- **Corrections**: an eegtable fix that changes what a stage computes makes
  that stage and its dependent checkpoints stale in the recipes it affects.
- **Missing acquisition reference**: ``reference.add_channels`` belongs to
  ``artifact-reference`` when artifact fitting re-references the data, so
  changing those electrodes also invalidates that reference and its fitted
  artifact model. Restored electrode names are recorded once in provenance.
- **Pending review file**: reviewer edits are kept while its ``parent_id``
  still matches the current parent. A changed parent requires a new review.
- **next**: runs one pending stage.
- **step STAGE**: runs that stage and requires its parents.
- **reset --from STAGE**: retires that stage and every stage that depends on
  it. Payloads stay on disk.

Command Reference
~~~~~~~~~~~~~~~~~

.. code-block:: text

   eegtable preprocess init CONFIG [--mode events|resting]
   eegtable preprocess check CONFIG [--recording LABEL]
   eegtable preprocess steps CONFIG
   eegtable preprocess status CONFIG [--recording LABEL] [--verify] [--json]
   eegtable preprocess step CONFIG STAGE --recording LABEL [--n-jobs N] [--overwrite]
   eegtable preprocess next CONFIG --recording LABEL [--n-jobs N] [--overwrite]
   eegtable preprocess run CONFIG [--until STAGE] [--recording LABEL] [--n-jobs N] [--overwrite] [--progress-json]
   eegtable preprocess inspect CONFIG STAGE --recording LABEL [--report | --json]
   eegtable preprocess review CONFIG raw|artifact|epochs [--recording LABEL] [--decisions FILE | --suggested]
   eegtable preprocess reset CONFIG --from STAGE --recording LABEL

``--recording`` is optional when the recipe selects one recording. Every
command and flag has ``--help``.

.. list-table::
   :header-rows: 1
   :widths: 12 88

   * - Exit
     - Meaning
   * - 0
     - Every selected recording reached the requested stage, or ``check`` passed.
   * - 1
     - At least one recording failed. The others still ran. Also an I/O failure.
   * - 2
     - The recipe or a prerequisite is wrong before any recording ran.
   * - 3
     - No recording failed and at least one awaits a review.

- ``--n-jobs``: passed to filtering and resampling.
- ``--overwrite``: lets ``export`` replace an existing bundle and remove its
  files the new one lacks. It is needed to republish an export that already
  matches the recipe, and to export after a recipe change and ``reset``, since
  the previous bundle is still in place. A deleted bundle is republished
  without it. A stale checkpoint is removed with ``reset``, not with
  ``--overwrite``.
- ``--progress-json``: writes one JSON event per line in the feature runner's
  format, with each recording as a subject.

Python
~~~~~~

- :func:`eegtable.preprocessing.load_recipe` returns one configuration per
  recording, keyed by label.
- :func:`eegtable.preprocessing.load_config` returns the single configuration of
  a one-recording recipe.
- The rest of the API is per recording.

.. code-block:: python

   from eegtable.preprocessing import load_recipe, open_workflow, run_until

   for label, config in load_recipe("preprocessing.yaml").items():
       outcome = run_until(open_workflow(config), "review-raw")
       print(label, outcome.state, outcome.next_action)

For an in-memory recording, :func:`eegtable.preprocessing.preprocess` takes
``ProcessingSettings`` and returns a ``PreprocessingResult`` with ``epochs``,
the event ledger, provenance, and an optional autoreject repair ledger. It
copies the input and writes no checkpoints or exports.

.. code-block:: python

   from eegtable.preprocessing import ProcessingSettings, preprocess
   from eegtable.preprocessing.config import EventEpochSettings, EventSettings

   settings = ProcessingSettings(epochs=EventEpochSettings(
       events=EventSettings(source="annotations", event_id={"stimulus": 1}),
       tmin=-0.5,
       tmax=1.5,
   ))
   result = preprocess(raw, settings)

This function has no unattended review policy. Raw and epoch review run only
when ``decisions`` contains ``"review-raw"`` or ``"review-epochs"``. An enabled
artifact correction requires an explicit ``"review-artifact"`` decision with
the fitted model's ``fit_id``. Use the separate fit/review/apply numerical
functions when the decision depends on inspecting that fit.

The Recipe
----------

A recipe is YAML.

- Relative paths resolve against the recipe file.
- Unknown keys are errors.
- Duplicate keys are errors.

.. code-block:: yaml

   input:
     root: sourcedata
     pattern: "sub-*/eeg/*_task-example_eeg.vhdr"
   output:
     directory: preprocessed
   workflow:
     raw_review: required
     artifact_review: suggested
     epoch_review: optional
   filter:
     l_freq: 0.1
     h_freq: 40.0
     notch_freqs: [60.0]
   epochs:
     kind: events
     events: {source: annotations, event_id: {stimulus: 1}}
     tmin: -0.5
     tmax: 1.5

``init --mode resting`` writes ``epochs.kind: fixed`` with ``duration: 2.0``
instead of the event block. A filled study file is
:file:`examples/preprocessing.yaml`.

Sections
~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Section
     - Keys
   * - ``input``
     - - ``kind: files`` (default): ``path``, one recording. Or ``root`` and
         ``pattern``, a directory and a glob below it; ``pattern`` has no default.
       - Exactly one of ``path`` and ``root``.
       - Every match must be a recording (``.fif``, ``.fif.gz``, ``.edf``,
         ``.bdf``, ``.vhdr``, ``.set``); another suffix is an error.
       - Hidden files and directories are skipped.
       - ``kind: bids``: ``root`` plus optional entity filters and
         ``canonical_channels``; see :doc:`bids`.
   * - ``output``
     - - ``directory``, and with ``path`` an optional ``name``.
       - Without ``name``, the export is named after the file with its suffix and
         one trailing ``_raw`` or ``_eeg`` removed.
       - With ``root``, ``name`` is not allowed and the tree below ``root`` is
         mirrored under ``directory``.
       - Two recordings that would share a bundle are an error.
   * - ``workflow``
     - - ``raw_review`` is ``required`` (default), ``suggested``, or ``disabled``.
       - ``artifact_review`` is ``required`` (default) or ``suggested``.
       - ``epoch_review`` is ``required``, ``optional``, or ``disabled``.
       - ``suggested`` saves the detectors' verdict as the decision when the
         gate is reached without one, bound to the same parent checkpoint as a
         saved review.
       - ``raw_review: disabled`` leaves detector candidates unapplied.
         Existing bad labels and explicitly configured ``channels.bads`` and
         ``annotations.bad_spans`` still apply.
   * - ``channels``
     - - ``rename``, ``types``, ``drop``, ``bads``, ``interpolate_bads``.
       - ``montage``: a standard name, or ``{path: ...}`` to a digitized FIF or
         any electrode file ``mne.channels.read_custom_montage`` reads,
         positions taken as written.
       - ``bipolar``: ``name``, ``anode``, ``cathode``, ``type`` of ``eog`` or
         ``ecg``.
       - ``projections``: ``error``, ``apply``, ``discard-inactive``.
   * - ``crop``
     - ``tmin``, ``tmax``, seconds from the start of the file. ``null`` skips it.
   * - ``annotations``
     - ``bad_spans`` (``onset``, ``duration``, ``description`` starting with
       ``BAD``), ``amplitude``, ``breaks``, ``muscle``. A null detector is
       skipped.
   * - ``bad_channels``
     - PyPREP.

       - ``method: pyprep``.
       - ``methods`` among ``flat``, ``deviation``, ``correlation``,
         ``high_frequency``, ``snr``. ``snr`` requires ``correlation`` and
         ``high_frequency``.
       - ``ransac``, ``random_state``.
       - ``repeats``: RANSAC draws that vote; a channel is a candidate when a
         strict majority flags it.
       - ``notch_freqs``: a notch on the diagnostic copy only, as PREP does
         before its deviation test.
   * - ``bridges``
     - ``true`` records bridged pairs and their median electrical distance as
       raw-review evidence. ``false`` skips it.
   * - ``stimulation``
     - ``event_ids``, ``channels``, ``tmin``, ``tmax``, ``mode``
       (``linear``, ``window``, ``constant``). ``constant`` requires ``baseline``.
   * - ``filter``
     - ``l_freq``, ``h_freq``, ``notch_freqs``. A null cutoff or an empty notch
       list skips that stage. ``l_freq`` must be below ``h_freq``.
   * - ``artifact``
     - ``method`` of ``ica``, ``ssp``, or ``regression``, the matching settings
       block, and ``reference`` (``average``, a list of EEG names, or ``null``).
       Regression requires a reference. ``null`` skips fitting, review, and apply.
   * - ``epochs``
     - - ``kind: events`` with ``events``, ``tmin``, ``tmax``. Or ``kind: fixed``
         with ``duration``, ``overlap``, ``start``, ``stop``.
       - Both accept ``padding``, ``baseline``, and ``detrend`` (``constant`` or
         ``linear``).
       - Event epochs also accept ``metadata``, a TSV with one row per input
         event, in the resolved event array's order before epoch rejection.
       - ``metadata`` and ``events.path`` accept ``{name}`` (the export name)
         and ``{parent}`` (the recording's directory), resolved per recording.
   * - ``epochs.events``
     - - ``source`` of ``annotations``, ``stim``, or ``file``.
       - ``event_id`` maps names to codes.
       - ``stim`` also takes ``stim_channel``, ``shortest_event``, and
         ``min_duration``. ``file`` takes ``path``.
       - A ``file`` source is a native MNE event file read by
         ``mne.read_events``, rather than a BIDS ``events.tsv`` sidecar.
         Event samples must be unique, strictly increasing, and inside the
         original recording before and after delay correction.
       - ``delay`` subtracts ``round(delay * sfreq)`` samples from every event,
         so a positive value moves events earlier.
   * - ``rejection``
     - - ``method: thresholds`` with ``reject`` and ``flat`` in volts, and an
         optional ``tmin`` and ``tmax`` inside the analysis window.
       - Or ``method: autoreject`` with ``n_interpolate``, ``consensus``,
         ``cv``, and ``random_state``.
   * - ``reference``
     - Final EEG reference. ``channels: average`` or a list of good EEG names.
       ``add_channels`` restores missing acquisition-reference electrodes as
       zeros before the first re-reference, including ``artifact.reference``.
       The configured montage is reapplied to locate the restored electrodes.
       Sources already marked as custom referenced cannot restore missing
       acquisition electrodes. Bipolar channels from ``channels.bipolar`` do
       not set that flag.
       ``null`` leaves the reference unchanged.
   * - ``sampling``
     - - ``method: decimate`` with integer ``factor``, which requires
         ``filter.h_freq``.
       - Or ``method: resample`` with ``sfreq`` and ``padding``. The padding
         must equal ``epochs.padding``.

Detector Settings
~~~~~~~~~~~~~~~~~

- ``annotations.amplitude`` takes ``peak`` and ``flat`` in volts for ``eeg``
  only, plus ``bad_percent`` and ``min_duration``.
- ``annotations.breaks`` takes ``min_break_duration``,
  ``t_start_after_previous``, and ``t_stop_before_next``, and requires event
  epochs.
- ``annotations.muscle`` takes ``filter_freq``, ``threshold``, and
  ``min_length_good``. Like the amplitude detector, it scores only the good
  EEG channels.

Artifact Settings
~~~~~~~~~~~~~~~~~

**ICA** settings:

- ``method``: ``fastica`` (the default), ``infomax`` with the extended update,
  or ``picard`` with ``ortho: false`` and ``extended: true``.
- ``l_freq``: default 1 Hz, on a copy.
- ``n_components``, ``random_state``, ``max_iter``, ``reject``, ``flat``,
  ``tstep``, ``eog_channels``, ``ecg_channel``, and ``iclabel``.
- ``max_iter``: a ``fastica`` or ``picard`` fit that reaches it fails as
  unconverged. MNE's ``infomax`` reports the limit whether or not it converged,
  so an ``infomax`` fit is not checked.

**ICLabel**: ``iclabel`` takes ``threshold`` (default 0.8) and ``keep``
(default ``[brain, other]``). It runs ICLabel on the training copy, without
the BAD spans the fit skipped, using the ONNX backend installed by
``preprocessing-auto`` and requires:

- ``infomax`` or ``picard``.
- ``artifact.reference: average``.
- Training data filtered to exactly 1–100 Hz, matching
  `MNE-ICALabel's documented requirements
  <https://mne.tools/mne-icalabel/stable/generated/api/mne_icalabel.iclabel.iclabel_label_components.html>`_.
  A source or analysis filter that removes more of this band is rejected;
  filtering cannot restore frequencies already removed.

**Fit evidence**: the fit checkpoint records every detector's verdict.

- ``scores`` per channel.
- ``iclabel`` labels and class probabilities.
- ``suggested`` per detector.
- ``suggested_exclude``: the union of the components MNE's EOG and ECG
  detectors flag and the components whose winning ICLabel class is outside
  ``keep`` at or above ``threshold``.

The fit excludes nothing.

**SSP** settings: ``n_eeg``, ``eog_channels`` or ``ecg_channel``, ``l_freq``,
``h_freq``, ``tmin``, ``tmax``, and ``reject``.

**Regression** settings: ``eog_channels``, ``tstep``, ``reject``, and ``flat``.

Stages
------

Stages run in this order. A disabled stage is omitted and its children read the
previous enabled parent. ``apply-artifact`` waits for both ``epoch`` and
``review-artifact``.

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Stage
     - What it does
   * - ``load``
     - Read the file; validate sampling and reject non-finite EEG/EOG/ECG data.
   * - ``prepare``
     - Rename and type channels, derive bipolar EOG or ECG, drop channels, set
       the montage, merge bad labels, and apply the projector policy.
   * - ``events``
     - Build the event table on the acquisition grid.
   * - ``crop-raw``
     - Keep ``[tmin, tmax]``. Event codes are unchanged.
   * - ``annotate``
     - Add the manual BAD spans, then amplitude, break, and muscle candidates.
   * - ``detect-bads``
     - PyPREP channel candidates and, when enabled, bridged pairs.
   * - ``review-raw``
     - Apply the saved bad channels and BAD spans.
   * - ``repair-stim``
     - ``mne.preprocessing.fix_stim_artifact`` on the declared windows.
   * - ``notch``, ``filter``
     - Zero-phase Hamming FIR (``firwin``) on physiology channels. BAD and edge
       annotations are skipped. A clean segment shorter than the filter is an error.
   * - ``artifact-reference``
     - Re-reference the continuous data for the artifact fit. Epochs are cut
       from this re-referenced data, so it carries through unless the final
       ``reference`` stage changes it.
   * - ``fit-artifact``, ``review-artifact``
     - Fit ICA, EEG SSP, or EOG regression. Review stores the components to
       exclude, the projectors to apply, or whether to apply the regression.
   * - ``epoch``
     - ``mne.Epochs`` with ``baseline=None``, ``proj=False``, and ``decim=1``.
       A BAD annotation anywhere in the padded epoch drops it. Padding lies
       outside the analysis window.
       From here on a checkpoint holds epochs, not the continuous data.
   * - ``apply-artifact``
     - Apply the reviewed operator. The fit is not repeated.
   * - ``fit-rejection``
     - Fit autoreject on the analysis window. Threshold rejection skips this stage.
   * - ``reject``
     - Apply peak-to-peak and flat thresholds, or the fitted autoreject operator,
       on the analysis window.
   * - ``review-epochs``
     - Drop epochs by original event row.
   * - ``interpolate``
     - Spline interpolation of bad EEG. Other bad labels stay.
   * - ``reference``
     - Average or named EEG reference.
   * - ``resample``
     - Integer decimation, after the low-pass guard, or polyphase resampling of
       the padded epochs. The output grid must contain the epoch origin.
   * - ``crop-epochs``, ``detrend``, ``baseline``
     - Remove padding, then optional SciPy detrending of EEG, then an optional
       baseline on the final grid.
   * - ``report``, ``export``
     - ``report`` validates the final epochs and ledger. ``export`` writes the
       HTML report and the files below.

Timing and Sampling
~~~~~~~~~~~~~~~~~~~

- **Event samples**: stay on the acquisition grid. The events table has the
  original sample, the delay-corrected sample, and ``event_sample_sfreq``.
- **final_sfreq**: in the manifest, the epoch rate after resampling.
- **Fixed-length epochs**: an epoch of ``duration`` seconds has
  ``round(duration * sfreq)`` samples, so its last time is
  ``(n_samples - 1) / sfreq``. Duration and the stride ``duration - overlap``
  must align to acquisition samples; a duration must contain at least two samples.
- **Decimation**: selecting every ``factor``-th sample does not itself filter.
  The configured low-pass cutoff must be at most one third of the target rate,
  with its full transition ending below the new Nyquist frequency.
- **Polyphase resampling**: downsampling only. The padded epoch must map to
  an integer number of output samples, its starting time must lie on the new
  grid, and padding must cover the antialiasing FIR's half-support. Adjust
  padding or the target rate when ``check`` rejects the grid.
- **Final bounds**: cropping uses available samples without extrapolation.
  The final endpoints may differ from the requested bounds by up to one output
  sample; the manifest records the actual bounds and rate.

The sampling choices follow MNE's `Filtering and resampling tutorial
<https://mne.tools/stable/auto_tutorials/preprocessing/30_filtering_resampling.html>`_.
EEGTable adds explicit guards for the low-pass transition, epoch grid, and
resampling padding.

Review
------

``run`` returns at the first enabled review that has no saved decision and
no ``suggested`` policy. The quick start above is the raw gate; the other gates
use the same steps and the fields below.

Filling a Pending File
~~~~~~~~~~~~~~~~~~~~~~

The pending file for a gate is
``<bundle directory>/.preprocessing/<name>/decisions/review-<target>.pending.yaml``,
YAML with a comment above each field.

- Leave ``parent_id`` and, for artifact review, ``fit_id`` as written.
- Replace every ``null``.
- Run ``review`` for that gate: a filled pending file is used before the viewer
  is opened.
- ``--decisions FILE`` reads another file instead.
- The Qt viewer (``preprocessing-gui``) writes the same decision when the dialog
  is accepted.

Decision Fields
~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Target
     - Decision
   * - ``raw``
     - - ``bads`` replaces ``info["bads"]``. ``[]`` clears it.
       - ``spans`` are appended. ``[]`` appends none.
       - Each span has ``onset``, ``duration``, and a ``description`` that
         starts with ``BAD``.
       - ``onset`` is seconds from the start of the recording, like
         ``annotations.bad_spans``, also after ``crop-raw``.
       - The viewer opens with the suggested bad channels and spans already
         marked and saves the spans added to them.
   * - ``artifact``, ICA
     - ``fit_id``. ``exclude``, component indices. ``[]`` excludes none.
   * - ``artifact``, SSP
     - ``fit_id``. ``include``, projector indices to apply.
   * - ``artifact``, regression
     - ``fit_id``. ``apply``, true or false.
   * - ``epochs``
     - ``exclude``, original event rows to drop.

Suggested Decisions
~~~~~~~~~~~~~~~~~~~

``review --suggested``, and a ``suggested`` policy in ``workflow``, save the
detectors' own verdict as the decision without a viewer or a file.

- ``raw``: the channels already marked bad plus the amplitude and PyPREP
  candidates, and the detected BAD spans.
- ``artifact``: the ICA ``suggested_exclude`` list, every SSP projector, or
  applying the regression.
- ``epochs``: there is no suggestion.

The saved decision is the same file a viewer or ``--decisions`` would write, so
provenance records the choice either way. A changed detector setting
invalidates it like any decision.

Keeping and Replacing Decisions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- A decision is kept. Replacing it requires ``reset --from`` that review stage.
- A new fit requires a new decision.
- ``parent_id`` has to match the checkpoint the decision was written for.

Inspecting a Checkpoint
~~~~~~~~~~~~~~~~~~~~~~~

``inspect STAGE`` opens that checkpoint in the browser. ``--report`` writes
``<stage>-<checkpoint id>-inspection.html`` inside the workspace and opens it.
It shows:

- the continuous data before epoching;
- the fitted operator at ``fit-artifact`` and ``review-artifact``;
- the epochs after.

Terminal Front End
~~~~~~~~~~~~~~~~~~

``tui/`` is an optional Go program for the same commands. The Python package
does not import it.

- **Recordings**: lists every recording and runs ``run`` with the log under the
  stage list.
- **Gates**: opens each gate as a checklist. The list starts ticked with the
  detectors' verdict. It shows PyPREP tests, ICLabel class and confidence, and
  peak-to-peak amplitude per epoch.
- **Decisions**: go through ``review --decisions``.
- **State**: comes from ``status --json`` and ``inspect STAGE --json``.
- **Viewer**: ``v`` opens the checkpoint in the MNE viewer when
  ``preprocessing-gui`` is installed.

.. code-block:: bash

   cd tui && go build -o eegtable-tui .     # Go 1.24+
   ./eegtable-tui preprocessing.yaml        # finds eegtable on PATH, or set EEGTABLE

**Keys**:

- ``Tab`` moves between recordings and stages.
- ``Enter`` runs the selected action.
- ``?`` lists the keys.
- ``l`` opens the log. ``PgUp`` and ``PgDn`` scroll. ``End`` jumps to the
  latest log line. ``Esc`` closes the log.
- ``Home`` and ``End`` also jump to the first and last row of a list.
- ``o`` sorts a review and keeps the same row selected.
- While a command is running, the header names it and other actions wait.

**JSON contract**: ``--json`` on ``status`` and ``inspect`` is a documented
contract for any front end.

- ``status --json`` lists each recording's stages and its ``next`` action
  (``review``, ``run`` or ``reset``).
- ``inspect review-raw --json`` returns the gate's ``items`` with ``suggested``
  flags and ``tags``, its ``parent`` checkpoint, and the ``parent_id`` a
  decision must echo.

Outputs
-------

Export writes into ``output.directory``. The manifest is published last, so a
directory without ``<name>_preprocessing.json`` has no finished export.

.. list-table::
   :header-rows: 1
   :widths: 36 64

   * - File
     - Contents
   * - ``<name>_epo.fif``
     - Double-precision epochs, with split FIF companions when needed.
       ``info["description"]`` names the manifest and provenance identity.
   * - ``<name>_events.tsv``
     - One row per resolved input event or fixed-window event, with
       ``original_row``, original and corrected samples, acquisition sample rate,
       ``retained``, ``epoch_row``, and ``drop_reason``. BIDS metadata is also retained.
   * - ``<name>_repairs.tsv``
     - Present when autoreject ran. Per-epoch channel repairs.
   * - ``<name>_report.html``
     - MNE report.
   * - ``<name>_preprocessing.json``
     - Provenance, stage order, settings, review decisions, and file hashes.
       Package versions enter the stage identities but are not listed.

Log and Checkpoints
~~~~~~~~~~~~~~~~~~~

- **Log**: ``run`` also keeps ``preprocess-run-<timestamp>.log`` at the common
  output root when MNE or a stage printed anything, and prints its path.
- **Checkpoints**: live in ``<bundle directory>/.preprocessing/<name>/``, where
  the bundle directory is ``output.directory`` plus the recording's path below
  ``input.root``. Hidden files the OS adds there, such as ``.DS_Store``, are
  ignored.
- **Handoff**: the feature runner skips that hidden directory. Point its
  ``inputs.root`` at ``output.directory`` and its pattern at ``**/*_epo.fif``;
  ``run`` prints that handoff when every recording is exported.

.. literalinclude:: ../../examples/preprocessing.yaml
   :language: yaml
   :caption: examples/preprocessing.yaml

Notes
-----

- Scalp EEG. There is no ASR, current-source density, source model, Maxwell
  filter, or EEG-fMRI gradient correction.
- ICA, SSP, EOG regression, and autoreject are fit on the recording being
  processed. A fit that must stay inside a training split uses the numerical
  functions in ``eegtable.preprocessing`` (:doc:`/api/preprocessing`) directly.
- Muscle and high-frequency detectors need the unfiltered recording to cover
  the band they measure.
- A notch removes a line and does not change the highpass or lowpass stored on
  the epochs. The feature runner's passband check does not see that hole.
- ``check`` runs ``load``, ``prepare``, and ``events`` in memory. Later stages
  are checked when they run.
- Padding can cause otherwise usable events near the recording boundary to
  be dropped. BAD annotations overlapping a padded epoch also drop it.
- Exports use a native FIF/TSV/JSON layout; they are not a validated BIDS
  derivative. :doc:`cohorts` explains the extraction and quality-report handoff.

The stage choices are grounded in MNE's official tutorials for `bad-channel
handling <https://mne.tools/stable/auto_tutorials/preprocessing/15_handling_bad_channels.html>`_
and `ICA artifact correction
<https://mne.tools/stable/auto_tutorials/preprocessing/40_artifact_correction_ica.html>`_.
In particular, ICA is fitted on a separately high-pass-filtered copy, then the
reviewed fit is applied to the analysis epochs.
