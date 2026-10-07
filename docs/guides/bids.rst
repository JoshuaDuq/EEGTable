BIDS EEG ingestion
==================

``eegtable.bids`` discovers and reads raw EEG recordings through MNE-BIDS.
Install ``eegtable[bids]``, or ``eegtable[bids,preprocessing]`` for the complete
preprocessing workflow. Discovery selects BrainVision ``.vhdr``, EDF ``.edf``,
BDF ``.bdf``, and EEGLAB ``.set`` EEG recordings. The dataset must contain
``dataset_description.json`` and the required recording and channel sidecars.
This integration reads raw EEG and its metadata; it does not convert datasets
to BIDS or validate every requirement of the BIDS specification.

Explicit discovery and reading
------------------------------

.. code-block:: python

   from pathlib import Path
   from eegtable.bids import BIDSQuery, discover_bids, read_bids

   paths = discover_bids(BIDSQuery(
       root=Path("/data/study"),
       subjects=("01", "02"),
       tasks=("motor",),
       sessions=("a",),
   ))
   recording = read_bids(paths[0])
   print(recording.event_id)  # names and codes to use for annotation epochs

``BIDSQuery`` optionally selects ``subjects``, ``sessions``, ``tasks``,
``acquisitions``, and ``runs``. Supply nonempty tuples of unique, unprefixed
alphanumeric string labels:
``"01"``, not ``"sub-01"``. Discovery calls
``mne_bids.find_matching_paths`` with EEG datatype and suffix restrictions,
excludes sidecars and derivative directories, returns deterministic path order,
and fails when no recording matches. A supplied ``BIDSPath`` must identify root,
subject, task, EEG datatype, and EEG suffix. A filesystem path may instead be
read with an explicit root: ``read_bids(path, root=Path("/data/study"))``.

``BIDSRecording`` contains:

- ``raw``: the preloaded MNE recording, including channel types, bad channels,
  annotations, geometry, acquisition information, and mapped subject information.
- ``entities``: BIDS subject, session, task, acquisition, run, and other supplied
  recording entities, retaining string identities such as ``"01"``.
- ``participant``: the complete matching row from ``participants.tsv``, including
  custom fields MNE cannot store in ``raw.info``. It is ``None`` when that optional
  table is absent; a present table must identify the subject exactly once.
- ``events``: the complete events sidecar as a DataFrame, or ``None`` when absent.
- ``channels``: channel metadata in the order used by the returned recording.
- ``event_id``: MNE-BIDS' annotation-to-code mapping when an events sidecar is
  present; otherwise ``None``.
- ``provenance``: reader and version, root, recording path, entities, participant
  row, event rows, channels, canonical order, and hashes of the relevant sidecars.

Channel names, event categories and custom descriptors retain their text,
including leading zeros. Standard numeric fields (onset, duration, sample,
response time and age) are parsed explicitly; ``n/a`` becomes missing, which an
event's duration and response time may be but its onset may not. Convert other
measured quantities to numbers when using them; model design builders convert
numeric targets.

Channel correspondence
----------------------

The reader uses ``on_ch_mismatch="raise"``. A recording whose channel names do
not agree with ``channels.tsv`` fails. By default, the returned channels follow
the sidecar's order. Supply an explicit canonical permutation
to compare recordings in the same sensor order:

.. code-block:: python

   recording = read_bids(
       paths[0], canonical_channels=("C3", "C4", "P3", "P4")
   )

This order must contain exactly the existing channel set. Missing, extra, or
duplicate names fail. Canonicalization reorders every named channel and its
metadata together; channel selection, renaming, dropping, and interpolation are
explicit preprocessing settings.

Preprocessing in memory
-----------------------

.. code-block:: python

   from eegtable.bids import preprocess_bids
   from eegtable.preprocessing.config import (
       EventEpochSettings, EventSettings, ProcessingSettings,
   )

   settings = ProcessingSettings(EventEpochSettings(
       events=EventSettings("annotations", {"left": 1, "right": 2}),
       tmin=-0.2,
       tmax=0.8,
   ))
   result = preprocess_bids(recording, settings)

The example assumes ``left`` and ``right`` are present in
``recording.event_id``. Use the mapping for the recording being read rather
than guessing codes from ``trial_type`` alone. ``preprocess_bids`` returns
epochs, an event ledger, and provenance without writing files; it uses the same
explicit ``decisions`` contract as :func:`eegtable.preprocessing.preprocess`.

The same numerical preprocessing stages handle ordinary and BIDS recordings.
BIDS event metadata is matched to the original acquisition samples. Annotation
identity disambiguates multiple sidecar events at the same sample. Matching
uses the original samples before applying event-delay correction, cropping,
rejection, or resampling. The optional sidecar ``sample`` column must agree with
onset on the acquisition sampling grid. Every selected event needs exactly one matching
metadata row; ambiguous duplicates fail. An unselected event occurring at the
same sample does not replace a selected trial's metadata.
MNE-BIDS may name annotations ``trial_type/value`` when one category contains
several event values. Use those names from ``recording.event_id``; the original
event columns remain in the aligned metadata.

Epoch metadata carries the event columns and ``bids_event_row`` (the original
events.tsv row). It also carries ``subject_id`` (for example ``sub-01``), present
``session``, ``task``, and ``run`` entities, every entity as ``bids_<entity>``, and
participant fields as ``participant_<field>``. The participant ID is already
represented by ``subject_id``. Duplicate column names fail. An explicitly supplied
``epochs.metadata`` table may add distinct columns with one row per selected
event. The event ledger contains the same metadata alongside retained rows and
drop reasons.

Fixed-length epochs carry recording and participant descriptors. They do not
invent an alignment between task event metadata and fixed windows. An events
sidecar may be absent for this mode, and that absence remains explicit in the
recording object. Event-locked BIDS epochs require the events sidecar.

Recipe and checkpoint workflow
-------------------------------

Enable BIDS ingestion explicitly with ``input.kind: bids``:

.. code-block:: yaml

   input:
     kind: bids
     root: /data/study
     subjects: ["01", "02"]
     sessions: ["a"]
     tasks: [motor]
     runs: ["01"]
     canonical_channels: [C3, C4, P3, P4]
   output:
     directory: preprocessed
   workflow:
     raw_review: required
     epoch_review: optional
   epochs:
     kind: events
     tmin: -0.2
     tmax: 0.8
     events:
       source: annotations
       event_id: {left: 1, right: 2}

Relative roots resolve against the recipe directory. Entity filters and
``canonical_channels`` are optional. Quote numerical labels so YAML preserves
their strings and leading zeros. ``input.path`` and glob ``input.pattern`` are
not BIDS-mode fields. The ordinary path/root input mode does not interpret BIDS
entities from filenames. BIDS output names are derived per recording and the
subject/session directory structure is mirrored below ``output.directory``.

Run the normal commands:

.. code-block:: console

   eegtable preprocess check preprocessing.yaml
   eegtable preprocess run preprocessing.yaml

The check command validates the BIDS sidecars and event sample alignment,
alongside the normal ``load``, ``prepare``, and ``events`` checks. Later-stage
and review requirements follow :doc:`preprocessing`.
The checkpointed workflow persists BIDS metadata in its state and final epochs.
Load-checkpoint identities include canonical order, the MNE-BIDS version, and
the relevant sidecar hashes: recording JSON, channels, events TSV/JSON,
participants TSV/JSON, dataset description, geometry, and scans when present.
Changing a sidecar invalidates the load checkpoint. Reset that recording from
``load``, then rerun and review its new checkpoints:

.. code-block:: console

   eegtable preprocess reset preprocessing.yaml --from load --recording LABEL
   eegtable preprocess run preprocessing.yaml --recording LABEL

BIDS checkpoint identities include absolute root and recording paths, so moving
the BIDS source tree also requires resetting the load checkpoint. The workflow
does not modify source recordings or sidecars. Exports mirror the recording
tree but use EEGTable's native FIF/TSV/JSON bundle; they are not a validated BIDS
derivative.

MNE-BIDS examples
-----------------

The integration follows the official `Read BIDS datasets example
<https://mne.tools/mne-bids/stable/auto_examples/read_bids_datasets.html>`_
and delegates matching and reading to the documented
`find_matching_paths
<https://mne.tools/mne-bids/stable/generated/mne_bids.find_matching_paths.html>`_
and `read_raw_bids
<https://mne.tools/mne-bids/stable/generated/mne_bids.read_raw_bids.html>`_ APIs.
MNE-BIDS can report custom participant columns that it cannot map to MNE Info;
these columns are preserved separately in ``BIDSRecording.participant`` and
epoch descriptors.
