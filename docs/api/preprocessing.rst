Preprocessing
=============

Available with the ``preprocessing`` extra. The recipe, stage order, command-line
workflow, review files, sampling contracts, and outputs are in
:doc:`/guides/preprocessing`. BIDS ingestion is described in :doc:`/guides/bids`.

The checkpoint API operates on one recording's ``Workflow`` at a time.
``load_recipe`` selects a cohort and returns configurations by recording label;
``load_config`` requires exactly one recording. Numerical functions operate on
MNE objects and typed settings without creating checkpoint files.

Workflow
--------

.. autofunction:: eegtable.preprocessing.load_recipe

.. autofunction:: eegtable.preprocessing.load_config

.. autofunction:: eegtable.preprocessing.open_workflow

.. autofunction:: eegtable.preprocessing.list_steps

.. autofunction:: eegtable.preprocessing.run_step

.. autofunction:: eegtable.preprocessing.run_next

.. autofunction:: eegtable.preprocessing.run_until

.. autofunction:: eegtable.preprocessing.read_checkpoint

.. autofunction:: eegtable.preprocessing.reset_from

.. autoclass:: eegtable.preprocessing.execution.Workflow

.. autoclass:: eegtable.preprocessing.execution.StepStatus

.. autoclass:: eegtable.preprocessing.execution.StepResult

.. autoclass:: eegtable.preprocessing.execution.RunOutcome

.. autoclass:: eegtable.preprocessing.checkpoints.Checkpoint

In-memory pipeline
------------------

``preprocess`` copies its input and returns epochs, the event ledger,
provenance, and optional autoreject repairs. It writes no files. Raw and epoch
reviews run only when their decisions are supplied; artifact correction requires
an explicit reviewed fit. Use the fit/review/apply operations below to inspect
or fit an operator within a training split.

.. autofunction:: eegtable.preprocessing.preprocess

.. autoclass:: eegtable.preprocessing.pipeline.PreprocessingResult

Settings
--------

Import individual settings from ``eegtable.preprocessing.config``. Constructors
validate types and ranges; recording-dependent constraints are checked during
preparation and numerical execution. Paths in directly constructed
settings are ``pathlib.Path`` objects, and name sequences are tuples.

.. autoclass:: eegtable.preprocessing.PreprocessingConfig

.. autoclass:: eegtable.preprocessing.ProcessingSettings

Inputs, outputs, and review policies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: eegtable.preprocessing.config.InputSettings

.. autoclass:: eegtable.preprocessing.config.BIDSInputSettings

.. autoclass:: eegtable.preprocessing.config.OutputSettings

.. autoclass:: eegtable.preprocessing.config.WorkflowSettings

Channels and continuous data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: eegtable.preprocessing.config.ChannelSettings

.. autoclass:: eegtable.preprocessing.config.BipolarSettings

.. autoclass:: eegtable.preprocessing.config.ReferenceSettings

.. autoclass:: eegtable.preprocessing.config.CropSettings

.. autoclass:: eegtable.preprocessing.config.FilterSettings

.. autoclass:: eegtable.preprocessing.config.AnnotationSettings

.. autoclass:: eegtable.preprocessing.config.BadSpan

.. autoclass:: eegtable.preprocessing.config.AmplitudeSettings

.. autoclass:: eegtable.preprocessing.config.BreakSettings

.. autoclass:: eegtable.preprocessing.config.MuscleSettings

.. autoclass:: eegtable.preprocessing.config.BadChannelSettings

.. autoclass:: eegtable.preprocessing.config.StimulationSettings

Events, epochs, rejection, and sampling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: eegtable.preprocessing.config.EventSettings

.. autoclass:: eegtable.preprocessing.config.EventEpochSettings

.. autoclass:: eegtable.preprocessing.config.FixedEpochSettings

.. autoclass:: eegtable.preprocessing.config.ThresholdSettings

.. autoclass:: eegtable.preprocessing.config.AutoRejectSettings

.. autoclass:: eegtable.preprocessing.config.DecimationSettings

.. autoclass:: eegtable.preprocessing.config.ResamplingSettings

Artifact models
~~~~~~~~~~~~~~~

.. autoclass:: eegtable.preprocessing.config.ArtifactSettings

.. autoclass:: eegtable.preprocessing.config.ICASettings

.. autoclass:: eegtable.preprocessing.config.ICLabelSettings

.. autoclass:: eegtable.preprocessing.config.SSPSettings

.. autoclass:: eegtable.preprocessing.config.RegressionSettings

Numerical operations
--------------------

Raw and epoch transforms copy their inputs; output functions explicitly write
files. Review operations apply the decision passed by the caller. Detectors
return candidates, and fitted artifact models exclude no components before review.

.. autofunction:: eegtable.preprocessing.raw.prepare_channels

.. autofunction:: eegtable.preprocessing.events.resolve_events

.. autofunction:: eegtable.preprocessing.raw.crop_raw

.. autofunction:: eegtable.preprocessing.raw.annotate_raw

.. autofunction:: eegtable.preprocessing.quality.detect_annotations

.. autofunction:: eegtable.preprocessing.quality.detect_bad_channels

.. autofunction:: eegtable.preprocessing.quality.detect_bridges

.. autofunction:: eegtable.preprocessing.quality.apply_raw_review

.. autofunction:: eegtable.preprocessing.quality.repair_stimulation

.. autofunction:: eegtable.preprocessing.raw.notch_raw

.. autofunction:: eegtable.preprocessing.raw.filter_raw

.. autofunction:: eegtable.preprocessing.artifacts.reference_artifact_data

.. autofunction:: eegtable.preprocessing.ica.fit_ica

.. autofunction:: eegtable.preprocessing.artifacts.fit_ssp

.. autofunction:: eegtable.preprocessing.artifacts.fit_eog_regression

.. autofunction:: eegtable.preprocessing.artifacts.review_artifact

.. autoclass:: eegtable.preprocessing.artifacts.ArtifactModel

.. autoclass:: eegtable.preprocessing.artifacts.ReviewedArtifact

.. autofunction:: eegtable.preprocessing.epochs.make_epochs

.. autofunction:: eegtable.preprocessing.artifacts.apply_artifact

.. autofunction:: eegtable.preprocessing.rejection.fit_rejection

.. autofunction:: eegtable.preprocessing.rejection.reject_epochs

.. autofunction:: eegtable.preprocessing.rejection.apply_rejection

.. autofunction:: eegtable.preprocessing.rejection.apply_epoch_review

.. autofunction:: eegtable.preprocessing.epochs.interpolate_channels

.. autofunction:: eegtable.preprocessing.epochs.reference_epochs

.. autofunction:: eegtable.preprocessing.sampling.resample_epochs

.. autofunction:: eegtable.preprocessing.sampling.crop_epochs

.. autofunction:: eegtable.preprocessing.epochs.detrend_epochs

.. autofunction:: eegtable.preprocessing.epochs.baseline_epochs

.. autofunction:: eegtable.preprocessing.report.build_report

.. autofunction:: eegtable.preprocessing.report.build_checkpoint_report

.. autofunction:: eegtable.preprocessing.io.write_result

BIDS ingestion
--------------

Available with the ``bids`` extra. ``preprocess_bids`` also requires the
preprocessing dependencies and uses the same numerical pipeline and explicit
review decisions as ``preprocess``.

.. autoclass:: eegtable.bids.BIDSQuery

.. autoclass:: eegtable.bids.BIDSRecording

.. autofunction:: eegtable.bids.discover_bids

.. autofunction:: eegtable.bids.read_bids

.. autofunction:: eegtable.bids.preprocess_bids
