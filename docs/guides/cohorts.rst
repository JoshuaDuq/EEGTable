Reproducible cohorts
====================

The study workflow has three explicit handoffs: :doc:`preprocessing` produces
epochs and a retention ledger; :doc:`runner` produces separate epoch and
cross-trial feature bundles; quality reports and design builders use those
bundles with matched descriptors. :doc:`bids` preserves BIDS participant and
event metadata through the first two handoffs. For ordinary files, supply
subject, session, condition, and outcome descriptors explicitly; filenames
alone do not supply these variables.

EEGTable keeps the numerical feature definition, its evidence and the recording
identity together. Newly written tables use schema 3: the sidecar declares the
exact payload filenames and their SHA-256 checksums. Reading a modified
checksummed payload raises. The reader also accepts the existing schema-less
``eegfeat`` format when it has the required row-identity manifest, but that format has no
payload checksums or general descriptor type manifest. Unsupported schemas or
missing row identities raise. Regenerate old outputs when the study requires
the current extraction provenance and integrity checks; see :doc:`tables`.

Extraction manifests record input content, input-relative recording identity,
resolved settings including defaults, installed scientific package versions and
the Python source identity. Every linked part of a split FIF recording is
checksummed; changing or removing any part invalidates the result.
When EEGTable preprocessing produced the epochs, its
manifest identity and payload checksums are verified and carried into extraction.
Externally prepared epochs explicitly have no EEGTable upstream manifest. Moving
the dataset preserves identities when its relative directory structure stays the
same. An implementation or environment change makes old results stale.

Quality policies
----------------

Coverage measures finite numerical input; it does not establish artifact-free
data. Inspect preprocessing retention, rejection decisions and feature-specific
flags as well. Choose a fixed exclusion policy before evaluating a model:

.. code-block:: python

   from eegtable.io import read_dataset
   from eegtable.quality import QualityPolicy
   from eegtable.model import build_design

   dataset = read_dataset(paths)
   # This cohort includes peak_frequency, whose edge_hit flag marks peaks on a band edge;
   # a flag no measure in the cohort raises is refused.
   policy = QualityPolicy(min_coverage=0.8, rejected_flags=("edge_hit",))
   design = build_design(dataset.table, dataset.targets, target="rating",
                         groups="subject_id", quality=policy)
   # design.meta, coverage, flags and quality_ledger preserve the evidence.

Choose ``rejected_flags`` from ``dataset.table.flags``; an unknown flag raises.
``min_coverage`` is a finite fraction in ``[0, 1]`` and rejects cells strictly
below the threshold. ``min_support`` does the same for the fraction of a
window a Morlet value rests on; values without temporal restriction always
pass it. Rejected cells become NaN, with a ``quality_rejected``
flag and a ledger of their reasons. Rows, feature definitions, original
coverage, and other flags are retained; the input table is unchanged.
Missingness learned from the cohort, imputation,
scaling, feature selection and supervised transforms still belong inside each
training fold.

Write an HTML report using :func:`eegtable.report.write_quality_report`, or from
current batch outputs:

.. code-block:: bash

   eegtable report recipe.toml quality.html --by recording
   eegtable report recipe.toml quality.html --by subject_id condition --min-coverage 0.8
   eegtable report recipe.toml groups.html --rows groups --by recording

The command requires every selected recording to have current ``done`` results
and the requested row kind to exist. ``--rows`` defaults to ``epochs`` and
``--by`` to ``recording``. Repeat ``--reject-flag`` to reject several emitted
flags. All requested grouping descriptors must exist and contain complete,
nonempty labels.

Reports contain cohort and feature missingness, coverage and flag fractions,
exclusion decisions, complete feature definitions and report provenance. The
descriptor rows passed to the Python reporting functions must already be aligned
to the feature rows. The command-line report also restores observed preprocessing
retention and artifact decisions from extraction manifests; unavailable upstream
evidence stays explicitly missing. Use :func:`eegtable.report.recording_quality`
to build the same recording summary for the Python reporting interface.
Suggested ICA component exclusions count proposed components, not rejected
epochs or an artifact score. Retention counts come from the preprocessing
ledger and provenance; feature missingness describes a separate stage.

Group samples
-------------

An across-trial coherence or ITPC estimate is one group sample. Its constituent
epochs do not become independent observations of that estimate. The runner saves
``recording`` and ``n_trials`` descriptors for every cross-trial row. Assemble
these outputs separately. The runner does not infer a group outcome or subject
identity from epoch metadata. In this example, ``group_outcomes`` has one row
per ``(recording, group)`` with ``subject_id`` and a numeric ``outcome``:

.. code-block:: python

   from eegtable.group import GroupDataset, build_group_design, read_group_dataset

   dataset = read_group_dataset(crosstrial_paths)
   descriptors = dataset.targets.merge(
       group_outcomes,
       on=["recording", "group"],
       how="left",
       sort=False,
       validate="one_to_one",
   )
   dataset = GroupDataset(dataset.table, descriptors)
   design = build_group_design(dataset, target="outcome", groups="subject_id")

Canonical group identities are ``(recording, group)``. Duplicate identities,
invalid trial counts and incorrectly aligned descriptors raise. Column unions
preserve different measured channel schemas with NaN values and zero coverage
where a recording did not measure a column. Use ``columns="identical"`` when
the study requires the same ordered feature definitions for every recording.
Matching ROI or global column names does not establish matching contributing
sensors; check the retained channels and ROI membership as well.

``n_trials`` is the count of retained epochs assigned to the group, not a
measure of independent information or estimator reliability. Overlapping
fixed-length epochs remain dependent, and estimator-specific trial requirements
still apply. ``build_group_design`` requires finite numeric targets and complete
group labels; it does not add covariate columns. Evaluate designs with the existing
group-disjoint classification or regression functions. For generalization to a
new study, group folds by study and keep all observations from that study out of
both fitting and model selection.

Repeated-session reliability
----------------------------

The :doc:`worked reliability tutorial </auto_tutorials/plot_session_reliability>`
extracts log alpha power, records trial aggregation, and compares agreement with
consistency in a balanced repeated-session simulation.

:func:`eegtable.intraclass_reliability` evaluates one explicitly aggregated estimate
per subject/session, with a complete balanced design and at least three subjects
and two sessions. It reports single-measure absolute agreement ICC(2,1) and
consistency ICC(3,1). Absolute agreement detects systematic session offsets that
consistency can ignore. Missing values, duplicate samples, unbalanced designs and
constant features raise; this API never silently drops subjects or fills values.
The estimates are descriptive and do not supply confidence intervals.
Apply any prespecified quality policy and aggregate epochs to one subject/session
row before calling this function. Use the same feature definition and aggregation
across sessions; neither row matching nor ICC calculation establishes that
measurement equivalence.

The formulas follow `Pingouin's documented two-way ANOVA implementation
<https://pingouin-stats.org/generated/pingouin.intraclass_corr.html>`_. Quality
reports use `MNE.Report <https://mne.tools/stable/generated/mne.Report.html>`_.

API signatures for quality policies, summaries, reports, and ICC are in
:doc:`/api/containers`. Group dataset loading and design construction are in
:doc:`/api/model`.
