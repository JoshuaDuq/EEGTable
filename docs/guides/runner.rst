Command Line Runner
===================

.. raw:: html

   <p class="hero-lede">
     Compute features for a folder of preprocessed MNE epochs files from one
     TOML recipe. Each written table loads back as a <code>FeatureTable</code>.
   </p>

A recipe sets feature measures, analysis windows, spectral estimation,
band-signal construction, and trial grouping. ``eegtable run`` applies that
recipe to every recording. Prepare the epochs beforehand, either in MNE or
with :doc:`preprocessing`; extraction does not perform raw preprocessing.

Quick Start
-----------

.. code-block:: bash

   eegtable init recipe.toml --template task   # a commented recipe to start from
   eegtable check recipe.toml                  # validate it, time it on the first recording
   eegtable run recipe.toml --workers 4        # compute features for every recording
   eegtable status recipe.toml                 # which recordings are done, and what to run next

``python -m eegtable`` is the same command.

``init`` **templates**
   ``basic``
      The default: a few spectral measures.
   ``task``
      Every measure family, for epochs around an event, with a baseline and a response window.
   ``resting``
      Every family that needs no event.

**check** writes nothing. It computes and times the first recording (see
`Checking a Recipe`_) and checks the channels used by ROIs and asymmetry pairs
in every recording's header. It catches:

- an ROI that names a missing channel,
- a window outside the epochs,
- a Morlet wavelet longer than the epoch.

Passing ``check`` does not verify all recordings' windows, numerical data,
metadata, or preprocessing manifests. Those are validated during their run.

The Recipe
----------

A recipe is a TOML file.

- Relative paths resolve against the recipe's directory.
- Unknown keys are errors.
- Every problem found is reported together.

.. code-block:: toml

   [inputs]
   root = "derivatives/preprocessed/eeg"
   pattern = "**/*_proc-clean_epo.fif"

   [output]
   root = "derivatives/eegtable"

   [windows]
   baseline = [-5.0, -1.0]
   stimulus = [0.0, 8.0]

   [trials]
   by = "event"

   [[features]]
   measure = "integrated_band_power"
   normalize = "log10"
   ratios = [["theta", "beta"]]

   [[features]]
   measure = "erds_mean"
   bands = ["alpha", "beta"]
   baseline = "baseline"
   spatial = ["global"]

   [[features]]
   measure = "itpc"
   bands = ["theta"]

.. rubric:: Sections

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Section
     - Keys and defaults
   * - ``[inputs]``
     - ``root`` (required); ``pattern`` = ``"**/*_epo.fif"``, relative to ``root`` and
       staying under it; ``picks`` = ``"eeg"`` (a channel type or a list of names);
       ``exclude_bads`` = ``true``. Hidden files are skipped, including the ``._`` files
       macOS leaves on external drives.
   * - ``[output]``
     - ``root`` (required); ``epoch_metadata`` = ``true`` copies each epoch's metadata
       into its row.
   * - ``[bands]``
     - ``name = [low, high]`` in Hz, half-open. Default: delta to gamma, 1–45 Hz.
   * - ``[windows]``
     - ``name = [tmin, tmax]`` in seconds. Default: the whole epoch, named ``all``,
       which is reserved.
   * - ``[rois]``
     - ``name = ["Ch1", "Ch2", ...]``, or ``name = { match = ["^P[z0-9]+$"] }`` for
       regular expressions matched against each recording's channels, which keep the
       recording's order. An ROI whose patterns match no channel fails that recording, and
       ``check`` names it. ``global`` is reserved.
   * - ``[spectra]``
     - ``method`` = ``"welch"``, ``"multitaper"`` or ``"morlet"``; ``fmin``/``fmax``
       default to the span of the bands, widened to cover the ``fit_range`` of every
       aperiodic fit (``aperiodic``, ``periodic_power``, adjusted ``peak_frequency``,
       ``spectral_parameterization``). An explicit ``fmin`` or ``fmax`` that leaves part of
       a ``fit_range`` out is a recipe error. See below.
   * - ``[band_signal]``
     - ``pad_sec`` = 0.5, ``pad_cycles`` = 3.0, passed to :meth:`eegtable.BandSignal.from_epochs`.
   * - ``[trials]``
     - How cross-trial measures group epochs: ``by`` = ``"all"`` (one group),
       ``"event"`` (by event name) or ``"metadata"`` with a ``column``.
       Every epoch must have a nonmissing value when grouping by metadata.
   * - ``[microstates]``
     - Parameters of :func:`~eegtable.microstates.segment`, fitted once per recording and shared by
       every microstate measure.
   * - ``[defaults]``
     - ``bands``, ``windows``, ``spatial`` and ``series`` for every entry that takes them
       and does not set its own. See below.
   * - ``[[features]]``
     - One entry per measure. See below.

Feature Entries
~~~~~~~~~~~~~~~

Each ``[[features]]`` entry names one measure or several.

- ``measure`` names a library function.
- ``measures = [...]`` names several instead. The entry's other keys apply to each of them, as
  if it were written once per measure.
- A key one of them does not take is an error for that measure.

Every other key is either one of that function's own keyword parameters, checked against its
annotations (``normalize``, ``fit_range``, ``threshold``, ``polarity``, ...), or one of the
runner's keys:

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Key
     - Meaning
   * - ``bands``
     - Band names. Default: every band. Single-band measures such as
       ``peak_frequency`` run once per band.
   * - ``windows``
     - Window names, and ``"all"`` for the whole epoch. Default: every window except the
       entry's baseline.
   * - ``baseline``
     - A window name. Power measures consume it for normalization; burst and ERDS
       measures calibrate against it.
   * - ``spatial``
     - Any of ``"channels"``, ``"rois"``, ``"global"``. Default ``["channels", "global"]``;
       for pairwise connectivity the levels are node sets, default ``["channels"]``.
   * - ``series``
     - Time-domain and complexity measures: ``"broadband"`` and band names (band
       envelopes). Default ``["broadband"]``.
   * - ``pairs``
     - ``pac`` and ``pac_surrogates``: ``[["theta", "gamma"], ...]`` as ``[phase, amplitude]``. PAC
       columns are named by amplitude band, so pairs in one entry need distinct ones.
   * - ``ratios``, ``asymmetry``
     - Power measures only: ``[[numerator, denominator], ...]`` band pairs and
       ``[[left, right], ...]`` channel pairs, through :func:`eegtable.band_ratio` and
       :func:`eegtable.asymmetry`.
   * - ``graph``, ``clustering_threshold``
     - ``envelope_correlation``, ``wpli``, ``spectral_connectivity`` and
       ``spectral_connectivity_time``: ``["global_efficiency",
       "clustering_coefficient"]``; the threshold is required for clustering.

Connectivity and PAC recipes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``spectral_connectivity`` reaches every library method: ``coh``, ``imcoh``,
``plv``, ``ciplv``, ``ppc``, ``pli``, ``wpli`` and ``wpli2_debiased``. ``method``
is required; ``mode`` is ``"multitaper"`` or ``"fourier"``. These estimates use
``[trials]`` and appear in the cross-trial table.

``spectral_connectivity_time`` requires ``method`` and ``freqs``. It accepts
``coh``, ``imcoh``, ``plv``, ``ciplv``, ``pli`` and ``wpli``. Each epoch is estimated
independently after removing full Morlet edge support, so the result and its
graph summaries appear in the per-epoch table. Analysis windows must be long
enough for the requested ``n_cycles`` and frequencies.

.. code-block:: toml

   [[features]]
   measure = "spectral_connectivity"
   method = "wpli2_debiased"
   bands = ["alpha"]
   graph = ["global_efficiency"]

   [[features]]
   measure = "spectral_connectivity_time"
   method = "coh"
   bands = ["alpha"]
   freqs = [8.0, 9.0, 10.0, 11.0, 12.0]
   n_cycles = 5.0

   [[features]]
   measure = "pac_surrogates"
   pairs = [["theta", "gamma"]]
   n_surrogates = 999
   surrogate = "blocks"
   min_shift_seconds = 0.5
   random_state = 42
   correction = "maxstat"

PAC inference requires the ``pac`` extra and returns raw, null-mean, null-standard-
deviation, corrected, z-score, empirical-p-value and adjusted-p-value columns.
Adjustments cover the returned spatial units and windows within each epoch and
band pair. Separate entries and spatial-level calls are separate families.
``pac`` retains its raw-estimate behavior.

Spectral, cycle and complexity recipes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``spectral_parameterization`` fits a PSD from ``welch`` or ``multitaper``;
``aperiodic_mode`` selects ``"fixed"`` or ``"knee"``. ``irasa`` and
``cycle_features`` require ``series = ["broadband"]`` and use their ``bands``
selection separately. Select IRASA and spectral-model bands inside ``fit_range``.
``permutation_entropy``, ``lempel_ziv_complexity`` and ``detrended_fluctuation``
accept broadband traces or named band envelopes through ``series``.

.. code-block:: toml

   [[features]]
   measure = "spectral_parameterization"
   bands = ["alpha", "beta"]
   fit_range = [2.0, 40.0]
   aperiodic_mode = "knee"

   [[features]]
   measure = "irasa"
   bands = ["alpha", "beta"]
   fit_range = [2.0, 40.0]
   series = ["broadband"]

   [[features]]
   measure = "cycle_features"
   bands = ["alpha"]
   burst_thresholds = {min_n_cycles = 3, amp_fraction_threshold = 0.3}

   [[features]]
   measures = ["permutation_entropy", "lempel_ziv_complexity", "detrended_fluctuation"]
   series = ["broadband"]

These entries require the ``spectral-model``, ``irasa``, ``cycles`` and
``complexity`` extras respectively. Backend and scientific validation errors
remain visible.

Shared Defaults
~~~~~~~~~~~~~~~

A ``[defaults]`` section sets ``bands``, ``windows``, ``spatial`` or ``series`` once.

.. code-block:: toml

   [defaults]
   windows = ["baseline", "response"]
   spatial = ["channels", "global"]

   [[features]]
   measures = ["variance", "line_length", "kurtosis", "hjorth_mobility"]

   [[features]]
   measures = ["erds_mean", "erd_magnitude", "ers_magnitude"]
   bands = ["alpha", "beta"]
   baseline = "baseline"      # computed on "response" only

**Inheritance rules**

- An entry inherits each key it takes and leaves out.
- A default ``bands`` does not reach ``aperiodic``.
- A default ``spatial`` keeps only the levels a measure has (``envelope_correlation`` has no
  ``global``).
- An entry with a baseline leaves that window out of the inherited windows, as leaving
  ``windows`` out does.
- Keys an entry sets win.

Spectra
~~~~~~~

Set by ``[spectra]``. Each method has its own defaults and constraints.

``welch``
   Hann-tapered segments of ``n_fft`` samples with 50% overlap, computed separately in each
   window.

   - The default segment is 2 s, capped by the shortest window any spectral entry uses, so
     every window shares one frequency grid.

``multitaper``
   Computed in each window with a frequency-smoothing ``bandwidth`` in Hz.

   - **Default**: 2.0 Hz. MNE's own default is ``8 / window_length`` Hz, which on a 1 s window
     smooths by ±4 Hz, wider than delta or theta.
   - **Lower limit**: a bandwidth below ``1.35 / window_length`` is rejected. In a narrower
     band no Slepian taper keeps 90% of its power, and MNE would fall back to a leaky one with
     only a warning.
   - **Grid**: the frequency grid follows the window length, so windows measured together
     must be the same length.
   - **Units**: the call uses MNE ``normalization="full"``, and the output is a density in
     V²/Hz.

``morlet``
   One time-frequency decomposition per recording.

   - **Defaults**: ``n_freqs`` = 40, ``spacing`` = ``"log"``, ``n_cycles = clip(f /
     n_cycles_factor, min_cycles, max_cycles)`` with factors (2.0, 3.0, 15.0), and ``decim`` = 4.
   - **Windows**: each window keeps the coefficients whose wavelet support lies inside it.
   - **Epoch length**: the lowest wavelet must fit inside the epoch.
   - **Units**: power is divided by the sampling rate of the recording. ``mean_tfr_power`` is
     then a smoothed density in V²/Hz. MNE's raw wavelet power scales with the sampling rate.

``window_statistic``
   How ``welch`` and ``morlet`` reduce a window: ``"mean"`` (the default) or ``"median"``.

   - **Welch**: takes the median over its segments.
   - **Morlet**: takes it over each frequency's retained time points.
   - **Why**: a burst that holds a minority of the window, such as movement or muscle
     artifact, barely moves a median, where it raises a mean in proportion to its power.
   - **Bias**: both medians are divided by their bias for Gaussian data, so on noise they
     estimate the same density as the mean.
   - **Steady oscillation**: it has constant power, and there Morlet's median reads up to 1.44
     times above the mean.
   - **Refusal**: Welch refuses the median for a window with fewer than 3 segments, because the
     median of one or two segments is their mean.
   - **Column names**: the median is recorded in each column's computation, so it gets its own
     column names.

Checking a Recipe
-----------------

``check`` computes the first recording and reports how long it took, which measures were
slowest, and a projection for the whole run:

.. code-block:: text

   Time           4 min for this recording
   Slowest        sample_entropy 2 min (40%) · multiscale_entropy 60 s (25%) · ...
   Projected      126 recordings ≈ 8 h 24 min one at a time, ≈ 52 min with --workers 10,
                  if the others are like it

**Reading the report**

- A measure's time includes building the inputs it is the first to need (spectra, band
  signals, the microstate fit), so shared costs show up on the first entry that uses them.
- The projection assumes each worker gets a full core. On processors with efficiency cores,
  or when memory bandwidth is shared, parallel runs take longer than that.
- ``--workers N`` sets the worker count it assumes (default: one per core, up to the number
  of recordings).

**Quick mode**

``check --quick`` computes only the first four epochs and scales the timing up by the epoch
count.

- It finds recipe errors in seconds.
- Its projection is rougher.
- Cross-trial measures see fewer trials than a run gives them.

Running in Parallel
-------------------

``eegtable run --workers N`` computes N recordings at once, each in its own process.

- Results are the same as one at a time, and the run log lists recordings in input order.
- Each worker's BLAS and OpenMP thread pools are sized to its share of the cores
  (``OMP_NUM_THREADS`` and the like), unless those variables are already set.
- Memory grows with the workers: each holds its recording's spectra and band signals (see
  `Things to Know`_).

**Worker crashes**

If a worker process dies (the operating system killing it for memory, a crash in compiled
code), recordings whose completion cannot be established are marked failed with
the original pool error. Submitted recordings are never retried automatically.
Completed results are retained, and recordings not yet submitted can continue in
a new pool. The failure sidecar and run log make the interrupted work explicit.

Outputs
-------

The input tree is mirrored under the output root. Only the row kinds produced
by the recipe are written. With both epoch and cross-trial measures, the input
``sub-01/eeg/sub-01_task-rest_epo.fif`` produces:

.. code-block:: text

   sub-01/eeg/sub-01_task-rest_features.tsv            one row per epoch
   sub-01/eeg/sub-01_task-rest_features_coverage.tsv
   sub-01/eeg/sub-01_task-rest_features.json           column metadata, flags, provenance
   sub-01/eeg/sub-01_task-rest_crosstrial.tsv          one row per trial group
   sub-01/eeg/sub-01_task-rest_crosstrial_coverage.tsv
   sub-01/eeg/sub-01_task-rest_crosstrial.json
   sub-01/eeg/sub-01_task-rest_failed.json             only while the recording fails: error, traceback
   eegtable_run.json                                    what happened to every recording

**Tables**

- ``_features`` holds measures estimated within an epoch. Its rows start with ``epoch``,
  ``selection`` and ``event``, then the epoch metadata, then a ``__eegtable_row_id`` column that
  ``read_table`` checks against the sidecar, then the features.
- ``_crosstrial`` holds measures estimated across trials (``itpc``, ``ppc``,
  ``envelope_correlation``, ``spectral_connectivity``, ``wpli`` and their graph
  summaries), keyed by ``group``, with ``recording`` and ``n_trials`` descriptors.
  The runner does not aggregate epoch metadata into these rows; add explicitly
  matched group outcomes and subject IDs as described in :doc:`cohorts`.
- The two files are separate. See :ref:`concepts-row-kinds`.
- Missing values are written ``n/a``.
- Values, coverage and optional support payloads are protected by SHA-256 checksums
  in schema 3 sidecars. Table readers validate them before returning results.
  The legacy ``eegfeat`` format described in :doc:`tables` lacks these checksums; merely
  loading it does not establish current extraction provenance.

The filenames follow TSV/JSON sidecar conventions. These outputs are not a
validated BIDS derivative.

Loading Results
~~~~~~~~~~~~~~~

Load a table back with its metadata:

.. code-block:: python

   import eegtable as ef
   from eegtable.io import read_table

   table = read_table("derivatives/eegtable/sub-01/eeg/sub-01_task-rest_features.tsv")
   alpha = table.select(band=ef.Band("alpha", 8.0, 13.0), space="global")

For modeling across recordings, pass several per-epoch ``*_features.tsv`` paths to
:func:`eegtable.io.read_dataset`.

- The loader stacks tables on the union of their feature columns.
- It restores the descriptor columns written by the runner, plus ``recording``, ``epoch``, and
  ``event`` from the stored row identity.
- Put target and grouping variables in the epoch metadata when
  :func:`eegtable.model.build_design` needs them.
- Descriptor columns are separate from feature columns.

.. code-block:: python

   from eegtable.io import read_dataset

   dataset = read_dataset([
       "derivatives/eegtable/sub-01/eeg/sub-01_task-rest_features.tsv",
       "derivatives/eegtable/sub-02/eeg/sub-02_task-rest_features.tsv",
   ])

Overwriting and Failures
~~~~~~~~~~~~~~~~~~~~~~~~

- A run refuses to write over earlier results unless given ``--overwrite``, which also removes
  result files the new recipe no longer produces.
- A recording that fails is logged with its error and traceback, and the run moves on. An
  input file that cannot be read, such as an empty or truncated one, fails only its own
  recording.
- Each recording's results are staged and read back before publication. A
  computation or validation failure during an overwrite retains its previous
  complete results and records the new failure.
- ``_failed.json`` records the failure beside the results it did not produce, and a later
  success removes it.
- The run log is rewritten after every recording, with ``finished`` false until the run ends,
  so a run that is interrupted still says what it did.

Resuming
--------

``eegtable status`` reads the results and the run log back, computing nothing, and puts
each recording in one of five states:

``done``
   Every table is complete, passes its payload checksums, and matches the current input,
   resolved recipe, software environment and Python source identity. The software
   environment includes the version of each installed package providing a plugin measure
   the recipe names.
``missing``
   No results.
``failed``
   No results, and the last run failed on it. The reason is that run's error, read from
   the recording's ``_failed.json``, so it survives another run into the same output root.
``stale``
   The recipe, recording identity, software environment, implementation, input
   content, or linked preprocessing manifest changed.
``partial``
   A table, coverage file, or sidecar is missing, unreadable, unsupported, or
   fails its integrity checks. Invalid upstream preprocessing evidence also
   makes the result partial.

**What "this recipe" means**

- It is what the recipe computes. Comments, formatting, and the three location keys
  (``inputs.root``, ``inputs.pattern``, ``output.root``) are left out, so moving the data and
  repointing ``inputs.root`` keeps results ``done``.
- Defaults are recorded along with explicit settings. Input and output payloads are
  checked by content, so changing timestamps alone does not invalidate results.
- Unsupported or incomplete manifests require regeneration. Preprocessing manifests
  and their payload checksums are verified when the epochs carry an EEGTable identity.

**Resuming a run**

- ``eegtable run --resume`` computes only the recordings that are not ``done``.
- Recordings that have stale or partial results still need ``--overwrite``, so nothing is
  replaced unasked.
- ``status`` ends with the command to run next.
- ``--json`` gives the same report as one object, ``{recipe, output_root, counts, recordings,
  next}``, where ``next`` is that command's arguments or ``null``.

Command Reference
-----------------

.. code-block:: text

   eegtable run RECIPE [--overwrite] [--resume] [--workers N] [--n-jobs N] [--progress-json]
   eegtable check RECIPE [--quick] [--workers N] [--n-jobs N]
   eegtable status RECIPE [--json]
   eegtable init [PATH] [--template basic|task|resting]
   eegtable report RECIPE OUTPUT [--rows epochs|groups] [--by COLUMN ...]
                                [--min-coverage FRACTION] [--reject-flag FLAG]

``--n-jobs`` is passed to MNE's filtering and spectral estimation.
``--workers`` defaults to 1 for ``run``. For ``check``, it controls only the
projection and defaults to the CPU count capped by the recording count.
``report`` requires every selected recording to be ``done`` and the requested
row kind to exist; see :doc:`cohorts`.

Python API
~~~~~~~~~~

The same workflow is available without the command-line front end:

.. code-block:: python

   from eegtable.runner import check, load_recipe, run, status

   recipe = load_recipe("recipe.toml")
   checked = check(recipe, quick=True)
   outcome = run(recipe, resume=True, workers=4)
   states = status(recipe)

``run`` returns a ``RunResult`` in input order, with individual recording
results, failures, and skipped recordings. Per-recording errors remain in
those results; inspect ``outcome.ok`` and each failed result's ``error`` and
``traceback``. Configuration and discovery failures raise before the batch
starts. Set ``overwrite=True`` explicitly to replace stale or partial results.

Exit Status
~~~~~~~~~~~

- **0**: everything succeeded.
- **1**: some recordings failed, or ``check``'s trial did, or it found recordings lacking
  channels the recipe names.
- **2**: nothing could start.
- ``status`` exits 0 whatever state the recordings are in.

Progress Protocol
~~~~~~~~~~~~~~~~~

``--progress-json`` prints one JSON object per line: ``start``, ``subject_start``,
``progress`` (``step``, ``current``, ``total``), ``log``, ``subject_done`` (``success``,
``elapsed``, ``eta``), ``complete`` and ``error``.

- This is the protocol of the EEG_fMRI_Pipeline terminal UI, with each recording standing in
  for a subject.
- ``eta`` is seconds left at the pace so far, null after the last recording.
- With ``--workers``, recordings start and finish in any order, and each event names its
  recording.
- The text output shows the same estimate after each recording, and names a recording whose
  result does not directly follow its start line.

Things to Know
--------------

- With ``exclude_bads``, each recording keeps its own good channels, so channel-level
  columns can differ between recordings. Interpolating bad EEG upstream can
  preserve a common sensor set. ROI and global names can match even when their
  contributing channels differ, so inspect retained channels and regex ROI
  membership before treating those estimates as comparable.
- Band signals are cached per recording for reuse across entries. Memory grows
  with the epoch count, channels, samples, requested bands, and time-frequency
  grid. Every worker holds its own recording and caches.
- Microstate templates are fitted per recording, not across a group.
- Only FIF epochs files (``mne.read_epochs``) are read. The default pattern
  selects ``*_epo.fif``; set ``inputs.pattern`` explicitly for other supported
  epoch names or compressed FIF files. Match each recording's primary file,
  rather than also matching its split companions.
