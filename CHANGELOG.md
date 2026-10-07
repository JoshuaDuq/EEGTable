# Changelog

Notable changes to EEGTable. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versions follow
[semantic versioning](https://semver.org/); before 1.0, a minor version may change
behaviour, and every such change is listed here.

## [Unreleased]

### Changed

- Extraction reuse compares recipe-scoped software and source identities.
  Unused optional package upgrades and unrelated preprocessing/modeling edits
  no longer invalidate results. The full analysis environment remains recorded
  separately. Existing extraction tables require one recomputation to adopt the
  scoped identities.
- Ordinary ICA no longer requires electrode positions or four good EEG channels;
  rank validation remains enforced, and ICLabel retains its geometry requirements.
  ICA application accepts changes to auxiliary channels and sampling rate while
  preserving checks on its fitted channels, reference and projectors.
- Export checkpoints now record the exported provenance identity. Existing export
  checkpoints require `reset CONFIG --from export` before republishing; upstream
  preprocessing checkpoints remain reusable.
- Feature metadata canonicalizes time-window bounds as floats, so equivalent
  integer and float bounds identify the same column and round-trip through files.
  Custom features previously constructed with integer bounds have new identifiers.
- Preprocessing checkpoint identities advance to the corrected implementation.
  Reset existing workflows from `load` before resuming, so saved stages cannot
  reuse the previous bipolar-reference or native sampling-rate behavior.
- Morlet computation identities include the wavelet's `zero_mean` setting.
  `Spectra.from_tfr` requires its explicit declaration because MNE does not retain
  it. Recompute existing Morlet tables before combining them with new results.
- Preprocessing source fingerprints include acquisition descriptions and subject
  information, and enabled scientific stages include their optional package
  versions. Existing checkpoint trees require resetting the load stage before reuse.
- ICLabel explicitly uses ONNX Runtime, already required by `preprocessing-auto`,
  so installing PyTorch cannot silently change its inference backend.
- Welch and multitaper constructors and runner features record sampling rate and
  the exact frequency grid in their column identities, as `from_spectrum` does.
  Recompute existing PSD tables before combining them with new results.
- Normalized custom spectral features record units on their resulting scale
  (`log10(unit)`, `log10 ratio`, `dB`, or `%`). Their corrected units change
  column identities; recompute those custom tables before combining them.
- Review checkpoint identities include their owning review policy. Existing
  review checkpoints and downstream stages need resetting before reuse.
- Morlet feature identities include the retained TFR time axis. Decimation and
  time offsets can change window averages and now produce distinct columns.
  Recompute existing Morlet tables before stacking them with new results.
- **Column names change once for two groups of features.** Tables computed before
  this release do not stack with new ones under `columns="identical"`, and the
  default `"union"` would keep both sets of columns side by side. Rerun with
  `eegtable run --overwrite`; existing results already read as stale after any
  code change.
  - Features built on band signals (ERD/ERS, bursts, ITPC, PPC, PAC, envelope
    correlation and the time-domain measures of a band) no longer hash `n_jobs`.
    Running with `--n-jobs 2` used to name identical values differently from a
    serial run.
  - Runner Welch and multitaper features hash only the settings their method reads.
    Morlet-only settings such as `n_freqs` or `decim` used to enter every Welch
    column's name.
- Finite window bounds that reach outside the data are refused instead of being
  clipped silently. A window of 0.5–40 s on epochs ending at 4 s used to measure
  0.5–4 s under a name claiming 0.5–40 s. A bound may reach one sample past the
  last one, whose interval it closes (0–30 s for a 30 s epoch ending at 29.99 s), and
  infinite bounds still run to the edge. The runner allows the same, where it used to
  allow half a sample.
- An ROI's or the global mean's coverage counts only the member channels its value
  was averaged over. An ROI whose `peak_frequency` came from one of three channels
  used to report full coverage. Values are unchanged.
- `Spectra.from_spectrum` checks the declared estimator parameters against the
  spectrum. It refuses keys MNE's estimator does not take, execution-only keys
  (`n_jobs`, `verbose`), and an `n_fft`, `fmin` or `fmax` that the frequency axis
  contradicts.
- `FeatureTable.to_dataframe()` indexes epoch rows by `(recording, epoch, event)`
  and group rows by their label. With that index,
  `frame.join(epochs.metadata, on="epoch")` stays aligned after MNE drops epochs.
- `FeatureTable.select()` matches a band by its name (`band="alpha"`) and warns
  when a requested value names no column at all.
- `repr(FeatureTable)` gives a one-line summary instead of every array.
- Passing epochs where `Spectra` are expected, a single signal outside a list, an
  unknown `normalize`, or a text target to a model now raises an error that names
  the fix, instead of an `AttributeError` or a `KeyError` from deep inside.
- `eegtable model run` prints each model's held-out scores beside its baselines.
  An external targets file that matches no sample shows example identities from
  both sides, and a fold too small for its inner splits reports both counts.
- pandas 3 is supported. Only the `cycles` extra keeps pandas below 3, because
  ByCycle 1.2.0 writes into arrays that pandas 3 makes read-only.
- The minimum MNE-Python version is 1.10. Versions 1.8 and 1.9 cannot be imported
  with SciPy 1.15 or later.
- The `spectral-model` extra accepts specparam's 2.0 releases from `2.0.0rc7`,
  instead of pinning that release candidate exactly.

- `aperiodic`, `aperiodic_ratio`, `periodic_power` and adjusted `peak_frequency`
  refuse a spectrum whose frequency axis does not span the whole `fit_range`. They
  used to fit only the bins present while recording the requested range: on a
  recipe with only an alpha band, the "2–40 Hz" fit ran on 7.5–13.5 Hz.
- A recipe's default `[spectra]` `fmin`/`fmax` also spans the `fit_range` of every
  aperiodic fit (`aperiodic`, `periodic_power`, adjusted `peak_frequency`,
  `spectral_parameterization`), not only the bands. Recipes whose bands are
  narrower get a wider spectrum and new spectral column names. An explicit
  `fmin`/`fmax` that leaves part of a fit range out is a recipe error.
- Welch spectra treat a channel with any non-finite sample in a window as missing
  in every epoch, as multitaper and Morlet do. MNE estimated around NaN that every
  epoch in the call shared, so one epoch's value depended on the others estimated
  with it, an infinity was not missing, and MNE 1.10–1.12 raised instead.
- Microstate templates come from the best of several modified k-means starts (the
  k-means centres and 20 random peak maps), by variance explained at the peaks. A
  start that leaves a cluster empty is discarded instead of ending the fit. Fitted
  templates, and so microstate columns, can change.
- Burst units name the threshold as a quantile (`quantile 0.75 of baseline`), not a
  percentile, which read as the 0.75th percentile. Burst columns are renamed.
- IRASA band powers no longer repeat their band among the shared computation
  parameters, so `band_ratio` pairs them; their columns are renamed.
- `detrended_fluctuation` needs at least 58 samples per window. Between 50 and 57,
  AntroPy fits a single block size and returns an exponent of exactly 0.
- Custom measures refuse a kernel without a qualified name, such as a
  `functools.partial` or a callable object. Its repr carried a memory address into
  the column identity, so the same feature had a new name in every process.
- Kernel SHAP runs without shap's default lasso, which left all but about 10
  features per row at exactly 0, and with shap's automatic sample count. It scores
  every feature and is about 20 times slower.
- BIDS events parse the standard `response_time` column as a number; the
  non-standard `reaction_time` is left as text like other custom columns.

- Preprocessing checkpoints go stale only where a correction below changes them:
  `annotate` with `annotations.muscle`, `fit-artifact` with ICLabel, and `epoch` with
  `epochs.padding` above 0 or a threshold `rejection.tmin`/`tmax`. Reset from that
  stage and review again; other recipes keep their checkpoints.

### Added

- `eegtable.extract(epochs, recipe, recording=...)` applies a recipe to epochs in
  memory, computing exactly what `eegtable run` writes for that recording.
  `load_recipe` also accepts a mapping with the TOML file's structure, in which
  `inputs` and `output` may be left out.
- `Spectra.welch`, `Spectra.multitaper` and `Spectra.morlet` compute spectra from
  epochs and record their own settings, so nothing is declared twice. They share
  the runner's estimator, so the same settings name the same columns either way.
- `spectral_measure` and `signal_measure` turn a kernel of your own into a measure
  with the library's bands, windows, ROIs, coverage, support and provenance. An
  installed package can register measures under the `eegtable.measures`
  entry-point group, and recipes can then name them.
- `FeatureTable.to_long()`: one row per cell, with its row identity, the column's
  metadata, value, coverage and support. This is the long format used by mixed
  models, R and plotting libraries.
- Morlet support reaches the results. `FeatureTable.support` holds the fraction of
  each requested window a value rests on, and bundles that have it gain a
  `<name>_support.tsv` file. `QualityPolicy(min_support=...)`,
  `eegtable report --min-support` and the model recipes' `quality.min_support`
  exclude values resting on too little of their window.
- `eegtable check` warns when recordings keep different channel sets, so their
  global means and pattern ROIs would split into separate columns. It also warns
  when a Welch window holds a single segment, making its spectrum one periodogram.
- Documentation:
  - six further executable tutorials on aperiodic backgrounds, IRASA, entropy,
    cycle waveforms, frozen microstate templates, and participant-disjoint CSP;
  - a collapsible Tutorials section using native Sphinx/Furo navigation, with
    the current tutorial's parent expanded and gallery links supplied automatically;
  - eight additional tutorials on resting alpha, Morlet support and quality,
    ERPs, beta bursts, phase consistency, sensor connectivity, session reliability,
    and participant-held-out modeling, with reference calculations and figures
    generated on every documentation build;
  - explicit finite-value and reduction checks in the motor ERD tutorial, with
    descriptive summaries that account for the repeated-observation sample unit;
  - a tutorial that measures single-trial ERD on a public dataset, with figures,
    run on every documentation build;
  - guides to the long format and reading bundles in R, to writing your own
    measures, and to EEGTable's place among related tools;
  - a page on versions, stability and what renames a column;
  - the changelog;
  - docstrings for every public modeling and preprocessing callable, parameter
    sections for the core measures, and an API page for the runner.
- `py.typed`, so type checkers read EEGTable's annotations.
- Package metadata for PyPI, a release workflow using trusted publishing,
  CONTRIBUTING, a code of conduct, CITATION.cff, issue and pull request templates,
  and Dependabot updates for workflow actions.
- CI tests on Windows and macOS, and at the oldest supported NumPy, SciPy, pandas
  and MNE releases.

### Performance

- A typical task recipe on one 64-channel recording takes 3.6 s of CPU instead of
  16.2 s, with values unchanged (relative differences below 1e-8).
  - Column hashes splice in the canonical computation text instead of re-encoding
    it per column.
  - The aperiodic fit is vectorized over cells (17 times faster), as is Higuchi's
    fractal dimension.
  - Welch input is split into blocks of epochs under the size at which MNE falls
    back to a per-row loop.
- Sidecars store each distinct computation once (schema 3), instead of once per
  column. A 64-channel envelope-correlation sidecar shrinks from 51 MB to 6.5 MB.
  Schema 2 bundles still read.
- Band signals release the filter padding instead of holding it for their lifetime.

### Fixed

- ICA review saves the exclusions selected in MNE's viewer after confirmation,
  rather than replacing them with a separate unchecked checklist. Plugin
  provider versions are read afresh when checking extraction reuse or publication.
- Plugin aliases of built-in measures retain their numerical dependency versions,
  so upgrades to those dependencies invalidate extraction reuse.
- Graph summaries preserve the minimum edge coverage and support, and any input
  flags, so derived networks cannot bypass the quality policy applied to their
  connectivity estimates.
- Modeling and scoring reject complex inputs before converting them to real
  arrays. Classification metrics reject out-of-range probabilities and class
  probabilities that do not sum to one.
- Prediction intervals validate that model predictions are finite, real, and
  aligned with the requested rows, preventing broadcasting or truncated complex
  values from producing misleading bounds.
- Resuming preprocessing refuses a valid export whose provenance differs from
  its checkpoint, instead of reporting another run's output as complete.
- Preprocessing bundle validation requires the epochs, event ledger, report and
  any recorded repair ledger in its manifest. An empty or incomplete inventory
  can no longer report missing outputs as a finished export.
- Univariate screening rescales centered observations before computing Pearson
  correlations, preventing extreme input units from producing spurious perfect
  or zero correlations through floating-point underflow or overflow.
- Permutation importance rejects misaligned, duplicate, or empty feature names.
  SHAP importance rejects duplicate output-to-input mappings instead of silently
  overwriting one transformed column's attribution with another's.
- Fold-local feature harmonization preserves the pipeline's trailing covariates
  during evaluation, inner tuning, and ridge permutation inference. Missing
  covariates can no longer cause EEG columns to be treated as covariates, and
  tuning cannot change the covariate layout under a shared harmonization mask.
- Batched ridge permutation correlations avoid underflow and overflow by scaling
  centered values before normalization, so target units do not change inference.
- The preprocessing `epoch` stage drops an epoch whose padding, or the part of its
  window outside `rejection.tmin`/`tmax`, overlaps a BAD span; MNE also narrows its
  annotation check to the threshold window it is given.
- `artifact.ica.method: infomax` no longer reports "ICA did not converge" for a fit
  that converged: MNE's infomax returns `max_iter` when its weight change
  converges, so only FastICA and Picard are held to the limit.
- ICLabel labels components on the recording without its BAD spans, which were left
  unfiltered and out of the ICA fit.
- The muscle detector scores only good EEG channels; one flat bad channel made
  every z-score NaN and silently disabled it.
- An `--overwrite` export removes the old bundle's files that the new one lacks,
  such as `_repairs.tsv` or split `_epo-N.fif` parts, and the existing-bundle check
  no longer matches other bundles whose names begin with `<name>_epo-`.
- An unexpected preprocessing failure reports its exception type, so a `KeyError`
  no longer reads as just the missing key.
- Subject-level r refuses a subject left with fewer than 3 trials once its folds are
  centred; three trials over two folds correlated at exactly ±1 whatever the data.
- Burst coverage is limited by the coverage of the samples a quantile threshold is
  calibrated on, as ERD/ERS coverage is by its baseline.
- `pac_surrogates` accepts NumPy integers for `n_surrogates` and `random_state`.
- BIDS events accept `n/a` durations, which the specification allows, and another
  participant's unparsable age (such as `89+`) no longer blocks every recording.
- A feature whose computation has a parameter named `method` can be read back.
- A zero-byte or corrupt `_epo.fif` no longer stops discovery: the run records that
  recording as failed and finishes the others, `status` reports it, and `check`
  warns about headers it cannot read.
- Upgrading the package that provides a plugin measure makes its results stale; its
  name and version are recorded with the software versions.
- The CLI no longer fails with `UnicodeEncodeError` when output goes to a legacy
  code page such as Windows cp1252.
- Epoch metadata with a `recording` column is refused when the table is written,
  instead of producing tables that `read_dataset`, `report` and `model` reject.
- An empty, absolute or `..` `inputs.pattern` is a recipe error instead of a
  traceback.
- `eegtable init` into a missing folder exits 2 with an error, and suggested
  commands are quoted for the platform's shell (POSIX, or cmd.exe and PowerShell
  on Windows) so paths with spaces can be pasted.
- The basic template's commented `[windows]` suggestion uses a 1 s baseline; the
  0.5 s one failed `check` on the template's own `peak_frequency` entry.
- Factory pipelines with covariates can be pickled, and the package's transformers
  follow scikit-learn's transformer contract: `get_feature_names_out()` works
  without arguments and `set_output(transform="pandas")` keeps column names.
- The TUI headlines a failed run with Python's error instead of MNE's last log
  line, builds on Windows, and honours `NO_COLOR`, which turned colour fully on.
- CI runs the recipe and passband tests that need optional packages, and a pull
  request's docs build can no longer cancel a pending Pages deployment.
- Documentation: custom spectral kernels receive trapezoid-rule weights, not bin
  widths; out-of-range finite TFR window bounds are refused, not intersected; and
  `fit_staged_residual_preprocessor` does not match what pooled cross-fitting does.
- Identity permutations reuse the observed fit's statistic, preserving exact ties
  across single-target and batched Ridge computations on different BLAS platforms.
- CI tests verify ICLabel classification without requiring a version-specific
  warning and compare undefined confidence intervals with NaN-aware assertions.
- Sign-flip inference counts floating-point ties inclusively in paired and
  maximum-statistic tests, preventing understated p-values from roundoff.
- Cross-fitting rejects duplicate or noninteger fold identifiers before fitting;
  duplicate identifiers could combine predictions from different fitted models
  during fold-centered scoring.
- Native preprocessing checkpoints and exports reject sampling-rate precision
  loss before publication instead of changing the time grid and provenance.
- Numeric categorical and object row descriptors retain numeric participant
  identities when recordings are combined; text labels and missing values remain
  distinct. Mixed text and numeric descriptor columns raise before writing.
- Bipolar EOG/ECG derivations preserve the retained EEG's reference metadata and
  projectors, including the configured policy for inactive projections.
- Custom features constructed with integer time-window bounds now round-trip
  through feature bundles without a spurious metadata-integrity error.
- Morlet constructors and recipe extraction reject complex analytic EEG samples
  instead of reporting their power on the real-signal density scale.
- Nuisance fitting and staged permutations reject fractional or repeated row
  indices and misaligned inputs before selecting trials or fitting models.
- Preprocessing checkpoints reject material numeric metadata loss during JSON
  serialization before publishing rounded event or epoch descriptors.
- External model targets preserve floating-point precision when read from TSV.
- Orphaned support files mark a recording as partial, so resumed extraction
  cannot skip recordings whose feature bundles are missing.
- SHAP preprocessing preserves caller-owned feature arrays and refuses PCA
  attribution even when component names match input feature names. Kernel SHAP
  seeds coalition sampling and preserves the caller's random streams.
- Cohort quality summaries reject grouping labels that collide with summary
  fields instead of overwriting the labels with counts or quality statistics.
- BIDS event alignment follows MNE-BIDS value labels when `trial_type` is entirely
  missing, retaining the correct metadata when unselected events share a sample.
- Preprocessing exports report metadata lost to native FIF JSON precision as an
  actionable error without publishing an incomplete bundle or accepting lost values.
- Thermal-pain marker repair preserves unrelated ledger text, including leading
  zeros in BIDS identifiers and numeric-looking trial labels.
- Signals combined into one feature table and PAC operands require exact time
  samples and sampling frequencies; relative tolerances could accept shifted data.
- Model tuning, classification responses, permutation importance and prediction
  intervals isolate reused feature arrays from in-place scikit-learn transforms.
- Feature readers reject malformed quality-flag row positions instead of
  truncating fractional indices or interpreting negative indices from the end.
- Preprocessing checkpoints become stale when acquisition descriptions or
  configured scientific package versions change, preserving current provenance.
- ERD/ERS coverage includes the baseline's finite-input coverage before spatial
  averaging, so incomplete baselines cannot pass a full-coverage quality threshold.
- Grouped fitting rejects active trial-wise early stopping in gradient boosting,
  stochastic MLP and SGD estimators, including replacements in parameter grids.
- Feature importance refuses preprocessing without an explicit output-column
  mapping instead of attributing transformed columns to unchanged input names.
- ICLabel fitting requires an applied common average reference over good EEG
  channels, including when called directly outside the preprocessing pipeline.
- Feature tables require boolean quality flags, preventing invalid flag fractions
  and lossy serialization. TSV writing refuses literal `n/a` row labels and
  descriptors because that token represents missing values.
- Model feature selection preserves time-window support, so minimum-support
  exclusions apply to the selected columns. Model designs and saved bundles
  retain support, and input support files are hashed and verified before publication.
- Covariate imputation refuses columns without finite training values instead
  of dropping them and shifting the columns used for deconfounding. Missingness
  checks also accept `ColumnTransformer` slice selectors.
- Grouped modeling and prediction intervals reject estimators with internal
  cross-validation that cannot receive groups and fit preprocessing per split.
- Trial grouping refuses missing or non-string identities, ITPC/PPC require
  integer minimum trial counts, and microstate templates reject complex values.
- Preprocessing refuses unknown or unused review decisions and validates
  resampling against MNE's independently rounded epoch endpoints.
- Split FIF continuations are discovered as parts of one recording, preventing
  repeated epochs from being counted under separate recording identities.
- Welch and multitaper constructors refuse invalid frequency bounds and complex
  epoch samples before clipping bounds or converting data.
- Connectivity ROI memberships must be disjoint, preventing self-connectivity
  from entering cross-ROI averages.
- Constant decimal features are removed by variance filtering, and constant
  fold cells stay exactly zero after centering in observed and permutation scores.
- Every subject-based inner scoring metric receives validation subjects,
  including metrics that do not choose the refitted candidate.
- Channel-only extraction leaves unused ROI patterns unresolved. Metadata trial
  grouping refuses distinct values that would become the same text label.
- Feature values, coverage and support retain exact floating-point precision
  when reading saved TSV bundles.
- Permutation p-values count numerical ties inclusively in either tail, so
  rounding differences cannot exclude an identity permutation.
- Inner tuning checks each training subject's missingness after its candidate's
  feature selection, including ordinary scores and grid-supplied preprocessing.
  All requested tuning metrics are validated, including under permutation.
- Stimulation repair after cropping uses retained event onsets while preserving
  the acquisition event ledger. Retained repair windows still require neighbors.
- Resumed preprocessing records current review policies in provenance.
- Simultaneous BIDS events with `value` labels and no `trial_type` retain the
  selected event's metadata.
- Custom measures reject complex results, unknown normalization modes and
  parameters that would overwrite the kernel's identity.
- Spectral power, feature values, coverage, support and scientific time/frequency
  axes reject complex inputs instead of accepting them or discarding imaginary parts.
- Table and signal cross-fitting reject malformed predictions and probabilities
  before returning results. Table cross-fitting also validates target shapes
  and requires fitted binary classifier classes.
- Cohort quality and subject-level scores omit unused categorical labels instead
  of creating empty groups or rejecting otherwise valid predictions.
- Cropped BrainVision conversions preserve the acquisition time of the exported
  first sample. Marker repair excludes macOS AppleDouble files from fixed sources.
- The preprocessing TUI forwards every requested worker count to Python,
  including negative counts supported by MNE and invalid zero counts.
- Custom spectral kernels receive non-finite bins as NaN as documented, so
  NaN-aware reductions omit infinities while preserving input coverage.
- Cross-fitting rejects missing or incorrectly shaped participant/run labels
  before checking group separation; missing labels could evade overlap checks.
- SHAP input attribution rejects covariate deconfounding, which mixes features
  with nuisance columns and previously reported zero nuisance importance even
  when those columns affected predictions.
- Muscle-detector scores excluded by existing BAD annotations are saved as JSON
  null, so annotation checkpoints remain writable without changing the scores.
- Raw review gates retain acquisition duration after cropping, keeping manual
  span bounds consistent with acquisition-relative onsets.
- Reviewing epochs after ICA application opens the epochs viewer instead of
  trying to plot ICA sources from unavailable raw data.
- The weekly public-dataset validation workflow never ran a test. A cache miss left
  no `MNE_DATA` directory for MNE to download into, and an unquoted version pin
  broke the shell command.
- A file named `:memory:.ses` was tracked, which Windows cannot check out. It and
  `.DS_Store` are no longer tracked.
