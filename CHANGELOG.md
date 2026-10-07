# Changelog

Notable changes to EEGTable. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versions follow
[semantic versioning](https://semver.org/); before 1.0, a minor version may change
behaviour, and every such change is listed here.

## [Unreleased]

### Changed

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
