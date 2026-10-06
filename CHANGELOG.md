# Changelog

Notable changes to EEGTable. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versions follow
[semantic versioning](https://semver.org/); before 1.0, a minor version may change
behaviour, and every such change is listed here.

## [Unreleased]

### Changed

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

- The weekly public-dataset validation workflow never ran a test. A cache miss left
  no `MNE_DATA` directory for MNE to download into, and an unquoted version pin
  broke the shell command.
- A file named `:memory:.ses` was tracked, which Windows cannot check out. It and
  `.DS_Store` are no longer tracked.
