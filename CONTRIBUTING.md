# Contributing to EEGTable

Bug reports, questions, fixes and new measures are welcome. Open an issue before a
large change so the design can be agreed first. Everyone taking part follows the
[code of conduct](CODE_OF_CONDUCT.md).

## Setting up

```bash
git clone https://github.com/JoshuaDuq/EEGTable.git
cd EEGTable
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,docs,model,connectivity,microstates,importance,references,preprocessing,preprocessing-auto,bids,spectral-model,irasa,cycles,complexity,pac,riemann]"
```

## Checks a pull request must pass

```bash
python -m pytest
python -m ruff check src tests
python -m black --check src tests
python -m mypy
python -m sphinx -b html -W --keep-going docs docs/_build/html
```

Changes to spectral, ERD/ERS, connectivity or modeling code should also pass the
public-dataset validation suite, which downloads about 350 MB on first use:

```bash
EEGTABLE_DATASETS=1 python -m pytest tests/validation -ra
```

CI runs the tests on Linux, Windows and macOS, both at the newest release of every
dependency and at the oldest ones `pyproject.toml` admits.

## How code is written here

- **Test first.** Write the test, watch it fail for the reason you expect, then make
  the change. A test's name states the behaviour; a comment states the scientific
  reason the assertion matters. Expected values come from hand-checked numbers or
  simulated signals with known answers, never from the code under test.
- **Comments say why.** Put a short comment at the line that needs justifying: the
  reason a guard exists, why a statistic is computed a particular way. Do not add
  docstrings that restate a signature. Public functions keep their numpydoc
  docstrings, which the API reference renders.
- **Refuse rather than repair.** Ambiguous input (rows that do not align, a window
  reaching outside the data, a declaration the data contradicts) raises an error
  that names the fix. A silently repaired input becomes a silently wrong result.
- **Identity holds what changes a value, and nothing else.** A column's name ends in
  a digest of its `FeatureMeta`, including the `ComputationSpec`. Only parameters
  that can change a value belong there. Settings that only change how a run executes
  (`n_jobs`, `verbose`, paths) never do: they would split one feature into a column
  per setting. A change to what enters an identity renames columns, so it needs a
  line under **Changed** in the [changelog](CHANGELOG.md).

## Adding a feature measure

1. **Compute it through the expansion helpers** in `src/eegtable/_expand.py`. They
   turn a per-cell kernel into a `FeatureTable` and handle bands, windows, ROIs and
   the global mean, coverage, Morlet support, flags and provenance:
   - `expand` takes a spectral kernel, `(data, freqs, weights) -> (values, flags)`;
   - `expand_signal` takes a time-series kernel,
     `(signal, trace, times, mask) -> {measure: values}`.
2. **Follow the signature conventions.** Name the first parameter for the input it
   reads: `spectra`, `series`, `signals`, `signal`, `phase_signal` or
   `segmentation`. The runner infers the input kind from that name. Make `bands`,
   `windows`, `groups`, `include_global` and `baseline` keyword-only, as the
   existing measures do. Any other keyword-only parameter with a type annotation
   becomes a recipe setting automatically.
3. **Export it.** Add it to `src/eegtable/__init__.py` and `__all__`, and to
   `EXPECTED` in `tests/test_public_api.py`.
4. **Let recipes name it.** Add it to `MEASURES` in
   `src/eegtable/runner/measures.py`. If it needs an optional package, record that
   package and its extra in the same file, so a recipe gets a clear error rather
   than an import failure.
5. **Document it.** Add an `autofunction` entry to the matching `docs/api/*.rst`
   page. In `docs/methods/*.rst`, give the formula, units, assumptions,
   missing-value behaviour and references.
6. **Validate it.** Add at least one check in `tests/validation/` marked with
   `@pytest.mark.validates(...)`, so the claim appears in the published scorecard.

## Releasing

1. Move the **Unreleased** entries in `CHANGELOG.md` under the new version and set
   `__version__` in `src/eegtable/__init__.py`.
2. Tag the commit `vX.Y.Z` and push the tag. The `release` workflow builds the
   sdist and wheel, checks that their version matches the tag, and publishes them
   to PyPI through trusted publishing.
3. Before the first release, register the publisher once on PyPI: project
   `eegtable`, repository `JoshuaDuq/EEGTable`, workflow `release.yml`,
   environment `pypi`.
