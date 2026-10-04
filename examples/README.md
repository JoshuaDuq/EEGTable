<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../assets/branding/eegtable-logo-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="../assets/branding/eegtable-logo.svg">
  <img src="../assets/branding/eegtable-logo.svg" width="201" height="48" alt="EEGTable">
</picture>

# Example output

Output of `examples/make_examples.py` from simulated recordings. Five subjects, two runs
each, sixteen trials per run, eight EEG channels, and fixed seed 11. The full cohort
is temporary; the generator copies one showcase recording and model outputs here,
replacing existing files. Regenerate with

```bash
python -m pip install -e ".[model]"
python examples/make_examples.py
```

## Feature extraction

[`recipe.toml`](recipe.toml) sets bands, windows, regions, and one entry per measure.
`eegtable run` applies it to every epochs file and writes, for each recording,

- [`sub-01_task-pain_run-01_features.tsv`](sub-01_task-pain_run-01_features.tsv) — one row per
  epoch. Leading columns are the epoch identity and the metadata attached to it
  (`subject`, `run`, `trial`, `intensity`, `rating`, `painful`). The remaining columns are
  features.
- [`sub-01_task-pain_run-01_features_coverage.tsv`](sub-01_task-pain_run-01_features_coverage.tsv)
  — aligned feature columns and row identities, without descriptive columns.
  Fraction of finite input behind each feature cell.
- [`sub-01_task-pain_run-01_features.json`](sub-01_task-pain_run-01_features.json) — sidecar.
  For each column, the measure, band, space, window, unit, normalization, computation
  parameters, and hash. Also flags and run provenance.
- [`sub-01_task-pain_run-01_crosstrial.tsv`](sub-01_task-pain_run-01_crosstrial.tsv) — measures
  defined on a set of trials. In this recipe, inter-trial phase coherence. One row per
  trial group. These rows are not written into the per-epoch table.
- [`sub-01_task-pain_run-01_crosstrial_coverage.tsv`](sub-01_task-pain_run-01_crosstrial_coverage.tsv)
  — finite-input coverage for the cross-trial table.
- [`sub-01_task-pain_run-01_crosstrial.json`](sub-01_task-pain_run-01_crosstrial.json) —
  cross-trial metadata and provenance.

## Modeling

Leave-one-subject-out ridge regression of `rating` on the band-power and ERDS columns.
Scores are per subject. The cohort correlation is tested against 200 within-subject
permutations.

- [`example_model_scores.tsv`](example_model_scores.tsv) — cohort correlation and the
  permutation *p*, then each subject's correlation. There is no interval: every held-out
  score comes from a model trained on the other subjects, so the scores are not independent.
- [`example_model_predictions.tsv`](example_model_predictions.tsv) — held-out prediction for
  every trial, with the fold index.

These numbers describe the simulation. Alpha power falls with stimulus intensity. `rating`
includes noise that these EEG features do not explain.

The generator uses the explicit `ridge_pipeline` and `ridge_grid` API. The current
modeling recipe starter uses training-scaled Ridge, so these simulated outputs are
not a benchmark of that starter. See the [modeling guide](https://joshuaduq.github.io/EEGTable/guides/modeling.html) for current workflows.
