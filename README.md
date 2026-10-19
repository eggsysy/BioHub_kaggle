# BioHub Cell Tracking — Kaggle Competition

Work for the Kaggle competition **[biohub-cell-tracking-during-development](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development)**:
3D + time cell tracking in zebrafish light-sheet microscopy.

## Result

| | Score |
|---|---|
| Starting point (`best_ensemble_0.935.ipynb`) | 0.935 public |
| **Final submission** (`sub_953/s953.ipynb`) | **0.95387 public / 0.91780 private** |
| Final rank | **939 / 3,558 teams** |

Honest attribution: essentially all of the gain came from adopting and tuning
stronger *public* pipelines (0.935 → 0.946 → 0.947 → 0.9539). None of the
custom training or hyper-parameter search produced a reproducible LB gain. The
section "What did not work" below records that in detail, because the negative
results were expensive to establish and are the more useful artifact.

## The metric

```
score = adjusted_edge_jaccard + 0.1 * division_jaccard
adj    = max(0, J * (1 - 0.1 * (T_pred - T_true) / T_true))
```

- Node matching is bipartite within a **7 µm** radius.
- Voxel scale: `z = 1.625 µm`, `y = x = 0.40625 µm`.
- Ground truth is **sparse** — only ~2.82% of real cells are annotated.
  Unmatched predicted nodes are therefore *not* false positives, but they
  still count toward `T_pred` and so are penalised by the `adj` term.

That asymmetry drives most of the design: over-prediction is cheap in Jaccard
but expensive in the track-count penalty.

## Pipeline

```
TemporalUNet3D (detector)
      -> SimpleNodeTransformer (edge scorer)
      -> ILP linking (ilpy / pyscipopt)
      -> post-processing: gap closing, motion relink,
         safe-division repair, short-track filtering
```

Test-time augmentation (TTA) was the single mechanism behind every real score
gain across every pipeline variant tried.

## Layout

### `harness/` — offline evaluation stack
The most reusable part of this repo. Lets you score post-processing changes in
seconds instead of burning a Kaggle GPU commit.

| File | Purpose |
|---|---|
| `extract.py` | Splits the monolithic notebook into importable modules (`pipeline_config.py`, `pipeline_funcs.py`, `baseline_env.py`) |
| `replay.py` | Replays post-processing + the **official** scorer offline. `score_stems_parallel` is a verified-identical 3.2x speedup |
| `score_submission.py` | Scores any `submission.csv` against the 4 public test samples using the official metric |
| `tune.py` | Optuna TPE search in three stages (`division` 11 knobs, `tracks` 7, `gaps` 10) |
| `embryo_split.py` | Embryo-disjoint train/test split — the only honest validation here |
| `verify_lb.py`, `validate_cache.py`, `stack_check.py` | Consistency guards |

`replay.py` contains `assert_effective(overrides)`, which fails loudly if an
override cannot possibly change the offline result (constant missing from
config, or never read by the pipeline). This was added after several silent
no-op "experiments".

### Diagnostics
`gt_stats.py`, `gt_divisions.py`, `gt_spatial.py`, `harness/edge_diag.py`,
`harness/fn_diag.py`, `harness/prune_oracle.py`, `harness/div_features.py`,
`harness/div_train.py`, `harness/mf_data.py`, `harness/mf_model.py`

### `public_kernels/` — 20 reference pipelines
Pulled public kernels used for comparison and adoption, named
`<author>_<kernel>`. Searching these **by score in the title** rather than by
recency or votes was the single highest-value action in the project.

### Submission / probe kernels
`sub_*/`, `probe_*/`, `rp_*/`, `top_kernel*/`, `w1_gt_export/`, `w2_cache/` —
each holds a notebook plus its `kernel-metadata.json`. These are the
experiment record; `sub_953/` is the final submission.

### `build_*.py`, `*_probe.py`
Notebook builders. The baseline notebook is a single 214 KB cell that sets its
own environment inline, so overrides are spliced in programmatically rather
than edited by hand. `reyhan_probe.py` splices **after the last `os.environ[`
line** for this reason. Each builder re-parses the notebook it produced to
verify the splice actually landed.

## What did not work

Each of these was closed with offline evidence, not guesswork:

- **Post-processing tuning.** FP/FN are already symmetric (51/49), precision
  0.9415 ≈ recall 0.9440. There was no slack left to tune.
- **The node-count exploit.** A pruning oracle showed +0.137 available, but
  the sparse GT is spatially uniform, so which nodes to drop is not learnable
  from any available signal.
- **A learned division classifier.** Only 52 positive examples; performed 14x
  worse than the hand-coded rule.
- **Multi-frame association.** W=4 scored worse than W=2.
- **Detector fine-tuning.** Three separate runs, all flat or negative.

## Known traps in this codebase

Recorded because each cost real time:

1. **Cell 8 of the baseline notebook both patches the predict script *and*
   launches prediction** via `subprocess.Popen`. A patch inserted as a later
   cell rewrites the file *after* inference has already run, and has zero
   effect. Modify the string cell 8 writes.
2. **Two separate guards** validate `BIDIRECTIONAL_EDGE_WEIGHT` — cell 5's
   `_EXPECTED_NUMERIC` and a hardcoded one in cell 8. Both must be updated.
3. Kaggle kernels need `"machine_shape": "NvidiaTeslaT4"` or they fail with
   *"no kernel image available for execution"*.
4. `kernelspec` requires `display_name` or `nbformat` validation rejects the
   notebook.
5. The deployed `train_unet_transformer.py` has a **dead division weight**:
   ```python
   weight[div_rows] = 1.0   # sets 1.0 to 1.0 — does nothing
   ```
6. `tracking_repo/weights/` is read-only on Kaggle; honour `CELLMOT_WEIGHTS_DIR`
   or build a writable `weights_ft/` tree and repoint `WEIGHTS_RELATIVE`.
7. T4 OOMs at batch 16 — use batch 6 with
   `PYTORCH_ALLOC_CONF=expandable_segments:True`.
8. Killing an Optuna parent leaves orphaned workers;
   `pkill -f harness/tune.py` catches the whole family.

## Methodological note

At one point the Optuna config was "validated" by splitting the same 50
samples it had been tuned on — not a holdout. It predicted +0.010; the
leaderboard returned −0.001. `harness/embryo_split.py` exists so that mistake
cannot be repeated silently.

## Not in version control

Excluded via `.gitignore` as regenerable or too large: virtualenvs and wheels,
the 50-sample prediction cache (`cache/`), ground-truth exports (`gt/`),
model weights (`*.pt`), `.geff` prediction graphs, kernel logs, Optuna
outputs, and `div_candidates.csv` (2.1 GB). `official_repo/` is a nested
clone and is excluded rather than vendored.
