# Shared ablation runner (`common/`)

Every ablation arm (`01`, `01b`, `02`, `03`, `04`, `06*`) is a `config.yaml` that only
describes its **feature set**, plus a three-line `run.py` that calls
`common.runner.main`. All training, search and data settings live in
`base_config.yaml`. An arm may deviate only through an explicit `overrides:` block, so
`common/check_configs.py` (or a plain `diff`) shows every difference between arms.

```bash
uv run python exp/Tab_exp/ablations/03_shap_plus_huim/run.py --seed 42
    [--target abs|late|3class]
    [--models MLP,AutoInt,ResNet,FTTransformer,LightGBM,XGBoost,CatBoost]
    [--reuse-params]                      # skip Phase 1 if best_params.json has a matching entry
    [--set training.phase2_frac=1.0] [--set features.extraction_dir=...]   # any config key
    [--out-dir DIR]                       # default: the arm directory
```

## Protocol

1. **Data** (`data.py`): `Flight_tab_2020.csv` → `dropna` → target → temporal split by day
   of month (train `FL_DAY ≤ 9`, val `10–12`, test `> 12`). The test split is never
   subsampled. A fingerprint of the test rows (`test_hash`) is checked against
   `../test_hashes.json`; an arm with different test rows raises `RuntimeError`.
2. **Features** (`features.py`): the arm's feature set; label maps, scaler and HUIM bins
   are fit on the train split only.
3. **Phase 1 — search**: `phase1_frac` (10%) of train, `n_iter` (10) candidates ×
   `cv` (3) stratified folds, early stopping on a `val_frac` sample of val, selection by
   ROC-AUC. Seeded by `run.phase1_seed` (42) for every seed, and stored in
   `best_params.json` with a signature of everything that influences it.
4. **Phase 2 — final fit**: best parameters, `phase2_frac` of train (0.1 by default;
   1.0 = all of train), early stopping on `val_frac` of val, seeded by `--seed`.
   `--reuse-params` reuses the stored Phase-1 result, so extra seeds (or a different
   `phase2_frac`) only repeat Phase 2.
5. **Evaluation** (`metrics.py`): on the full test set — ROC-AUC from probabilities,
   accuracy, macro-F1 (+ `auc_hard`, the hard-label AUC older results used). Test
   probabilities are saved to `<arm>/predictions/` (gitignored) for paired tests.

Models: mambular **MLP, AutoInt, ResNet, FT-Transformer** (grid `d_model ∈ {64,128,256}`,
`n_layers ∈ {2,6,10}`, `lr ∈ {1e-5,1e-4,1e-3}`) and **LightGBM, XGBoost, CatBoost**
(`gbdt.py`: own grids, up to 2000 trees, early stopping on validation AUC with patience
50, same search budget; CatBoost receives the categorical columns natively, the other two
the same integer codes as the deep models).

## Files

| file | role |
|---|---|
| `base_config.yaml` | single source of training / search / data settings |
| `runner.py` | CLI, Phase 1, Phase 2, `best_params.json`, `results.csv`, locking |
| `data.py` | loading, target (`abs` / `late` / `3class`), split, `test_hash` |
| `features.py` | feature sets `all`, `shap_only`, `llm_only`, `shap_plus_huim`, `huim_only` |
| `metrics.py` | metrics; chunked inference aligned to the batch size |
| `seeding.py` | `set_seed()` — python, numpy, torch (+CUDA), Lightning workers, deterministic cuDNN |
| `gbdt.py` | LightGBM / XGBoost / CatBoost behind the same interface as the mambular models |
| `fast_loader.py` | batched DataLoader fetch for mambular (speed only, identical outputs) |
| `pretab_patch.py` | row-independent PLE encoding (library bug fix, see below) |
| `mambular_patch.py` | batch-first attention for mambular's transformer layer (library bug fix) |
| `verify_fast_loader.py` | checks that the batched loader and chunked inference change nothing |
| `check_configs.py` | diffs the resolved configs of all arms against `01b` |

`results.csv` columns: `arm, model, seed, target, phase2_frac, n_train, n_val, n_test,
test_pos_rate, test_hash, auc, accuracy, f1_macro, auc_hard, extraction, n_features,
best_params, phase1_sig, timestamp`. A run whose `(model, seed, target, phase2_frac,
extraction)` is already present is skipped, so commands can be re-issued safely.

**Parallel runs.** `results.csv` / `best_params.json` are updated under a lock file and
re-read before each write. A model in progress is claimed with a `.running__…` marker
holding the PID; other runs skip it while that process lives and take it over if it died.
After a machine restart, delete leftover `.running__*` files (PIDs may be reused). On a
32 GB machine keep at most three or four concurrent runs: a mambular FT-Transformer
search on HUIM-augmented features peaks at 6–8 GB RAM.

## Library issues handled here

These are bugs or pitfalls in the libraries (or in how they were used before this
runner); each changes results, so all results produced by this runner include the fixes.

1. **Hyper-parameter search did not search.** mambular's `set_params()` updates
   `config_kwargs` but not the already-built `config`, so `RandomizedSearchCV`
   (clone + `set_params`) trained the *default* configuration for every candidate. The
   runner constructs each candidate with `cls(**params)` (same `ParameterSampler`
   draws, `StratifiedKFold(cv)`).
2. MLP and ResNet have no `n_layers`: mapped to `layer_sizes = [d_model] * n_layers`
   (+ `num_blocks = n_layers` for ResNet).
3. Search selects by ROC-AUC (accuracy is logged as well).
4. Categorical label maps and the scaler are fit on train only (a per-split
   `LabelEncoder` gives the same airport different codes in train and test). Unseen
   categories → `-1`.
5. Reported AUC is computed from probabilities (hard-label AUC ≈ balanced accuracy).
6. `FL_DAY` is excluded as a feature: test days never occur in train.
7. HUIM columns are float 0/1; patterns that never fire in train are dropped.
   `KBinsDiscretizer` uses exact quantiles (`subsample=None`) — the sklearn default
   subsamples 200k rows with `random_state=None`, so bins changed between runs.
8. **pretab PLE**: `PLETransformer.transform` clamps its last fitted threshold to the
   maximum of the batch being transformed, so a row's encoding depends on its
   batch-mates (max difference 47 in the self-test). `pretab_patch` keeps the fitted
   thresholds; it is bit-identical to the original whenever the clamp does not fire
   (`python -m common.pretab_patch`).
9. **mambular FT-Transformer**: `CustomTransformerEncoderLayer` keeps PyTorch's
   `batch_first=False` while its input is `(batch, features, d_model)`, so attention ran
   across the *rows* of each mini-batch: memory O(batch²) (out of memory on 16 GB at
   batch 2048) and predictions depending on batch-mates (max |Δp| 0.02 when re-chunking).
   `mambular_patch` makes it batch-first: |Δp| 3e-5 (TF32 noise, as for the other models),
   peak memory 1083 → 284 MiB (`python -m common.mambular_patch`).
10. Label-encoded categoricals are always categorical for mambular
    (`preprocessing.cat_cutoff`); pretab's ratio rule made `OP_CARRIER_FL_NUM` numerical
    on 10% samples and categorical on the full train set.
11. Memory: Lightning/mambular objects stay alive through reference cycles; a
    `gc.collect()` after each Phase-1 fold keeps long searches from growing ~1 GB/h.
12. Inference is chunked (bounded RAM) with chunks rounded to a multiple of the batch
    size, so GPU batches equal those of a single call.

## Determinism

- Same command twice → identical AUC/accuracy/F1 to the last digit (checked for 01 and 03
  with a reduced search, and for every GBDT).
- Twelve seed-42 runs (01b, 03 × 2024, 04 × 2020-train; four models each) were repeated
  hours later under a different parallel load: all bit-identical to the original rows.
- A run whose result row was lost was re-run from scratch (Phase 1 + Phase 2) and
  reproduced it exactly (AUC 0.610473, ACC 0.596104, F1 0.562788).
- `verify_fast_loader.py` (CPU and GPU): all four mambular models give identical
  predictions with and without the batched loader, and chunked inference equals a
  single call.
