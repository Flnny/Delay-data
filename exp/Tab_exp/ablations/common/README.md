# Shared ablation runner (`common/`)

Every mambular arm (`01`–`06_*`) is now `config.yaml` (feature set only) + a 3-line
`run.py` that calls `common.runner.main`. Training settings live in
`base_config.yaml`; an arm may deviate only through an explicit `overrides:` block,
so `diff armA/config.yaml armB/config.yaml` shows every difference.

```bash
uv run python exp/Tab_exp/ablations/03_shap_plus_huim/run.py --seed 42
    [--target abs|late|3class] [--models MLP,AutoInt,ResNet,FTTransformer]
    [--reuse-params]                      # skip Phase 1 if best_params.json matches
    [--set training.n_iter=2] [--set features.extraction_dir=...]
    [--out-dir DIR]                       # default: the arm directory
```

| file | role |
|---|---|
| `base_config.yaml` | single source of training/search/data settings |
| `seeding.py` | `set_seed()` — python, numpy, torch(+CUDA), Lightning workers, cuDNN/deterministic algorithms |
| `data.py` | load → dropna → target (`abs`/`late`/`3class`) → day split → `test_hash` + registry check |
| `features.py` | feature sets (`all`, `shap_only`, `llm_only`, `shap_plus_huim`, `huim_only`); all fitting on TRAIN |
| `metrics.py` | AUC from probabilities (+ legacy hard-label `auc_hard`), accuracy, macro-F1; chunked inference |
| `runner.py` | CLI, Phase 1 search, Phase 2 retrain, `best_params.json`, `results.csv` |
| `fast_loader.py` | batched DataLoader fetch for mambular (speed only) |
| `pretab_patch.py` | row-independent PLE transform (bug fix, see below) |
| `verify_fast_loader.py` | proves `fast_loader` + chunked inference are bit-identical |

`results.csv` columns: `arm, model, seed, target, phase2_frac, n_train, n_val, n_test,
test_pos_rate, test_hash, auc, accuracy, f1_macro` + `auc_hard, extraction, n_features,
best_params, phase1_sig, timestamp`. A run is skipped if `(model, seed, target,
phase2_frac, extraction)` is already present. Pre-runner results are kept as
`results_legacy.csv`.

`../test_hashes.json` records the test-set fingerprint per (file, target, split). An
arm whose test rows differ raises `RuntimeError`.

## Behaviour changes vs. the legacy scripts

1. **Hyper-parameter search actually searches.** mambular's `set_params()` updates
   `config_kwargs` but not the already-built `config`, so
   `RandomizedSearchCV` (clone + set_params) trained the *default* config for every
   candidate and for Phase 2. The runner builds each candidate with `cls(**params)`
   (same `ParameterSampler` draws, `StratifiedKFold(cv)`).
2. MLP/ResNet have no `n_layers`: mapped to `layer_sizes=[d_model]*n_layers`
   (+ `num_blocks=n_layers` for ResNet).
3. Search selects by `roc_auc` (accuracy logged too).
4. Categorical label maps and the scaler are fit on TRAIN only (01/02/03/06 fit a
   `LabelEncoder` per split, so the same airport got different codes in train/test;
   01 also fit a `StandardScaler` per split). Unseen categories → `-1`.
5. Reported `auc` uses probabilities (legacy AUC used hard labels ≈ balanced accuracy).
6. `FL_DAY` excluded as a feature (test days never occur in train).
7. HUIM columns are float 0/1 in both 03 and 04; patterns that never fire in train
   are dropped. `KBinsDiscretizer` uses exact quantiles (`subsample=None`); the
   sklearn default sub-sampled 200k rows with `random_state=None`, so legacy bins
   changed between runs.
8. pretab's `PLETransformer.transform` clamps its last fitted threshold to the max of
   the batch being transformed, so a row's encoding depended on its batch-mates
   (max diff 47 in the self-test). `pretab_patch` keeps fitted thresholds; it is
   bit-identical to the original whenever the clamp does not fire
   (`python -m common.pretab_patch`).

## Verification (TODO item 0)

Acceptance run: 01 and 03, seed 42, MLP, `training.n_iter=2`, each twice.

| arm | AUC run a | AUC run b | test_hash |
|---|---|---|---|
| 01_baseline_all_features | 0.5934517313447859 | 0.5934517313447859 | 7cab2c005e188b58 |
| 03_shap_plus_huim | 0.604066142787261 | 0.604066142787261 | 7cab2c005e188b58 |

`verify_fast_loader.py` (CPU and GPU): MLP/AutoInt/ResNet predictions identical with
and without the batched loader, and chunked inference identical to a single call.
