# Data & training conventions (must preserve)

- **Two dataset years — don't conflate them.** The **ablations** train/evaluate on **2020** (`Flight_tab_2020.csv`, `data_info_2020.yaml`). SHAP/HUIM patterns come from **2020 train days** (`02_2020_train`, primary/default) or, as a cross-year check, **2024** (`01_2024_baseline`) — see `shap-huim-method.md`.
- **Temporal split by day-of-month — never shuffle across days** (prevents leakage): train `FL_DAY ≤ 9`, val `10 ≤ FL_DAY ≤ 12`, test `FL_DAY > 12`. The test split is never subsampled; every arm must report the same `test_hash` (`ablations/test_hashes.json`, enforced by the runner).
- **Excluded columns:** `FLIGHTS`, `FL_YEAR`, `FL_MONTH`, and `FL_DAY` (split key only — test days never occur in train).
- **Encoding (train-fit only):** categorical label maps learned on train (unseen → `-1`), `StandardScaler` on train, HUIM bins (`KBinsDiscretizer`, exact quantiles) on train. Label-encoded columns are forced categorical for mambular (`preprocessing.cat_cutoff`).
- **Target:** `run.target` — `abs` (`|ARR_DELAY| > 15`, default), `late` (`> 15`), `3class` (early/on-time/late).
- Feature lists come from `data_info_<year>.yaml`, not hardcoded.

## Shared runner (`exp/Tab_exp/ablations/common/`)
Every arm = `config.yaml` (experiment + features, optional `overrides:`) + a thin `run.py`. Training settings come from `common/base_config.yaml`; check equality with `common/check_configs.py` (only `01_baseline_all_features` has legacy overrides; `01b_baseline_matched_training` is the reference arm).

- **Phase 1:** 10% of train, `n_iter=10` candidates × `cv=3` StratifiedKFold over `d_model ∈ {64,128,256}`, `n_layers ∈ {2,6,10}`, `lr ∈ {1e-5,1e-4,1e-3}`, selected by `roc_auc`. Each candidate is constructed fresh (`RandomizedSearchCV` + mambular `set_params` silently trained the default config). MLP/ResNet map `n_layers` to `layer_sizes` (+ `num_blocks`). Seeded by `run.phase1_seed` (42) and stored in `best_params.json` with a signature; `--reuse-params` skips Phase 1 for extra seeds.
- **Phase 2:** best params, `phase2_frac` of train, early stopping on `val_frac` of val, seeded by `--seed`. GPU, `16-mixed`.
- **Evaluation:** AUC from probabilities (+ `auc_hard` for legacy comparison), accuracy, macro-F1 on the full test set → `<arm>/results.csv`; probabilities → `<arm>/predictions/` (gitignored).
- Library patches enabled by default (all verified, see `common/README.md`): batched DataLoader (speed), row-independent pretab PLE, batch-first attention for FT-Transformer.
- Models: MLP, AutoInt, ResNet, FTTransformer.
