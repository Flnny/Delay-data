# Tabular ablations (2020)

Feature-set ablations for binary delay prediction (`|ARR_DELAY| > 15`) on the 2020
tabular snapshot, with mambular deep models (MLP, AutoInt, ResNet, FT-Transformer) and
GBDT baselines (LightGBM, XGBoost, CatBoost). All arms run through one seeded runner
(`common/`, see `common/README.md` for its design and the library bugs it fixes).

| Arm | Feature set |
|---|---|
| `01b_baseline_matched_training` | all raw features — **the reference** |
| `01_baseline_all_features` | same features, legacy training settings (documented, not compared) |
| `02_shap_only` | top-10 SHAP features |
| `03_shap_plus_huim` | top-10 SHAP features + 20 driver / 20 protector HUIM pattern columns (proposed) |
| `04_huim_only` | HUIM pattern columns only |
| `06_llm_top10`, `06_1_llm_top5_datafree`, `06_2_llm_top5_datainformed` | LLM-Select rankings from `05_baseline_100_data/` |

02/03/04 take their SHAP/HUIM files from `features.extraction_dir`:
`../../shap_huim_extractions/01_2024_baseline/csvs` (2024, default) or
`../../shap_huim_extractions/02_2020_train/csvs` (2020 train days only).

Results: `aggregate/ABLATIONS.md` (mean ± std over seeds 42/43/44, Δ tables) and
`aggregate/paired_tests.md` (paired DeLong tests). Representation bake-off (50k sample,
all years): `representation/bakeoff/RESULTS_GBDT.md`.

## How to reproduce

Run from the repo root. Data: download the Kaggle dataset into
`Datasets/Aeolus/Flight_Tab/Tab/` (see the top-level README); `uv sync` installs everything.

```bash
# 0. (optional) self-checks of the runner's library patches — all should print OK/IDENTICAL
uv run python exp/Tab_exp/ablations/common/verify_fast_loader.py --accelerator gpu
(cd exp/Tab_exp/ablations && uv run python -m common.pretab_patch && uv run python -m common.mambular_patch)

# 1. SHAP + HUIM extractions (02/03/04 read these). 01_2024_baseline is committed as-is.
uv run python exp/Tab_exp/shap_huim_extractions/02_2020_train/run.py            # ~30 min

# 2. LLM-Select rankings for the 06 arms (committed; needs a local Ollama qwen2.5:7b)
uv run python exp/Tab_exp/ablations/05_baseline_100_data/run.py
uv run python exp/Tab_exp/ablations/05_baseline_100_data/compare.py             # SHAP vs LLM, both extractions

# 3. Seed 42 — full protocol (Phase 1 search + Phase 2) for every arm.
A=exp/Tab_exp/ablations
E20="--set features.extraction_dir=../../shap_huim_extractions/02_2020_train/csvs"
uv run python $A/01b_baseline_matched_training/run.py --seed 42
for arm in 02_shap_only 03_shap_plus_huim 04_huim_only; do
  uv run python $A/$arm/run.py --seed 42            # 2024 extraction
  uv run python $A/$arm/run.py --seed 42 $E20       # 2020-train extraction
done
for arm in 06_llm_top10 06_1_llm_top5_datafree 06_2_llm_top5_datainformed; do
  uv run python $A/$arm/run.py --seed 42
done
#    GBDT baselines: same commands with  --models LightGBM,XGBoost,CatBoost

# 4. Seeds 43 and 44 — reuse the stored Phase-1 params, repeat Phase 2 only.
#    Same commands as step 3 with  --seed 43 --reuse-params  (and 44).

# 5. Tables
uv run python $A/aggregate.py          # -> aggregate/ABLATIONS.md, ablation_table.csv, deltas.csv
uv run python $A/paired_test.py        # -> aggregate/paired_tests.md
uv run python $A/common/check_configs.py   # arms differ only in their feature set?

# 6. Bake-off GBDT rows (rebuilds the 50k sample if artifacts/ is missing)
uv run python $A/representation/bakeoff/prep/build_sample.py
uv run python $A/representation/bakeoff/run_gbdt.py
```

Notes:
- Every run is deterministic (seeded; same AUC to the last digit on re-run). A finished
  `(model, seed, target, phase2_frac, extraction)` is skipped, so commands can be re-issued.
- Runs can go in parallel (results/params writes are locked; in-progress work is claimed
  with `.running__*` markers). On a 32 GB machine keep **≤ 3** concurrent arms: a
  mambular FT-Transformer search peaks at 6–8 GB RAM.
- Budget on an RTX 5060 Ti: one arm at seed 42 with the 4 mambular models ≈ 3–6 h
  (FT-Transformer/AutoInt searches dominate); a seed-43/44 rerun ≈ 20–25 min; the three
  GBDTs for one arm ≈ 5–15 min.
- `--target late|3class` gives the other targets; results are keyed by target.
