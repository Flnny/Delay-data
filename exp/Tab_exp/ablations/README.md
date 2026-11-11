# Tabular ablations

Which features — and which models — predict US domestic flight delay
(`|ARR_DELAY| > 15 min`, binary) from the 2020 tabular data? The ablations vary the
**feature set** (all features, SHAP-selected, SHAP + HUIM patterns, HUIM only,
LLM-selected) and evaluate each with four mambular deep models (MLP, AutoInt, ResNet,
FT-Transformer) and three GBDTs (LightGBM, XGBoost, CatBoost) under one protocol:

- temporal split by day of month — train days 1–9, validation 10–12, test 13–31 (all
  2,547,088 test rows, identical for every arm);
- hyper-parameter search on 10% of train (10 candidates × 3 folds, ROC-AUC), then a
  final fit on 10% of train (`phase2_frac` 0.1) and, separately, on all of it (1.0);
- seeds 42 / 43 / 44, reported as mean ± std; AUC from predicted probabilities.

The shared, seeded runner and the library issues it handles are described in
[`common/README.md`](common/README.md).

## Arms

| folder | feature set | README |
|---|---|---|
| `01b_baseline_matched_training` | all raw features — **reference** | [→](01b_baseline_matched_training/README.md) |
| `01_baseline_all_features` | same features, original (non-comparable) training settings | [→](01_baseline_all_features/README.md) |
| `02_shap_only` | top-10 SHAP features | [→](02_shap_only/README.md) |
| `03_shap_plus_huim` | top-10 SHAP features + 40 HUIM pattern columns (proposed method) | [→](03_shap_plus_huim/README.md) |
| `04_huim_only` | HUIM pattern columns only | [→](04_huim_only/README.md) |
| `05_baseline_100_data` | LLM feature ranking (no training) feeding the 06 arms | [→](05_baseline_100_data/README.md) |
| `06_llm_top10` | top-10 LLM-selected features | [→](06_llm_top10/README.md) |
| `06_1_llm_top5_datafree` | top-5 LLM features, prompt without data | [→](06_1_llm_top5_datafree/README.md) |
| `06_2_llm_top5_datainformed` | top-5 LLM features, prompt with 100 sample rows | [→](06_2_llm_top5_datainformed/README.md) |
| `representation/` | representation × head on a 50k all-years sample (bake-off, TabFM head) | [→](representation/README.md) |
| `aggregate/` | generated tables over all arms, paired tests | [→](aggregate/README.md) |

All models are trained and tested on 2020. 02 / 03 / 04 read their SHAP ranking and
HUIM patterns from `features.extraction_dir` under `../shap_huim_extractions/`:
**`02_2020_train`** (default, primary — mined on 2020 training days only, so no
validation or test day influences the selection) or `01_2024_baseline` (mined on 2024,
the method's original source; kept as a cross-year transfer check).

Each arm folder holds `config.yaml` (its feature set), `run.py`, `results.csv`,
`best_params.json` (Phase-1 selection), a generated `RESULTS.md`, and — gitignored —
`predictions/` with the test probabilities.

## Main findings (seeds 42/43/44, final fit on 10% of train)

| feature set | best deep model | best GBDT |
|---|---|---|
| all features (01b) | 0.629 (AutoInt, FT) | **0.646** (CatBoost) |
| SHAP top-10 (02) | 0.626 | 0.642 |
| SHAP top-10 + HUIM (03) | 0.624 | 0.642 |
| HUIM only (04) | 0.588 | 0.589 |
| LLM top-10 (06) | 0.627 | 0.636 |

(02–04: SHAP ranking and HUIM patterns mined on the 2020 training days.)

**Final fit on all of train** (`phase2_frac` 1.0, seeds 42/43/44; search unchanged):

| feature set | best deep model | best GBDT |
|---|---|---|
| all features (01b) | 0.663 (AutoInt) | **0.683** (XGBoost) |
| SHAP top-10 (02) | 0.674 (FT-Transformer) | **0.683** (XGBoost) |
| SHAP top-10 + HUIM (03) | 0.670 (AutoInt) | **0.683** (XGBoost) |
| HUIM only (04) | 0.590 | 0.591 |
| LLM top-10 (06) | 0.661 | 0.675 |

With all of train the SHAP top-10 lifts every deep model above its all-features score
(FT-Transformer +0.023, ResNet +0.029, MLP +0.024, AutoInt +0.004), while XGBoost stays at
0.683 on every feature set. HUIM still adds nothing measurable (03 − 02 between −0.011
and +0.009 for the deep models, ~0 for LightGBM / XGBoost; CatBoost +0.022 only because
its 02 score is unusually low). One cell is incomplete: FT-Transformer on 03 did not
train on seed 43 (AUC 0.500, validation loss flat from the first epoch) and stalled on
seed 44 (no epoch finished in hours, run twice), so that cell holds 2 seeds.

1. GBDTs beat every deep model on every feature set.
2. The HUIM pattern columns add no measurable signal: 03 − 02 is within the seed spread
   or negative for all seven models.
3. Ten SHAP features keep nearly all the signal of the full feature set.
4. Training data volume matters more than the feature set: fitting on all of train
   instead of 10% adds 2–5 AUC points to the strong models (e.g. XGBoost 0.644 → 0.683,
   AutoInt 0.629 → 0.663, 3 seeds).
5. Seed-to-seed std (0.001–0.010) far exceeds the test-sampling error (DeLong SE
   ≈ 0.0002); judge differences against the seed std.

### Patterns from another year (2024)

The method was first run with the SHAP ranking and HUIM patterns mined on **2024** data
and applied to 2020 models. To check whether patterns found in one year still help in
another, arms 02–04 were also run with that 2024 extraction (models still trained and
tested on 2020):

| feature set | best deep model | best GBDT |
|---|---|---|
| all features (01b), for comparison | 0.629 | 0.646 |
| SHAP top-10, 2024 ranking (02) | 0.626 | 0.645 |
| SHAP top-10 + HUIM, 2024 patterns (03) | 0.621 | 0.645 |
| HUIM only, 2024 patterns (04) | 0.553 | 0.553 |

The 2024 ranking still selects useful features, but the 2024 patterns transfer poorly:
on their own they reach 0.553, against 0.589 for patterns mined on 2020.

Full tables: [`aggregate/ABLATIONS.md`](aggregate/ABLATIONS.md).

## How to reproduce

Run from the repository root after downloading the dataset into
`Datasets/Aeolus/Flight_Tab/Tab/` and `uv sync`.

```bash
A=exp/Tab_exp/ablations
E24="--set features.extraction_dir=../../shap_huim_extractions/01_2024_baseline/csvs"

# 0. optional self-checks of the runner's library patches (print OK / IDENTICAL)
uv run python $A/common/verify_fast_loader.py --accelerator gpu
(cd $A && uv run python -m common.pretab_patch && uv run python -m common.mambular_patch)

# 1. inputs (both committed): 2020-train SHAP/HUIM extraction, LLM rankings
uv run python exp/Tab_exp/shap_huim_extractions/02_2020_train/run.py      # ~30 min
uv run python $A/05_baseline_100_data/run.py                              # needs Ollama qwen2.5:7b

# 2. seed 42: search + final fit for every arm (deep models; add
#    --models LightGBM,XGBoost,CatBoost for the GBDTs)
uv run python $A/01b_baseline_matched_training/run.py --seed 42
for arm in 02_shap_only 03_shap_plus_huim 04_huim_only; do
  uv run python $A/$arm/run.py --seed 42          # 2020-train extraction (default)
  uv run python $A/$arm/run.py --seed 42 $E24     # cross-year check (2024 extraction)
done
for arm in 06_llm_top10 06_1_llm_top5_datafree 06_2_llm_top5_datainformed; do
  uv run python $A/$arm/run.py --seed 42
done

# 3. seeds 43, 44 and the all-of-train final fit reuse the stored search:
#    same commands with  --seed 43 --reuse-params
#    and with           --reuse-params --set training.phase2_frac=1.0

# 4. tables
uv run python $A/aggregate.py          # aggregate/*, <arm>/RESULTS.md
uv run python $A/paired_test.py        # aggregate/paired_tests.*
uv run python $A/common/check_configs.py
```

Notes:
- Every run is deterministic; a finished `(model, seed, target, phase2_frac, extraction)`
  is skipped, so commands can be re-issued.
- Runs can go in parallel (locked writes, `.running__*` claims). On a 32 GB machine keep
  at most three or four concurrent runs; after a restart delete leftover `.running__*`.
- Runtime on one RTX 5060 Ti: search + final fit of the four deep models for one arm
  3–6 h; an extra seed 20–25 min; the three GBDTs 5–15 min per arm; an all-of-train final
  fit of the deep models 10–60 min per arm.
- `--target late|3class` selects other targets; results are kept per target.
