# 06 — Top-10 LLM-selected features

**Question.** Is a feature set chosen by a 7B LLM from feature *names* as predictive as
the SHAP-selected one (arm 02)?

**Feature set** (`features.mode: llm_only`, `top_n: 10`): the first 10 rows of
`../05_baseline_100_data/output/llm_feature_importance_datafree.csv` —
OP_CARRIER, FL_WEEK, ORIGIN_INDEX, DEST_INDEX, CRS_DEP_TIME_MIN, CRS_ARR_TIME_MIN,
CRS_ELAPSED_TIME, O_PRCP, O_WSPD, D_PRCP. At top-10 the data-free and data-informed
prompts select the identical set, so this one arm covers both
(they diverge at top-5: [06.1](../06_1_llm_top5_datafree/README.md),
[06.2](../06_2_llm_top5_datainformed/README.md)).

Training settings: `../common/base_config.yaml`, no overrides.

```bash
uv run python exp/Tab_exp/ablations/06_llm_top10/run.py --seed 42
uv run python exp/Tab_exp/ablations/06_llm_top10/run.py --seed 42 --models LightGBM,XGBoost,CatBoost
```

## Findings (3 seeds, final fit on 10% of train)

| model | LLM top-10 | 2024-SHAP top-10 (02) | all features (01b) |
|---|---|---|---|
| MLP | 0.567 | 0.589 | 0.591 |
| AutoInt | **0.627** | 0.626 | 0.629 |
| ResNet | 0.563 | 0.587 | 0.593 |
| FT-Transformer | **0.625** | 0.625 | 0.629 |
| XGBoost | 0.634 | 0.645 | 0.644 |
| CatBoost | 0.636 | 0.644 | 0.646 |

- On the attention models (AutoInt, FT-Transformer) the LLM set matches SHAP.
- It trails SHAP by ~0.01 with GBDTs and by ~0.02 with MLP / ResNet. Unlike the SHAP
  set it contains no coordinates or temperatures (SHAP ranks O_LONGITUDE first).

Always-current numbers: [`RESULTS.md`](RESULTS.md).
