# 06.1 — Top-5 LLM features, data-free prompt

**Question.** Paired with [06.2](../06_2_llm_top5_datainformed/README.md): does showing
the LLM 100 real training rows change which features it picks — and are those picks
better? Top-5 is where the two prompt variants diverge (they share only ORIGIN_INDEX and
O_PRCP; at top-10 they agree completely, see [06](../06_llm_top10/README.md)).

**Feature set** (`features.mode: llm_only`, `top_n: 5`), from
`../05_baseline_100_data/output/llm_feature_importance_datafree.csv` (feature name +
description + task context only): **OP_CARRIER, ORIGIN_INDEX, CRS_DEP_TIME_MIN, O_PRCP,
O_WSPD**.

Training settings: `../common/base_config.yaml`, no overrides.

```bash
uv run python exp/Tab_exp/ablations/06_1_llm_top5_datafree/run.py --seed 42
```

## Findings (3 seeds, final fit on 10% of train)

| MLP | AutoInt | ResNet | FT-Transformer | LightGBM | XGBoost | CatBoost |
|---|---|---|---|---|---|---|
| 0.530 | 0.603 | 0.532 | 0.604 | 0.602 | 0.603 | **0.606** |

- Five features cost 2–4 AUC points against all features for the strong models; MLP and
  ResNet fall to near the base rate (accuracy ≈ majority-class share).
- Slightly better than the data-informed top-5 for AutoInt / FT / CatBoost
  (0.603 vs 0.596, 0.604 vs 0.598, 0.606 vs 0.603), worse for MLP / ResNet.

Always-current numbers: [`RESULTS.md`](RESULTS.md).
