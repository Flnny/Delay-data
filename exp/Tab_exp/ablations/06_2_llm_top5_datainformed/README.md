# 06.2 — Top-5 LLM features, data-informed prompt

**Question.** Paired with [06.1](../06_1_llm_top5_datafree/README.md): the same LLM-Score
prompt, but with 100 raw training rows in context. Does seeing data lead to better
picks?

**Feature set** (`features.mode: llm_only`, `top_n: 5`), from
`../05_baseline_100_data/output/llm_feature_importance_datainformed.csv`:
**FL_WEEK, ORIGIN_INDEX, CRS_ARR_TIME_MIN, CRS_ELAPSED_TIME, O_PRCP**.

Training settings: `../common/base_config.yaml`, no overrides.

```bash
uv run python exp/Tab_exp/ablations/06_2_llm_top5_datainformed/run.py --seed 42
```

## Findings (3 seeds, final fit on 10% of train)

| MLP | AutoInt | ResNet | FT-Transformer | LightGBM | XGBoost | CatBoost |
|---|---|---|---|---|---|---|
| 0.558 | 0.596 | 0.556 | 0.598 | 0.604 | **0.605** | 0.603 |

- Showing the LLM data did not produce a better feature set for the models that learn
  from five features at all: AutoInt, FT-Transformer and CatBoost are 0.003–0.007 below
  the data-free top-5, LightGBM / XGBoost +0.002. MLP and ResNet are +0.02–0.03 higher,
  but both stay far below every other model.

Always-current numbers: [`RESULTS.md`](RESULTS.md).
