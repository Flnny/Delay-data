# LLM-Select vs SHAP — per-extraction summary

Top-10 Jaccard is LLM top-10 vs SHAP top-10.

| extraction | data | rho_datafree | tau_datafree | rho_datainformed | tau_datainformed | jaccard_datafree | jaccard_datainformed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 01_2024_baseline | 2024 | -0.054 | -0.03 | -0.127 | -0.077 | 0.333 | 0.333 |
| 02_2020_train | 2020 train days | -0.353 | -0.238 | -0.325 | -0.226 | 0.333 | 0.333 |

## SHAP agreement between extractions

- **01_2024_baseline vs 02_2020_train**: Spearman rho = 0.457 over 18 shared features; top-10 Jaccard = 0.54
  - only in 01_2024_baseline: ['CRS_ARR_TIME_MIN', 'OP_CARRIER', 'OP_CARRIER_FL_NUM']
  - only in 02_2020_train: ['DEST_INDEX', 'D_LATITUDE', 'O_WSPD']
