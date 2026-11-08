# LLM-Select vs SHAP + HUIM — feature-importance comparison

Method: **LLM-Score** (taekb/llm-select) on local Ollama **qwen2.5:7b**, two prompt variants (data-free, data-informed), 3 repeats/feature. Compared against the 2020 train days SHAP importances (02_2020_train) and HUIM delay-driver / delay-protector itemsets.

## 1. Per-feature comparison table

| Feature | LLM_datafree | LLM_datainformed | SHAP_mean_abs | SHAP_rank | HUIM_driver_freq | HUIM_protector_freq |
| --- | --- | --- | --- | --- | --- | --- |
| O_LONGITUDE | 0.233 | 0.167 | 0.134 | 1 | 0.53 | 0.38 |
| O_LATITUDE | 0.383 | 0.3 | 0.102 | 2 | 0.44 | 0.33 |
| CRS_ELAPSED_TIME | 0.633 | 0.767 | 0.098 | 3 | 0.21 | 0.23 |
| O_TEMP | 0.317 | 0.233 | 0.083 | 4 | 0.28 | 0.25 |
| D_LONGITUDE | 0.183 | 0.167 | 0.07 | 5 | 0.21 | 0.19 |
| D_LATITUDE | 0.183 | 0.167 | 0.051 | 6 | 0.21 | 0.1 |
| DEST_INDEX | 0.683 | 0.633 | 0.043 | 7 | 0.01 | 0.04 |
| ORIGIN_INDEX | 0.8 | 0.75 | 0.035 | 8 | 0.0 | 0.0 |
| O_WSPD | 0.75 | 0.583 | 0.03 | 9 | 0.04 | 0.06 |
| CRS_DEP_TIME_MIN | 0.717 | 0.6 | 0.03 | 10 | 0.05 | 0.04 |
| OP_CARRIER_FL_NUM | 0.117 | 0.133 | 0.028 | 11 | 0.0 | 0.0 |
| D_TEMP | 0.317 | 0.217 | 0.026 | 12 | 0.02 | 0.03 |
| D_WSPD | 0.283 | 0.367 | 0.023 | 13 | 0.02 | 0.06 |
| CRS_ARR_TIME_MIN | 0.6 | 0.733 | 0.018 | 14 | 0.01 | 0.01 |
| FL_WEEK | 0.65 | 0.633 | 0.017 | 15 | 0.0 | 0.03 |
| OP_CARRIER | 0.717 | 0.583 | 0.014 | 16 | 0.0 | 0.0 |
| D_PRCP | 0.517 | 0.617 | 0.01 | 17 | 0.49 | 0.0 |
| O_PRCP | 0.767 | 0.65 | 0.003 | 18 | 0.0 | 0.36 |
| FL_DAY | 0.233 | 0.133 |  |  | 0.0 | 0.0 |

`HUIM_*_freq` = fraction of the top-100 itemsets (by utility) that contain the feature.

## 2. Rank agreement with SHAP

- LLM data-free vs SHAP:      Spearman rho = **-0.353**, Kendall tau = -0.238
- LLM data-informed vs SHAP:  Spearman rho = **-0.325**, Kendall tau = -0.226

## 3. Top-10 feature-set overlap (Jaccard)

- LLM data-free ∩ SHAP:            ['CRS_DEP_TIME_MIN', 'CRS_ELAPSED_TIME', 'DEST_INDEX', 'ORIGIN_INDEX', 'O_WSPD']  (Jaccard 0.33)
- LLM data-informed ∩ SHAP:        ['CRS_DEP_TIME_MIN', 'CRS_ELAPSED_TIME', 'DEST_INDEX', 'ORIGIN_INDEX', 'O_WSPD']  (Jaccard 0.33)
- LLM data-free ∩ HUIM (combined): ['CRS_DEP_TIME_MIN', 'CRS_ELAPSED_TIME', 'D_PRCP', 'O_PRCP', 'O_WSPD']  (Jaccard 0.33)
- data-free ∩ data-informed:       ['CRS_ARR_TIME_MIN', 'CRS_DEP_TIME_MIN', 'CRS_ELAPSED_TIME', 'DEST_INDEX', 'D_PRCP', 'FL_WEEK', 'OP_CARRIER', 'ORIGIN_INDEX', 'O_PRCP', 'O_WSPD']  (Jaccard 1.00)

## 4. Where LLM and SHAP agree / disagree

- **Closest rank agreement:** ['D_TEMP', 'DEST_INDEX', 'D_WSPD', 'CRS_ELAPSED_TIME', 'CRS_ARR_TIME_MIN']
- **Largest rank gaps (LLM data-free rank vs SHAP rank):**
  - O_PRCP: LLM #2 vs SHAP #18
  - O_LONGITUDE: LLM #15 vs SHAP #1
  - OP_CARRIER: LLM #4 vs SHAP #16
  - D_LONGITUDE: LLM #16 vs SHAP #5
  - D_LATITUDE: LLM #16 vs SHAP #6

## 5. Do the LLM's important features drive the HUIM itemsets?

For each high-utility itemset we show the mean LLM (data-free) score of its constituent features — high values mean the LLM independently rated the itemset's features as predictive.

### Delay drivers (top 15 by utility)

| # | Itemset | Utility | Mean LLM (data-free) score of its features |
|---|---------|--------:|:------------------------------------------:|
| 1 | O_LONGITUDE=VeryHigh | 503177 | 0.23 |
| 2 | O_LONGITUDE=VeryHigh + D_PRCP=VeryLow | 491619 | 0.37 |
| 3 | O_LATITUDE=High + O_LONGITUDE=VeryHigh | 375912 | 0.31 |
| 4 | CRS_ELAPSED_TIME=VeryHigh | 367914 | 0.63 |
| 5 | O_LATITUDE=High + O_LONGITUDE=VeryHigh + D_PRCP=VeryLow | 357978 | 0.38 |
| 6 | CRS_ELAPSED_TIME=VeryHigh + D_PRCP=VeryLow | 356958 | 0.57 |
| 7 | D_LONGITUDE=VeryHigh + O_LONGITUDE=VeryHigh | 311062 | 0.21 |
| 8 | D_LONGITUDE=VeryHigh + O_LONGITUDE=VeryHigh + D_PRCP=VeryLow | 310048 | 0.31 |
| 9 | O_TEMP=VeryLow + D_PRCP=VeryLow | 303086 | 0.42 |
| 10 | O_TEMP=VeryLow | 299991 | 0.32 |
| 11 | O_LATITUDE=VeryHigh + D_PRCP=VeryLow | 287750 | 0.45 |
| 12 | O_LATITUDE=VeryHigh | 285387 | 0.38 |
| 13 | O_LONGITUDE=Mid + O_LATITUDE=VeryHigh | 267328 | 0.31 |
| 14 | O_LONGITUDE=Mid + O_LATITUDE=VeryHigh + D_PRCP=VeryLow | 264025 | 0.38 |
| 15 | O_TEMP=VeryLow + O_LATITUDE=VeryHigh | 263328 | 0.35 |

### Delay protectors (top 15 by utility)

| # | Itemset | Utility | Mean LLM (data-free) score of its features |
|---|---------|--------:|:------------------------------------------:|
| 1 | O_LONGITUDE=VeryLow | 324585 | 0.23 |
| 2 | CRS_ELAPSED_TIME=VeryLow | 302100 | 0.63 |
| 3 | O_LONGITUDE=Low | 266966 | 0.23 |
| 4 | O_LATITUDE=VeryLow | 247402 | 0.38 |
| 5 | O_LONGITUDE=Low + O_PRCP=VeryLow | 247006 | 0.50 |
| 6 | CRS_ELAPSED_TIME=VeryLow + O_PRCP=VeryLow | 246009 | 0.70 |
| 7 | O_LONGITUDE=VeryLow + O_PRCP=VeryLow | 235732 | 0.50 |
| 8 | O_LATITUDE=Low | 217210 | 0.38 |
| 9 | O_TEMP=VeryHigh | 217107 | 0.32 |
| 10 | O_LATITUDE=VeryLow + O_PRCP=VeryLow | 197438 | 0.57 |
| 11 | O_TEMP=VeryHigh + O_PRCP=VeryLow | 195957 | 0.54 |
| 12 | O_LATITUDE=Low + O_PRCP=VeryLow | 190252 | 0.57 |
| 13 | O_LATITUDE=VeryLow + O_TEMP=VeryHigh | 166070 | 0.35 |
| 14 | O_TEMP=High | 165340 | 0.32 |
| 15 | D_LATITUDE=VeryLow | 152877 | 0.18 |

## 6. Caveats

- SHAP/HUIM were mined on 2020 train days only (no val/test days), the same year as the LLM sample rows and the ablations.
- **Small local model:** qwen2.5:7b is far weaker than the paper's GPT-4; treat LLM scores as a directional signal (3 repeats + std quantify run-to-run variance).
- **HUIM is itemset-level:** the per-feature frequency is a projection of multi-feature patterns onto single features.
