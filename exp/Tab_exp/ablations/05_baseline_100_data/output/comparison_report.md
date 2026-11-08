# LLM-Select vs SHAP + HUIM — feature-importance comparison

Method: **LLM-Score** (taekb/llm-select) on local Ollama **qwen2.5:7b**, two prompt variants (data-free, data-informed), 3 repeats/feature. Compared against the 2024 SHAP importances (01_2024_baseline) and HUIM delay-driver / delay-protector itemsets.

## 1. Per-feature comparison table

| Feature | LLM_datafree | LLM_datainformed | SHAP_mean_abs | SHAP_rank | HUIM_driver_freq | HUIM_protector_freq |
| --- | --- | --- | --- | --- | --- | --- |
| O_LONGITUDE | 0.233 | 0.167 | 0.11 | 1 | 0.59 | 0.52 |
| OP_CARRIER | 0.717 | 0.583 | 0.065 | 2 | 0.09 | 0.29 |
| CRS_ELAPSED_TIME | 0.633 | 0.767 | 0.063 | 3 | 0.37 | 0.35 |
| O_TEMP | 0.317 | 0.233 | 0.05 | 4 | 0.12 | 0.27 |
| OP_CARRIER_FL_NUM | 0.117 | 0.133 | 0.047 | 5 | 0.0 | 0.0 |
| CRS_DEP_TIME_MIN | 0.717 | 0.6 | 0.044 | 6 | 0.37 | 0.29 |
| O_LATITUDE | 0.383 | 0.3 | 0.038 | 7 | 0.28 | 0.27 |
| CRS_ARR_TIME_MIN | 0.6 | 0.733 | 0.035 | 8 | 0.3 | 0.19 |
| D_LONGITUDE | 0.183 | 0.167 | 0.029 | 9 | 0.06 | 0.26 |
| ORIGIN_INDEX | 0.8 | 0.75 | 0.028 | 10 | 0.05 | 0.08 |
| FL_DAY | 0.233 | 0.133 | 0.027 | 11 | 0.0 | 0.06 |
| O_WSPD | 0.75 | 0.583 | 0.025 | 12 | 0.12 | 0.2 |
| O_PRCP | 0.767 | 0.65 | 0.019 | 13 | 0.03 | 0.55 |
| D_TEMP | 0.317 | 0.217 | 0.018 | 14 | 0.23 | 0.09 |
| DEST_INDEX | 0.683 | 0.633 | 0.018 | 15 | 0.0 | 0.03 |
| D_LATITUDE | 0.183 | 0.167 | 0.015 | 16 | 0.05 | 0.19 |
| D_PRCP | 0.517 | 0.617 | 0.012 | 17 | 0.0 | 0.47 |
| D_WSPD | 0.283 | 0.367 | 0.007 | 18 | 0.04 | 0.14 |
| FL_WEEK | 0.65 | 0.633 | 0.001 | 19 | 0.04 | 0.03 |

`HUIM_*_freq` = fraction of the top-100 itemsets (by utility) that contain the feature.

## 2. Rank agreement with SHAP

- LLM data-free vs SHAP:      Spearman rho = **-0.054**, Kendall tau = -0.030
- LLM data-informed vs SHAP:  Spearman rho = **-0.127**, Kendall tau = -0.077

## 3. Top-10 feature-set overlap (Jaccard)

- LLM data-free ∩ SHAP:            ['CRS_ARR_TIME_MIN', 'CRS_DEP_TIME_MIN', 'CRS_ELAPSED_TIME', 'OP_CARRIER', 'ORIGIN_INDEX']  (Jaccard 0.33)
- LLM data-informed ∩ SHAP:        ['CRS_ARR_TIME_MIN', 'CRS_DEP_TIME_MIN', 'CRS_ELAPSED_TIME', 'OP_CARRIER', 'ORIGIN_INDEX']  (Jaccard 0.33)
- LLM data-free ∩ HUIM (combined): ['CRS_ARR_TIME_MIN', 'CRS_DEP_TIME_MIN', 'CRS_ELAPSED_TIME', 'D_PRCP', 'OP_CARRIER', 'O_PRCP']  (Jaccard 0.43)
- data-free ∩ data-informed:       ['CRS_ARR_TIME_MIN', 'CRS_DEP_TIME_MIN', 'CRS_ELAPSED_TIME', 'DEST_INDEX', 'D_PRCP', 'FL_WEEK', 'OP_CARRIER', 'ORIGIN_INDEX', 'O_PRCP', 'O_WSPD']  (Jaccard 1.00)

## 4. Where LLM and SHAP agree / disagree

- **Closest rank agreement:** ['CRS_ARR_TIME_MIN', 'D_LATITUDE', 'OP_CARRIER', 'CRS_DEP_TIME_MIN', 'D_TEMP']
- **Largest rank gaps (LLM data-free rank vs SHAP rank):**
  - O_LONGITUDE: LLM #15 vs SHAP #1
  - OP_CARRIER_FL_NUM: LLM #19 vs SHAP #5
  - FL_WEEK: LLM #7 vs SHAP #19
  - O_PRCP: LLM #2 vs SHAP #13
  - ORIGIN_INDEX: LLM #1 vs SHAP #10

## 5. Do the LLM's important features drive the HUIM itemsets?

For each high-utility itemset we show the mean LLM (data-free) score of its constituent features — high values mean the LLM independently rated the itemset's features as predictive.

### Delay drivers (top 15 by utility)

| # | Itemset | Utility | Mean LLM (data-free) score of its features |
|---|---------|--------:|:------------------------------------------:|
| 1 | O_PRCP=VeryLow | 99388 | 0.77 |
| 2 | ORIGIN_INDEX=JFK + O_LONGITUDE=VeryHigh | 98337 | 0.52 |
| 3 | OP_CARRIER=DL + O_LONGITUDE=VeryHigh | 98208 | 0.47 |
| 4 | CRS_ARR_TIME_MIN=Mid + O_LATITUDE=High + O_LONGITUDE=VeryHigh | 98085 | 0.41 |
| 5 | CRS_ELAPSED_TIME=High | 97904 | 0.63 |
| 6 | O_WSPD=VeryHigh + O_LATITUDE=High + O_LONGITUDE=VeryHigh | 96911 | 0.46 |
| 7 | O_PRCP=VeryLow + O_LONGITUDE=VeryHigh | 95991 | 0.50 |
| 8 | CRS_ELAPSED_TIME=VeryHigh + O_WSPD=VeryHigh + O_LONGITUDE=VeryHigh | 95721 | 0.54 |
| 9 | CRS_DEP_TIME_MIN=High + D_TEMP=VeryHigh + O_LONGITUDE=VeryHigh | 95364 | 0.42 |
| 10 | FL_WEEK=1 + O_LONGITUDE=VeryHigh | 95122 | 0.44 |
| 11 | CRS_ARR_TIME_MIN=Mid + O_LONGITUDE=VeryHigh | 95117 | 0.42 |
| 12 | OP_CARRIER=B6 + O_LONGITUDE=VeryHigh | 94550 | 0.47 |
| 13 | CRS_ELAPSED_TIME=VeryHigh + O_WSPD=VeryHigh | 93895 | 0.69 |
| 14 | OP_CARRIER=AA + CRS_ELAPSED_TIME=VeryHigh | 93825 | 0.68 |
| 15 | O_TEMP=VeryLow + CRS_ELAPSED_TIME=VeryHigh + O_LONGITUDE=VeryHigh | 93056 | 0.39 |

### Delay protectors (top 15 by utility)

| # | Itemset | Utility | Mean LLM (data-free) score of its features |
|---|---------|--------:|:------------------------------------------:|
| 1 | CRS_ARR_TIME_MIN=VeryLow + O_LATITUDE=VeryLow + CRS_DEP_TIME_MIN=VeryLow + O_LONGITUDE=High + O_PRCP=VeryLow + D_PRCP=VeryLow | 9999 | 0.54 |
| 2 | FL_DAY=22 + O_LATITUDE=Low + O_LONGITUDE=Low + O_PRCP=VeryLow + D_PRCP=VeryLow | 9999 | 0.43 |
| 3 | O_WSPD=VeryLow + CRS_ARR_TIME_MIN=VeryLow + CRS_ELAPSED_TIME=Low + O_PRCP=VeryLow + D_PRCP=VeryLow | 9999 | 0.65 |
| 4 | O_TEMP=VeryHigh + FL_WEEK=6 + O_PRCP=VeryLow | 9999 | 0.58 |
| 5 | CRS_ARR_TIME_MIN=VeryLow + O_TEMP=Mid + O_LONGITUDE=Mid | 9999 | 0.38 |
| 6 | CRS_DEP_TIME_MIN=VeryLow + O_LATITUDE=Mid + CRS_ELAPSED_TIME=Low + O_PRCP=VeryLow + D_PRCP=VeryLow | 9998 | 0.60 |
| 7 | OP_CARRIER=YX + O_WSPD=Low + CRS_ELAPSED_TIME=Low + O_PRCP=VeryLow + D_PRCP=VeryLow | 9997 | 0.68 |
| 8 | D_LONGITUDE=VeryLow + CRS_DEP_TIME_MIN=Low + O_LONGITUDE=Low + O_PRCP=VeryLow + D_PRCP=VeryLow | 9997 | 0.48 |
| 9 | O_WSPD=VeryLow + CRS_ARR_TIME_MIN=VeryLow + CRS_DEP_TIME_MIN=VeryLow + O_LONGITUDE=High + O_PRCP=VeryLow | 9994 | 0.61 |
| 10 | FL_DAY=29 + O_LATITUDE=VeryLow + O_PRCP=VeryLow + D_PRCP=VeryLow | 9992 | 0.47 |
| 11 | D_LONGITUDE=VeryLow + CRS_ELAPSED_TIME=VeryLow + OP_CARRIER=WN + D_PRCP=VeryLow | 9991 | 0.51 |
| 12 | D_LATITUDE=VeryHigh + O_TEMP=Mid + O_LONGITUDE=Low + O_PRCP=VeryLow | 9991 | 0.37 |
| 13 | D_LONGITUDE=VeryLow + O_LONGITUDE=VeryLow + D_LATITUDE=VeryLow + O_LATITUDE=VeryLow + O_TEMP=High + O_PRCP=VeryLow | 9991 | 0.34 |
| 14 | CRS_DEP_TIME_MIN=VeryLow + O_LATITUDE=Low + D_PRCP=VeryLow | 9988 | 0.54 |
| 15 | O_WSPD=Low + D_LONGITUDE=Mid + O_LONGITUDE=Low | 9988 | 0.39 |

## 6. Caveats

- **Year mismatch:** SHAP/HUIM were mined on 2024 data; the LLM sample rows are 2020. Feature semantics are year-independent, so the feature-level comparison holds, but exact magnitudes are not strictly aligned.
- **Small local model:** qwen2.5:7b is far weaker than the paper's GPT-4; treat LLM scores as a directional signal (3 repeats + std quantify run-to-run variance).
- **HUIM is itemset-level:** the per-feature frequency is a projection of multi-feature patterns onto single features.
