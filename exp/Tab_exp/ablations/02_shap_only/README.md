# 02 — Top-10 SHAP features

**Question.** Does restricting the model to the 10 features with the largest mean |SHAP|
help or hurt? Paired with 03 it separates the effect of *SHAP feature selection* from the
effect of *adding HUIM patterns*.

**Feature set** (`features.mode: shap_only`, `top_shap_n: 10`): the first 10 rows of
`shap_feature_importance.csv` in `features.extraction_dir`. Two extractions are used
(see `../../shap_huim_extractions/`):

| extraction | mined on | top-10 features |
|---|---|---|
| `01_2024_baseline` (default) | 2024 | O_LONGITUDE, OP_CARRIER, CRS_ELAPSED_TIME, O_TEMP, OP_CARRIER_FL_NUM, CRS_DEP_TIME_MIN, O_LATITUDE, CRS_ARR_TIME_MIN, D_LONGITUDE, ORIGIN_INDEX |
| `02_2020_train` | 2020, train days only | O_LONGITUDE, O_LATITUDE, CRS_ELAPSED_TIME, O_TEMP, D_LONGITUDE, D_LATITUDE, DEST_INDEX, ORIGIN_INDEX, O_WSPD, CRS_DEP_TIME_MIN |

Training settings: `../common/base_config.yaml`, no overrides.

## Run

```bash
A=exp/Tab_exp/ablations/02_shap_only
E20="--set features.extraction_dir=../../shap_huim_extractions/02_2020_train/csvs"
uv run python $A/run.py --seed 42           # 2024 extraction
uv run python $A/run.py --seed 42 $E20      # 2020-train extraction
# GBDTs: --models LightGBM,XGBoost,CatBoost ; more seeds: --seed 43 --reuse-params
```

## Findings (3 seeds, final fit on 10% of train; Δ = 02 − 01b)

| model | 2024 SHAP | Δ | 2020-train SHAP | Δ |
|---|---|---|---|---|
| MLP | 0.589 | −0.001 | 0.605 | **+0.014** |
| AutoInt | 0.626 | −0.003 | 0.625 | −0.004 |
| ResNet | 0.587 | −0.006 | 0.596 | +0.002 |
| FT-Transformer | 0.625 | −0.004 | 0.626 | −0.002 |
| XGBoost | 0.645 | +0.001 | 0.642 | −0.003 |
| CatBoost | 0.644 | −0.002 | 0.637 | −0.009 |

- Ten features keep almost all the signal: 2024-SHAP top-10 is within ~0.005 of all
  features for every model.
- The 2020-train selection helps the plain MLP (+0.014) but not the stronger models.
- With all of train in the final fit, the 2020-train selection helps the deep models
  (seed 42: FT-Transformer 0.674 vs 0.651 on all features, ResNet 0.651 vs 0.610) —
  see `RESULTS.md` once all seeds are in.

Always-current numbers: [`RESULTS.md`](RESULTS.md).
