# 02 — Top-10 SHAP features

**Question.** Does restricting the model to the 10 features with the largest mean |SHAP|
help or hurt? Paired with 03 it separates the effect of *SHAP feature selection* from the
effect of *adding HUIM patterns*.

**Feature set** (`features.mode: shap_only`, `top_shap_n: 10`): the first 10 rows of
`shap_feature_importance.csv` in `features.extraction_dir` (see
`../../shap_huim_extractions/`):

| extraction | mined on | top-10 features |
|---|---|---|
| **`02_2020_train`** (default, primary) | 2020, training days only | O_LONGITUDE, O_LATITUDE, CRS_ELAPSED_TIME, O_TEMP, D_LONGITUDE, D_LATITUDE, DEST_INDEX, ORIGIN_INDEX, O_WSPD, CRS_DEP_TIME_MIN |
| `01_2024_baseline` (cross-year check) | 2024 | O_LONGITUDE, OP_CARRIER, CRS_ELAPSED_TIME, O_TEMP, OP_CARRIER_FL_NUM, CRS_DEP_TIME_MIN, O_LATITUDE, CRS_ARR_TIME_MIN, D_LONGITUDE, ORIGIN_INDEX |

The models are always trained and tested on 2020. The primary extraction comes from the
same year's training days, so no validation or test day influences the selection; the
2024 extraction checks whether a ranking from another year transfers.

Training settings: `../common/base_config.yaml`, no overrides.

## Run

```bash
A=exp/Tab_exp/ablations/02_shap_only
E24="--set features.extraction_dir=../../shap_huim_extractions/01_2024_baseline/csvs"
uv run python $A/run.py --seed 42           # 2020-train extraction (default)
uv run python $A/run.py --seed 42 $E24      # cross-year check with the 2024 extraction
# GBDTs: --models LightGBM,XGBoost,CatBoost ; more seeds: --seed 43 --reuse-params
```

## Findings (3 seeds, final fit on 10% of train)

SHAP ranking from the 2020 training days; Δ = 02 − 01b (all features), seed-matched:

| model | AUC | Δ vs all features |
|---|---|---|
| MLP | 0.605 | **+0.014** |
| AutoInt | 0.625 | −0.004 |
| ResNet | 0.596 | +0.002 |
| FT-Transformer | 0.626 | −0.002 |
| XGBoost | 0.642 | −0.003 |
| CatBoost | 0.637 | −0.009 |

- Ten features keep almost all the signal of the full feature set.
- With 10% of train in the final fit, the selection helps the plain MLP (+0.014) but not
  the stronger models.
- With all of train in the final fit (3 seeds) it lifts every deep model above all
  features: FT-Transformer 0.674 vs 0.651, ResNet 0.647 vs 0.618, MLP 0.625 vs 0.601,
  AutoInt 0.667 vs 0.663; XGBoost is unchanged at 0.683.

### Ranking from another year (2024)

The same arm with the SHAP ranking taken from the 2024 extraction instead — a different
year than the one the models are trained and tested on (2020). It checks whether a
ranking found in another year still selects good features for 2020.

| model | AUC | Δ vs all features |
|---|---|---|
| MLP | 0.589 | −0.001 |
| AutoInt | 0.626 | −0.003 |
| ResNet | 0.587 | −0.006 |
| FT-Transformer | 0.625 | −0.004 |
| XGBoost | 0.645 | +0.001 |
| CatBoost | 0.644 | −0.002 |

The 2024 ranking also keeps nearly all the signal (within ~0.006 of all features) but,
unlike the 2020 ranking, gives the MLP no gain.

Always-current numbers: [`RESULTS.md`](RESULTS.md).
