# 03 — Top-10 SHAP features + HUIM patterns (proposed method)

**Question.** Do high-utility itemsets mined from SHAP values add predictive signal on
top of the SHAP-selected features?

**Feature set** (`features.mode: shap_plus_huim`):
- the 10 SHAP features of [02](../02_shap_only/README.md), plus
- one binary column per pattern for the top-20 **delay-driver** patterns (mined from
  positive SHAP values) and the top-20 **delay-protector** patterns (negative SHAP
  values), read from `huim_delay_drivers.csv` / `huim_delay_protectors.csv` in
  `features.extraction_dir`. A pattern such as `O_LONGITUDE=VeryHigh + D_PRCP=VeryLow`
  is 1 for a flight when all its conditions hold; continuous conditions use 5 quantile
  bins fit on the train split. Patterns that never fire in train are dropped (3 of 40
  for the 2024 extraction, 0 for 2020-train).

Extractions: `01_2024_baseline` (2024, default) and `02_2020_train` (2020 train days
only, leakage-safe). The 2024 pattern files are sorted lexicographically by utility (a
PAMI quirk), so their "top-20" are not the 20 highest-utility patterns; the 2020-train
files are sorted numerically. See `../../shap_huim_extractions/`.

Training settings: `../common/base_config.yaml`, no overrides.

## Run

```bash
A=exp/Tab_exp/ablations/03_shap_plus_huim
E20="--set features.extraction_dir=../../shap_huim_extractions/02_2020_train/csvs"
uv run python $A/run.py --seed 42
uv run python $A/run.py --seed 42 $E20
# GBDTs: --models LightGBM,XGBoost,CatBoost ; more seeds: --seed 43 --reuse-params
```

## Findings (3 seeds, final fit on 10% of train)

Δ HUIM = 03 − 02 (same SHAP features, with vs without patterns), seed-matched:

| model | 2024 patterns: AUC | Δ HUIM | 2020-train patterns: AUC | Δ HUIM |
|---|---|---|---|---|
| MLP | 0.589 | −0.001 | 0.608 | +0.003 |
| AutoInt | 0.612 | **−0.014** | 0.620 | −0.005 |
| ResNet | 0.585 | −0.002 | 0.600 | +0.004 |
| FT-Transformer | 0.621 | −0.004 | 0.624 | −0.003 |
| LightGBM | 0.643 | −0.000 | 0.640 | −0.000 |
| XGBoost | 0.645 | −0.001 | 0.642 | +0.000 |
| CatBoost | 0.642 | −0.002 | 0.636 | −0.001 |

- The pattern columns add no measurable signal: every Δ HUIM is within the
  seed-to-seed spread (0.001–0.010) or negative; with 2024 patterns AutoInt loses 0.014.
- Against all features (01b): the deep models are below the reference (AutoInt −0.017
  with 2024 patterns), except MLP / ResNet with 2020-train patterns, whose gain comes
  from the SHAP selection (02); with 2024 patterns LightGBM / XGBoost are within +0.001
  and CatBoost 0.004 below, with 2020-train patterns the GBDTs are 0.002–0.010 below.
- Paired DeLong tests: `../aggregate/paired_tests.md`. Seed noise, not test sampling,
  dominates (DeLong SE ≈ 0.0002).

Always-current numbers (incl. `phase2_frac` 1.0): [`RESULTS.md`](RESULTS.md).
