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
  bins fit on the train split. Patterns that never fire in train are dropped.

Extractions (see `../../shap_huim_extractions/`); the models are always trained and
tested on 2020:
- **`02_2020_train` (default, primary)** — mined on 2020 training days only, so no
  validation or test day influences the patterns; all 40 patterns fire in train.
- `01_2024_baseline` (cross-year check) — mined on 2024. 3 of its 40 patterns never fire
  in 2020 train, and its pattern files are sorted lexicographically by utility (a PAMI
  quirk), so its "top-20" are not the 20 highest-utility patterns.

Training settings: `../common/base_config.yaml`, no overrides.

## Run

```bash
A=exp/Tab_exp/ablations/03_shap_plus_huim
E24="--set features.extraction_dir=../../shap_huim_extractions/01_2024_baseline/csvs"
uv run python $A/run.py --seed 42           # 2020-train patterns (default)
uv run python $A/run.py --seed 42 $E24      # cross-year check with 2024 patterns
# GBDTs: --models LightGBM,XGBoost,CatBoost ; more seeds: --seed 43 --reuse-params
```

## Findings (3 seeds, final fit on 10% of train)

Δ HUIM = 03 − 02 (same SHAP features, with vs without patterns), seed-matched:

| model | 2020-train patterns: AUC | Δ HUIM | 2024 patterns: AUC | Δ HUIM |
|---|---|---|---|---|
| MLP | 0.608 | +0.003 | 0.589 | −0.001 |
| AutoInt | 0.620 | −0.005 | 0.612 | **−0.014** |
| ResNet | 0.600 | +0.004 | 0.585 | −0.002 |
| FT-Transformer | 0.624 | −0.003 | 0.621 | −0.004 |
| LightGBM | 0.640 | −0.000 | 0.643 | −0.000 |
| XGBoost | 0.642 | +0.000 | 0.645 | −0.001 |
| CatBoost | 0.636 | −0.001 | 0.642 | −0.002 |

- The pattern columns add no measurable signal: every Δ HUIM is within the
  seed-to-seed spread (0.001–0.010) or negative.
- Against all features (01b): with 2020-train patterns MLP / ResNet gain, but that gain
  comes from the SHAP selection (02); AutoInt and FT-Transformer are 0.005–0.009 below,
  the GBDTs 0.002–0.010 below.
- Cross-year check: 2024 patterns are no better and hurt AutoInt (−0.014 vs 02,
  −0.017 vs 01b).
- With all of train in the final fit (GBDTs, 3 seeds), Δ HUIM stays ~0 for XGBoost and
  LightGBM; CatBoost gains +0.022 over 02, but 02 is unusually weak for CatBoost — against
  all features (01b, 0.677) 03 reaches 0.678.
- Paired DeLong tests: `../aggregate/paired_tests.md`. Seed noise, not test sampling,
  dominates (DeLong SE ≈ 0.0002).

Always-current numbers (incl. `phase2_frac` 1.0): [`RESULTS.md`](RESULTS.md).
