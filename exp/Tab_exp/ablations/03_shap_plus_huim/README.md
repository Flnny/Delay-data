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

Patterns mined on the 2020 training days. Δ HUIM = 03 − 02 (same SHAP features, with vs
without patterns), seed-matched:

| model | AUC | Δ HUIM |
|---|---|---|
| MLP | 0.608 | +0.003 |
| AutoInt | 0.620 | −0.005 |
| ResNet | 0.600 | +0.004 |
| FT-Transformer | 0.624 | −0.003 |
| LightGBM | 0.640 | −0.000 |
| XGBoost | 0.642 | +0.000 |
| CatBoost | 0.636 | −0.001 |

- The pattern columns add no measurable signal: every Δ HUIM is within the
  seed-to-seed spread (0.001–0.010) or negative.
- Against all features (01b): MLP / ResNet gain, but that gain comes from the SHAP
  selection (02); AutoInt and FT-Transformer are 0.005–0.009 below, the GBDTs
  0.002–0.010 below.
- With all of train in the final fit (3 seeds), Δ HUIM is +0.009 (MLP), +0.004 (AutoInt),
  −0.011 (ResNet), ~0 for XGBoost / LightGBM; CatBoost gains +0.022 over 02, but 02 is
  unusually weak for CatBoost — against all features (01b, 0.677) 03 reaches 0.678.
- **Incomplete cell:** FT-Transformer with all of train did not train on seed 43 (AUC
  0.500, validation loss flat at ~0.68 from the first epoch, early-stopped at epoch 6)
  and stalled on seed 44 (no epoch finished in several hours, on two attempts), while
  seed 42 trained normally (0.666). Its selected configuration (10 layers, lr 1e-3) looks
  unstable on the full training set; the cell keeps 2 seeds and is not used for
  conclusions.
- Paired DeLong tests: `../aggregate/paired_tests.md`. Seed noise, not test sampling,
  dominates (DeLong SE ≈ 0.0002).

### Patterns from another year (2024)

The same arm with the SHAP ranking and HUIM patterns taken from the 2024 extraction — a
different year than the one the models are trained and tested on (2020). It checks
whether patterns found in another year still help.

| model | AUC | Δ HUIM |
|---|---|---|
| MLP | 0.589 | −0.001 |
| AutoInt | 0.612 | **−0.014** |
| ResNet | 0.585 | −0.002 |
| FT-Transformer | 0.621 | −0.004 |
| LightGBM | 0.643 | −0.000 |
| XGBoost | 0.645 | −0.001 |
| CatBoost | 0.642 | −0.002 |

The 2024 patterns add nothing either and hurt AutoInt (−0.014 vs 02, −0.017 vs all
features).

Always-current numbers (incl. `phase2_frac` 1.0): [`RESULTS.md`](RESULTS.md).
