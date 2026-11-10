# 04 — HUIM patterns only

**Question.** How much of the delay signal do the mined patterns carry on their own,
without any original feature?

**Feature set** (`features.mode: huim_only`): only the binary pattern columns of
[03](../03_shap_plus_huim/README.md) — top-20 delay-driver + top-20 delay-protector
patterns from `features.extraction_dir`, evaluated on the 2020 rows with 5 quantile bins
fit on train; patterns that never fire in train are dropped. Extractions:
**`02_2020_train`** (default, primary — mined on 2020 training days) and
`01_2024_baseline` (cross-year check — mined on 2024). The models are always trained and
tested on 2020.

Training settings: `../common/base_config.yaml`, no overrides.

## Run

```bash
A=exp/Tab_exp/ablations/04_huim_only
uv run python $A/run.py --seed 42     # 2020-train patterns (default)
uv run python $A/run.py --seed 42 --set features.extraction_dir=../../shap_huim_extractions/01_2024_baseline/csvs
```

## Findings (3 seeds, final fit on 10% of train)

Patterns mined on the 2020 training days:

| MLP | AutoInt | ResNet | FT | LightGBM | XGBoost | CatBoost |
|---|---|---|---|---|---|---|
| 0.588 | 0.582 | 0.582 | 0.587 | 0.589 | 0.588 | 0.589 |

- The weakest arm: 0.04–0.06 AUC below all features (01b) for AutoInt, FT-Transformer
  and the GBDTs; level with all features only for the plain MLP (−0.002).
- All seven models land on nearly the same value, and training on all of train gains
  almost nothing (+0.001–0.003 for the GBDTs): the ceiling is set by the pattern columns,
  not by the model.

### Patterns from another year (2024)

The same arm with the patterns taken from the 2024 extraction — a different year than the
one the models are trained and tested on (2020). It checks whether patterns found in
another year still carry signal in 2020.

| MLP | AutoInt | ResNet | FT | LightGBM | XGBoost | CatBoost |
|---|---|---|---|---|---|---|
| 0.553 | 0.551 | 0.552 | 0.552 | 0.553 | 0.553 | 0.553 |

The 2024 patterns carry clearly less signal: about 0.035 below the 2020 patterns, and
0.04–0.09 below all features for every model.

Always-current numbers: [`RESULTS.md`](RESULTS.md).
