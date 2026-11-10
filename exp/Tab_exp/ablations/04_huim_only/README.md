# 04 — HUIM patterns only

**Question.** How much of the delay signal do the mined patterns carry on their own,
without any original feature?

**Feature set** (`features.mode: huim_only`): only the binary pattern columns of
[03](../03_shap_plus_huim/README.md) — top-20 delay-driver + top-20 delay-protector
patterns from `features.extraction_dir`, matched with 5 quantile bins fit on train;
patterns that never fire in train are dropped. Extractions: `01_2024_baseline`
(default) and `02_2020_train`.

Training settings: `../common/base_config.yaml`, no overrides.

## Run

```bash
A=exp/Tab_exp/ablations/04_huim_only
uv run python $A/run.py --seed 42
uv run python $A/run.py --seed 42 --set features.extraction_dir=../../shap_huim_extractions/02_2020_train/csvs
```

## Findings (3 seeds, final fit on 10% of train)

| patterns | MLP | AutoInt | ResNet | FT | LightGBM | XGBoost | CatBoost |
|---|---|---|---|---|---|---|---|
| 2024 | 0.553 | 0.551 | 0.552 | 0.552 | 0.553 | 0.553 | 0.553 |
| 2020-train | 0.588 | 0.582 | 0.582 | 0.587 | 0.589 | 0.588 | 0.589 |

- The weakest arm: with 2024 patterns 0.04–0.09 AUC below all features (01b) for every
  model; with 2020-train patterns 0.04–0.06 below for AutoInt, FT-Transformer and the
  GBDTs, and level with all features only for the plain MLP (−0.002).
- All seven models land on nearly the same value, and for the GBDTs training on all of
  train gains almost nothing (+0.001–0.003): the ceiling is set by the pattern columns,
  not by the model.
- Patterns mined on 2020 train days carry clearly more signal than the 2024 ones.

Always-current numbers: [`RESULTS.md`](RESULTS.md).
