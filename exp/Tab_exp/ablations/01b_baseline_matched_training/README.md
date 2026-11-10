# 01b — All features (reference arm)

**Question.** How well do the models predict delay from every raw tabular feature? Every
other arm is compared against this one.

**Feature set** (`features.mode: all`): all columns listed in `data_info_2020.yaml`
except `FLIGHTS`, `FL_YEAR`, `FL_MONTH` (constant or meaningless per flight) and
`FL_DAY` (it defines the split, so test days never occur in train):

- categorical (label-encoded on train): `OP_CARRIER`, `OP_CARRIER_FL_NUM`, `FL_WEEK`,
  `ORIGIN_INDEX`, `DEST_INDEX`
- continuous (standardised on train): scheduled departure / arrival time, scheduled
  elapsed time, origin & destination temperature, precipitation, wind speed,
  latitude, longitude

**Training.** No overrides: everything comes from `../common/base_config.yaml`
(search on 10% of train, 10 candidates × 3 folds, final fit on `phase2_frac` of train,
early stopping on 10% of val, full test set). See `../common/README.md`.

## Run

```bash
A=exp/Tab_exp/ablations/01b_baseline_matched_training
uv run python $A/run.py --seed 42                                    # 4 deep models, search + fit
uv run python $A/run.py --seed 42 --models LightGBM,XGBoost,CatBoost  # GBDTs
uv run python $A/run.py --seed 43 --reuse-params                     # more seeds: final fit only
uv run python $A/run.py --seed 42 --reuse-params --set training.phase2_frac=1.0   # all of train
```

## Findings (seeds 42/43/44, final fit on 10% of train)

| MLP | AutoInt | ResNet | FT-Transformer | LightGBM | XGBoost | CatBoost |
|---|---|---|---|---|---|---|
| 0.5906 ± 0.0024 | 0.6289 ± 0.0027 | 0.5931 ± 0.0032 | 0.6286 ± 0.0007 | 0.6417 ± 0.0005 | 0.6443 ± 0.0004 | **0.6459 ± 0.0005** |

- GBDTs are the strongest models; AutoInt and FT-Transformer are the best deep models.
- With all of train in the final fit (`phase2_frac` 1.0, seed 42) every model gains
  1.2–3.8 AUC points (XGBoost 0.6828, CatBoost 0.6776, AutoInt 0.6619) — more than most
  feature-set changes in the other arms.

Full, always-current numbers (incl. `phase2_frac` 1.0 and the selected
hyper-parameters): [`RESULTS.md`](RESULTS.md).
