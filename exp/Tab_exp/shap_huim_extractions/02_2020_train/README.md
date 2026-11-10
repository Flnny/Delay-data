# SHAP + HUIM extraction — 2020, train days only

Produces the SHAP feature ranking and the HUIM delay-driver / delay-protector patterns
that ablation arms 02, 03 and 04 consume (`features.extraction_dir` →
`../../shap_huim_extractions/02_2020_train/csvs`). Unlike `01_2024_baseline` (mined on
another year), everything here comes from the **2020 training days**, so the patterns
used by the 2020 ablations never see validation or test days.

## Pipeline (`run.py`, settings in `config.yaml`)

1. Load `Flight_tab_2020.csv`, keep `FL_DAY ≤ 9` only (rows of later days are dropped
   right after reading), take a 10% sample (seed 42).
2. Split inside the training days: **FL_DAY 1–7** → MLP fit and SHAP background,
   **FL_DAY 8–9** → early stopping and the rows SHAP explains. Encoders are fit on days
   1–7 only.
3. Train a mambular MLP (`d_model` 64, lr 1e-3, up to 50 epochs, patience 5).
4. SHAP (`shap.Explainer`, 200 background rows, 1,000 explained rows) on the model's
   predicted label (`shap.output: label`, as in `01_2024_baseline`; `proba` is available).
5. Items: categorical values as `COL=<raw value>`, continuous values as 5 quantile bins
   (`VeryLow … VeryHigh`) fit on the full training split — the same bins the ablations
   use to match patterns.
6. Two transaction streams — positive SHAP (delay drivers) and negative SHAP (delay
   protectors), utility = `int(|SHAP| × 10000)` — mined with PAMI EFIM at
   `minUtil` = 90th percentile of transaction utilities.
7. Patterns are sorted by numeric utility (PAMI returns utilities as strings; sorting
   those lexicographically puts e.g. 9999 above 99875).

Feature exclusions match the ablations: `FLIGHTS`, `FL_YEAR`, `FL_MONTH`, `FL_DAY`.
pretab's batch-dependent PLE encoding is patched (`../../ablations/common/pretab_patch.py`),
which matters here because SHAP evaluates the model on small perturbed batches.

## Outputs

| file | content |
|---|---|
| `csvs/shap_feature_importance.csv` | features ranked by mean \|SHAP\| |
| `csvs/huim_delay_drivers.csv`, `csvs/huim_delay_protectors.csv` | `Pattern, ItemIDs, Utility, Size`, highest utility first |
| `csvs/huim_item_labels.csv` | item id → `COL=value` |
| `csvs/extraction_meta.json` | config, row counts, day ranges, MLP validation AUC, minUtil, runtime |
| `txt/shap_utility_{positive,negative}.txt` | PAMI transaction files |

```bash
uv run python exp/Tab_exp/shap_huim_extractions/02_2020_train/run.py     # ~30 min
```

## Result

- MLP on the early-stopping days: AUC 0.614.
- SHAP top-10: O_LONGITUDE, O_LATITUDE, CRS_ELAPSED_TIME, O_TEMP, D_LONGITUDE, D_LATITUDE,
  DEST_INDEX, ORIGIN_INDEX, O_WSPD, CRS_DEP_TIME_MIN (2024 top-10 overlap: Jaccard 0.54).
- 27,942 driver and 27,757 protector patterns; the top patterns are short, e.g.
  `O_LONGITUDE=VeryHigh` and `O_LONGITUDE=VeryHigh + D_PRCP=VeryLow` (drivers).
- All 40 patterns used by the ablations (top-20 + top-20) fire on the training days.
