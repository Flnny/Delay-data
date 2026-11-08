# The SHAP + HUIM method (core novelty)

Two extractions live in `exp/Tab_exp/shap_huim_extractions/`, each writing `csvs/`
(`shap_feature_importance.csv`, `huim_delay_drivers.csv`, `huim_delay_protectors.csv`,
`huim_item_labels.csv`) and `txt/` (PAMI transaction files):

- `01_2024_baseline/` — mined on **2024** (different year from the 2020 ablations). Kept
  unchanged because legacy 03/04 results used exactly these files.
- `02_2020_train/` — mined on **2020 train days only** (MLP fit on FL_DAY 1–7, early
  stopping + explained rows from FL_DAY 8–9; val/test days dropped after reading).
  Leakage-safe source for the 2020 ablations.

Pipeline (both): sample 10% → MLP → SHAP (`bg_samples=200`, `test_samples=1000`) →
5 quantile bins for continuous features → two HUIM streams (**positive SHAP = delay
drivers**, **negative SHAP = delay protectors**, utility = `int(|SHAP| × 10000)`) →
PAMI EFIM with `minUtil` = p90 of transaction utilities.

Hand-off: ablations 02/03/04 read the CSVs from `features.extraction_dir` (relative to
the arm dir, default `../../shap_huim_extractions/01_2024_baseline/csvs`). Switch
extraction with `--set features.extraction_dir=../../shap_huim_extractions/02_2020_train/csvs`.
There is no `exp/Tab_exp/SHAP_HUIM/` copy step anymore.

Known quirks:
- PAMI returns utilities as **strings**. `01_2024_baseline` sorted them
  lexicographically ("9999" > "99875"), so its "top-N" pattern rows are not the top-N
  by utility. `02_2020_train` casts to int (correct order).
- `01_2024_baseline` explains `model.predict` (hard labels) with an unpatched pretab
  PLE (batch-dependent encoding); `02_2020_train` keeps `shap.output: label` for
  comparability but runs with `common/pretab_patch.py`.
- Item labels are `COL=<raw value>` (e.g. `OP_CARRIER_FL_NUM=7.0`, `FL_WEEK=1`) or
  `COL=VeryLow..VeryHigh`; `ablations/common/features.py` matches them against raw values
  with bins fit on the train split.
