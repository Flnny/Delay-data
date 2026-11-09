# Architecture

## `Datasets/` — data construction (raw BTS CSV → model-ready representations)
- `Flight_tab.py` — tabular representation (categorical: carrier, airport indices, calendar; continuous: weather, coordinates, scheduled times).
- `Flight_chain.py` — flight chains linking consecutive flights by aircraft/crew, for temporal delay propagation.
- `Flight_networks.py` — DGL graphs: airports as nodes, flights as edges, for network modeling.
- `data_extract.py`, `data_pro.py` — ETL: cleaning, outlier removal, datetime→minutes, feature engineering.

## `exp/Tab_exp/` — tabular experiments (the actively developed area)
- `Deep_Model_Classifier.py` / `Deep_Model_Regressor.py` / `Deep_Model_LSS.py` — standalone benchmarks of all `mambular` models for classification / regression / distributional tasks.
- `ablations/` — feature-set ablations on 2020, all driven by `ablations/common/` (shared seeded runner, see its README):
  - `01b_baseline_matched_training` (reference: all features), `01_baseline_all_features` (same features, legacy training overrides), `02_shap_only`, `03_shap_plus_huim` (proposed), `04_huim_only`, `06_llm_top10`, `06_1_llm_top5_datafree`, `06_2_llm_top5_datainformed` (LLM-Select rankings from `05_baseline_100_data/`).
  - Each arm: `config.yaml` + thin `run.py` → `results.csv`, `best_params.json`, `predictions/`; pre-runner outputs kept as `*_legacy.*`.
  - `aggregate.py` → `aggregate/ABLATIONS.md` (mean ± std over seeds, Δ table); `paired_test.py` → paired DeLong tests.
  - `representation/` — representation ablation on a 50k all-years sample (own configs, not the runner): `bakeoff/` (embeddings x fixed MLP head, GBDT rows) and `skrub_head/` (skrub + TabFM head, Colab notebook). See `representation/README.md`.
- `shap_huim_extractions/` — SHAP+HUIM mining: `01_2024_baseline/` (+ variant `01_2024_v2/`), `02_2020_train/` (2020 train days only); each writes `csvs/`, `txt/`.

## `util/` — analysis & figures
- `check/shap_check.py` — recompute SHAP global feature-importance ranking.
- `check/mi.py` — mutual-information ranking. ⚠️ contains **hardcoded stale `D:\project\...` paths**; fix paths before use.
- `figures/` — plotting scripts: `propagation.py` (delay-propagation Sankey), `month_delay.py`, `geo_delay.py`, `ts_delay.py` (temporal/geographic trends), `Table_delay.py`, `airport.py`, `radar.py`. Output PNGs live in `util/figures/<subdir>/`.
