# Setup & Commands

Environment is managed with **uv** (Python 3.12+). The PyTorch CUDA 12.8 wheel index is pinned in `pyproject.toml`.

```bash
uv sync                 # install all dependencies into .venv
```

Run any script with `uv run python <script>` (or activate `.venv` first). Each experiment is a self-contained directory with its own `config.yaml` + `run.py`; **`run.py` resolves paths relative to its own location**, so it can be launched from anywhere:

```bash
# Tabular ablations — shared runner (exp/Tab_exp/ablations/common/), all accept:
#   --seed N  --target abs|late|3class  --models MLP,AutoInt,ResNet,FTTransformer
#   --reuse-params (skip Phase 1)  --set key.sub=value  --out-dir DIR
uv run python exp/Tab_exp/ablations/01b_baseline_matched_training/run.py --seed 42  # reference: all features
uv run python exp/Tab_exp/ablations/02_shap_only/run.py --seed 42                   # top-10 SHAP features
uv run python exp/Tab_exp/ablations/03_shap_plus_huim/run.py --seed 42              # SHAP + HUIM (proposed)
uv run python exp/Tab_exp/ablations/04_huim_only/run.py --seed 42                   # HUIM patterns only
#   default extraction = 02_2020_train; cross-year check: --set features.extraction_dir=../../shap_huim_extractions/01_2024_baseline/csvs
uv run python exp/Tab_exp/ablations/aggregate.py      # -> ablations/aggregate/ABLATIONS.md
uv run python exp/Tab_exp/ablations/paired_test.py    # paired DeLong tests on saved predictions
uv run python exp/Tab_exp/ablations/common/check_configs.py   # arms differ only in features?

# SHAP + HUIM pattern extraction (produces the CSVs the ablations consume)
uv run python exp/Tab_exp/shap_huim_extractions/02_2020_train/run.py     # 2020 train days only
uv run python exp/Tab_exp/shap_huim_extractions/01_2024_baseline/run.py  # 2024 (legacy)
uv run python exp/Tab_exp/shap_huim_extractions/01_2024_baseline/visualize.py

# Other tabular benchmarks (not config-driven; constants near top of file)
uv run python exp/Tab_exp/Deep_Model_Classifier.py   # binary classification, all features
uv run python exp/Tab_exp/Deep_Model_Regressor.py    # regress exact DEP_DELAY (MSE/MAE)
uv run python exp/Tab_exp/Deep_Model_LSS.py          # distributional (location-scale) delay, NLL
```

There is **no test suite, linter, or build step** — this is a research repo. "Running" means executing an experiment `run.py` and inspecting its `results.csv` / generated figures. Self-checks for the runner's library patches: `uv run python exp/Tab_exp/ablations/common/verify_fast_loader.py --accelerator gpu`, and from `exp/Tab_exp/ablations/`: `uv run python -m common.pretab_patch`, `uv run python -m common.mambular_patch`.

Chain and network experiments are **Jupyter notebooks**, run them in Jupyter/VS Code:
- `exp/Chain_exp/`: `LSTM.ipynb`, `GRU.ipynb`, `CNN_LSTM.ipynb`, `MogrifierLSTM.ipynb`
- `exp/Network_exp/`: `AFM-node_embedding.ipynb`
