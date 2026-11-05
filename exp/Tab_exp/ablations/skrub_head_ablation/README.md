# Skrub Head Ablation

Follow-up to `../representation_bakeoff/`. There the head was fixed (one MLP) and the
representation varied; **skrub `TableVectorizer` won** (AUC 0.5982 ± 0.0014). Here the
**skrub representation is frozen** and only the **classifier head** varies.

Everything else is identical to the bake-off: the same 50k `sample.parquet`, temporal
`FL_DAY` split (train ≤9 / val 10–12 / test >12), target `|ARR_DELAY| > 15`, skrub fit on
train only, and AUC / F1-macro / accuracy as mean ± std over seeds `[42, 43, 44]`.

| Head | What it is |
|---|---|
| `mlp` | The bake-off head, unchanged (`[256,128]`, dropout 0.1, Adam 1e-3). Reference row, expect ≈ 0.598 |
| `mlp_bn_wide` | `[512,256,128]` + BatchNorm, dropout 0.2, AdamW |
| `resnet` | Residual MLP (Gorishniy et al., 2021), 3 blocks |
| `ft_transformer` | Each skrub column becomes a token, then [CLS] and a 3-layer Transformer |
| `logreg` | Logistic regression (linear floor) |
| `hist_gbm` | sklearn HistGradientBoosting (non-neural reference) |

## Run

Everything is in **`skrub_head_ablation.ipynb`**, which runs on its own (no imports from the
repo), so it can run on a remote box, Colab or Kaggle.

1. Build the sample once, locally or remotely:
   `uv run python exp/Tab_exp/representation_bakeoff/prep/build_sample.py`
   This produces `representation_bakeoff/artifacts/sample.parquet`.
2. Copy `sample.parquet` to the machine that will run the notebook, then set `SAMPLE_PATH` in
   the config cell. If you set `TAB_DIR` instead (the folder with `Flight_tab_YYYY.csv` +
   `data_info_YYYY.yaml`), the notebook rebuilds the identical sample itself.
3. Run all cells. Outputs go to `outputs/`: `results.csv`, `emb_skrub.npy` (cached
   embedding) and `figures/head_ablation.png`.

To try another head, add it to `HEADS` in the config cell. For a new architecture, also add
a class and one line in `build_torch_head`. Heads already in `results.csv` are skipped
unless `FORCE_RERUN = True`.
