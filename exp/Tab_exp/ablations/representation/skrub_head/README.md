# Skrub + TabFM Head

Follow-up to `../bakeoff/`. The bake-off fed one fixed MLP head with different
representations, and **skrub `TableVectorizer` won** (test AUC 0.5982 ± 0.0014). This experiment
keeps the skrub representation and **replaces the MLP head with TabFM**.

[TabFM](https://github.com/google-research/tabfm) is Google's pretrained tabular foundation
model. Nothing is trained here: TabFM learns *in context*. At prediction time it sees the train
rows with their labels and predicts the test rows directly. The weights come from Hugging Face
`google/tabfm-1.0.0-pytorch` under the **non-commercial** `tabfm-non-commercial-v1.0` license.
The code is Apache-2.0.

## Setup (identical to the bake-off)

Only the head differs, so the comparison is head against head.

- **Sample:** the bake-off's fixed 50k `sample.parquet` (years 2016–2024, stratified by year and label).
- **Split:** temporal by `FL_DAY`: train ≤ 9 / val 10–12 / test > 12 (14,690 / 4,985 / 30,325 rows). Days are never shuffled across splits.
- **Target:** `|ARR_DELAY| > 15 min` (40.8% positive).
- **Features:** 6 categorical + 13 continuous. `FLIGHTS`, `FL_YEAR` and `FL_MONTH` are excluded.
- **Representation:** skrub `TableVectorizer(cardinality_threshold=40)` + `StandardScaler`, both fit on train only. The result has 95 dimensions.
- **Metrics:** test AUC (primary), F1-macro and accuracy, as mean ± std over seeds `[42, 43, 44]`.

The MLP is not re-run. Its numbers come from `../bakeoff/results.csv` (row `skrub`).

## TabFM head

`fit` stores the train rows as context and prepares TabFM's preprocessing. It does not train
anything. Each of the `n_estimators = 32` ensemble members sees a differently shuffled and
normalised view of the context, and their predictions are averaged. The seed is passed as
`random_state` and controls only this ensemble randomness. Val rows are used only to report a
val AUC; nothing is selected on them. Test rows are predicted in chunks of 4,096 to limit
memory use.

## Results

| Head | Test AUC | F1-macro | Accuracy |
|---|---|---|---|
| skrub + MLP (bake-off) | 0.5982 ± 0.0014 | 0.5371 ± 0.0060 | 0.6005 ± 0.0014 |
| **skrub + TabFM** | **0.6336 ± 0.0001** | **0.5451 ± 0.0004** | **0.6168 ± 0.0002** |

- On the same representation, TabFM adds **+0.0354 AUC** over the MLP head. That gap is about
  35σ of the pooled seed std, far above the noise level of roughly 2σ.
- Val AUC is 0.6246, close to test, so the context did not overfit.
- The seed spread is tiny because the seed only changes the ensemble shuffling. No weights are trained.
- Each seed took about 1,290 s (~21.5 min) on a Colab GPU. That run used a TabFM build without
  `cache_context`, so the context was re-encoded on every predict call. The pinned GitHub
  build below avoids this and should be faster.

## Run

Everything is in **`skrub_head_ablation.ipynb`**. It has no imports from the repo, so it runs on
Colab, Kaggle or a remote box.

1. **Install.** On Colab, run the first cell. It removes any PyPI `tabfm` and installs
   `skrub>=0.10.0` plus `tabfm[pytorch]` pinned to GitHub commit `fbb6655`. The PyPI 1.0.1
   release lacks `cache_context`. Then restart the session once and run from the top. Elsewhere,
   install a CUDA build of torch first, then:
   ```bash
   pip uninstall -y tabfm
   pip install "skrub>=0.10.0" "tabfm[pytorch] @ git+https://github.com/google-research/tabfm.git@fbb665569425fd2f490c6576b3af967876fe11ff"
   ```
   You also need pandas, scikit-learn, pyarrow and pyyaml.
2. **Get the sample.** Build it once with
   `uv run python exp/Tab_exp/ablations/representation/bakeoff/prep/build_sample.py`, which writes
   `../bakeoff/artifacts/sample.parquet`. Copy it to the run machine and set
   `SAMPLE_PATH` in the config cell. The default is a Google Drive path. Off Colab, skip the
   `drive.mount` cell. If you set `TAB_DIR` instead (the folder with `Flight_tab_YYYY.csv` and
   `data_info_YYYY.yaml` for 2016–2024), the notebook rebuilds the identical sample itself.
3. **Run all cells.** Outputs go to `outputs/`:
   - `results_per_seed.csv`: one row per seed, appended as soon as each seed finishes. A Colab
     disconnect loses only the seed in progress; re-run to continue. Seeds already in the file are
     skipped unless `FORCE_RERUN = True`.
   - `results.csv`: the aggregated TabFM row (mean ± std, val AUC, time, params).
   - `emb_skrub.npy`: the cached skrub matrix, reused while `REUSE_EMBEDDING = True` and the row count matches.
   - `sample.parquet`: written only when the sample was rebuilt from `TAB_DIR`.

If the GPU runs out of memory, lower `TABFM["n_estimators"]` (e.g. 8) or cap the context with
`TABFM["max_num_rows"]` (e.g. 8000). The default `None` uses all ~14.7k train rows.
