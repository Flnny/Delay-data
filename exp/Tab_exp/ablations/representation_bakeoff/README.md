# Representation Bake-off + HUIM Hybrid

Controlled comparison of tabular representations feeding **one fixed MLP head**,
then a HUIM delay-cost hybrid on the winner. Only the input embedding varies;
head / split / sample / seeds / metric are identical across every row.

- **Data:** all years under `Datasets/Aeolus/Flight_Tab`, 50k stratified sample.
- **Split:** temporal by `FL_DAY` (train ≤9 / val 10–12 / test >12). No day-shuffle.
- **Target:** `|ARR_DELAY| > 15 min` (binary, ~43% positive).
- **Metric:** AUC (primary) + F1-macro + accuracy, mean ± std over seeds `[42,43,44]`.

## Run order

```bash
# 0. Prep (one-time; heavy) — process all raw years, then draw the fixed sample
uv run python exp/Tab_exp/representation_bakeoff/prep/build_tab_years.py
uv run python exp/Tab_exp/representation_bakeoff/prep/build_sample.py

# 1. Bake-off: A=sentence_transformer, B=skrub, C=tabpfn  -> results.csv
uv run python exp/Tab_exp/representation_bakeoff/run_bakeoff.py

# 2. HUIM hybrid (D): mine delay-cost patterns, add to the winner
uv run python exp/Tab_exp/representation_bakeoff/huim/mine.py
uv run python exp/Tab_exp/representation_bakeoff/run_hybrid.py
```

All knobs live in `config.yaml`. Artifacts (sample, cached embeddings, HUIM
outputs) land in `artifacts/`; the comparison table is `results.csv`.

## New dependencies
`sentence-transformers`, `skrub`, `tabpfn`, `pyarrow` (+ optional `faiss-cpu`).
Representation A downloads the MiniLM model on first run (needs internet).
