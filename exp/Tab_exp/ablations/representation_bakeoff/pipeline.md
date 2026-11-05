# Pipeline — Representation Bake-off + HUIM Hybrid

What this experiment does, end to end, and what was built to run it. This is the
"how it was made" companion to `RESULTS.md` (the findings) and `README.md` (run order).

## Goal

Answer one controlled question — *which tabular representation best feeds a fixed MLP
classifier for flight-delay prediction?* — then test whether adding HUIM delay-cost
patterns to the winner helps. **Only the input embedding varies**; head, split, sample,
seeds and metric are identical across every experiment, so any difference measured is
representation quality and nothing else.

## Data flow

```
raw flight_with_weather_YYYY.csv (2016–2024, 9 files, ~1.1–1.9 GB each)
        │  prep/build_tab_years.py   (reuses Datasets/Flight_tab.py: outlier clip,
        │                             date→Y/M/D, times→minutes, select 24 cols)
        ▼
Datasets/Aeolus/Flight_Tab/Tab/Flight_tab_YYYY.csv  +  data_info_YYYY.yaml  (9 years)
        │  prep/build_sample.py      (proportional per-year quota, stratified by label,
        │                             temporal FL_DAY split baked in)
        ▼
artifacts/sample.parquet   50k rows | tr 14,690 / val 4,985 / test 30,325 | 40.8% delayed
        │
        ├─ embeddings/embed_*.py → dense matrix per representation → artifacts/emb_<name>.npy
        │        │  head/mlp_head.py  (one fixed plain-torch MLP, 3 seeds)
        │        ▼
        │   run_bakeoff.py → results.csv  (rows A, B, C)
        │
        └─ huim/mine.py → artifacts/huim_drivers.csv / huim_protectors.csv
                 │  run_hybrid.py  (winner emb + HUIM binary cols → same head)
                 ▼
             results.csv  (row D)
```

## Components built (`exp/Tab_exp/representation_bakeoff/`)

| File | Role |
|---|---|
| `config.yaml` | All knobs: years, sample size, split days, head hyperparams, seeds, per-rep + HUIM settings. |
| `common.py` | Shared loaders: sample, feature lists (from `data_info` yaml minus excludes), split masks, numeric encoder. |
| `prep/build_tab_years.py` | Generalises `Datasets/Flight_tab.py` (was hardcoded to one year) to process every raw year; skips already-built years. |
| `prep/build_sample.py` | Draws the single fixed 50k sample; persists parquet + `split` column. |
| `head/mlp_head.py` | **THE fixed head** — plain torch MLP over a dense matrix. StandardScaler fit on train; early-stop on val AUC; `run_multiseed` returns mean±std. Deliberately **not** mambular's MLPClassifier (it re-embeds raw columns). |
| `embeddings/embed_sentence_transformer.py` | **A** — row→"col=value; …" text → MiniLM (384-d). No fit. |
| `embeddings/embed_skrub.py` | **B** — `TableVectorizer` fit on train → bounded numeric matrix (95-d). |
| `embeddings/embed_tabpfn.py` | **C** — TabPFN frozen embedder (context ≤10k, batched). Built and ready; license-gated at runtime. |
| `run_bakeoff.py` | Loops A/B/C through the fixed head; caches each embedding; writes `results.csv`. Per-rep try/except so one failure doesn't sink the rest. |
| `huim/mine.py` | Synthesises delay-cost utility, mines high-utility itemsets per class with PAMI EFIM. |
| `huim/pattern_features.py` | Turns patterns into binary `HUIM_*` columns (categorical match on value, continuous on quantile bin). |
| `run_hybrid.py` | Auto-picks winner from `results.csv`, concatenates HUIM columns, retrains the fixed head → row D. |

## Fixed decisions (settled with the user before building)

- **Years:** all of 2016–2024 (only 2020 & 2024 shipped pre-processed; the other 7 were generated).
- **Sample:** 50k total, quota proportional to each year's size, stratified by label.
- **Split:** temporal by `FL_DAY` (train ≤9 / val 10–12 / test >12) — repo convention, no same-day leakage.
- **Metric:** AUC primary (+ F1-macro, accuracy), mean±std over seeds `[42,43,44]`.
- **HUIM utility (no natural value column exists):** synthesised delay cost — item utility = `|ARR_DELAY|`
  in a delayed flight, on-time margin `(15−|ARR_DELAY|)` in an on-time flight. Fit on **train only**.

## Method reality-corrections applied

The original plan assumed retail-transaction data. Corrected against the real data:
CSV not xlsx; 4–7M rows/year not ~1M; **binary near-balanced** target not multi-class;
~19 usable features not ~200; **no utility column** (synthesised); every representation
tool was a **new dependency** (`sentence-transformers`, `skrub`, `tabpfn`, `pyarrow` added).

## Issues hit and how they were fixed

1. **TabPFN 8.x is license-gated.** Needs a one-time Prior Labs token; can't prompt in a
   headless shell (`OSError WinError 10038`). The runner caught it; A/B/D completed. Left
   as *not evaluated* per the user's choice. Headless path: set `TABPFN_TOKEN`.
2. **HUIM combinatorial blow-up.** First pass put all 19 features (incl. flight number, thousands
   of unique values) as equal-utility items in every transaction → EFIM enumerated **4.4M**
   patterns and wrote a **1 GB** CSV. Fixed by: excluding flight number as an item, capping
   transactions to 4,000/class, raising minUtil to p98, and keeping only **size ≤ 3** patterns
   (top 200) — CSVs dropped to ~14 KB with frequent, interpretable patterns.
3. **Prep memory/time.** Each raw year (~6M rows, datetime parsing) processed one at a time
   with gc between; ~3–4 min/year, run in the background.

## Results (see `RESULTS.md`)

| # | Representation | dim | AUC (mean ± std) |
|---|---|---|---|
| **B** | **skrub TableVectorizer** | 95 | **0.5982 ± 0.0014** (winner) |
| A | SentenceTransformer | 384 | 0.5726 ± 0.0042 (weakest, as predicted) |
| D | skrub + HUIM | 135 | 0.5919 ± 0.0012 (HUIM hurt −0.6%) |
| C | TabPFN | — | not evaluated (license-gated) |

**Takeaways:** skrub is the best representation by a clean, low-variance margin;
SentenceTransformer is weakest because text tokenisation mangles the numbers; HUIM adds
no lift on this data (a small real negative, reported as-is). Overall ceiling ~0.60 AUC,
consistent with the repo's other tabular baselines.

## Reproduce

```bash
uv run python exp/Tab_exp/representation_bakeoff/prep/build_tab_years.py   # once (heavy)
uv run python exp/Tab_exp/representation_bakeoff/prep/build_sample.py
uv run python exp/Tab_exp/representation_bakeoff/run_bakeoff.py
uv run python exp/Tab_exp/representation_bakeoff/huim/mine.py
uv run python exp/Tab_exp/representation_bakeoff/run_hybrid.py
```

Artifacts (`sample.parquet`, `emb_*.npy`, HUIM files) land in `artifacts/` (gitignored).
`results.csv` is the comparison table.
