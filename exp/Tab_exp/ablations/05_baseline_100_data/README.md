# 05 — LLM-Select feature ranking (no model training)

**Question.** Can a small local LLM rank the features as well as SHAP, from their names
and descriptions alone (and optionally a few sample rows)? The rankings produced here
are the feature sets of the 06 arms.

**Method.** LLM-Score from *LLM-Select* (Jeong, Lipton & Ravikumar, TMLR 2025): the LLM
gives every feature an independent 0–1 importance score, 3 repeats per feature, averaged.
Model: `qwen2.5:7b` via a local Ollama server. Two prompt variants:

- **data-free** — feature name, description (`helpers/feature_descriptions.yaml`) and
  task context only (as in the paper);
- **data-informed** — the same plus 100 raw training rows
  (`output/llm_sample_raw.csv`, train days only; `DEP_DELAY` and the raw target are
  never shown).

## Files

| file | role |
|---|---|
| `helpers/export_raw_for_llm.py` | draws the 100 raw training rows |
| `helpers/llm_client.py`, `helpers/config.yaml` | Ollama client and prompt settings |
| `run.py` | scores every feature → `output/llm_feature_importance_{datafree,datainformed}.csv` (+ raw replies in `output/raw_llm_outputs/`) |
| `compare.py` | LLM vs SHAP vs HUIM: rank correlation, top-10 overlap, figures |

## Run

```bash
B=exp/Tab_exp/ablations/05_baseline_100_data
uv run python $B/helpers/export_raw_for_llm.py
uv run python $B/run.py                       # needs `ollama serve` with qwen2.5:7b
uv run python $B/compare.py                   # both SHAP extractions
uv run python $B/compare.py --extraction 02_2020_train
```

`compare.py` writes `output/` (2024 SHAP), `output/02_2020_train/` (2020-train SHAP) and
`output/extraction_summary.md` (both side by side, plus SHAP-vs-SHAP agreement).

## Findings

| | vs 2024 SHAP | vs 2020-train SHAP |
|---|---|---|
| Spearman ρ, data-free / data-informed | −0.05 / −0.13 | −0.35 / −0.33 |
| top-10 Jaccard, data-free / data-informed | 0.33 / 0.33 | 0.33 / 0.33 |

- The LLM ranking barely agrees with SHAP; the two SHAP extractions agree only
  moderately with each other (ρ 0.46, top-10 Jaccard 0.54).
- Both prompt variants pick the same top-10 set (only the order differs); they diverge
  at top-5. How well these sets *predict* is measured in 06, 06.1 and 06.2.
