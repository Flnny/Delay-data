# aggregate/ — combined results of all ablation arms

Everything here is generated; do not edit by hand.

```bash
uv run python exp/Tab_exp/ablations/aggregate.py     # tables below + <arm>/RESULTS.md
uv run python exp/Tab_exp/ablations/paired_test.py   # paired DeLong tests
```

| file | content |
|---|---|
| `ABLATIONS.md` | main report: feature sets × models side by side (AUC mean ± std over seeds), all metrics per arm/model, seed-matched Δ table, representation bake-off |
| `ablation_table.csv` | arm × model × extraction × target × `phase2_frac`: mean / std of AUC, accuracy, macro-F1, number of seeds |
| `deltas.csv` | seed-matched AUC differences per extraction and model: Δ SHAP selection (02 − 01b), Δ HUIM (03 − 02), Δ total (03 − 01b), 04 − 01b |
| `paired_tests.md` / `.csv` | paired DeLong test per seed on the shared 2.55M test rows: ΔAUC, its standard error, p-value; summarised over seeds |

## How to read them

- All arms are evaluated on the same test rows (one `test_hash`; `aggregate.py` refuses
  to mix test sets). AUC comes from predicted probabilities.
- Rows are separated by `phase2_frac`: 0.1 = final fit on 10% of the training split,
  1.0 = on all of it. Compare arms only within the same `phase2_frac`.
- **Noise:** the seed-to-seed std is 0.001–0.010 AUC, while the DeLong standard error on
  2.55M rows is ~0.0002. Training randomness, not test sampling, dominates, so a tiny
  per-seed p-value alone does not make a difference meaningful — compare Δ with the
  seed std and check that its sign is the same for every seed.
