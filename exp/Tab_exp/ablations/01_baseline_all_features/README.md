# 01 — All features, legacy training settings

Same feature set as [`01b`](../01b_baseline_matched_training/README.md), but trained with
the settings this baseline originally used, kept through an explicit `overrides:` block
in `config.yaml`:

| setting | 01 (this arm) | every other arm |
|---|---|---|
| `phase2_frac` | 1.0 | 0.1 |
| `val_frac` | 1.0 | 0.1 |
| `batch_size` | 4096 | 2048 |
| `patience_phase2` | 3 | 5 |
| Phase-1 validation | internal 50% split of the search sample | 10% sample of the val split |

Because these differ from the other arms, **01 is not used for comparisons** — use
`01b`. The arm stays runnable to document what the original baseline did;
`../common/check_configs.py` reports it as the only arm that differs outside its
feature set.

`results_legacy.csv` holds the numbers from before the shared runner (hard-label AUC,
one seed, a search that did not change the model). They are not comparable with any
current result.

```bash
uv run python exp/Tab_exp/ablations/01_baseline_all_features/run.py --seed 42
```
