# Bake-off with GBDT rows

50k sample, split 14690/4985/30325, seeds [42, 43, 44]; mean ± std over seeds. GBDTs use one fixed configuration with early stopping on validation AUC (no search, like the fixed MLP head). The TabFM row is copied from `../skrub_head/README.md`.

| Representation / head | dim | AUC | F1-macro | Accuracy |
|---|---|---|---|---|
| skrub + TabFM (head ablation) | 95 | 0.6336 ± 0.0001 | 0.5451 ± 0.0004 | 0.6168 ± 0.0002 |
| raw + CatBoost | 19 | 0.6009 ± 0.0016 | 0.4776 ± 0.0348 | 0.6016 ± 0.0015 |
| skrub + LightGBM | 95 | 0.5994 ± 0.0006 | 0.4933 ± 0.0230 | 0.6029 ± 0.0003 |
| skrub + MLP (rerun) | 95 | 0.5982 ± 0.0015 | 0.5370 ± 0.0050 | 0.6001 ± 0.0013 |
| skrub | 95 | 0.5982 ± 0.0014 | 0.5371 ± 0.0060 | 0.6005 ± 0.0014 |
| D_skrub_plus_huim | 135 | 0.5919 ± 0.0012 | 0.5282 ± 0.0055 | 0.5991 ± 0.0015 |
| sentence_transformer | 384 | 0.5726 ± 0.0042 | 0.4698 ± 0.0413 | 0.5943 ± 0.0016 |
