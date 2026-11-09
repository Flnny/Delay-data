# Notes for interpreting results

Source of truth: `exp/Tab_exp/ablations/aggregate/ABLATIONS.md` (mean ± std over seeds
42/43/44, from `aggregate.py`) and `aggregate/paired_tests.md` (paired DeLong). Numbers
below are 2020, target `abs`, `phase2_frac` 0.1, as of the shared-runner reruns.

- **Legacy numbers are not comparable.** `results_legacy.csv` files (and older handoff
  pages) used hard-label "AUC" (≈ balanced accuracy), per-split label encoding, a hyper-
  parameter search that never changed the model, and one seed. Their "SHAP+HUIM adds 1–2%
  AUC" does not survive the fixed pipeline.
- **GBDTs are the strongest models**: CatBoost on all features 0.6459 ± 0.0005, XGBoost
  0.6443, LightGBM 0.6417; best mambular models (AutoInt, FT-Transformer) ≈ 0.629.
- **HUIM pattern columns add nothing measurable.** 03 − 02 (same SHAP features ± HUIM) is
  between −0.0015 and +0.003 for every model and both extractions, within seed noise
  (except AutoInt with 2024 patterns: −0.012). HUIM-only (04) is the worst arm.
- **2024-mined patterns** (`01_2024_baseline`) never beat all features (03 − 01b < 0 for all
  mambular models). **2020-train patterns** help only weak models: MLP +0.017 (mostly from
  the SHAP selection, 02 − 01b = +0.014); with GBDTs the 2020 SHAP top-10 is ~0.002–0.009
  *below* all features.
- **LLM top-10** matches SHAP top-10 on AutoInt/FT (0.627 vs 0.626) but trails it with GBDTs
  (0.636 vs 0.645). Top-5 sets cost 2–4 points.
- **Noise:** seed-to-seed std is 0.001–0.010 AUC; the DeLong SE on 2.5M test rows is ~0.0002.
  Training noise dominates, so per-seed DeLong p-values overstate significance — judge
  differences against the seed std (and sign consistency across seeds).
- Bake-off (50k sample, all years, `representation/bakeoff/RESULTS_GBDT.md`): skrub + TabFM
  0.6336 ≫ raw + CatBoost 0.6009 ≈ skrub + LightGBM 0.5994 ≈ skrub + MLP 0.5982 > skrub + HUIM 0.5919.
- Checkpoints are temporary (deleted after each run); predictions live in
  `<arm>/predictions/` (gitignored).
