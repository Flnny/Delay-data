"""
run_gbdt.py — GBDT rows for the representation bake-off
=====================================================================
Same 50k sample, same temporal split, same seeds (head.seeds) as the MLP-head rows:

  skrub + LightGBM   skrub TableVectorizer matrix (identical to row B) -> LightGBM
  raw + CatBoost     raw Tab columns, categoricals as native CatBoost cat_features

The bake-off head has no hyper-parameter search, so neither do these rows: one fixed,
conventional configuration each, early-stopped on validation AUC. Row/column
subsampling is on so that the seed actually matters (otherwise std would be 0 by
construction). Also re-runs `skrub + MLP` as a check that the rebuilt sample and
embedding reproduce the committed row B.

Writes rows to results.csv (same columns as run_bakeoff.py) and the combined
comparison (incl. the TabFM head from ../skrub_head) to RESULTS_GBDT.md.

    PYTHONPATH=. uv run python exp/Tab_exp/ablations/representation/bakeoff/run_gbdt.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

EXP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP_DIR))
import common  # noqa: E402
from embeddings import embed_skrub  # noqa: E402
from head.mlp_head import run_multiseed  # noqa: E402

THREADS = 4
MAX_ROUNDS, EARLY = 2000, 50
LGB_PARAMS = dict(num_leaves=31, learning_rate=0.05, min_child_samples=50,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8)
CB_PARAMS = dict(depth=6, learning_rate=0.05, l2_leaf_reg=3, rsm=0.8,
                 bootstrap_type="Bernoulli", subsample=0.8)
# TabFM head on the same skrub representation (../skrub_head/README.md, 3 seeds)
TABFM_ROW = {"Representation": "skrub + TabFM (head ablation)", "dim": 95,
             "auc_mean": 0.6336, "auc_std": 0.0001, "f1_mean": 0.5451, "f1_std": 0.0004,
             "acc_mean": 0.6168, "acc_std": 0.0002}


def metrics(y, p):
    pred = (p > 0.5).astype(int)
    return roc_auc_score(y, p), f1_score(y, pred, average="macro"), accuracy_score(y, pred)


def summarise(name, dim, per_seed):
    a = np.array(per_seed)
    return {"Representation": name, "dim": dim,
            "auc_mean": a[:, 0].mean(), "auc_std": a[:, 0].std(),
            "f1_mean": a[:, 1].mean(), "f1_std": a[:, 1].std(),
            "acc_mean": a[:, 2].mean(), "acc_std": a[:, 2].std()}


def lightgbm_rows(Z, y, masks, seeds):
    import lightgbm as lgb
    tr, va, te = masks
    out = []
    for s in seeds:
        m = lgb.LGBMClassifier(n_estimators=MAX_ROUNDS, random_state=s, n_jobs=THREADS,
                               deterministic=True, force_col_wise=True, verbose=-1, **LGB_PARAMS)
        m.fit(Z[tr], y[tr], eval_set=[(Z[va], y[va])], eval_metric="auc",
              callbacks=[lgb.early_stopping(EARLY, verbose=False)])
        r = metrics(y[te], m.predict_proba(Z[te])[:, 1])
        print(f"    seed={s}: AUC={r[0]:.4f} F1m={r[1]:.4f} ACC={r[2]:.4f} (trees={m.best_iteration_})")
        out.append(r)
    return out


def catboost_rows(df, cat, cont, y, masks, seeds):
    from catboost import CatBoostClassifier
    tr, va, te = masks
    X = df[cat + cont].copy()
    for c in cat:
        X[c] = X[c].astype(str)  # raw categories, no encoding
    out = []
    for s in seeds:
        m = CatBoostClassifier(iterations=MAX_ROUNDS, random_seed=s, thread_count=THREADS,
                               eval_metric="AUC", od_type="Iter", od_wait=EARLY, use_best_model=True,
                               verbose=False, allow_writing_files=False, cat_features=cat, **CB_PARAMS)
        m.fit(X[tr], y[tr], eval_set=(X[va], y[va]))
        r = metrics(y[te], m.predict_proba(X[te])[:, 1])
        print(f"    seed={s}: AUC={r[0]:.4f} F1m={r[1]:.4f} ACC={r[2]:.4f} (trees={m.get_best_iteration()})")
        out.append(r)
    return out


def main():
    cfg = common.load_config()
    df = common.load_sample(cfg)
    cat, cont = common.feature_lists(cfg)
    masks = common.split_masks(df)
    tr, va, te = masks
    y = df["label"].to_numpy()
    seeds = cfg["head"]["seeds"]
    print(f"Sample {len(df)} rows, split {tr.sum()}/{va.sum()}/{te.sum()}, seeds {seeds}")

    art = (EXP_DIR / cfg["paths"]["work_dir"]).resolve()
    art.mkdir(parents=True, exist_ok=True)
    emb = art / "emb_skrub.npy"
    Z = np.load(emb) if emb.exists() else embed_skrub.build(df, cat, cont, tr, y, cfg)
    np.save(emb, Z)

    rows = []
    print("skrub + MLP (reproduction check of row B)")
    m = run_multiseed(Z[tr], y[tr], Z[va], y[va], Z[te], y[te], cfg["head"])
    rows.append({"Representation": "skrub + MLP (rerun)", "dim": Z.shape[1],
                 **{k: m[k] for k in ("auc_mean", "auc_std", "f1_mean", "f1_std", "acc_mean", "acc_std")}})
    print("skrub + LightGBM")
    rows.append(summarise("skrub + LightGBM", Z.shape[1], lightgbm_rows(Z, y, masks, seeds)))
    print("raw + CatBoost")
    rows.append(summarise("raw + CatBoost", len(cat) + len(cont), catboost_rows(df, cat, cont, y, masks, seeds)))

    res_csv = EXP_DIR / "results.csv"
    res = pd.read_csv(res_csv)
    res = res[~res["Representation"].isin([r["Representation"] for r in rows])]
    res = pd.concat([res, pd.DataFrame(rows[1:])], ignore_index=True)  # the rerun is a check, not a row
    res.to_csv(res_csv, index=False)

    comp = pd.concat([res, pd.DataFrame([rows[0], TABFM_ROW])], ignore_index=True)
    comp = comp.sort_values("auc_mean", ascending=False)
    lines = ["# Bake-off with GBDT rows\n",
             f"50k sample, split {tr.sum()}/{va.sum()}/{te.sum()}, seeds {seeds}; mean ± std over seeds. "
             "GBDTs use one fixed configuration with early stopping on validation AUC (no search, like the "
             "fixed MLP head). The TabFM row is copied from `../skrub_head/README.md`.\n",
             "| Representation / head | dim | AUC | F1-macro | Accuracy |", "|---|---|---|---|---|"]
    for r in comp.itertuples(index=False):
        lines.append(f"| {r.Representation} | {r.dim} | {r.auc_mean:.4f} ± {r.auc_std:.4f} | "
                     f"{r.f1_mean:.4f} ± {r.f1_std:.4f} | {r.acc_mean:.4f} ± {r.acc_std:.4f} |")
    (EXP_DIR / "RESULTS_GBDT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
