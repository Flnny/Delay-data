"""
run_bakeoff.py
==============
Phase 1 of the bake-off. For each representation (A/B/C), build a dense matrix,
feed it to THE fixed MLP head over the identical temporal split, and record
mean +/- std test metrics over the configured seeds.

Each representation's matrix is cached to artifacts/emb_<name>.npy so the HUIM
hybrid (run_hybrid.py) can reuse the winner without recomputing.

Run (after prep):
    uv run python exp/Tab_exp/ablations/representation/bakeoff/run_bakeoff.py
    uv run python exp/Tab_exp/ablations/representation/bakeoff/run_bakeoff.py --only skrub
"""
import argparse
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from exp.Tab_exp.ablations.representation.bakeoff.embeddings import embed_sentence_transformer, embed_skrub

EXP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP_DIR))

import exp.Tab_exp.ablations.representation.bakeoff.common as common  # noqa: E402
from exp.Tab_exp.ablations.representation.bakeoff.head.mlp_head import run_multiseed  # noqa: E402
from exp.Tab_exp.ablations.representation.bakeoff.embeddings import embed_tabpfn  # noqa: E402

REPRESENTATIONS = {
    "sentence_transformer": embed_sentence_transformer,
    "skrub": embed_skrub,
    "tabpfn": embed_tabpfn,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", choices=list(REPRESENTATIONS),
                    help="run only these representations")
    args = ap.parse_args()

    cfg = common.load_config()
    df = common.load_sample(cfg)
    cat, cont = common.feature_lists(cfg)
    tr_m, va_m, te_m = common.split_masks(df)
    y = df["label"].to_numpy()
    y_tr, y_va, y_te = y[tr_m], y[va_m], y[te_m]
    head_cfg = cfg["head"]

    art = (EXP_DIR / cfg["paths"]["work_dir"]).resolve()
    art.mkdir(parents=True, exist_ok=True)
    results_csv = EXP_DIR / "results.csv"
    cols = ["Representation", "dim", "auc_mean", "auc_std",
            "f1_mean", "f1_std", "acc_mean", "acc_std"]
    results = pd.read_csv(results_csv) if results_csv.exists() else pd.DataFrame(columns=cols)

    print(f"Sample: {len(df)} rows | split tr/va/te = {tr_m.sum()}/{va_m.sum()}/{te_m.sum()}")
    print(f"Features: {len(cat)} cat + {len(cont)} cont | delayed frac = {y.mean():.3f}\n")

    todo = args.only or list(REPRESENTATIONS)
    for name in todo:
        if name in results["Representation"].values:
            print(f"Skipping {name} — already in results.csv")
            continue
        print(f"{'='*60}\n  {name}\n{'='*60}")
        try:
            Z = REPRESENTATIONS[name].build(df, cat, cont, tr_m, y, cfg)
            np.save(art / f"emb_{name}.npy", Z)
            m = run_multiseed(Z[tr_m], y_tr, Z[va_m], y_va, Z[te_m], y_te, head_cfg)
            row = {"Representation": name, "dim": Z.shape[1], **{
                k: m[k] for k in ("auc_mean", "auc_std", "f1_mean", "f1_std", "acc_mean", "acc_std")}}
            print(f"  -> AUC {m['auc_mean']:.4f} +/- {m['auc_std']:.4f} | "
                  f"F1m {m['f1_mean']:.4f} | ACC {m['acc_mean']:.4f}")
        except Exception:
            print(f"  !! {name} FAILED:\n{traceback.format_exc()}")
            continue
        results = pd.concat([results, pd.DataFrame([row])], ignore_index=True)
        results.to_csv(results_csv, index=False)

    print(f"\nResults -> {results_csv}")
    if not results.empty:
        print(results.to_string(index=False))


if __name__ == "__main__":
    main()
