"""
run_hybrid.py
=============
Phase 2 (experiment D). Take the winning representation from run_bakeoff.py,
concatenate the HUIM delay-cost pattern columns (from huim/mine.py) onto it, and
retrain THE fixed head on the identical split. Appends the D row to results.csv.

Run (after run_bakeoff.py and huim/mine.py):
    uv run python exp/Tab_exp/ablations/representation/bakeoff/run_hybrid.py
    uv run python exp/Tab_exp/ablations/representation/bakeoff/run_hybrid.py --winner skrub
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer

EXP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP_DIR))

import exp.Tab_exp.ablations.representation.bakeoff.common as common  # noqa: E402
from exp.Tab_exp.ablations.representation.bakeoff.head.mlp_head import run_multiseed  # noqa: E402
from exp.Tab_exp.ablations.representation.bakeoff.huim.pattern_features import add_huim_features  # noqa: E402


def pick_winner(results: pd.DataFrame, candidates) -> str:
    sub = results[results["Representation"].isin(candidates)]
    if sub.empty:
        raise SystemExit("No base representations in results.csv — run run_bakeoff.py first.")
    return sub.loc[sub["auc_mean"].idxmax(), "Representation"]


def load_patterns(art: Path, top_d: int, top_p: int):
    pats = []
    for fname, top in (("huim_drivers.csv", top_d), ("huim_protectors.csv", top_p)):
        fp = art / fname
        if fp.exists():
            pats += pd.read_csv(fp).head(top)["Pattern"].tolist()
    if not pats:
        raise SystemExit(f"No HUIM patterns in {art} — run huim/mine.py first.")
    return pats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--winner", choices=["sentence_transformer", "skrub", "tabpfn"],
                    help="override the auto-picked winner")
    args = ap.parse_args()

    cfg = common.load_config()
    df = common.load_sample(cfg)
    cat, cont = common.feature_lists(cfg)
    tr_m, va_m, te_m = common.split_masks(df)
    y = df["label"].to_numpy()
    art = (EXP_DIR / cfg["paths"]["work_dir"]).resolve()

    results_csv = EXP_DIR / "results.csv"
    results = pd.read_csv(results_csv)
    base = ["sentence_transformer", "skrub", "tabpfn"]
    winner = args.winner or pick_winner(results, base)
    print(f"Winner representation: {winner}")

    emb = np.load(art / f"emb_{winner}.npy")

    # HUIM binary columns — discretizer fit on TRAIN only (matches huim/mine.py)
    h = cfg["huim"]
    disc = KBinsDiscretizer(n_bins=h["n_bins"], encode="ordinal", strategy="quantile")
    disc.fit(df[tr_m][cont])
    patterns = load_patterns(art, h["top_drivers_n"], h["top_protectors_n"])
    H = add_huim_features(df, patterns, cat, cont, disc)

    # drop patterns that never fire on train (zero variance -> no signal)
    keep = [c for c in H.columns if H.loc[tr_m, c].var() > 1e-8]
    H = H[keep]
    print(f"HUIM columns: {len(keep)}/{len(patterns)} kept (zero-variance dropped)")

    Z = np.concatenate([emb, H.to_numpy(np.float32)], axis=1)
    print(f"Hybrid dim = {Z.shape[1]} ({emb.shape[1]} emb + {len(keep)} HUIM)")

    m = run_multiseed(Z[tr_m], y[tr_m], Z[va_m], y[va_m], Z[te_m], y[te_m], cfg["head"])

    name = f"D_{winner}_plus_huim"
    row = {"Representation": name, "dim": Z.shape[1],
           "auc_mean": m["auc_mean"], "auc_std": m["auc_std"],
           "f1_mean": m["f1_mean"], "f1_std": m["f1_std"],
           "acc_mean": m["acc_mean"], "acc_std": m["acc_std"]}
    results = results[results["Representation"] != name]
    results = pd.concat([results, pd.DataFrame([row])], ignore_index=True)
    results.to_csv(results_csv, index=False)

    print(f"\n  {name} -> AUC {m['auc_mean']:.4f} +/- {m['auc_std']:.4f} | "
          f"F1m {m['f1_mean']:.4f} | ACC {m['acc_mean']:.4f}")
    base_auc = results.loc[results["Representation"] == winner, "auc_mean"].iloc[0]
    print(f"  lift over {winner}: {m['auc_mean'] - base_auc:+.4f} AUC")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
