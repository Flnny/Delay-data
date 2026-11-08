"""
paired_test.py — paired DeLong test of AUC differences on the shared test set
=============================================================================
Uses the per-row test probabilities the runner saves (<arm>/predictions/*.npy) and
the labels in _ytest/<target>__<test_hash>.npy. Both arms are scored on the SAME
2.5M test rows, so the comparison is paired (DeLong et al. 1988; fast O(n log n)
algorithm of Sun & Xu 2014).

For every requested pair and every seed both arms have, it reports AUC_a, AUC_b,
ΔAUC, its standard error, z and the two-sided p-value. Seeds are also combined by
averaging the per-seed z statistics' underlying ΔAUC (mean Δ, mean SE as a
conservative SE since seeds share the test set).

    uv run python exp/Tab_exp/ablations/paired_test.py                       # default pairs
    uv run python exp/Tab_exp/ablations/paired_test.py --pair 03_shap_plus_huim 01b_baseline_matched_training
Output: aggregate/paired_tests.{csv,md}
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
DEFAULT_PAIRS = [("03_shap_plus_huim", "01b_baseline_matched_training"),
                 ("03_shap_plus_huim", "02_shap_only"),
                 ("02_shap_only", "01b_baseline_matched_training")]
FNAME = re.compile(r"(?P<model>[^_]+(?:Transformer)?)__s(?P<seed>\d+)__(?P<target>[^_]+)__p(?P<p2>[\d.]+)__(?P<ext>.+)\.npy")


def _midrank(x):
    order = np.argsort(x, kind="mergesort")
    z = x[order]
    n = len(x)
    ranks = np.empty(n)
    # vectorised tie handling: group equal values
    diff = np.r_[True, z[1:] != z[:-1], True]
    starts = np.flatnonzero(diff[:-1])
    ends = np.flatnonzero(diff[1:]) + 1
    mid = (starts + ends + 1) / 2.0  # 1-based average rank
    ranks_sorted = np.repeat(mid, ends - starts)
    ranks[order] = ranks_sorted
    return ranks


def delong(y, pa, pb):
    """Paired DeLong: returns auc_a, auc_b, delta, se, z, p."""
    y = np.asarray(y).astype(bool)
    m, n = int(y.sum()), int((~y).sum())  # python ints: numpy bool sums are int32 on Windows -> m*n overflows
    out = []
    v10s, v01s = [], []
    for p in (pa, pb):
        pos, neg = p[y], p[~y]
        tx, ty, tz = _midrank(pos), _midrank(neg), _midrank(np.r_[pos, neg])
        auc = (tz[:m].sum() - m * (m + 1) / 2) / (m * n)
        v01s.append((tz[:m] - tx) / n)         # structural components, positives
        v10s.append(1.0 - (tz[m:] - ty) / m)   # negatives
        out.append(auc)
    sx = np.cov(np.vstack(v01s))
    sy = np.cov(np.vstack(v10s))
    s = sx / m + sy / n
    var = s[0, 0] + s[1, 1] - 2 * s[0, 1]
    delta = out[0] - out[1]
    se = float(np.sqrt(max(var, 1e-300)))
    z = delta / se
    return out[0], out[1], delta, se, z, 2 * norm.sf(abs(z))


def index_preds(arm_dir: Path):
    rows = []
    for f in (arm_dir / "predictions").glob("*.npy"):
        mt = FNAME.match(f.name)
        if mt:
            rows.append({**mt.groupdict(), "seed": int(mt["seed"]), "path": f})
    return pd.DataFrame(rows)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", nargs=2, action="append", metavar=("ARM_A", "ARM_B"))
    ap.add_argument("--target", default="abs")
    a = ap.parse_args()
    pairs = a.pair or DEFAULT_PAIRS
    ys = list((HERE / "_ytest").glob(f"{a.target}__*.npy"))
    if len(ys) != 1:
        raise SystemExit(f"expected one _ytest/{a.target}__<hash>.npy, found {ys}")
    y = np.load(ys[0]).astype(int)
    test_hash = ys[0].stem.split("__")[1]

    res = []
    for arm_a, arm_b in pairs:
        A, B = index_preds(HERE / arm_a), index_preds(HERE / arm_b)
        if A.empty or B.empty:
            print(f"skip {arm_a} vs {arm_b}: no saved predictions")
            continue
        A, B = A[A.target == a.target], B[B.target == a.target]
        # 01b has no extraction: pair it with every extraction of the other arm
        on = ["model", "seed", "p2"]
        mg = A.merge(B, on=on, suffixes=("_a", "_b"))
        mg = mg[(mg.ext_a == mg.ext_b) | (mg.ext_a == "-") | (mg.ext_b == "-")]
        for r in mg.itertuples(index=False):
            pa = np.load(r.path_a).astype(np.float64)
            pb = np.load(r.path_b).astype(np.float64)
            if pa.shape[0] != len(y) or pb.shape[0] != len(y):
                raise SystemExit("prediction length != test length (different test set?)")
            pa = pa[:, -1] if pa.ndim == 2 else pa
            pb = pb[:, -1] if pb.ndim == 2 else pb
            auc_a, auc_b, d, se, z, p = delong(y, pa, pb)
            res.append({"arm_a": arm_a, "arm_b": arm_b, "extraction": r.ext_a if r.ext_a != "-" else r.ext_b,
                        "model": r.model, "seed": r.seed, "phase2_frac": r.p2,
                        "auc_a": auc_a, "auc_b": auc_b, "delta": d, "se": se, "z": z, "p": p})
            print(f"{arm_a} vs {arm_b} [{res[-1]['extraction']}] {r.model} s{r.seed}: "
                  f"Δ={d:+.4f} ± {se:.4f}  z={z:+.1f}  p={p:.2g}")
    if not res:
        raise SystemExit("nothing compared")
    df = pd.DataFrame(res)
    out = HERE / "aggregate"
    out.mkdir(exist_ok=True)
    df.to_csv(out / "paired_tests.csv", index=False)
    g = df.groupby(["arm_a", "arm_b", "extraction", "model", "phase2_frac"])
    summ = g.agg(n_seeds=("seed", "nunique"), delta_mean=("delta", "mean"), delta_std=("delta", "std"),
                 se_mean=("se", "mean"), p_max=("p", "max")).reset_index()
    lines = ["# Paired DeLong tests (same test rows)\n",
             f"test_hash {test_hash}, n_test {len(y)}. `p_max` = largest per-seed p-value; "
             "`delta_std` = spread of ΔAUC across seeds (training noise), `se_mean` = DeLong SE "
             "(test-sampling noise).\n",
             "| A | B | extraction | model | p2 | seeds | ΔAUC mean | ΔAUC std (seeds) | DeLong SE | p_max |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in summ.itertuples(index=False):
        lines.append(f"| {r.arm_a} | {r.arm_b} | {r.extraction} | {r.model} | {r.phase2_frac} | {r.n_seeds} | "
                     f"{r.delta_mean:+.4f} | {'' if pd.isna(r.delta_std) else f'{r.delta_std:.4f}'} | "
                     f"{r.se_mean:.4f} | {r.p_max:.2g} |")
    (out / "paired_tests.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
