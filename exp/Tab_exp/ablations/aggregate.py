"""
aggregate.py — one command for the whole ablation table
=========================================================================
Collects every <arm>/results.csv written by common/runner.py (results_legacy.csv is
ignored) and writes, under ablations/aggregate/:

  ablation_table.csv / .md   arm x model x extraction x target x phase2_frac:
                             mean ± std of auc / accuracy / f1_macro over seeds
  deltas.csv / .md           per extraction and model (seed-matched where possible):
                             Δ SHAP selection = 02 − 01b, Δ HUIM = 03 − 02,
                             Δ total = 03 − 01b, plus 04 − 01b
  (both .md files are concatenated into ABLATIONS.md)
  <arm>/RESULTS.md           per-arm results + Phase-1 selections (linked from each arm README)

Checks: all rows of a target share one test_hash (else it aborts).

    uv run python exp/Tab_exp/ablations/aggregate.py [--target abs] [--phase2-frac 0.1]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "aggregate"

# display order + short labels (experiment.name -> label)
ARMS = {
    "baseline_matched_training": "01b all features (ref)",
    "shap_only": "02 SHAP top-10",
    "shap_plus_huim": "03 SHAP + HUIM",
    "huim_only": "04 HUIM only",
    "llm_top10": "06 LLM top-10",
    "llm_top5_datafree": "06.1 LLM top-5 (data-free)",
    "llm_top5_datainformed": "06.2 LLM top-5 (data-informed)",
    "baseline_all_features": "01 all features (legacy cfg)",
}
MODELS = ["MLP", "AutoInt", "ResNet", "FTTransformer", "LightGBM", "XGBoost", "CatBoost"]

# Where the SHAP ranking / HUIM patterns of arms 02-04 come from. The primary
# extraction is shown in the main tables; the other-year one gets its own section.
PRIMARY_EXT = "02_2020_train"
OTHER_YEAR_EXT = "01_2024_baseline"
OTHER_YEAR_NOTE = (
    "**What this table is.** Arms 02, 03 and 04 need a SHAP feature ranking and HUIM "
    "patterns. In the main tables these are mined from the 2020 training days — the same "
    "year the models are trained and tested on. The rows below repeat 02, 03 and 04 with "
    "the ranking and patterns mined from **2024** instead (the method's original source); "
    "the models are still trained and tested on 2020. The question is whether patterns "
    "found in another year still help in 2020. The all-features reference (01b) is "
    "repeated for comparison.")
METRICS = ["auc", "accuracy", "f1_macro"]


def load() -> pd.DataFrame:
    frames = []
    for f in sorted(HERE.glob("*/results.csv")):
        df = pd.read_csv(f)
        if "test_hash" not in df.columns:  # not a runner file
            continue
        df["source"] = f.parent.name
        frames.append(df)
    if not frames:
        raise SystemExit("no results.csv from the runner found")
    df = pd.concat(frames, ignore_index=True)
    for tgt, g in df.groupby("target"):
        if g["test_hash"].nunique() != 1:
            raise SystemExit(f"target={tgt}: multiple test_hash values {g['test_hash'].unique()} — "
                             "results are not on the same test set")
    return df


def fmt(m, s, n):
    return f"{m:.4f}" if n < 2 or pd.isna(s) else f"{m:.4f} ± {s:.4f}"


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join("" if pd.isna(v) else str(v) for v in r) + " |"
              for r in df.itertuples(index=False, name=None)]
    return "\n".join(lines)


def summary(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["arm", "model", "extraction", "target", "phase2_frac"]
    g = df.groupby(keys, dropna=False)
    out = g[METRICS].agg(["mean", "std"])
    out.columns = [f"{m}_{s}" for m, s in out.columns]
    out["n_seeds"] = g["seed"].nunique()
    out["seeds"] = g["seed"].apply(lambda s: ",".join(map(str, sorted(s.unique()))))
    out = out.reset_index()
    out["arm_order"] = out["arm"].map({a: i for i, a in enumerate(ARMS)}).fillna(99)
    out["model_order"] = out["model"].map({m: i for i, m in enumerate(MODELS)}).fillna(99)
    return out.sort_values(["target", "phase2_frac", "arm_order", "extraction", "model_order"]).drop(
        columns=["arm_order", "model_order"])


def pivot(s: pd.DataFrame) -> pd.DataFrame:
    """Rows: feature set (arm + extraction); columns: models; cells: AUC mean ± std, best in bold."""
    rows = []
    multi = s["target"].nunique() > 1 or s["phase2_frac"].nunique() > 1
    for (tgt, p2, arm, ext), g in s.groupby(["target", "phase2_frac", "arm", "extraction"], sort=False):
        name = ARMS.get(arm, arm)
        if ext == OTHER_YEAR_EXT:
            name += " — 2024 patterns"
        r = {"feature set": name}
        if multi:
            r["target/p2"] = f"{tgt}/{p2}"
        best = g.loc[g["auc_mean"].idxmax(), "model"]
        for m in MODELS:
            x = g[g.model == m]
            cell = fmt(x.auc_mean.iloc[0], x.auc_std.iloc[0], x.n_seeds.iloc[0]) if len(x) else ""
            r[m] = f"**{cell}**" if cell and m == best else cell
        rows.append(r)
    return pd.DataFrame(rows)


def deltas(df: pd.DataFrame) -> pd.DataFrame:
    """Seed-matched AUC differences. 01b has no extraction; it is the reference for all."""
    rows = []
    exts = sorted(e for e in df["extraction"].dropna().unique()
                  if e != "-" and not str(e).startswith("llm_"))
    for (tgt, p2), d in df.groupby(["target", "phase2_frac"]):
        ref = d[d.arm == "baseline_matched_training"].set_index(["model", "seed"])["auc"]
        for ext in exts:
            arm_auc = {a: d[(d.arm == a) & (d.extraction == ext)].set_index(["model", "seed"])["auc"]
                       for a in ("shap_only", "shap_plus_huim", "huim_only")}
            for model in MODELS:
                def pair(a, b):
                    """mean/std of a-b over seeds present in both, and n."""
                    if a is None or b is None:
                        return np.nan, np.nan, 0
                    sa = a[a.index.get_level_values(0) == model].droplevel(0)
                    sb = b[b.index.get_level_values(0) == model].droplevel(0)
                    common = sa.index.intersection(sb.index)
                    if len(common) == 0:
                        return np.nan, np.nan, 0
                    diff = sa[common] - sb[common]
                    return diff.mean(), diff.std() if len(diff) > 1 else np.nan, len(common)

                s02, s03, s04 = arm_auc["shap_only"], arm_auc["shap_plus_huim"], arm_auc["huim_only"]
                r = {"target": tgt, "phase2_frac": p2, "extraction": ext, "model": model}
                for name, (a, b) in {"Δ SHAP sel (02−01b)": (s02, ref), "Δ HUIM (03−02)": (s03, s02),
                                     "Δ total (03−01b)": (s03, ref), "Δ HUIM-only (04−01b)": (s04, ref)}.items():
                    m, s, n = pair(a, b)
                    r[name] = np.nan if n == 0 else round(m, 4)
                    r[name + " std"] = s
                    r[name + " n"] = n
                for a, col in ((ref, "AUC 01b"), (s02, "AUC 02"), (s03, "AUC 03"), (s04, "AUC 04")):
                    v = a[a.index.get_level_values(0) == model]
                    r[col] = round(v.mean(), 4) if len(v) else np.nan
                if any(pd.notna(r[c]) for c in ("AUC 02", "AUC 03", "AUC 04")):
                    rows.append(r)
    return pd.DataFrame(rows)


def write_arm_results(s: pd.DataFrame, df: pd.DataFrame) -> list[Path]:
    """<arm>/RESULTS.md: this arm's rows (all targets / phase2_frac / extractions) and
    the Phase-1 parameters stored in its best_params.json."""
    import json
    written = []
    src_of = df.groupby("arm")["source"].first().to_dict()
    for arm, g in s.groupby("arm", sort=False):
        d = HERE / src_of[arm]
        lines = [f"# Results — {ARMS.get(arm, arm)}\n",
                 "Generated by `../aggregate.py` from `results.csv` (do not edit by hand). "
                 "Test set: all 2,547,088 test-day rows of 2020. Mean ± std over seeds; "
                 "AUC from predicted probabilities. `phase2_frac` = share of the training "
                 "split used for the final fit (0.1 or 1.0).\n"]
        def table(h):
            return md_table(pd.DataFrame({
                "model": h["model"],
                "AUC": [fmt(x, y, n) for x, y, n in zip(h.auc_mean, h.auc_std, h.n_seeds)],
                "accuracy": [fmt(x, y, n) for x, y, n in zip(h.accuracy_mean, h.accuracy_std, h.n_seeds)],
                "macro-F1": [fmt(x, y, n) for x, y, n in zip(h.f1_macro_mean, h.f1_macro_std, h.n_seeds)],
                "seeds": h["seeds"]}))

        main, other = g[g.extraction != OTHER_YEAR_EXT], g[g.extraction == OTHER_YEAR_EXT]
        if len(other):
            lines.append("SHAP ranking / HUIM patterns mined on the **2020 training days** "
                         "(primary). Results with patterns mined on 2024 are in the last section.\n")
        for (tgt, p2), h in main.groupby(["target", "phase2_frac"]):
            lines += [f"## target `{tgt}`, phase2_frac {p2}\n", table(h), ""]
        if len(other):
            lines += ["## Patterns mined on another year (2024)\n",
                      "Same arm, but the SHAP ranking / HUIM patterns come from **2024** "
                      "(the method's original source) instead of the 2020 training days. The "
                      "models are still trained and tested on 2020; this checks whether "
                      "patterns from another year still help.\n"]
            for (tgt, p2), h in other.groupby(["target", "phase2_frac"]):
                lines += [f"### target `{tgt}`, phase2_frac {p2}\n", table(h), ""]
        bp = d / "best_params.json"
        if bp.exists():
            lines.append("## Phase-1 selection (shared by all seeds and phase2_frac values)\n")
            rows = []
            for key, e in json.loads(bp.read_text()).items():
                model, tgt, ext = key.split("|")
                best = max(e.get("search", []), key=lambda r: r.get(f"mean_{e.get('scoring', 'roc_auc')}", -1),
                           default={})
                ext = {PRIMARY_EXT: "2020 train", OTHER_YEAR_EXT: "2024"}.get(ext, ext)
                rows.append({"model": model, "target": tgt, "patterns from": ext,
                             "params": json.dumps(e["params"]),
                             "CV AUC": round(best.get("mean_roc_auc", float("nan")), 4)})
            lines += [md_table(pd.DataFrame(rows)), ""]
        if (d / "results_legacy.csv").exists():
            lines.append("`results_legacy.csv` holds this arm's results from before the shared runner "
                         "(hard-label AUC, one seed, no effective search); they are not comparable.\n")
        (d / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        written.append(d / "RESULTS.md")
    return written


def main():
    if hasattr(sys.stdout, "reconfigure"):  # Windows consoles default to cp125x (no "Δ", "±")
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--target")
    ap.add_argument("--phase2-frac", type=float)
    a = ap.parse_args()
    df = load()
    if a.target:
        df = df[df.target == a.target]
    if a.phase2_frac is not None:
        df = df[np.isclose(df.phase2_frac, a.phase2_frac)]
    OUT.mkdir(exist_ok=True)

    s = summary(df)
    s.to_csv(OUT / "ablation_table.csv", index=False)
    disp = pd.DataFrame({
        "arm": s["arm"].map(ARMS).fillna(s["arm"]), "extraction": s["extraction"],
        "model": s["model"], "target": s["target"], "p2_frac": s["phase2_frac"],
        **{m.upper() if m == "auc" else m: [fmt(x, y, n) for x, y, n in zip(s[f"{m}_mean"], s[f"{m}_std"], s["n_seeds"])]
           for m in METRICS},
        "seeds": s["seeds"],
    })
    d = deltas(df)
    d.to_csv(OUT / "deltas.csv", index=False)

    is_other = s["extraction"] == OTHER_YEAR_EXT
    is_ref = s["arm"] == "baseline_matched_training"
    dcols = ["target", "phase2_frac", "model", "AUC 01b", "AUC 02", "AUC 03", "AUC 04",
             "Δ SHAP sel (02−01b)", "Δ HUIM (03−02)", "Δ total (03−01b)", "Δ HUIM-only (04−01b)"]
    md = ["# Ablation results\n",
          f"Generated by `aggregate.py` from {df['source'].nunique()} arm result files, "
          f"{len(df)} runs. Test set hash(es): {sorted(df['test_hash'].unique())}. "
          "Values are mean ± std over seeds (std omitted for a single seed). "
          "AUC is computed from predicted probabilities. Models are always trained and "
          "tested on 2020; arms 02–04 use SHAP rankings / HUIM patterns mined on the 2020 "
          "training days. Results with patterns mined on 2024 are in a separate section "
          "at the end.\n",
          "## Side by side: AUC per feature set (rows) and model (columns)\n",
          "mean ± std over seeds; **bold** = best model for that feature set.\n",
          md_table(pivot(s[~is_other])), "",
          "## Arm × model (all metrics)\n",
          md_table(disp[~is_other.to_numpy()].drop(columns="extraction")), ""]
    if len(d):
        md += ["## Where does the gain come from? (seed-matched AUC deltas)\n",
               "Δ SHAP sel isolates feature selection; Δ HUIM isolates adding the HUIM pattern "
               "columns on top of the same SHAP features. Empty cells: that arm has not run yet.\n",
               md_table(d[d.extraction == PRIMARY_EXT][dcols]), ""]
    if is_other.any():
        md += ["## Patterns mined on another year (2024)\n", OTHER_YEAR_NOTE + "\n",
               "### AUC per feature set and model\n",
               md_table(pivot(s[is_other | is_ref])), "",
               "### All metrics\n",
               md_table(disp[(is_other | is_ref).to_numpy()].drop(columns="extraction")), ""]
        if len(d):
            md += ["### Seed-matched AUC deltas with 2024 patterns\n",
                   md_table(d[d.extraction == OTHER_YEAR_EXT][dcols]), ""]
    bake = HERE / "representation" / "bakeoff" / "RESULTS_GBDT.md"
    if bake.exists():
        body = bake.read_text(encoding="utf-8").split("\n", 1)[1]
        md += ["## Representation bake-off (50k sample, all years) incl. TabFM head\n", body]
    (OUT / "ABLATIONS.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    if a.target is None and a.phase2_frac is None:  # per-arm files only from the full set
        write_arm_results(s, df)
    print("\n".join(md))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
