"""
05_baseline_100_data/compare.py
===============================
Qualitatively compare the LLM-Select (LLM-Score) feature ranking against the
existing SHAP importances and HUIM itemset patterns from the 2024 baseline
extraction.

Inputs
  - output/llm_feature_importance_datafree.csv        (from run.py)
  - output/llm_feature_importance_datainformed.csv     (from run.py)
  - ../../shap_huim_extractions/01_2024_baseline/csvs/shap_feature_importance.csv
  - ../../shap_huim_extractions/01_2024_baseline/csvs/huim_delay_drivers.csv
  - ../../shap_huim_extractions/01_2024_baseline/csvs/huim_delay_protectors.csv

Outputs (in output/)
  - comparison.csv           per-feature merged table across methods
  - comparison_report.md     narrative: rank correlations, overlaps, itemset check
  - figures/*.png            grouped bar, scatter, heatmap

Note: the SHAP/HUIM signals were mined on 2024 data while the LLM sample rows are
2020; feature *semantics* are year-independent, so this qualitative feature-level
comparison is valid. See the caveats section of the report.

Run:
    uv run python exp/Tab_exp/ablations/05_baseline_100_data/compare.py            # all extractions
    uv run python exp/Tab_exp/ablations/05_baseline_100_data/compare.py --extraction 02_2020_train

--extraction picks a folder under shap_huim_extractions/ (repeatable; default: every
folder in DEFAULT_EXTRACTIONS that has csvs/). 01_2024_baseline writes to output/
(unchanged location); any other extraction writes to output/<extraction>/. When more
than one extraction is compared, output/extraction_summary.md puts their Spearman /
Jaccard numbers side by side, plus SHAP-vs-SHAP agreement between the extractions.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

SCRIPT_DIR = Path(__file__).parent
LLM_DIR = SCRIPT_DIR / "output"   # run.py outputs (inputs here), fixed
OUT_DIR = SCRIPT_DIR / "output"
FIG_DIR = OUT_DIR / "figures"
EXTRACTIONS_ROOT = (SCRIPT_DIR / "../../shap_huim_extractions").resolve()
DEFAULT_EXTRACTIONS = ["01_2024_baseline", "02_2020_train"]
EXTRACTION_YEAR = {"01_2024_baseline": "2024", "01_2024_v2": "2024", "02_2020_train": "2020 train days"}
# set per extraction by run_one()
SHAP_HUIM_DIR = EXTRACTIONS_ROOT / "01_2024_baseline" / "csvs"
EXT_LABEL = "2024"

# how many top-utility itemsets define the HUIM feature-frequency signal / get listed
TOP_PATTERNS = 100      # for per-feature appearance frequency
TOP_ITEMSETS_SHOW = 15  # for the itemset listing in the report
TOP_K = 10              # top-k feature set used for overlap / Jaccard


# ── helpers ─────────────────────────────────────────────────────────────────
def pattern_features(pattern: str) -> list[str]:
    """'O_PRCP=VeryLow + O_LONGITUDE=VeryHigh' -> ['O_PRCP', 'O_LONGITUDE']."""
    return [cond.split("=")[0].strip() for cond in str(pattern).split(" + ") if "=" in cond]


def feature_freq(patterns: pd.Series, features: list[str], top_n: int) -> dict[str, float]:
    """Fraction of the top-n patterns (by file order = by utility) containing each feature."""
    top = patterns.head(top_n)
    counts = {f: 0 for f in features}
    for pat in top:
        for f in set(pattern_features(pat)):
            if f in counts:
                counts[f] += 1
    n = max(len(top), 1)
    return {f: counts[f] / n for f in features}


def minmax(s: pd.Series) -> pd.Series:
    lo, hi = s.min(), s.max()
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo == 0:
        return s * 0.0
    return (s - lo) / (hi - lo)


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a | b) else float("nan")


def df_to_markdown(df: pd.DataFrame) -> str:
    """Minimal GitHub-flavored markdown table (avoids the `tabulate` dependency)."""
    cols = list(df.columns)
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = ["| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |"
            for row in df.itertuples(index=False, name=None)]
    return "\n".join([head, sep, *body])


# ── load ────────────────────────────────────────────────────────────────────
def load_inputs():
    df_free_path = LLM_DIR / "llm_feature_importance_datafree.csv"
    df_info_path = LLM_DIR / "llm_feature_importance_datainformed.csv"
    for p in (df_free_path, df_info_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing {p}. Run run.py first.")

    llm_free = pd.read_csv(df_free_path)[["Feature", "LLM_Score"]].rename(
        columns={"LLM_Score": "LLM_datafree"})
    llm_info = pd.read_csv(df_info_path)[["Feature", "LLM_Score"]].rename(
        columns={"LLM_Score": "LLM_datainformed"})

    shap = pd.read_csv(SHAP_HUIM_DIR / "shap_feature_importance.csv").rename(
        columns={"Mean|SHAP|": "SHAP_mean_abs"})
    drivers = pd.read_csv(SHAP_HUIM_DIR / "huim_delay_drivers.csv")
    protectors = pd.read_csv(SHAP_HUIM_DIR / "huim_delay_protectors.csv")
    return llm_free, llm_info, shap, drivers, protectors


def build_table(llm_free, llm_info, shap, drivers, protectors) -> pd.DataFrame:
    df = llm_free.merge(llm_info, on="Feature", how="outer").merge(
        shap, on="Feature", how="outer")

    features = df["Feature"].tolist()
    df["HUIM_driver_freq"] = df["Feature"].map(
        feature_freq(drivers["Pattern"], features, TOP_PATTERNS))
    df["HUIM_protector_freq"] = df["Feature"].map(
        feature_freq(protectors["Pattern"], features, TOP_PATTERNS))

    df["SHAP_rank"] = df["SHAP_mean_abs"].rank(ascending=False, method="min").astype("Int64")
    return df.sort_values("SHAP_mean_abs", ascending=False, ignore_index=True)


# ── figures ─────────────────────────────────────────────────────────────────
def make_figures(df: pd.DataFrame):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    order = df.sort_values("SHAP_mean_abs", ascending=True)  # ascending for barh
    feats = order["Feature"].tolist()
    y = np.arange(len(feats))

    # 1) grouped horizontal bar of min-max-normalized importance
    norm = pd.DataFrame({
        "LLM (data-free)": minmax(order["LLM_datafree"]),
        "LLM (data-informed)": minmax(order["LLM_datainformed"]),
        "SHAP": minmax(order["SHAP_mean_abs"]),
    })
    fig, ax = plt.subplots(figsize=(10, 8))
    h = 0.25
    for i, col in enumerate(norm.columns):
        ax.barh(y + (i - 1) * h, norm[col].values, height=h, label=col)
    ax.set_yticks(y)
    ax.set_yticklabels(feats)
    ax.set_xlabel("min-max normalized importance")
    ax.set_title("Feature importance: LLM-Score vs SHAP (normalized)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "01_importance_bars.png", dpi=130)
    plt.close(fig)

    # 2) scatter LLM vs SHAP (two panels)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, col, title in zip(
        axes,
        ["LLM_datafree", "LLM_datainformed"],
        ["LLM (data-free) vs SHAP", "LLM (data-informed) vs SHAP"],
    ):
        sub = df.dropna(subset=[col, "SHAP_mean_abs"])
        ax.scatter(sub["SHAP_mean_abs"], sub[col], color="tab:blue")
        for _, r in sub.iterrows():
            ax.annotate(r["Feature"], (r["SHAP_mean_abs"], r[col]),
                        fontsize=7, alpha=0.8)
        rho = spearmanr(sub["SHAP_mean_abs"], sub[col])[0]
        ax.set_xlabel(f"SHAP Mean|SHAP| ({EXT_LABEL})")
        ax.set_ylabel(col)
        ax.set_title(f"{title}\nSpearman rho = {rho:.2f}")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_llm_vs_shap_scatter.png", dpi=130)
    plt.close(fig)

    # 3) heatmap of normalized methods
    methods = ["LLM_datafree", "LLM_datainformed", "SHAP_mean_abs",
               "HUIM_driver_freq", "HUIM_protector_freq"]
    hm = df.set_index("Feature")[methods].apply(minmax)
    fig, ax = plt.subplots(figsize=(9, 9))
    im = ax.imshow(hm.values, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(["LLM\n(data-free)", "LLM\n(data-informed)", "SHAP",
                        "HUIM\ndrivers", "HUIM\nprotectors"])
    ax.set_yticks(range(len(hm.index)))
    ax.set_yticklabels(hm.index)
    ax.set_title("Normalized feature importance across methods")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_method_heatmap.png", dpi=130)
    plt.close(fig)


# ── report ──────────────────────────────────────────────────────────────────
def top_k_set(df: pd.DataFrame, col: str, k: int = TOP_K) -> set:
    return set(df.dropna(subset=[col]).nlargest(k, col)["Feature"])


def corr_line(df, col):
    sub = df.dropna(subset=[col, "SHAP_mean_abs"])
    rho = spearmanr(sub["SHAP_mean_abs"], sub[col])[0]
    tau = kendalltau(sub["SHAP_mean_abs"], sub[col])[0]
    return rho, tau


def itemset_block(title, patterns: pd.Series, utils: pd.Series, llm_scores: dict) -> str:
    lines = [f"### {title} (top {TOP_ITEMSETS_SHOW} by utility)", "",
             "| # | Itemset | Utility | Mean LLM (data-free) score of its features |",
             "|---|---------|--------:|:------------------------------------------:|"]
    for i, (pat, u) in enumerate(zip(patterns.head(TOP_ITEMSETS_SHOW),
                                     utils.head(TOP_ITEMSETS_SHOW)), 1):
        feats = pattern_features(pat)
        vals = [llm_scores.get(f) for f in feats if llm_scores.get(f) is not None]
        mean_llm = f"{np.mean(vals):.2f}" if vals else "n/a"
        lines.append(f"| {i} | {pat} | {int(u)} | {mean_llm} |")
    return "\n".join(lines) + "\n"


def write_report(df, drivers, protectors):
    llm_scores = df.set_index("Feature")["LLM_datafree"].to_dict()

    rho_f, tau_f = corr_line(df, "LLM_datafree")
    rho_i, tau_i = corr_line(df, "LLM_datainformed")

    shap_top = top_k_set(df, "SHAP_mean_abs")
    free_top = top_k_set(df, "LLM_datafree")
    info_top = top_k_set(df, "LLM_datainformed")
    df["HUIM_combined"] = df["HUIM_driver_freq"].fillna(0) + df["HUIM_protector_freq"].fillna(0)
    huim_top = top_k_set(df, "HUIM_combined")

    disp = df[["Feature", "LLM_datafree", "LLM_datainformed", "SHAP_mean_abs",
               "SHAP_rank", "HUIM_driver_freq", "HUIM_protector_freq"]].copy()
    for c in ["LLM_datafree", "LLM_datainformed", "SHAP_mean_abs",
              "HUIM_driver_freq", "HUIM_protector_freq"]:
        disp[c] = disp[c].round(3)

    # rank-disagreement narrative
    tmp = df.dropna(subset=["LLM_datafree", "SHAP_mean_abs"]).copy()
    tmp["llm_rank"] = tmp["LLM_datafree"].rank(ascending=False, method="min")
    tmp["shap_rank"] = tmp["SHAP_mean_abs"].rank(ascending=False, method="min")
    tmp["rank_gap"] = (tmp["llm_rank"] - tmp["shap_rank"]).abs()
    agree = tmp.nsmallest(5, "rank_gap")["Feature"].tolist()
    disagree = tmp.nlargest(5, "rank_gap")[["Feature", "llm_rank", "shap_rank"]]

    md = []
    md.append("# LLM-Select vs SHAP + HUIM — feature-importance comparison\n")
    md.append("Method: **LLM-Score** (taekb/llm-select) on local Ollama "
              "**qwen2.5:7b**, two prompt variants (data-free, data-informed), "
              f"3 repeats/feature. Compared against the {EXT_LABEL} SHAP importances "
              f"({SHAP_HUIM_DIR.parent.name}) and "
              "HUIM delay-driver / delay-protector itemsets.\n")

    md.append("## 1. Per-feature comparison table\n")
    md.append(df_to_markdown(disp))
    md.append("\n`HUIM_*_freq` = fraction of the top-%d itemsets (by utility) that "
              "contain the feature.\n" % TOP_PATTERNS)

    md.append("## 2. Rank agreement with SHAP\n")
    md.append(f"- LLM data-free vs SHAP:      Spearman rho = **{rho_f:.3f}**, "
              f"Kendall tau = {tau_f:.3f}")
    md.append(f"- LLM data-informed vs SHAP:  Spearman rho = **{rho_i:.3f}**, "
              f"Kendall tau = {tau_i:.3f}\n")

    md.append("## 3. Top-%d feature-set overlap (Jaccard)\n" % TOP_K)
    md.append(f"- LLM data-free ∩ SHAP:            {sorted(free_top & shap_top)}  "
              f"(Jaccard {jaccard(free_top, shap_top):.2f})")
    md.append(f"- LLM data-informed ∩ SHAP:        {sorted(info_top & shap_top)}  "
              f"(Jaccard {jaccard(info_top, shap_top):.2f})")
    md.append(f"- LLM data-free ∩ HUIM (combined): {sorted(free_top & huim_top)}  "
              f"(Jaccard {jaccard(free_top, huim_top):.2f})")
    md.append(f"- data-free ∩ data-informed:       {sorted(free_top & info_top)}  "
              f"(Jaccard {jaccard(free_top, info_top):.2f})\n")

    md.append("## 4. Where LLM and SHAP agree / disagree\n")
    md.append(f"- **Closest rank agreement:** {agree}")
    md.append("- **Largest rank gaps (LLM data-free rank vs SHAP rank):**")
    for _, r in disagree.iterrows():
        md.append(f"  - {r['Feature']}: LLM #{int(r['llm_rank'])} vs SHAP "
                  f"#{int(r['shap_rank'])}")
    md.append("")

    md.append("## 5. Do the LLM's important features drive the HUIM itemsets?\n")
    md.append("For each high-utility itemset we show the mean LLM (data-free) "
              "score of its constituent features — high values mean the LLM "
              "independently rated the itemset's features as predictive.\n")
    md.append(itemset_block("Delay drivers", drivers["Pattern"], drivers["Utility"], llm_scores))
    md.append(itemset_block("Delay protectors", protectors["Pattern"], protectors["Utility"], llm_scores))

    md.append("## 6. Caveats\n")
    if "2024" in EXT_LABEL:
        md.append("- **Year mismatch:** SHAP/HUIM were mined on 2024 data; the LLM "
                  "sample rows are 2020. Feature semantics are year-independent, so "
                  "the feature-level comparison holds, but exact magnitudes are not "
                  "strictly aligned.")
    else:
        md.append(f"- SHAP/HUIM were mined on {EXT_LABEL} only (no val/test days), "
                  "the same year as the LLM sample rows and the ablations.")
    md.append("- **Small local model:** qwen2.5:7b is far weaker than the paper's "
              "GPT-4; treat LLM scores as a directional signal (3 repeats + std "
              "quantify run-to-run variance).")
    md.append("- **HUIM is itemset-level:** the per-feature frequency is a "
              "projection of multi-feature patterns onto single features.\n")

    (OUT_DIR / "comparison_report.md").write_text("\n".join(md), encoding="utf-8")


def run_one(ext: str) -> dict:
    """Full comparison for one extraction; returns headline numbers."""
    global SHAP_HUIM_DIR, OUT_DIR, FIG_DIR, EXT_LABEL
    SHAP_HUIM_DIR = EXTRACTIONS_ROOT / ext / "csvs"
    EXT_LABEL = EXTRACTION_YEAR.get(ext, ext)
    OUT_DIR = SCRIPT_DIR / "output" if ext == "01_2024_baseline" else SCRIPT_DIR / "output" / ext
    FIG_DIR = OUT_DIR / "figures"

    llm_free, llm_info, shap, drivers, protectors = load_inputs()
    df = build_table(llm_free, llm_info, shap, drivers, protectors)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "comparison.csv", index=False)
    print(f"[{ext}] Wrote {OUT_DIR / 'comparison.csv'}")
    make_figures(df)
    write_report(df, drivers, protectors)
    print(f"[{ext}] Wrote {OUT_DIR / 'comparison_report.md'} + figures")

    rho_f, tau_f = corr_line(df, "LLM_datafree")
    rho_i, tau_i = corr_line(df, "LLM_datainformed")
    shap_top = top_k_set(df, "SHAP_mean_abs")
    return {
        "extraction": ext, "data": EXT_LABEL,
        "rho_datafree": rho_f, "tau_datafree": tau_f,
        "rho_datainformed": rho_i, "tau_datainformed": tau_i,
        "jaccard_datafree": jaccard(top_k_set(df, "LLM_datafree"), shap_top),
        "jaccard_datainformed": jaccard(top_k_set(df, "LLM_datainformed"), shap_top),
        "shap_top": sorted(shap_top),
        "shap": shap.set_index("Feature")["SHAP_mean_abs"],
    }


def write_summary(results: list[dict]):
    out = SCRIPT_DIR / "output"
    rows = [{k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()
             if k not in ("shap", "shap_top")} for r in results]
    md = ["# LLM-Select vs SHAP — per-extraction summary\n",
          f"Top-{TOP_K} Jaccard is LLM top-{TOP_K} vs SHAP top-{TOP_K}.\n",
          df_to_markdown(pd.DataFrame(rows)), "",
          "## SHAP agreement between extractions\n"]
    for i in range(len(results)):
        for j in range(i + 1, len(results)):
            a, b = results[i], results[j]
            common = a["shap"].index.intersection(b["shap"].index)
            rho = spearmanr(a["shap"][common], b["shap"][common])[0]
            ta, tb = set(a["shap_top"]), set(b["shap_top"])
            md.append(f"- **{a['extraction']} vs {b['extraction']}**: Spearman rho = {rho:.3f} "
                      f"over {len(common)} shared features; top-{TOP_K} Jaccard = {jaccard(ta, tb):.2f}")
            md.append(f"  - only in {a['extraction']}: {sorted(ta - tb)}")
            md.append(f"  - only in {b['extraction']}: {sorted(tb - ta)}")
    (out / "extraction_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    pd.DataFrame(rows).to_csv(out / "extraction_summary.csv", index=False)
    print(f"Wrote {out / 'extraction_summary.md'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extraction", action="append",
                    help="folder under shap_huim_extractions/ (repeatable)")
    args = ap.parse_args()
    exts = args.extraction or [e for e in DEFAULT_EXTRACTIONS
                               if (EXTRACTIONS_ROOT / e / "csvs" / "shap_feature_importance.csv").exists()]
    results = [run_one(e) for e in exts]
    if len(results) > 1:
        write_summary(results)
    for r in results:
        print(f"{r['extraction']:>18}: rho free/informed = {r['rho_datafree']:.3f}/{r['rho_datainformed']:.3f}  "
              f"Jaccard free/informed = {r['jaccard_datafree']:.2f}/{r['jaccard_datainformed']:.2f}")


if __name__ == "__main__":
    main()
