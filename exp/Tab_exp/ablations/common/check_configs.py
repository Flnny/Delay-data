"""Show how the RESOLVED configs of the ablation arms differ.

Each arm's config is merged exactly as the runner does (base -> arm -> overrides)
and every leaf is compared against the reference arm. Differences under
`experiment` and `features` are expected (that is what an arm is); anything else
means the arms are not trained under the same conditions.

    uv run python exp/Tab_exp/ablations/common/check_configs.py [--ref 01b_baseline_matched_training]
Exit code 1 if a compared arm differs outside experiment/features.
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common.runner import load_config, parse_args  # noqa: E402

ARMS = ["01_baseline_all_features", "01b_baseline_matched_training", "02_shap_only",
        "03_shap_plus_huim", "04_huim_only", "06_llm_top10", "06_1_llm_top5_datafree",
        "06_2_llm_top5_datainformed"]
FEATURE_SECTIONS = ("experiment", "features")


def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        else:
            out[key] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="01b_baseline_matched_training")
    ap.add_argument("--expect-legacy", nargs="*", default=["01_baseline_all_features"],
                    help="arms allowed to differ in training (reported, not failed)")
    a = ap.parse_args()
    root = HERE.parent
    cfgs = {arm: flatten(load_config(root / arm, parse_args([]))) for arm in ARMS if (root / arm).exists()}
    ref = cfgs[a.ref]
    bad = False
    for arm, c in cfgs.items():
        if arm == a.ref:
            continue
        keys = sorted(set(ref) | set(c))
        diff = [(k, ref.get(k, "<absent>"), c.get(k, "<absent>")) for k in keys if ref.get(k) != c.get(k)]
        feat = [d for d in diff if d[0].split(".")[0] in FEATURE_SECTIONS]
        other = [d for d in diff if d[0].split(".")[0] not in FEATURE_SECTIONS]
        status = "OK" if not other else ("LEGACY (expected)" if arm in a.expect_legacy else "MISMATCH")
        bad |= status == "MISMATCH"
        print(f"\n## {arm} vs {a.ref}: {status}")
        print(f"   feature-set keys differing: {[k for k, _, _ in feat if not k.startswith('experiment')]}")
        for k, r, v in other:
            print(f"   !! {k}: {r!r} -> {v!r}")
    print("\nALL NON-LEGACY ARMS MATCH OUTSIDE FEATURES" if not bad else "\nMISMATCH FOUND")
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
