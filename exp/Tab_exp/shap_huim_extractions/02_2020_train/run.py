"""
02_2020_train/run.py — SHAP + HUIM extraction on the 2020 TRAIN split only
==========================================================================
Same method / output format as 01_2024_baseline/run.py, with these changes:

  * Data: 2020, and ONLY train days (FL_DAY <= train_days_max). Val/test days are
    dropped right after loading, so no downstream ablation is contaminated.
      FL_DAY <= fit_days_max  -> MLP fit + SHAP background
      remaining train days    -> MLP early stopping + the rows SHAP explains
  * Encoding is fit on the fit part only (01_2024 fit LabelEncoder/StandardScaler
    per split). HUIM items are labelled from RAW values (same "COL=value" format).
  * Continuous items use a KBinsDiscretizer fit on the full train split (exact
    quantiles) — the same bins ablations/common/features.py uses to match patterns.
  * The SHAP target function rebuilds a typed DataFrame (so categorical columns stay
    int for mambular). shap.output=label reproduces 01_2024 (explains predict()).
  * exclude_cols matches ablations/common/base_config.yaml (adds FL_DAY).
  * Patterns are sorted by NUMERIC utility. 01_2024_baseline sorted PAMI's string
    utilities lexicographically, so its "top-N" rows are not the top-N by utility.
  * pretab's PLE batch-max clamp is fixed (common/pretab_patch.py). This matters
    most here: SHAP evaluates the model on small perturbed batches, so with the stock
    transform attributions partly reflect batch composition (01_2024 ran unpatched).

Outputs (csvs/): shap_feature_importance.csv, huim_delay_drivers.csv,
huim_delay_protectors.csv, huim_item_labels.csv, extraction_meta.json
            (txt/): shap_utility_positive.txt, shap_utility_negative.txt

Run:
    uv run python exp/Tab_exp/shap_huim_extractions/02_2020_train/run.py
"""
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import yaml
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import KBinsDiscretizer

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parents[1] / "ablations"))
from common import data as D  # noqa: E402
from common import fast_loader, pretab_patch  # noqa: E402
from common.features import BIN_LABELS, _encode  # noqa: E402
from common.metrics import positive_scores  # noqa: E402
from common.seeding import set_seed  # noqa: E402

warnings.filterwarnings("ignore")

cfg = yaml.safe_load((SCRIPT_DIR / "config.yaml").read_text(encoding="utf-8"))
d, sh, hu, mcfg = cfg["data"], cfg["shap"], cfg["huim"], cfg["mlp"]
CSVS_DIR = SCRIPT_DIR / cfg["outputs"]["csvs_dir"]
TXT_DIR = SCRIPT_DIR / cfg["outputs"]["txt_dir"]
CSVS_DIR.mkdir(exist_ok=True)
TXT_DIR.mkdir(exist_ok=True)
BIN_NAMES = {v: k for k, v in BIN_LABELS.items()}
SEED = d["seed"]
t0 = time.time()


def step(msg):
    print("\n" + "=" * 60 + f"\n  {msg}   [{time.time() - t0:.0f}s]\n" + "=" * 60, flush=True)


# ── 1. data: train days only ─────────────────────────────────────────────────
step("STEP 1: loading 2020, train days only")
set_seed(SEED)
# SHAP calls predict on small perturbed batches: without the PLE fix the encoding of a
# row would depend on the other rows in that batch (see common/pretab_patch.py).
pretab_patch.enable()
fast_loader.enable()
loader_cfg = {
    "data": {"file": d["file"], "yaml_info": d["yaml_info"],
             "train_days_max": d["train_days_max"], "val_days_max": d["train_days_max"]},
    "run": {"target": "abs", "delay_threshold": 15.0},
    "features": {"exclude_cols": cfg["features"]["exclude_cols"]},
    "paths": {"data_dir": cfg["paths"]["data_dir"]},
}
train_full, _, _, cat_cols, cont_cols, _ = D.load_splits(loader_cfg)
assert train_full["FL_DAY"].max() <= d["train_days_max"]
print(f"  train split rows: {len(train_full)} (FL_DAY {train_full['FL_DAY'].min()}..{train_full['FL_DAY'].max()})")
features = cat_cols + cont_cols
print(f"  features ({len(features)}): {features}")

# Continuous-item bins: fit on the FULL train split (what the ablations use for matching)
disc = KBinsDiscretizer(n_bins=hu["n_bins"], encode="ordinal", strategy="quantile",
                        quantile_method="linear", subsample=None).fit(train_full[cont_cols])

df = train_full.sample(frac=d["sample_frac"], random_state=SEED)
del train_full
fit_raw = df[df["FL_DAY"] <= d["fit_days_max"]].copy()
es_raw = df[df["FL_DAY"] > d["fit_days_max"]].copy()
print(f"  sample {len(df)} rows -> fit {len(fit_raw)} (days <= {d['fit_days_max']}), "
      f"early-stop/explain {len(es_raw)}")
print(f"  target rate: fit {fit_raw[D.TARGET_COL].mean():.3f}  es {es_raw[D.TARGET_COL].mean():.3f}")

fit_enc, es_enc = fit_raw.copy(), es_raw.copy()
_encode([fit_enc, es_enc], cat_cols, cont_cols)
X_fit, y_fit = fit_enc[features], fit_enc[D.TARGET_COL]
X_es, y_es = es_enc[features], es_enc[D.TARGET_COL]

# ── 2. MLP ───────────────────────────────────────────────────────────────────
step("STEP 2: training MLP")
from mambular.models import MLPClassifier  # noqa: E402

import tempfile  # noqa: E402

model = MLPClassifier(d_model=mcfg["d_model"])
_ckpt = tempfile.TemporaryDirectory(prefix="shap_mlp_")
model.fit(X_fit, y_fit, X_val=X_es, y_val=y_es, max_epochs=mcfg["max_epochs"], lr=mcfg["lr"],
          patience=mcfg["patience"], batch_size=mcfg["batch_size"], random_state=SEED,
          checkpoint_path=_ckpt.name, logger=False)
p_es = positive_scores(model.predict_proba(X_es))
mlp_auc, mlp_acc = roc_auc_score(y_es, p_es), accuracy_score(y_es, (p_es > 0.5).astype(int))
print(f"  MLP on early-stop days — AUC(proba): {mlp_auc:.4f}  ACC: {mlp_acc:.4f}")

# ── 3. SHAP ──────────────────────────────────────────────────────────────────
step(f"STEP 3: SHAP (output={sh['output']})")
dtypes = X_fit.dtypes


def f_model(A):
    X = pd.DataFrame(np.asarray(A), columns=features).astype(dtypes)
    if sh["output"] == "label":
        return np.asarray(model.predict(X), dtype=float)
    if sh["output"] == "proba":
        return positive_scores(model.predict_proba(X))
    raise ValueError(f"shap.output must be label or proba, got {sh['output']!r}")


X_bg = X_fit.sample(n=min(sh["bg_samples"], len(X_fit)), random_state=SEED)
explain_idx = X_es.sample(n=min(sh["test_samples"], len(X_es)), random_state=SEED).index
X_explain = X_es.loc[explain_idx]
explain_raw = es_raw.loc[explain_idx]
print(f"  background {len(X_bg)} (fit days), explaining {len(X_explain)} (early-stop days)")
explainer = shap.Explainer(f_model, X_bg, seed=SEED)
shap_values = np.asarray(explainer(X_explain).values)
print(f"  SHAP shape {shap_values.shape}, range [{shap_values.min():.4f}, {shap_values.max():.4f}]")

importance_df = pd.DataFrame({"Feature": features, "Mean|SHAP|": np.abs(shap_values).mean(axis=0)}
                             ).sort_values("Mean|SHAP|", ascending=False)
print(importance_df.head(10).to_string(index=False))

# ── 4. items ─────────────────────────────────────────────────────────────────
step("STEP 4: items")
binned = disc.transform(explain_raw[cont_cols]).astype(int)
item_id, item_label, nxt = {}, {}, 1
for col in cat_cols:
    for v in sorted(explain_raw[col].astype(str).unique()):
        item_id[(col, v)] = nxt
        item_label[nxt] = f"{col}={v}"
        nxt += 1
for col in cont_cols:
    for b in range(hu["n_bins"]):
        item_id[(col, b)] = nxt
        item_label[nxt] = f"{col}={BIN_NAMES[b]}"
        nxt += 1
print(f"  {nxt - 1} items")

# ── 5. transactions ──────────────────────────────────────────────────────────
step("STEP 5: HUIM transactions")
scale = hu["shap_scale"]


def transactions(sign):
    lines, skipped = [], 0
    for r in range(len(explain_raw)):
        items, utils = [], []
        for j, col in enumerate(features):
            sv = shap_values[r, j]
            if (sign > 0 and sv > 0) or (sign < 0 and sv < 0):
                key = (col, str(explain_raw[col].iloc[r])) if col in cat_cols \
                    else (col, int(binned[r, cont_cols.index(col)]))
                items.append(item_id[key])
                utils.append(max(int(abs(sv) * scale), 1))
        if items:
            lines.append("\t".join(map(str, items)) + f":{sum(utils)}:" + "\t".join(map(str, utils)))
        else:
            skipped += 1
    print(f"    [{'positive' if sign > 0 else 'negative'}] {len(lines)} transactions, {skipped} empty")
    return lines


pos_file, neg_file = TXT_DIR / "shap_utility_positive.txt", TXT_DIR / "shap_utility_negative.txt"
pos_file.write_text("\n".join(transactions(+1)))
neg_file.write_text("\n".join(transactions(-1)))

# ── 6. EFIM ──────────────────────────────────────────────────────────────────
step("STEP 6: EFIM mining")
from PAMI.highUtilityPattern.basic import EFIM  # noqa: E402


def mine(path: Path, label: str):
    tu = [int(l.split(":")[1]) for l in path.read_text().splitlines() if l.strip()]
    if not tu:
        print(f"  [{label}] no transactions")
        return pd.DataFrame(), None
    used = None
    for pct in (hu["min_util_percentile"], 75):
        min_util = int(np.percentile(tu, pct))
        obj = EFIM.EFIM(str(path), min_util, "\t")
        obj.mine()
        pats = obj.getPatterns()
        print(f"  [{label}] p{pct} minUtil={min_util}: {len(pats)} patterns")
        if pats:
            used = {"percentile": pct, "min_util": min_util}
            break
    if not pats:
        return pd.DataFrame(), used
    rows = []
    for pstr, util in pats.items():
        ids = [int(x) for x in str(pstr).split("\t") if x.strip().isdigit()]
        # PAMI returns utilities as str: without int() the sort below is lexicographic
        # ("9999" > "99875"), which is what happened in 01_2024_baseline.
        rows.append({"Pattern": " + ".join(item_label[i] for i in ids), "ItemIDs": ids,
                     "Utility": int(util), "Size": len(ids)})
    out = pd.DataFrame(rows).sort_values("Utility", ascending=False, kind="mergesort")
    print(out.head(10)[["Utility", "Pattern"]].to_string(index=False))
    return out, used


drivers, drv_thr = mine(pos_file, "drivers")
protectors, prt_thr = mine(neg_file, "protectors")

# ── 7. save ──────────────────────────────────────────────────────────────────
step("STEP 7: saving")
importance_df.to_csv(CSVS_DIR / "shap_feature_importance.csv", index=False)
if not drivers.empty:
    drivers.to_csv(CSVS_DIR / "huim_delay_drivers.csv", index=False)
if not protectors.empty:
    protectors.to_csv(CSVS_DIR / "huim_delay_protectors.csv", index=False)
pd.DataFrame([{"ItemID": k, "Label": v} for k, v in item_label.items()]).to_csv(
    CSVS_DIR / "huim_item_labels.csv", index=False)
meta = {
    "config": cfg, "features": features,
    "rows": {"sample": len(df), "fit": len(fit_raw), "early_stop": len(es_raw),
             "background": len(X_bg), "explained": len(X_explain)},
    "fl_day_range": {"fit": [int(fit_raw.FL_DAY.min()), int(fit_raw.FL_DAY.max())],
                     "explain": [int(explain_raw.FL_DAY.min()), int(explain_raw.FL_DAY.max())]},
    "mlp_early_stop_auc": mlp_auc, "mlp_early_stop_acc": mlp_acc,
    "n_drivers": len(drivers), "n_protectors": len(protectors),
    "min_util": {"drivers": drv_thr, "protectors": prt_thr},
    "seconds": round(time.time() - t0),
}
(CSVS_DIR / "extraction_meta.json").write_text(json.dumps(meta, indent=2, default=str))
print(json.dumps({k: v for k, v in meta.items() if k != "config"}, indent=2, default=str))
