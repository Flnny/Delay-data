"""Feature-set construction. Every fitted transform (label maps, scaler,
discretizer, variance filter) is fit on the TRAIN split only.

Modes
-----
all             all raw categorical + continuous columns
shap_only       top_shap_n features of <extraction_dir>/shap_feature_importance.csv
llm_only        top_n features of `importance_file` (same Feature-ranked CSV shape)
shap_plus_huim  top_shap_n SHAP features + top driver/protector HUIM patterns
huim_only       HUIM patterns only

HUIM columns are float 0/1 in both modes (so 03 vs 04 differ only by the SHAP
features). Patterns that never fire in TRAIN are dropped: they carry no
information and a zero-variance float column becomes NaN in mambular's scaler.
Patterns are matched against the raw schema, so a pattern may reference a column
that is excluded as a feature (e.g. FL_DAY).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler

SHAP_FILE = "shap_feature_importance.csv"
DRIVERS_FILE = "huim_delay_drivers.csv"
PROTECTORS_FILE = "huim_delay_protectors.csv"
BIN_LABELS = {"VeryLow": 0, "Low": 1, "Mid": 2, "High": 3, "VeryHigh": 4}
UNSEEN = -1  # code for categories absent from train; mambular maps unknowns to its reserved 0


def _resolve(arm_dir: Path, p: str) -> Path:
    p = Path(p)
    return (p if p.is_absolute() else arm_dir / p).resolve()


def extraction_dir(fcfg: dict, arm_dir: Path) -> Path | None:
    return _resolve(arm_dir, fcfg["extraction_dir"]) if fcfg.get("extraction_dir") else None


def _top_ranked(path: Path, n: int, cat_cols, cont_cols):
    top = pd.read_csv(path).head(n)["Feature"].tolist()
    missing = [f for f in top if f not in cat_cols + cont_cols]
    if missing:
        print(f"  !! ranked features not in schema (ignored): {missing}")
    kept_cat = [c for c in cat_cols if c in top]
    kept_cont = [c for c in cont_cols if c in top]
    print(f"  ranking {path.name}: top {n} -> {len(kept_cat)} cat + {len(kept_cont)} cont")
    return kept_cat, kept_cont


def _load_patterns(ext: Path, fcfg: dict) -> list[str]:
    d = pd.read_csv(ext / DRIVERS_FILE).head(fcfg["top_drivers_n"])["Pattern"].tolist()
    p = pd.read_csv(ext / PROTECTORS_FILE).head(fcfg["top_protectors_n"])["Pattern"].tolist()
    print(f"  HUIM: {len(d)} drivers + {len(p)} protectors from {ext}")
    return d + p


def huim_col(pattern: str) -> str:
    return "HUIM_" + pattern.replace(" + ", "_").replace("=", "_")


def _parse(pattern: str) -> list[tuple[str, str]]:
    return [tuple(c.strip().split("=", 1)) for c in pattern.split(" + ")]


def _add_huim(splits, patterns, schema, n_bins):
    """Float 0/1 match columns; matching uses RAW values (before encoding).
    The discretizer is fit on TRAIN, exact quantiles (subsample=None) so bins are
    deterministic, only on the continuous columns the patterns reference."""
    conds = {p: _parse(p) for p in patterns}
    for p, cs in conds.items():
        for col, _ in cs:
            if col not in schema["cat"] and col not in schema["cont"]:
                raise ValueError(f"HUIM pattern {p!r} references column {col!r} not in the data schema")
    cont_used = sorted({c for cs in conds.values() for c, _ in cs if c in schema["cont"]})
    disc = None
    if cont_used:
        disc = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="quantile",
                                quantile_method="linear", subsample=None)
        disc.fit(splits[0][cont_used])
    for df in splits:
        binned = disc.transform(df[cont_used]) if disc is not None else None
        for pat, cs in conds.items():
            m = np.ones(len(df), dtype=bool)
            for col, val in cs:
                if col in schema["cont"]:
                    m &= binned[:, cont_used.index(col)] == BIN_LABELS[val]
                else:
                    try:
                        m &= (pd.to_numeric(df[col], errors="coerce") == float(val)).fillna(False).to_numpy()
                    except ValueError:
                        m &= (df[col].astype(str) == val).to_numpy()
            df[huim_col(pat)] = m.astype(float)
    return [huim_col(p) for p in patterns]


def _encode(splits, cat_cols, cont_cols):
    """Train-fitted label maps (unseen -> -1) and StandardScaler."""
    tr = splits[0]
    for col in cat_cols:
        mapping = {v: i for i, v in enumerate(sorted(tr[col].astype(str).unique()))}
        for df in splits:
            df[col] = df[col].astype(str).map(mapping).fillna(UNSEEN).astype(np.int64)
    if cont_cols:
        scaler = StandardScaler().fit(tr[cont_cols])
        for df in splits:
            df[cont_cols] = scaler.transform(df[cont_cols])


def build_features(cfg: dict, arm_dir: Path, df_train, df_val, df_test, cat_cols, cont_cols, schema):
    """Mutates the split frames in place; returns X_train, X_val, X_test, info."""
    fcfg = cfg["features"]
    mode = fcfg["mode"]
    splits = [df_train, df_val, df_test]
    ext = extraction_dir(fcfg, arm_dir)
    info = {"mode": mode, "extraction": ext.parent.name if ext and ext.name == "csvs" else (ext.name if ext else "-")}

    if mode == "all":
        kcat, kcont, huim = cat_cols, cont_cols, []
    elif mode == "shap_only":
        kcat, kcont = _top_ranked(ext / SHAP_FILE, fcfg["top_shap_n"], cat_cols, cont_cols)
        huim = []
    elif mode == "llm_only":
        kcat, kcont = _top_ranked(_resolve(arm_dir, fcfg["importance_file"]), fcfg["top_n"], cat_cols, cont_cols)
        huim = []
        info["extraction"] = Path(fcfg["importance_file"]).stem
    elif mode in ("shap_plus_huim", "huim_only"):
        huim = _add_huim(splits, _load_patterns(ext, fcfg), schema, fcfg["n_bins"])
        var = df_train[huim].var()
        dropped = [c for c in huim if var[c] <= 1e-8]
        huim = [c for c in huim if c not in dropped]
        info["huim_dropped"] = dropped
        print(f"  HUIM: {len(huim)} kept, {len(dropped)} dropped (never fire in train): {dropped}")
        if mode == "shap_plus_huim":
            kcat, kcont = _top_ranked(ext / SHAP_FILE, fcfg["top_shap_n"], cat_cols, cont_cols)
        else:
            kcat, kcont = [], []
    else:
        raise ValueError(f"unknown features.mode {mode!r}")

    _encode(splits, kcat, kcont)
    features = kcat + huim + kcont

    print(f"  features ({len(features)}): {features}")
    info["features"] = features
    info["cat_features"] = list(kcat)  # label-encoded categoricals (CatBoost cat_features)
    return df_train[features], df_val[features], df_test[features], info
