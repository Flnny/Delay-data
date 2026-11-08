"""Loading, target construction, temporal split and test-set fingerprint."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

TARGET_COL = "__target__"
TARGETS = ("abs", "late", "3class")


def make_target(arr_delay: pd.Series, target: str, thr: float) -> pd.Series:
    if target == "abs":
        y = arr_delay.abs() > thr
    elif target == "late":
        y = arr_delay > thr
    elif target == "3class":
        # 0 = early (< -thr), 1 = on time, 2 = late (> thr)
        return pd.Series(np.select([arr_delay < -thr, arr_delay > thr], [0, 2], 1),
                         index=arr_delay.index).astype(int)
    else:
        raise ValueError(f"unknown target {target!r}; expected one of {TARGETS}")
    return y.astype(int)


def positive_class(target: str) -> int:
    """Class whose rate is reported as test_pos_rate (the 'late' class for 3class)."""
    return 2 if target == "3class" else 1


def load_splits(cfg: dict):
    """Return df_train, df_val, df_test (full, unsampled), cat/cont feature lists
    (after exclude_cols) and the full schema {"cat": [...], "cont": [...]}.

    dropna and the split happen on the full frame BEFORE any feature selection, so
    every arm sees exactly the same rows; test_hash checks that.
    """
    d, r = cfg["data"], cfg["run"]
    data_dir = cfg["paths"]["data_dir"]
    df = pd.read_csv(os.path.join(data_dir, d["file"]))
    df = df.dropna()
    df[TARGET_COL] = make_target(df["ARR_DELAY"], r["target"], r["delay_threshold"])

    with open(os.path.join(data_dir, d["yaml_info"]), "r") as f:
        info = yaml.safe_load(f)["columns_info"]
    exclude = set(cfg["features"]["exclude_cols"])
    cat_cols = [c for c in info["Categorical Features"] if c not in exclude]
    cont_cols = [c for c in info["Continuous Features"] if c not in exclude]

    tmax, vmax = d["train_days_max"], d["val_days_max"]
    df_train = df[df["FL_DAY"] <= tmax].copy()
    df_val = df[(df["FL_DAY"] > tmax) & (df["FL_DAY"] <= vmax)].copy()
    df_test = df[df["FL_DAY"] > vmax].copy()
    schema = {"cat": list(info["Categorical Features"]), "cont": list(info["Continuous Features"])}
    return df_train, df_val, df_test, cat_cols, cont_cols, schema


def test_hash(df_test: pd.DataFrame) -> str:
    """sha256 over test row indices + labels (first 16 hex chars)."""
    h = hashlib.sha256()
    h.update(np.asarray(df_test.index, dtype=np.int64).tobytes())
    h.update(np.asarray(df_test[TARGET_COL], dtype=np.int64).tobytes())
    return h.hexdigest()[:16]


def check_test_hash(registry: Path, cfg: dict, digest: str, arm: str) -> None:
    """All arms with the same data/target/split must share one test set; else fail."""
    d = cfg["data"]
    key = (f"{d['file']}|target={cfg['run']['target']}|thr={cfg['run']['delay_threshold']}"
           f"|train<={d['train_days_max']}|val<={d['val_days_max']}")
    reg = json.loads(registry.read_text()) if registry.exists() else {}
    entry = reg.get(key)
    if entry is None:
        reg[key] = {"test_hash": digest, "first_arm": arm}
        registry.write_text(json.dumps(reg, indent=2))
    elif entry["test_hash"] != digest:
        raise RuntimeError(
            f"test_hash mismatch for {key}: arm {arm!r} got {digest}, "
            f"registry has {entry['test_hash']} (from {entry['first_arm']!r}). "
            f"Test sets differ across arms — comparison would be invalid."
        )
