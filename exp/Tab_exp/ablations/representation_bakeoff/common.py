"""
common.py
=========
Shared loading / feature helpers for the representation bake-off.
Every experiment reads the SAME sample.parquet and the SAME feature lists here,
so the only thing that varies across A/B/C/D is the embedding step.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

EXP_DIR = Path(__file__).resolve().parent


def load_config() -> dict:
    with open(EXP_DIR / "config.yaml", "r") as f:
        return yaml.safe_load(f)


def load_sample(cfg: dict) -> pd.DataFrame:
    work = (EXP_DIR / cfg["paths"]["work_dir"]).resolve()
    return pd.read_parquet(work / "sample.parquet")


def feature_lists(cfg: dict) -> tuple[list[str], list[str]]:
    """Categorical / continuous feature names, minus exclude_cols. Read from any
    year's data_info yaml (schema is identical across years)."""
    tab_dir = Path(cfg["paths"]["tab_dir"])
    year = cfg["data"]["years"][0]
    with open(tab_dir / f"data_info_{year}.yaml", "r") as f:
        info = yaml.safe_load(f)
    exclude = set(cfg["data"]["exclude_cols"])
    cat = [c for c in info["columns_info"]["Categorical Features"] if c not in exclude]
    cont = [c for c in info["columns_info"]["Continuous Features"] if c not in exclude]
    return cat, cont


def split_masks(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        (df["split"] == "train").to_numpy(),
        (df["split"] == "val").to_numpy(),
        (df["split"] == "test").to_numpy(),
    )


def encode_numeric(df: pd.DataFrame, cat: list[str], cont: list[str],
                   train_mask: np.ndarray) -> np.ndarray:
    """Label-encode categoricals (fit on TRAIN; unseen -> -1) + raw continuous.
    Produces a plain numeric matrix (used by the TabPFN embedder)."""
    cols = []
    for c in cat:
        tr_vals = df.loc[train_mask, c].astype(str)
        mapping = {v: i for i, v in enumerate(sorted(tr_vals.unique()))}
        cols.append(df[c].astype(str).map(mapping).fillna(-1).to_numpy())
    for c in cont:
        cols.append(df[c].astype(float).to_numpy())
    return np.column_stack(cols).astype(np.float32)
