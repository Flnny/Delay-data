"""
huim/pattern_features.py
========================
Turn mined HUIM patterns into binary HUIM_* columns on the sample. A row's column
is 1 iff every condition in the pattern holds. Categorical conditions match on the
original value string; continuous conditions match the row's quantile bin.

The KBinsDiscretizer MUST be the one fit on the TRAIN split (same as huim/mine.py),
so pass it in from the caller.
"""
import numpy as np
import pandas as pd

BIN_LABELS = {"VeryLow": 0, "Low": 1, "Mid": 2, "High": 3, "VeryHigh": 4}


def add_huim_features(df, patterns, cat, cont, discretizer) -> pd.DataFrame:
    cont_binned = discretizer.transform(df[cont]).astype(int)
    cat_arr = {c: df[c].astype(str).to_numpy() for c in cat}
    out = {}
    for pat in patterns:
        matches = np.ones(len(df), dtype=bool)
        for cond in (c.strip() for c in pat.split(" + ")):
            col, val = cond.split("=", 1)
            if col in cat:
                matches &= cat_arr[col] == val
            elif col in cont:
                ci = cont.index(col)
                matches &= cont_binned[:, ci] == BIN_LABELS[val]
        name = "HUIM_" + pat.replace(" + ", "_").replace("=", "_")
        out[name] = matches.astype(np.float32)
    return pd.DataFrame(out, index=df.index)
