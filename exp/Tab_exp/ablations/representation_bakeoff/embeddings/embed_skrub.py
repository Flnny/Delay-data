"""
embeddings/embed_skrub.py  —  Representation B
==============================================
skrub TableVectorizer: raw dataframe -> bounded numeric matrix. High-cardinality
columns (OP_CARRIER_FL_NUM, airport indices) are handled by skrub's default
encoders, so the output width stays modest. Fit on TRAIN rows only.
"""
import numpy as np


def build(df, cat, cont, train_mask, y, cfg) -> np.ndarray:
    from skrub import TableVectorizer

    X = df[cat + cont].copy()
    for c in cat:                       # force categorical dtype for skrub
        X[c] = X[c].astype(str)

    tv = TableVectorizer(
        cardinality_threshold=cfg["representations"]["skrub"]["cardinality_threshold"]
    )
    tv.fit(X[train_mask])               # fit on train only
    Z = tv.transform(X)
    Z = np.asarray(getattr(Z, "values", Z), dtype=np.float32)
    print(f"    [skrub] output dim = {Z.shape[1]}")
    return Z
