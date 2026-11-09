"""
embeddings/embed_tabpfn.py  —  Representation C
===============================================
TabPFN (Prior Labs, local inference) as a FROZEN embedder. Fit the in-context set
on a train subsample (<= context_max), then extract the per-sample transformer
embedding for every row via get_embeddings(X, data_source="test").

get_embeddings returns (n_estimators, n_samples, dim); we average over the
ensemble axis. The 50k rows are emitted in batches to bound GPU memory (the fixed
context is re-attended for each batch). Downloads model weights on first run.
"""
import numpy as np

from exp.Tab_exp.ablations.representation.bakeoff.common import encode_numeric


def build(df, cat, cont, train_mask, y, cfg) -> np.ndarray:
    from tabpfn import TabPFNClassifier

    tcfg = cfg["representations"]["tabpfn"]
    X = encode_numeric(df, cat, cont, train_mask)     # numeric matrix, TabPFN-ready
    y = np.asarray(y)
    tr_idx = np.where(train_mask)[0]

    rng = np.random.default_rng(cfg["data"]["seed"])
    if len(tr_idx) > tcfg["context_max"]:
        tr_idx = rng.choice(tr_idx, size=tcfg["context_max"], replace=False)

    clf = TabPFNClassifier(
        n_estimators=tcfg.get("n_estimators", 2),
        auto_scale_n_estimators=False,
        ignore_pretraining_limits=True,
    )
    clf.fit(X[tr_idx], y[tr_idx])

    bs = tcfg["batch_size"]
    chunks = []
    for start in range(0, len(X), bs):
        e = np.asarray(clf.get_embeddings(X[start:start + bs]), dtype=np.float32)
        if e.ndim == 3:                    # (n_estimators, batch, dim) -> mean over ensemble
            e = e.mean(axis=0)
        chunks.append(e)
    emb = np.concatenate(chunks, axis=0)
    print(f"    [tabpfn] output dim = {emb.shape[1]} over {len(emb)} rows")
    return emb
