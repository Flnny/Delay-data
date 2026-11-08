"""Test-set metrics. AUC is computed from probabilities (fix for the legacy
hard-label AUC, which is really balanced accuracy); the hard-label value is kept
as `auc_hard` so old tables can still be bridged."""
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def positive_scores(proba: np.ndarray) -> np.ndarray:
    """mambular returns (n, 1) sigmoid outputs for binary tasks, (n, k) softmax otherwise."""
    proba = np.asarray(proba)
    if proba.ndim == 1:
        return proba
    return proba[:, 0] if proba.shape[1] == 1 else proba[:, 1]


def predict_proba_chunked(model, X, chunk: int, batch_size: int) -> np.ndarray:
    """mambular expands every numerical column (PLE -> ~20 dims, float64) for the
    whole input at once; on the 2.5M-row test set that exhausts RAM.

    The chunk is rounded down to a multiple of the inference batch size so the
    GPU batches are exactly the ones an unchunked call would form (mixed-precision
    kernels can differ in the last bits for a different batch composition)."""
    chunk = max(batch_size, (chunk // batch_size) * batch_size) if chunk > 0 else 0
    if chunk <= 0 or len(X) <= chunk:
        return np.asarray(model.predict_proba(X), dtype=np.float64)
    parts = [np.asarray(model.predict_proba(X.iloc[i:i + chunk]), dtype=np.float64)
             for i in range(0, len(X), chunk)]
    return np.concatenate(parts, axis=0)


def evaluate(model, X_test, y_test, n_classes: int, chunk: int, batch_size: int) -> dict:
    y = np.asarray(y_test)
    proba = predict_proba_chunked(model, X_test, chunk, batch_size)
    if n_classes == 2:
        p = positive_scores(proba)
        pred = (p > 0.5).astype(int)
        auc = roc_auc_score(y, p)
        auc_hard = roc_auc_score(y, pred) if len(np.unique(pred)) > 1 else 0.5
    else:
        pred = proba.argmax(axis=1)
        auc = roc_auc_score(y, proba, multi_class="ovr", average="macro")
        auc_hard = np.nan
    return {
        "auc": float(auc),
        "accuracy": float(accuracy_score(y, pred)),
        "f1_macro": float(f1_score(y, pred, average="macro")),
        "auc_hard": float(auc_hard),
        "proba": proba,  # (n_test, 1|k) — popped by the runner, saved for paired tests
    }
