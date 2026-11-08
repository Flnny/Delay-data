"""Row-independent, vectorised replacement for pretab's PLETransformer.transform.

Bug in pretab 0.0.1 (pretab/transformers/ple/ple.py, transform):

    if locations[-1] > np.max(feature):
        locations[-1] = np.max(feature)

The last FITTED threshold is overwritten with the max of the batch being
transformed, so a row's encoding depends on which other rows are transformed with
it. Consequences: test predictions change with the inference chunk size, and SHAP
(which calls predict on small perturbed batches) attributes partly to batch
composition. The original loop is also O(rows) in Python, which dominates
preprocessing time.

patched_transform() keeps the fitted thresholds fixed and computes the same
piecewise-linear encoding with numpy. It is bit-identical to the original whenever
the clamp does not fire (np.max(feature) >= last threshold), which
`python -m common.pretab_patch` (run from ablations/) verifies, together with
batch-invariance.
"""
import re

import numpy as np
from sklearn.utils.validation import check_array

_PATCHED = False
_ORIGINAL = None


def _locations(conds, pattern):
    locs = []
    for s in conds:
        locs.extend(re.findall(pattern, s))
    return np.sort(list({float(x) for x in locs}))


def patched_transform(self, X):
    X = check_array(X, dtype=np.float64, ensure_2d=True, ensure_all_finite=False)
    out = []
    for col in range(X.shape[1]):
        feature = X[:, col]  # name used by the eval'd fitted conditions
        enc = np.sum(np.stack([eval(c) * (i + 1) for i, c in enumerate(self.conditions_[col])]).T, axis=1)
        b = np.asarray(enc - 1, dtype=np.int64)
        L = _locations(self.conditions_[col], self.pattern)  # fitted, never clamped
        n, k = len(feature), len(L) + 1
        ple = np.zeros((n, k))
        ge = feature >= L[-1]
        le = ~ge & (feature <= L[0])
        mid = ~ge & ~le
        rows = np.arange(n)
        # prefix of ones for bins below the active one (ge and mid branches)
        prefix = (np.arange(k)[None, :] < b[:, None]) & (ge | mid)[:, None]
        ple[prefix] = 1.0
        val = feature.copy()  # ge / le branches store the raw value
        bm = b[mid]
        val[mid] = (feature[mid] - L[bm - 1]) / (L[bm] - L[bm - 1])
        ple[rows, b] = val
        if ple.shape[1] == 1:
            ple = np.zeros([n, self.n_bins])
        out.append(ple)
    return np.hstack(out).astype(np.float32)


def enable() -> None:
    global _PATCHED, _ORIGINAL
    if _PATCHED:
        return
    from pretab.transformers.ple.ple import PLETransformer
    _ORIGINAL = PLETransformer.transform
    PLETransformer.transform = patched_transform
    _PATCHED = True


def _selftest():
    from pretab.transformers.ple.ple import PLETransformer
    orig = PLETransformer.transform
    rng = np.random.default_rng(0)
    n = 20000
    X = np.column_stack([rng.normal(size=n), rng.exponential(size=n), rng.integers(0, 2, n).astype(float),
                         np.round(rng.normal(size=n), 1), rng.integers(0, 50, n).astype(float)])
    y = (X[:, 0] + X[:, 2] + rng.normal(size=n) > 0.7).astype(int)
    t = PLETransformer(n_bins=20, task="classification").fit(X[:15000], y[:15000])
    T = X[15000:]
    # 1) identical to the original on a batch where the clamp cannot fire
    #    (append one row holding each column's max over everything).
    big = np.vstack([T, X.max(axis=0)])
    a, b = orig(t, big), patched_transform(t, big)
    same = np.array_equal(a, b)
    print(f"identical to original when clamp inactive: {same}")
    # 2) batch invariance of the patched transform
    full = patched_transform(t, T)
    parts = np.vstack([patched_transform(t, T[i:i + 37]) for i in range(0, len(T), 37)])
    inv = np.array_equal(full, parts)
    print(f"patched batch-invariant: {inv}")
    # 3) the original is NOT batch-invariant (documents the bug)
    o_full = orig(t, T)
    o_parts = np.vstack([orig(t, T[i:i + 37]) for i in range(0, len(T), 37)])
    print(f"original batch-invariant: {np.array_equal(o_full, o_parts)} "
          f"(max |diff| {np.abs(o_full - o_parts).max():.3g})")
    ok = same and inv
    print("PRETAB PATCH OK" if ok else "PRETAB PATCH FAILED")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if _selftest() else 1)
