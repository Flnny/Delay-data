"""Fix mambular's transformer encoder layer attending across ROWS of a batch.

mambular 1.5.0, arch_utils/transformer_utils.py: CustomTransformerEncoderLayer
subclasses nn.TransformerEncoderLayer without batch_first=True (PyTorch default
False), but FTTransformer's embedding layer emits (batch, n_features, d_model).
nn.MultiheadAttention therefore treats the BATCH as the sequence: every row
attends to the other rows of its mini-batch instead of to its own feature tokens.

Effects: memory grows with batch_size**2 (FT at batch 2048, d_model 64, 10 layers
needs >16 GB), and a flight's prediction depends on which flights share its batch.
AutoInt is not affected (it builds its own MultiheadAttention with
batch_first=True); MLP/ResNet have no attention.

enable() makes the layer's attention batch-first. Run
`python -m common.mambular_patch --accelerator gpu` (from ablations/) to check
that every registered model's predictions are invariant (up to float noise, TOL) to
how the test rows are chunked (chunks deliberately NOT aligned to the batch size).
Measured on this machine: FT max|dprob| 0.0204 unpatched -> 3.2e-5 patched, peak GPU
memory 1083 -> 284 MiB; MLP/AutoInt/ResNet 1e-7..1.3e-4 (TF32 noise) either way.
"""
import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np

_PATCHED = False
TOL = 1e-3  # max |dprob| tolerated between chunkings (float noise, not leakage)


def enable() -> None:
    global _PATCHED
    if _PATCHED:
        return
    from mambular.arch_utils.transformer_utils import CustomTransformerEncoderLayer as L

    orig_init = L.__init__

    def __init__(self, config):
        orig_init(self, config)
        self.batch_first = True
        self.self_attn.batch_first = True

    L.__init__ = __init__
    _PATCHED = True


def _selftest(accelerator: str, models: list[str], patch: bool) -> bool:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import pandas as pd
    import torch
    from common import fast_loader, pretab_patch
    from common.metrics import positive_scores
    from common.runner import _model_registry
    from common.seeding import set_seed

    pretab_patch.enable()
    fast_loader.enable()
    if patch:
        enable()
    rng = np.random.default_rng(0)
    n = 9000
    X = pd.DataFrame({"cat_a": rng.integers(0, 12, n), "cat_b": rng.integers(0, 400, n),
                      **{f"x{i}": rng.normal(size=n) for i in range(8)}})
    y = pd.Series(((X.x0 + X.cat_a / 12 + rng.normal(size=n)) > 0.6).astype(int))
    T = X[7000:]
    ok = True
    for name in models:
        cls, init = _model_registry()[name]
        set_seed(42)
        m = cls(**{**init, "cat_cutoff": 10**9})
        torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
        with tempfile.TemporaryDirectory() as ck:
            m.fit(X[:6000], y[:6000], X_val=X[6000:7000], y_val=y[6000:7000], max_epochs=2,
                  batch_size=512, checkpoint_path=ck, accelerator=accelerator, devices=1,
                  logger=False, enable_progress_bar=False)
        full = positive_scores(m.predict_proba(T))
        parts = np.concatenate([positive_scores(m.predict_proba(T.iloc[i:i + 333]))
                                for i in range(0, len(T), 333)])  # 333: not a batch multiple
        diff = float(np.abs(full - parts).max())
        # Unaligned chunks change GEMM shapes, so TF32/cuBLAS kernel choice adds float
        # noise (<=1e-4 here, every model). Cross-row attention shows up as ~1e-2.
        same = diff < TOL
        ok &= same
        peak = torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else float("nan")
        print(f"{name:14s} chunk-invariant={same} max|diff|={diff:.3g} peak_gpu={peak:.0f} MiB")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--accelerator", default="gpu")
    ap.add_argument("--models", default="MLP,AutoInt,ResNet,FTTransformer")
    ap.add_argument("--no-patch", action="store_true", help="show the unpatched behaviour")
    a = ap.parse_args()
    ok = _selftest(a.accelerator, a.models.split(","), patch=not a.no_patch)
    print("MAMBULAR PATCH OK" if ok else "NOT CHUNK-INVARIANT")
    raise SystemExit(0 if ok else 1)
