"""Check that common.fast_loader and chunked inference leave mambular outputs bit-identical.

Trains each model twice on the same synthetic data with the same seed — once with
mambular's stock per-row loader, once with the batched loader (separate processes,
so the patch cannot leak) — and compares predict_proba exactly. Each child also
checks metrics.predict_proba_chunked against a single predict_proba call.

    uv run python exp/Tab_exp/ablations/common/verify_fast_loader.py [--accelerator cpu|gpu]
"""
import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def child(fast: bool, model_name: str, accelerator: str, out: str):
    sys.path.insert(0, str(HERE.parent))
    import pandas as pd
    from common.seeding import set_seed
    from common import fast_loader, pretab_patch
    from common.runner import _model_registry
    pretab_patch.enable()  # both arms: makes chunked == unchunked possible at all
    if fast:
        fast_loader.enable()
    rng = np.random.default_rng(0)
    n = 6000
    X = pd.DataFrame({"cat_a": rng.integers(0, 12, n), "cat_b": rng.integers(0, 40, n),
                      **{f"x{i}": rng.normal(size=n) for i in range(6)},
                      **{f"h{i}": rng.integers(0, 2, n).astype(float) for i in range(5)}})
    y = pd.Series(((X.x0 + 0.5 * X.cat_a / 12 + rng.normal(size=n)) > 0.5).astype(int))
    tr, va, te = slice(0, 4000), slice(4000, 5000), slice(5000, n)
    cls, init = _model_registry()[model_name]
    set_seed(42)
    m = cls(**init)
    t = time.time()
    with tempfile.TemporaryDirectory() as ck:
        m.fit(X[tr], y[tr], X_val=X[va], y_val=y[va], max_epochs=3, batch_size=256, patience=5,
              random_state=42, checkpoint_path=ck, accelerator=accelerator, devices=1,
              logger=False, deterministic="warn", enable_progress_bar=False)
    p = np.asarray(m.predict_proba(X[te]), dtype=np.float64)
    np.save(out, p)
    # chunked inference (chunk rounded to a batch multiple) must equal one call
    from common.metrics import predict_proba_chunked
    pc = predict_proba_chunked(m, X[te], chunk=300, batch_size=256)  # -> 256-row chunks
    print(json.dumps({"fast": fast, "seconds": round(time.time() - t, 1),
                      "chunked_identical": bool(np.array_equal(p, pc))}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--accelerator", default="cpu")
    ap.add_argument("--models", default="MLP,AutoInt,ResNet")
    ap.add_argument("--child", nargs=3)  # fast model out
    a = ap.parse_args()
    if a.child:
        return child(a.child[0] == "1", a.child[1], a.accelerator, a.child[2])
    ok = True
    with tempfile.TemporaryDirectory() as d:
        for name in a.models.split(","):
            paths = {}
            for fast in (0, 1):
                paths[fast] = str(Path(d) / f"{name}_{fast}.npy")
                r = subprocess.run([sys.executable, __file__, "--accelerator", a.accelerator,
                                    "--child", str(fast), name, paths[fast]],
                                   capture_output=True, text=True)
                if r.returncode:
                    print(r.stdout[-2000:], r.stderr[-4000:])
                    raise SystemExit(f"{name} fast={fast} failed")
                line = [l for l in r.stdout.splitlines() if l.startswith("{")][-1]
                ok &= json.loads(line)["chunked_identical"]
                print(name, line)
            p0, p1 = np.load(paths[0]), np.load(paths[1])
            same = p0.shape == p1.shape and np.array_equal(p0, p1)
            ok &= same
            print(f"{name}: identical={same} shape={p0.shape} max|diff|={np.abs(p0 - p1).max():.3g}")
    print("ALL IDENTICAL" if ok else "MISMATCH")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
