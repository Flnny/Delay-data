"""Shared two-phase runner for the mambular ablation arms.

Each arm's run.py is a thin wrapper:  main(Path(__file__).parent)

CLI
---
  --seed 42                  Phase-2 seed (model init, loader order, phase-2 subsamples)
  --target abs|late|3class
  --models MLP,AutoInt       subset of the registry below
  --reuse-params             skip Phase 1 when best_params.json has a matching entry
  --set training.n_iter=2    generic config override (value parsed as YAML), repeatable
  --out-dir DIR              where results.csv / best_params.json go (default: arm dir)

Phase-1 note: the search is implemented here instead of RandomizedSearchCV because
mambular's set_params() updates config_kwargs but NOT the already-built config
object, so clone().set_params() silently trains the default config for every
candidate. Candidates are drawn with sklearn's ParameterSampler and scored with
StratifiedKFold (= what RandomizedSearchCV(cv=int) does for classifiers).
"""
import argparse
import contextlib
import copy
import dataclasses
import gc
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import ParameterSampler, StratifiedKFold

from .seeding import set_seed
from . import data as D
from . import fast_loader, gbdt, mambular_patch, pretab_patch
from .features import build_features
from .metrics import evaluate, positive_scores

warnings.filterwarnings("ignore")
torch.set_float32_matmul_precision("high")

COMMON_DIR = Path(__file__).resolve().parent
ABLATIONS_DIR = COMMON_DIR.parent
TEST_HASH_REGISTRY = ABLATIONS_DIR / "test_hashes.json"

RESULT_COLUMNS = [
    "arm", "model", "seed", "target", "phase2_frac", "n_train", "n_val", "n_test",
    "test_pos_rate", "test_hash", "auc", "accuracy", "f1_macro",
    # extras
    "auc_hard", "extraction", "n_features", "best_params", "phase1_sig", "timestamp",
]
SKIP_KEY = ["model", "seed", "target", "phase2_frac", "extraction"]


def _model_registry():
    from mambular.models import (AutoIntClassifier, FTTransformerClassifier,
                                 MLPClassifier, ResNetClassifier)
    # (class, constructor defaults) — defaults match the legacy run.py scripts
    return {
        "MLP": (MLPClassifier, {"d_model": 128}),
        "AutoInt": (AutoIntClassifier, {"d_model": 128, "n_layers": 4}),
        "ResNet": (ResNetClassifier, {}),
        "FTTransformer": (FTTransformerClassifier, {}),
        # GBDT baselines (common/gbdt.py): same runner interface, own grids
        **{name: (gcls, {}) for name, gcls in gbdt.MODELS.items()},
    }


@contextlib.contextmanager
def _file_lock(path: Path, timeout: float = 600.0):
    """Cross-process lock via an O_EXCL lock file, so parallel runs of the same arm
    (e.g. different extractions) cannot drop each other's results/best_params rows."""
    lock = Path(str(path) + ".lock")
    t0 = time.time()
    while True:
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            if time.time() - t0 > timeout:
                raise TimeoutError(f"could not acquire {lock}; delete it if no run is active")
            time.sleep(0.2)
    try:
        yield
    finally:
        os.close(fd)
        lock.unlink(missing_ok=True)


def _pid_alive(pid: int) -> bool:
    try:
        import psutil
        return psutil.pid_exists(pid)
    except ImportError:  # be conservative: assume alive
        return True


def _claim(out_dir: Path, key: dict) -> Path | None:
    """Mark (model, seed, target, phase2_frac, extraction) as in progress so parallel
    queues skip it. A marker left by a dead process is taken over."""
    marker = out_dir / (".running__" + "__".join(str(key[k]) for k in SKIP_KEY))
    for _ in range(2):
        try:
            fd = os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"{os.getpid()} {time.strftime('%Y-%m-%d %H:%M:%S')}".encode())
            os.close(fd)
            return marker
        except FileExistsError:
            try:
                pid = int(marker.read_text().split()[0])
            except (OSError, ValueError, IndexError):
                pid = -1
            if pid > 0 and _pid_alive(pid):
                print(f"Skipping {key['model']}: running in process {pid} ({marker.name})")
                return None
            print(f"Stale marker {marker.name} (pid {pid} gone): taking over")
            marker.unlink(missing_ok=True)
    return None


# ── config ────────────────────────────────────────────────────────────────────
def _deep_merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in (b or {}).items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _set_dotted(cfg: dict, dotted: str, value):
    keys = dotted.split(".")
    node = cfg
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value


def load_config(arm_dir: Path, args) -> dict:
    base = yaml.safe_load((COMMON_DIR / "base_config.yaml").read_text(encoding="utf-8"))
    arm_path = Path(args.config).resolve() if args.config else arm_dir / "config.yaml"
    arm = yaml.safe_load(arm_path.read_text(encoding="utf-8"))
    overrides = arm.pop("overrides", {}) or {}
    cfg = _deep_merge(_deep_merge(base, arm), overrides)
    for s in args.set or []:
        k, v = s.split("=", 1)
        _set_dotted(cfg, k, yaml.safe_load(v))
    if args.seed is not None:
        cfg["run"]["seed"] = args.seed
    if args.target is not None:
        cfg["run"]["target"] = args.target
    if args.models:
        cfg["run"]["models"] = args.models.split(",")
    if cfg["run"]["target"] not in D.TARGETS:
        raise ValueError(f"target must be one of {D.TARGETS}")
    return cfg


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int)
    p.add_argument("--target", choices=D.TARGETS)
    p.add_argument("--models")
    p.add_argument("--reuse-params", action="store_true")
    p.add_argument("--set", action="append", metavar="KEY=VALUE")
    p.add_argument("--config", help="alternate arm config.yaml (relative paths still resolve against the arm dir)")
    p.add_argument("--out-dir")
    return p.parse_args(argv)


# ── training helpers ──────────────────────────────────────────────────────────
def _grid(t: dict) -> dict:
    return {"d_model": t["d_model_options"], "n_layers": t["n_layers_options"], "lr": t["lr_options"]}


def _materialize(cls, params: dict) -> dict:
    """Translate the shared grid point into constructor kwargs the model really uses.

    MLP/ResNet configs have no `n_layers`; their depth/width live in `layer_sizes`
    (+ `num_blocks` for ResNet). Every key returned must be a config field, so a
    grid value can never be silently ignored.
    """
    if getattr(cls, "is_gbdt", False):
        return dict(params)
    fields = {f.name for f in dataclasses.fields(cls().config)}
    out = dict(params)
    if "n_layers" in out and "n_layers" not in fields:
        n = out.pop("n_layers")
        out["layer_sizes"] = [out["d_model"]] * n
        if "num_blocks" in fields:
            out["num_blocks"] = n
    unknown = [k for k in out if k not in fields]
    if unknown:
        raise ValueError(f"{cls.__name__}: grid keys {unknown} are not config fields")
    return out


def _build(cls, init_kw: dict, params: dict):
    return cls(**{**init_kw, **_materialize(cls, params)})


def _fold_scores(model, X, y) -> dict:
    proba = np.asarray(model.predict_proba(X), dtype=np.float64)
    if proba.ndim == 2 and proba.shape[1] > 2:
        pred = proba.argmax(axis=1)
        auc = roc_auc_score(y, proba, multi_class="ovr", average="macro")
    else:
        p = positive_scores(proba)
        pred = (p > 0.5).astype(int)
        auc = roc_auc_score(y, p)
    return {"roc_auc": float(auc), "accuracy": float(accuracy_score(y, pred))}


def _fit_kwargs(t: dict, patience: int, ckpt_dir: str, X_val=None, y_val=None, rs=101) -> dict:
    kw = {
        "max_epochs": t["max_epochs"], "rebuild": True, "patience": patience,
        "batch_size": t["batch_size"], "random_state": rs,
        "dataloader_kwargs": {"num_workers": t["num_workers"], "pin_memory": False,
                              "persistent_workers": t["num_workers"] > 0},
        "checkpoint_path": ckpt_dir,
        # Trainer kwargs
        "accelerator": t["accelerator"], "devices": 1, "precision": t["precision"],
        "logger": False, "deterministic": "warn",
    }
    if X_val is not None:
        kw.update(X_val=X_val, y_val=y_val)
    else:
        kw["val_size"] = t["phase1_val_size"]
    return kw


def phase1_signature(cfg: dict, model_name: str, extraction: str, gbdt_grid: dict | None = None) -> str:
    t = cfg["training"]
    keys = ["phase1_frac", "phase1_val_mode", "phase1_val_size", "val_frac", "max_epochs",
            "patience_phase1", "batch_size", "precision", "n_iter", "cv", "search_scoring",
            "d_model_options", "n_layers_options", "lr_options"]
    sig = {k: t.get(k) for k in keys}
    sig.update(model=model_name, target=cfg["run"]["target"], phase1_seed=cfg["run"]["phase1_seed"],
               features=cfg["features"], extraction=extraction, data=cfg["data"],
               preprocessing=cfg.get("preprocessing", {}),
               patches={k: t.get(k) for k in ("row_independent_ple", "batch_first_attention")})
    if gbdt_grid is not None:  # only for GBDTs, so existing mambular signatures stay valid
        sig.update(gbdt_grid=gbdt_grid, gbdt_budget=[gbdt.MAX_ROUNDS, gbdt.EARLY_STOP, gbdt.THREADS])
    return hashlib.sha256(json.dumps(sig, sort_keys=True, default=str).encode()).hexdigest()[:12]


def run_search(cls, init_kw, grid, X, y, X_val, y_val, t, ckpt_dir, seed):
    """Random search with fresh construction per candidate. Selects by
    t['search_scoring'] (roc_auc | accuracy); logs both. Returns (best_params, table)."""
    scoring = t["search_scoring"]
    if scoring not in ("roc_auc", "accuracy"):
        raise ValueError(f"search_scoring must be roc_auc or accuracy, got {scoring!r}")
    cands = list(ParameterSampler(grid, n_iter=t["n_iter"], random_state=seed))
    folds = list(StratifiedKFold(n_splits=t["cv"]).split(X, y))
    rows = []
    for i, params in enumerate(cands):
        fold = []
        for tr, te in folds:
            m = _build(cls, init_kw, params)
            m.fit(X.iloc[tr], y.iloc[tr],
                  **_fit_kwargs(t, t["patience_phase1"], ckpt_dir, X_val, y_val, rs=seed))
            fold.append(_fold_scores(m, X.iloc[te], y.iloc[te]))
            del m
            gc.collect()  # Lightning/mambular keep reference cycles; RSS grew ~1 GB/h without this
            torch.cuda.empty_cache()
        row = {"params": params}
        for s in ("roc_auc", "accuracy"):
            v = [f[s] for f in fold]
            row[f"mean_{s}"], row[f"std_{s}"] = float(np.mean(v)), float(np.std(v))
        rows.append(row)
        print(f"    cand {i+1}/{len(cands)} {params} -> "
              f"auc {row['mean_roc_auc']:.4f}  acc {row['mean_accuracy']:.4f}")
    # first max wins (same tie-break as sklearn's rank_test_score)
    best = max(range(len(rows)), key=lambda i: (rows[i][f"mean_{scoring}"], -i))
    return cands[best], rows


# ── main ──────────────────────────────────────────────────────────────────────
def main(arm_dir, argv=None):
    arm_dir = Path(arm_dir).resolve()
    args = parse_args(argv)
    cfg = load_config(arm_dir, args)
    r, t = cfg["run"], cfg["training"]
    arm = cfg["experiment"]["name"]
    seed, p1_seed, target = r["seed"], r["phase1_seed"], r["target"]
    out_dir = Path(args.out_dir).resolve() if args.out_dir else arm_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    results_file, params_file = out_dir / "results.csv", out_dir / "best_params.json"

    print(f"[{arm}] seed={seed} target={target} models={r['models']} out={out_dir}")
    if t.get("fast_loader", False):
        fast_loader.enable()
    if t.get("row_independent_ple", False):
        pretab_patch.enable()
    if t.get("batch_first_attention", False):
        mambular_patch.enable()

    # data + features (deterministic, no RNG involved)
    df_train, df_val, df_test, cat_cols, cont_cols, schema = D.load_splits(cfg)
    digest = D.test_hash(df_test)
    D.check_test_hash(TEST_HASH_REGISTRY, cfg, digest, arm)
    print(f"  rows train/val/test = {len(df_train)}/{len(df_val)}/{len(df_test)}  test_hash={digest}")
    X_train, X_val, X_test, finfo = build_features(cfg, arm_dir, df_train, df_val, df_test,
                                                   cat_cols, cont_cols, schema)
    y_train, y_val, y_test = (d[D.TARGET_COL].copy() for d in (df_train, df_val, df_test))
    # Raw split frames are no longer needed (X_* are independent copies): free RAM.
    del df_train, df_val, df_test
    gc.collect()
    n_classes = int(y_train.nunique())
    extraction = finfo["extraction"]

    # Phase-1 samples: fixed phase1_seed (shared by every Phase-2 seed)
    X_search = X_train.sample(frac=t["phase1_frac"], random_state=p1_seed)
    y_search = y_train.loc[X_search.index]
    explicit = t["phase1_val_mode"] == "explicit"
    if explicit:
        X_val_p1 = X_val if t["val_frac"] >= 1 else X_val.sample(frac=t["val_frac"], random_state=p1_seed)
        y_val_p1 = y_val.loc[X_val_p1.index]
    else:
        X_val_p1 = y_val_p1 = None

    # Phase-2 samples: depend on the run seed. Test is never sampled.
    X_tr2 = X_train if t["phase2_frac"] >= 1 else X_train.sample(frac=t["phase2_frac"], random_state=seed)
    y_tr2 = y_train.loc[X_tr2.index]
    X_va2 = X_val if t["val_frac"] >= 1 else X_val.sample(frac=t["val_frac"], random_state=seed)
    y_va2 = y_val.loc[X_va2.index]

    results = pd.read_csv(results_file) if results_file.exists() else pd.DataFrame(columns=RESULT_COLUMNS)
    best_params = json.loads(params_file.read_text()) if params_file.exists() else {}
    registry = _model_registry()
    ckpt_root = arm_dir / "model_checkpoints"
    ckpt_root.mkdir(exist_ok=True)

    for name in r["models"]:
        key = {"model": name, "seed": seed, "target": target,
               "phase2_frac": t["phase2_frac"], "extraction": extraction}
        # re-read every time: parallel queues may have finished this model meanwhile
        results = pd.read_csv(results_file) if results_file.exists() else pd.DataFrame(columns=RESULT_COLUMNS)
        if len(results) and (results[SKIP_KEY].astype(str) == pd.Series(key).astype(str)).all(axis=1).any():
            print(f"Skipping {name}: {key} already in {results_file.name}")
            continue
        marker = _claim(out_dir, key)
        if marker is None:
            continue
        cls, init_kw = registry[name]
        if getattr(cls, "is_gbdt", False):
            init_kw = {"cat_features": finfo["cat_features"]}
            grid = cls.grid()
            sig = phase1_signature(cfg, name, extraction, gbdt_grid=grid)
        else:
            init_kw = {**init_kw, **cfg.get("preprocessing", {})}
            grid = _grid(t)
            sig = phase1_signature(cfg, name, extraction)
        pkey = f"{name}|{target}|{extraction}"
        ckpt_dir = tempfile.mkdtemp(prefix=f"{name}_s{seed}_", dir=ckpt_root)
        print(f"\n{'=' * 60}\n  {name}\n{'=' * 60}")
        try:
            entry = best_params.get(pkey)
            if args.reuse_params and entry and entry["phase1_sig"] == sig:
                bp = entry["params"]
                print(f"Phase 1: reusing {bp} from {params_file.name}")
            else:
                if args.reuse_params:
                    print("Phase 1: no matching best_params entry -> searching")
                print(f"Phase 1: search on {len(X_search)} rows, {t['n_iter']} cand x {t['cv']} folds")
                set_seed(p1_seed)
                bp, table = run_search(cls, init_kw, grid, X_search, y_search, X_val_p1, y_val_p1,
                                       t, ckpt_dir, p1_seed)
                entry = {"params": bp, "phase1_sig": sig, "scoring": t["search_scoring"], "search": table}
                best_params[pkey] = entry
                with _file_lock(params_file):  # re-read: another process may have written meanwhile
                    cur = json.loads(params_file.read_text()) if params_file.exists() else {}
                    cur[pkey] = entry
                    params_file.write_text(json.dumps(cur, indent=2, default=float))
            print(f"Best params: {bp}")

            print(f"Phase 2: train on {len(X_tr2)} rows, early-stop on {len(X_va2)} val rows (seed {seed})")
            set_seed(seed)
            model = _build(cls, init_kw, bp)
            print(f"  constructor kwargs: {_materialize(cls, bp)}")
            model.fit(X_tr2, y_tr2, **_fit_kwargs(t, t["patience_phase2"], ckpt_dir, X_va2, y_va2, rs=seed))
            m = evaluate(model, X_test, y_test, n_classes,
                         chunk=t["predict_chunk"], batch_size=t["batch_size"])
            proba = m.pop("proba")
            if r.get("save_test_predictions", False):
                # row order = df_test order (fixed by test_hash); used by paired_test.py
                pdir = out_dir / "predictions"
                pdir.mkdir(exist_ok=True)
                np.save(pdir / f"{name}__s{seed}__{target}__p{t['phase2_frac']}__{extraction}.npy",
                        proba.astype(np.float32))
                ydir = ABLATIONS_DIR / "_ytest"
                ydir.mkdir(exist_ok=True)
                yf = ydir / f"{target}__{digest}.npy"
                if not yf.exists():
                    np.save(yf, np.asarray(y_test, dtype=np.int8))
        finally:
            shutil.rmtree(ckpt_dir, ignore_errors=True)
            if sys.exc_info()[0] is not None:  # failing: release the claim for other queues
                marker.unlink(missing_ok=True)

        auc_col = m["auc_hard"] if r["legacy_metrics"] else m["auc"]
        print(f"{name} RESULT  AUC: {auc_col:.6f}  ACC: {m['accuracy']:.6f}  F1m: {m['f1_macro']:.6f}")
        row = {
            "arm": arm, "model": name, "seed": seed, "target": target,
            "phase2_frac": t["phase2_frac"], "n_train": len(X_tr2), "n_val": len(X_va2),
            "n_test": len(X_test),
            "test_pos_rate": float((y_test == D.positive_class(target)).mean()),
            "test_hash": digest, "auc": auc_col, "accuracy": m["accuracy"], "f1_macro": m["f1_macro"],
            "auc_hard": m["auc_hard"], "extraction": extraction, "n_features": X_test.shape[1],
            "best_params": json.dumps(bp, default=float), "phase1_sig": sig,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with _file_lock(results_file):  # re-read: another process may have appended meanwhile
            cur = pd.read_csv(results_file) if results_file.exists() else pd.DataFrame(columns=RESULT_COLUMNS)
            results = pd.concat([cur, pd.DataFrame([row])], ignore_index=True)[RESULT_COLUMNS]
            results.to_csv(results_file, index=False)
        marker.unlink(missing_ok=True)
        del model
        torch.cuda.empty_cache()
        gc.collect()

    print(f"\nResults -> {results_file}")
    print(results[["arm", "model", "seed", "target", "auc", "accuracy", "f1_macro", "test_hash"]].to_string(index=False))
    return results
