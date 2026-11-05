"""
head/mlp_head.py
================
THE fixed classifier head for the representation bake-off.

A plain PyTorch MLP over a dense feature matrix. This is deliberately NOT
mambular's MLPClassifier: mambular re-embeds raw tabular columns internally and
would mangle a precomputed dense embedding. Here every representation (A/B/C/D)
is fed as-is; only the input dimension changes. Architecture, optimiser,
schedule, standardisation and seeds are identical for all of them, so any metric
difference reflects representation quality and nothing else.

Contract:
    run_multiseed(Z_tr, y_tr, Z_va, y_va, Z_te, y_te, head_cfg)
        -> {"auc_mean","auc_std","f1_mean","f1_std","acc_mean","acc_std","seeds":[...]}
"""
from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: list[int], dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _to_tensor(a: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(np.asarray(a, dtype=np.float32), device=device)


def train_eval_once(Z_tr, y_tr, Z_va, y_va, Z_te, y_te, head_cfg, seed, device) -> dict:
    """Train the fixed MLP once with a given seed; return test metrics."""
    _set_seed(seed)

    # Standardise on TRAIN only (fit on train, apply to val/test) — same for every rep.
    scaler = StandardScaler().fit(Z_tr)
    Z_tr = scaler.transform(Z_tr)
    Z_va = scaler.transform(Z_va)
    Z_te = scaler.transform(Z_te)

    Xtr = _to_tensor(Z_tr, device)
    ytr = _to_tensor(np.asarray(y_tr), device)
    Xva = _to_tensor(Z_va, device)
    yva = np.asarray(y_va)
    Xte = _to_tensor(Z_te, device)
    yte = np.asarray(y_te)

    model = MLP(Xtr.shape[1], head_cfg["hidden"], head_cfg["dropout"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=head_cfg["lr"])
    loss_fn = nn.BCEWithLogitsLoss()

    ds = torch.utils.data.TensorDataset(Xtr, ytr)
    gen = torch.Generator().manual_seed(seed)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=head_cfg["batch_size"], shuffle=True, generator=gen
    )

    best_auc, best_state, no_improve = -np.inf, None, 0
    for _epoch in range(head_cfg["max_epochs"]):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            va_prob = torch.sigmoid(model(Xva)).cpu().numpy()
        # AUC needs both classes present in val
        auc = roc_auc_score(yva, va_prob) if len(np.unique(yva)) > 1 else 0.5
        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= head_cfg["patience"]:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        te_prob = torch.sigmoid(model(Xte)).cpu().numpy()
    te_pred = (te_prob > 0.5).astype(int)

    return {
        "auc": roc_auc_score(yte, te_prob) if len(np.unique(yte)) > 1 else float("nan"),
        "f1_macro": f1_score(yte, te_pred, average="macro"),
        "acc": accuracy_score(yte, te_pred),
    }


def run_multiseed(Z_tr, y_tr, Z_va, y_va, Z_te, y_te, head_cfg, device=None) -> dict:
    """Train the fixed head once per seed; return mean +/- std of test metrics."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    aucs, f1s, accs = [], [], []
    for seed in head_cfg["seeds"]:
        m = train_eval_once(Z_tr, y_tr, Z_va, y_va, Z_te, y_te, head_cfg, seed, device)
        aucs.append(m["auc"]); f1s.append(m["f1_macro"]); accs.append(m["acc"])
        print(f"    seed={seed}: AUC={m['auc']:.4f} F1m={m['f1_macro']:.4f} ACC={m['acc']:.4f}")
    return {
        "auc_mean": float(np.mean(aucs)), "auc_std": float(np.std(aucs)),
        "f1_mean": float(np.mean(f1s)), "f1_std": float(np.std(f1s)),
        "acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs)),
        "seeds": list(head_cfg["seeds"]),
    }
