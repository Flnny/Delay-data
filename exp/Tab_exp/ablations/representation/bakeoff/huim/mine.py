"""
huim/mine.py
============
Mine High-Utility Itemsets from the TRAIN split with a SYNTHESISED delay-cost
utility (there is no natural value column in flight data).

- Item  = (feature, quantile-bin) for continuous, (feature, value) for categorical.
- Utility (drivers) : each item in a DELAYED flight carries the flight's
  |ARR_DELAY| (delay cost). High-utility itemsets = feature combos co-occurring
  in the most-delayed flights.
- Utility (protectors): each item in an ON-TIME flight carries its on-time margin
  (threshold - |ARR_DELAY|). High-utility itemsets = combos in the earliest flights.
- Mined per class with PAMI EFIM (minUtil = Nth percentile, fallback 75th).

All fitting (bins, item vocabulary, utilities) uses TRAIN rows only.

Run (after build_sample):
    uv run python exp/Tab_exp/ablations/representation/bakeoff/huim/mine.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer

EXP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EXP_DIR))
import exp.Tab_exp.ablations.representation.bakeoff.common as common  # noqa: E402

BIN_LABELS = {0: "VeryLow", 1: "Low", 2: "Mid", 3: "High", 4: "VeryHigh"}


def build_item_maps(train, cat, cont, n_bins):
    item_id, item_label, nid = {}, {}, 1
    for col in cat:
        for val in sorted(train[col].astype(str).unique()):
            item_id[(col, val)] = nid
            item_label[nid] = f"{col}={val}"
            nid += 1
    for col in cont:
        for b in range(n_bins):
            item_id[(col, b)] = nid
            item_label[nid] = f"{col}={BIN_LABELS[b]}"
            nid += 1
    return item_id, item_label


def write_transactions(rows, cont_bins, weights, cat, cont, item_id, path):
    """One PAMI line per row: items:totalUtility:u1<tab>u2..., all items share the
    row's synthesised weight."""
    lines = []
    cat_arr = {c: rows[c].astype(str).to_numpy() for c in cat}
    for r in range(len(rows)):
        items = []
        for c in cat:
            key = (c, cat_arr[c][r])
            if key in item_id:
                items.append(item_id[key])
        for j, c in enumerate(cont):
            key = (c, int(cont_bins[r, j]))
            if key in item_id:
                items.append(item_id[key])
        if not items:
            continue
        u = int(max(weights[r], 1))
        utils = [u] * len(items)
        items_s = "\t".join(map(str, items))
        utils_s = "\t".join(map(str, utils))
        lines.append(f"{items_s}:{sum(utils)}:{utils_s}")
    Path(path).write_text("\n".join(lines))
    return len(lines)


def mine_file(path, label, item_label, pctile, max_size, save_top):
    from PAMI.highUtilityPattern.basic import EFIM

    trans_utils = []
    for line in Path(path).read_text().splitlines():
        parts = line.split(":")
        if len(parts) >= 2:
            trans_utils.append(int(parts[1]))
    if not trans_utils:
        print(f"  [{label}] no transactions"); return pd.DataFrame()

    min_util = int(np.percentile(trans_utils, pctile))
    print(f"  [{label}] util range [{min(trans_utils)},{max(trans_utils)}] | "
          f"minUtil(p{pctile})={min_util}")
    obj = EFIM.EFIM(str(path), min_util, "\t"); obj.mine()
    patterns = obj.getPatterns()
    if not patterns:
        min_util = int(np.percentile(trans_utils, 75))
        print(f"  [{label}] none at p{pctile}, retry p75 minUtil={min_util}")
        obj = EFIM.EFIM(str(path), min_util, "\t"); obj.mine()
        patterns = obj.getPatterns()
    if not patterns:
        print(f"  [{label}] no patterns"); return pd.DataFrame()

    rows = []
    for pat_str, util in patterns.items():
        ids = [int(x) for x in str(pat_str).split("\t") if x.strip().isdigit()]
        if not ids or len(ids) > max_size:        # keep only short, frequent patterns
            continue
        rows.append({
            "Pattern": " + ".join(item_label.get(i, f"?{i}") for i in ids),
            "ItemIDs": ids, "Utility": util, "Size": len(ids),
        })
    if not rows:
        print(f"  [{label}] no patterns with size <= {max_size}"); return pd.DataFrame()
    out = (pd.DataFrame(rows).sort_values("Utility", ascending=False)
           .head(save_top).reset_index(drop=True))
    print(f"  [{label}] kept {len(out)} (size<= {max_size}, top {save_top}) | "
          f"top: {out.iloc[0]['Pattern']}")
    return out


def main():
    cfg = common.load_config()
    df = common.load_sample(cfg)
    cat, cont = common.feature_lists(cfg)
    tr_m, _, _ = common.split_masks(df)
    h = cfg["huim"]; n_bins = h["n_bins"]; thr = float(cfg["data"]["delay_threshold"])
    pctile = h["min_util_percentile"]
    max_size = h.get("max_pattern_size", 3)
    save_top = h.get("save_top", 200)
    max_txn = h.get("max_txn", 0)
    item_exclude = set(h.get("item_exclude_cols", []))
    cat_h = [c for c in cat if c not in item_exclude]   # drop noisy high-card items (flight number)
    art = (EXP_DIR / cfg["paths"]["work_dir"]).resolve(); art.mkdir(parents=True, exist_ok=True)

    train = df[tr_m].reset_index(drop=True)
    disc = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="quantile")
    train_bins = disc.fit_transform(train[cont]).astype(int)

    item_id, item_label = build_item_maps(train, cat_h, cont, n_bins)
    print(f"Vocabulary: {len(item_id)} items ({len(cat_h)} cat + {len(cont)} cont) | "
          f"train rows: {len(train)}", flush=True)

    arr = train["ARR_DELAY"].abs().to_numpy()
    lbl = train["label"].to_numpy()
    w_drivers = np.maximum(arr.astype(int), 1)                 # delay cost
    w_protect = np.maximum((thr - arr).astype(int), 1)         # on-time margin

    rng = np.random.default_rng(int(cfg["data"]["seed"]))
    def pick(mask):
        idx = np.where(mask)[0]
        if max_txn and len(idx) > max_txn:
            idx = rng.choice(idx, size=max_txn, replace=False)
        return idx
    d_idx, p_idx = pick(lbl == 1), pick(lbl == 0)

    pos_path = art / "huim_txn_drivers.txt"
    neg_path = art / "huim_txn_protectors.txt"
    n_d = write_transactions(train.iloc[d_idx].reset_index(drop=True), train_bins[d_idx],
                             w_drivers[d_idx], cat_h, cont, item_id, pos_path)
    n_p = write_transactions(train.iloc[p_idx].reset_index(drop=True), train_bins[p_idx],
                             w_protect[p_idx], cat_h, cont, item_id, neg_path)
    print(f"Transactions: drivers={n_d}, protectors={n_p}", flush=True)

    drivers = mine_file(pos_path, "Delay Drivers", item_label, pctile, max_size, save_top)
    protectors = mine_file(neg_path, "Delay Protectors", item_label, pctile, max_size, save_top)

    if not drivers.empty:
        drivers.to_csv(art / "huim_drivers.csv", index=False)
    if not protectors.empty:
        protectors.to_csv(art / "huim_protectors.csv", index=False)
    pd.DataFrame([{"ItemID": k, "Label": v} for k, v in item_label.items()]) \
        .to_csv(art / "huim_item_labels.csv", index=False)
    print(f"Saved drivers/protectors to {art}")


if __name__ == "__main__":
    main()
