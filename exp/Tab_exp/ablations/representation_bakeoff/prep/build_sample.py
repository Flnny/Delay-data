"""
prep/build_sample.py
====================
Draw the single fixed sample used by every experiment in the bake-off.

- Pools all years in config, allocates `sample_total` rows proportional to each
  year's size, and draws a per-year stratified subsample by binary label.
- Assigns the temporal split by FL_DAY (train <=9 / val 10-12 / test >12) — the
  repo convention, no shuffling across days.
- Persists artifacts/sample.parquet with every Tab column plus `label`, `split`,
  and raw `ARR_DELAY` (kept for the HUIM delay-cost utility downstream).

Run (after build_tab_years):
    uv run python exp/Tab_exp/representation_bakeoff/prep/build_sample.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

SCRIPT_DIR = Path(__file__).resolve().parent
EXP_DIR = SCRIPT_DIR.parent

with open(EXP_DIR / "config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

TAB_DIR = Path(cfg["paths"]["tab_dir"])
WORK_DIR = (EXP_DIR / cfg["paths"]["work_dir"]).resolve()
WORK_DIR.mkdir(parents=True, exist_ok=True)

YEARS = cfg["data"]["years"]
SAMPLE_TOTAL = int(cfg["data"]["sample_total"])
SEED = int(cfg["data"]["seed"])
THR = float(cfg["data"]["delay_threshold"])
TMAX = int(cfg["data"]["train_days_max"])
VMAX = int(cfg["data"]["val_days_max"])


def year_rowcount(year: int) -> int:
    """Read row count from data_info_YYYY.yaml (shape=[rows, cols])."""
    info = TAB_DIR / f"data_info_{year}.yaml"
    with open(info, "r") as f:
        return int(yaml.safe_load(f)["data_summary"]["shape"][0])


def allocate_quota(counts: dict[int, int], total: int) -> dict[int, int]:
    """Largest-remainder allocation of `total` proportional to per-year counts."""
    grand = sum(counts.values())
    exact = {y: total * c / grand for y, c in counts.items()}
    floor = {y: int(np.floor(v)) for y, v in exact.items()}
    remainder = total - sum(floor.values())
    # hand out the leftover to the largest fractional parts
    frac_order = sorted(exact, key=lambda y: exact[y] - floor[y], reverse=True)
    for y in frac_order[:remainder]:
        floor[y] += 1
    return floor


def split_of(fl_day: pd.Series) -> pd.Series:
    return np.select(
        [fl_day <= TMAX, fl_day <= VMAX], ["train", "val"], default="test"
    )


def main():
    counts = {y: year_rowcount(y) for y in YEARS}
    quota = allocate_quota(counts, SAMPLE_TOTAL)
    print("Per-year quota (proportional to size):")
    for y in YEARS:
        print(f"  {y}: {quota[y]:>6} of {counts[y]:>10,} rows")

    parts = []
    for y in YEARS:
        n = quota[y]
        if n <= 0:
            continue
        df = pd.read_csv(TAB_DIR / f"Flight_tab_{y}.csv")
        df = df.dropna()
        df["label"] = (df["ARR_DELAY"].abs() > THR).astype(int)
        df["FL_YEAR"] = y  # authoritative year from filename

        # Stratified subsample by label, preserving this year's delayed fraction.
        n = min(n, len(df))
        sub, _ = train_test_split(
            df, train_size=n, stratify=df["label"], random_state=SEED
        )
        parts.append(sub)
        print(f"  [{y}] sampled {len(sub)} (delayed frac={sub['label'].mean():.3f})")
        del df

    sample = pd.concat(parts, ignore_index=True)
    sample["split"] = split_of(sample["FL_DAY"])

    out = WORK_DIR / "sample.parquet"
    sample.to_parquet(out, index=False)

    print("\n=== SAMPLE SUMMARY ===")
    print(f"total rows      : {len(sample)}")
    print(f"delayed frac    : {sample['label'].mean():.3f}")
    print(f"split sizes     : {sample['split'].value_counts().to_dict()}")
    print(f"years present   : {sorted(sample['FL_YEAR'].unique())}")
    print(f"written         : {out}")


if __name__ == "__main__":
    main()
