"""
prep/build_tab_years.py
=======================
Generate model-ready Flight_tab_YYYY.csv (+ data_info_YYYY.yaml) for every year
listed in config.yaml, by reusing the existing raw->tabular transform in
Datasets/Flight_tab.py (process_file / save_year_data).

Only 2020 and 2024 ship pre-processed; this fills in the rest so the bake-off
sample can span all available years. Years whose Tab CSV already exists are
skipped (pass --force to regenerate).

Run:
    uv run python exp/Tab_exp/representation_bakeoff/prep/build_tab_years.py
    uv run python exp/Tab_exp/representation_bakeoff/prep/build_tab_years.py --years 2016 2017
"""
import argparse
import gc
import importlib.util
import os
import sys
from pathlib import Path

import yaml

# ── Resolve paths relative to this script (repo convention) ────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
EXP_DIR = SCRIPT_DIR.parent
REPO_ROOT = EXP_DIR.parents[2]  # exp/Tab_exp/representation_bakeoff -> repo root

with open(EXP_DIR / "config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

RAW_DIR = Path(cfg["paths"]["raw_dir"])
TAB_DIR = Path(cfg["paths"]["tab_dir"])


# ── Import process_file / save_year_data from Datasets/Flight_tab.py ───────────
def _load_flight_tab():
    mod_path = REPO_ROOT / "Datasets" / "Flight_tab.py"
    spec = importlib.util.spec_from_file_location("flight_tab", mod_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="*", default=cfg["data"]["years"])
    ap.add_argument("--force", action="store_true", help="regenerate even if Tab CSV exists")
    args = ap.parse_args()

    flight_tab = _load_flight_tab()
    TAB_DIR.mkdir(parents=True, exist_ok=True)

    for year in args.years:
        out_csv = TAB_DIR / f"Flight_tab_{year}.csv"
        if out_csv.exists() and not args.force:
            print(f"[{year}] skip — {out_csv.name} already exists")
            continue

        raw = RAW_DIR / f"flight_with_weather_{year}.csv"
        if not raw.is_file():
            print(f"[{year}] MISSING raw file {raw} — skipping")
            continue

        print(f"[{year}] processing {raw.name} ...", flush=True)
        df = flight_tab.process_file(str(raw))
        if df is None:
            print(f"[{year}] process_file returned None — skipped")
            continue

        flight_tab.save_year_data(df, year, str(TAB_DIR))
        print(f"[{year}] done — {len(df):,} rows -> {out_csv.name}", flush=True)
        del df
        gc.collect()

    print("build_tab_years complete.")


if __name__ == "__main__":
    main()
