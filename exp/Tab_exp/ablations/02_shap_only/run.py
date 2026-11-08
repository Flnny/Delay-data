"""
02_shap_only/run.py — thin wrapper around the shared ablation runner.
Feature set is defined in ./config.yaml; training defaults in ../common/base_config.yaml.

Run:
    uv run python exp/Tab_exp/ablations/02_shap_only/run.py --seed 42 [--target abs|late|3class]
                 [--models MLP,AutoInt,ResNet] [--reuse-params] [--set training.n_iter=2]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.runner import main  # noqa: E402

if __name__ == "__main__":
    main(Path(__file__).resolve().parent)
