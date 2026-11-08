"""Single place where every RNG is seeded."""
import os
import random

# Must be set before the first cuBLAS call for deterministic GEMMs.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from lightning.pytorch import seed_everything  # noqa: E402


def set_seed(seed: int) -> None:
    """Seed python, numpy, torch (CPU+CUDA) and Lightning (incl. dataloader workers).

    sklearn objects do not read global state: pass `random_state=` explicitly
    (the runner derives every random_state from the same seed values).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    seed_everything(seed, workers=True, verbose=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
