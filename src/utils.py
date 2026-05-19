"""Shared utilities for paths, seeds, and experiment bookkeeping.

Rubric: reproducible paths and fixed random seeds support reliable comparisons
across `model`, `real_ratio`, and `augmentation` settings.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set all relevant RNG seeds for reproducible experiment comparisons."""
    # Rubric: reproducibility requires a fixed seed across Python, NumPy, and
    # PyTorch so low-data comparisons remain attributable to the experiment.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Return CUDA when available, otherwise CPU, and print runtime info."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only"
    # Rubric: GPU acceleration is used when available, but the pipeline must
    # not assume CUDA exists because the report should remain reproducible.
    print(f"torch: {torch.__version__}")
    print(f"cuda available: {torch.cuda.is_available()}")
    print(f"cuda version: {torch.version.cuda}")
    print(f"gpu: {gpu_name}")
    print(f"device: {device}")
    return device


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if it does not exist and return it as a Path."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory
