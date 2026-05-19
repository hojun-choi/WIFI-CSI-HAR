# Wi-Fi CSI-based HAR on UT-HAR

Project-specific experimental framework for Wi-Fi CSI-based Human Activity Recognition on `UT-HAR`, focused on real training data ratio, data augmentation, and sequence model comparison.

Tested with Python `3.10.11`.

## Project Motivation

Wi-Fi CSI-based HAR often struggles with `generalization` because CSI signals are sensitive to environment changes, subject differences, device placement, room layout, and noise. In practice, collecting enough labeled CSI data for every new setup is expensive, so a useful system should remain reasonably robust even when real labeled training data is limited.

This project studies that low-data robustness problem on `UT-HAR` by intentionally reducing the real training data ratio and measuring performance degradation, then testing whether `train-only augmentation` can recover some of that loss. The original benchmark code is preserved under `original_baseline/`, while this repository builds a clean, report-friendly experimental pipeline focused on interpretable comparison.

## Preprocessing Strategy

The project will include a small preprocessing ablation because preprocessing can affect robustness under limited data. The main experiments will use one selected preprocessing policy consistently, and any normalization statistics will be computed only from the train split to avoid leakage.

## Environment Setup

Use Windows PowerShell and Python `3.10.11`.

Create and activate a virtual environment:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
```

Install CUDA-enabled PyTorch for an NVIDIA GPU.

Note: Choose the CUDA-enabled PyTorch command from the official PyTorch Get Started page if this command changes.

Default command:

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Fallback if `cu128` fails:

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

Then install the remaining project dependencies:

```powershell
pip install -r requirements.txt
```

Verify GPU availability:

```powershell
python -c "import torch; print('torch:', torch.__version__); print('cuda available:', torch.cuda.is_available()); print('cuda version:', torch.version.cuda); print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

Do not run long training until the dry-run command succeeds.

The expected laptop GPU is `NVIDIA RTX 4050 Laptop GPU`, but the code must still fall back to CPU if CUDA is not available.

## Dataset Placement

Place the UT-HAR files at the following paths:

```text
data/UT_HAR/data/X_train.csv
data/UT_HAR/data/X_val.csv
data/UT_HAR/data/X_test.csv
data/UT_HAR/label/y_train.csv
data/UT_HAR/label/y_val.csv
data/UT_HAR/label/y_test.csv
```

These files keep the `.csv` extension from the benchmark, but they are NumPy binary files and must be loaded with `np.load`.

## Quick Start

```bash
python experiments/01_check_data.py
```

Install dependencies first if needed:

```bash
pip install -r requirements.txt
```

## Repository Layout

- `original_baseline/`: copied benchmark code preserved for direct comparison with the original implementation.
- `src/` and `experiments/`: clean project-specific code for the final course experiments and report-ready analysis.
