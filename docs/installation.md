# Installation Guide

This guide covers every supported method for installing the Lightweight Gravitational Transformer (LGT) and verifying the installation.

---

## Table of Contents

- [System Requirements](#system-requirements)
- [Installation Methods](#installation-methods)
  - [From Source (Recommended)](#from-source-recommended)
  - [Editable Install (Development)](#editable-install-development)
  - [Using pip (Published Package)](#using-pip-published-package)
- [GPU Support](#gpu-support)
- [Verifying the Installation](#verifying-the-installation)
- [Troubleshooting](#troubleshooting)

---

## System Requirements

| Component | Minimum | Recommended |
|---|---|---|
| Python | 3.9 | 3.11+ |
| PyTorch | 2.0.0 | 2.2+ |
| NumPy | 1.24.0 | 1.26+ |
| SciPy | 1.10.0 | 1.12+ |
| RAM | 2 GB | 8 GB+ |
| Disk | 200 MB | 1 GB (for exported models) |
| GPU | Optional | CUDA 11.8+ / ROCm 5.6+ |

---

## Installation Methods

### From Source (Recommended)

Installing from source gives you the latest version and allows you to inspect and modify the code.

```bash
# 1. Clone the repository
git clone https://github.com/MASSIVEMAGNETICS/Lightweight-Gravitational-Transformer.git
cd Lightweight-Gravitational-Transformer

# 2. Create a virtual environment
python -m venv .venv

# 3. Activate the environment
#    Linux / macOS:
source .venv/bin/activate
#    Windows (Command Prompt):
# .venv\Scripts\activate.bat
#    Windows (PowerShell):
# .venv\Scripts\Activate.ps1

# 4. Upgrade pip
pip install --upgrade pip

# 5. Install dependencies
pip install -r requirements_lgt.txt
```

### Editable Install (Development)

If you plan to modify the source code or run tests:

```bash
# After cloning (step 1-4 above), install in editable mode with dev extras
pip install -e ".[dev]"
```

This installs `pytest` and `pytest-cov` alongside the package.

### Using pip (Published Package)

Once the package is published to PyPI:

```bash
pip install lightweight-gravitational-transformer
```

To install with development tools:

```bash
pip install "lightweight-gravitational-transformer[dev]"
```

---

## GPU Support

LGT works on both CPU and GPU. To use a CUDA-enabled GPU:

```bash
# Install PyTorch with CUDA 12.1 support (adjust for your CUDA version)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Then install LGT dependencies
pip install -r requirements_lgt.txt
```

Check available CUDA:

```python
import torch
print(torch.cuda.is_available())      # True if GPU is accessible
print(torch.cuda.get_device_name(0))  # GPU name
```

To run LGT on GPU, pass the device when creating tensors or move the model:

```python
import torch
from lightweight_gravitational_transformer import LightweightGravitationalTransformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LightweightGravitationalTransformer(dim_model=128).to(device)

x = torch.randn(2, 32, 128, device=device)
output, _ = model(x)
```

---

## Verifying the Installation

Run the following verification script to confirm everything is installed correctly:

```python
# verify_install.py
import sys
print(f"Python: {sys.version}")

import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

import numpy as np
print(f"NumPy: {np.__version__}")

import scipy
print(f"SciPy: {scipy.__version__}")

# Core LGT imports
from gravitational_attention import MultiHeadGravitationalAttention
from fractal_position_embedding import FractalPositionEmbedding
from lightweight_gravitational_transformer import LightweightGravitationalTransformer
from victorcos_module import Ledger, MirrorLayer, LGTVictorOSModule
from training import TrainingLoop, ContainmentProtocol
from tri_model import TriModelTransformer
from export_edge_model import build_model

# Smoke test
model = LightweightGravitationalTransformer(vocab_size=1000, dim_model=64, num_layers=2)
x = torch.randint(0, 1000, (1, 8))
out, diag = model(x, return_diagnostics=True)
assert out.shape == (1, 8, 64), f"Unexpected shape: {out.shape}"
print(f"\nLGT smoke test passed ✓  output shape: {out.shape}")
```

Run with:

```bash
python verify_install.py
```

Or run the full test suite:

```bash
pytest tests/ -v
```

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'torch'`

PyTorch is not installed. Install it for your platform from [pytorch.org](https://pytorch.org/get-started/locally/).

### `ModuleNotFoundError: No module named 'gravitational_attention'`

You are not running Python from the repository root directory, or you haven't installed the package. Ensure you are in the `Lightweight-Gravitational-Transformer/` directory, or install the package:

```bash
cd Lightweight-Gravitational-Transformer
pip install -e .
```

### `RuntimeError: CUDA error: no kernel image is available for execution on the device`

Your PyTorch build does not match your CUDA version. Reinstall PyTorch with the correct CUDA version from [pytorch.org](https://pytorch.org/get-started/locally/).

### Import errors after editing source files

When running scripts directly (not as a package), Python must be able to find the LGT modules. Either run scripts from the repository root, or add the root to `PYTHONPATH`:

```bash
export PYTHONPATH=/path/to/Lightweight-Gravitational-Transformer:$PYTHONPATH
```

### Tests fail with `AttributeError` on a fresh clone

Ensure you have installed all dependencies:

```bash
pip install -r requirements_lgt.txt
pip install pytest
pytest tests/ -v
```
