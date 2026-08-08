"""Inventory security application package."""

import os
import sys


# Some PyTorch operations are not implemented by Metal Performance Shaders.
# Allow those individual operations to fall back to CPU on Apple Silicon.
if sys.platform == "darwin":
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
