"""Select a safe PyTorch inference device across mixed team hardware."""

from __future__ import annotations

from typing import Dict


def resolve_torch_device(requested: str | None = "auto") -> str:
    """Prefer NVIDIA CUDA, then Apple MPS, then portable CPU inference."""
    normalized = str(requested or "auto").strip().lower()
    import torch

    if normalized == "auto":
        if torch.cuda.is_available():
            return "cuda:0"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if normalized.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    if normalized == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        return "cpu"
    return normalized


def device_info(requested: str | None = "auto") -> Dict[str, object]:
    """Return dashboard-safe information about the selected device."""
    import torch

    selected = resolve_torch_device(requested)
    info: Dict[str, object] = {
        "requested": requested or "auto",
        "selected": selected,
        "cuda_available": torch.cuda.is_available(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": None,
        "gpu_memory_gb": None,
    }
    if selected.startswith("cuda"):
        index = int(selected.split(":", 1)[1]) if ":" in selected else 0
        properties = torch.cuda.get_device_properties(index)
        info["gpu_name"] = properties.name
        info["gpu_memory_gb"] = round(properties.total_memory / (1024 ** 3), 1)
    elif selected == "mps":
        info["gpu_name"] = "Apple Silicon GPU (Metal/MPS)"
    return info
