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
        info["memory"] = gpu_memory_status(selected, torch_module=torch)
    elif selected == "mps":
        info["gpu_name"] = "Apple Silicon GPU (Metal/MPS)"
    return info


def gpu_memory_status(
    device: str, torch_module=None, pressure_threshold: float = 0.85
) -> Dict[str, object]:
    """Return CUDA memory pressure without allocating accelerator memory."""
    if not str(device).startswith("cuda"):
        return {"available": False, "device": device, "pressure": "unavailable"}
    if torch_module is None:
        import torch as torch_module

    index = int(device.split(":", 1)[1]) if ":" in device else 0
    total = int(torch_module.cuda.get_device_properties(index).total_memory)
    allocated = int(torch_module.cuda.memory_allocated(index))
    reserved = int(torch_module.cuda.memory_reserved(index))
    utilization = reserved / total if total else 0.0
    gib = 1024 ** 3
    return {
        "available": True,
        "device": device,
        "allocated_gb": round(allocated / gib, 2),
        "reserved_gb": round(reserved / gib, 2),
        "total_gb": round(total / gib, 2),
        "utilization": round(utilization, 3),
        "pressure": "high" if utilization >= pressure_threshold else "normal",
    }


def release_accelerator_cache(device: str, torch_module=None) -> bool:
    """Release unused framework cache for the selected accelerator."""
    if torch_module is None:
        import torch as torch_module

    if str(device).startswith("cuda"):
        torch_module.cuda.empty_cache()
        return True
    if device == "mps" and hasattr(torch_module, "mps"):
        empty_cache = getattr(torch_module.mps, "empty_cache", None)
        if callable(empty_cache):
            empty_cache()
            return True
    return False
