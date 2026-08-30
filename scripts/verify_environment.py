"""Verify that a team member's local environment can run the application."""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


REQUIRED_IMPORTS = (
    "cv2",
    "yaml",
    "ultralytics",
    "deepface",
    "streamlit",
    "transformers",
    "einops",
    "pygame",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--expected-device", choices=("cuda", "mps", "cpu"), default=None
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if sys.version_info[:2] != (3, 11):
        raise SystemExit(f"Python 3.11 is required; found {sys.version.split()[0]}")

    failures = []
    for module_name in REQUIRED_IMPORTS:
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            failures.append(f"{module_name}: {exc}")
    if failures:
        raise SystemExit("Import verification failed:\n- " + "\n- ".join(failures))

    import torch

    cuda_available = torch.cuda.is_available()
    if args.expected_device == "cuda" and not cuda_available:
        raise SystemExit("CUDA profile was installed, but torch.cuda.is_available() is False")
    if args.expected_device == "cpu" and "+cpu" not in torch.__version__:
        raise SystemExit(f"CPU profile expected, but installed PyTorch is {torch.__version__}")
    mps_available = bool(
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    )
    if args.expected_device == "mps" and not mps_available:
        raise SystemExit(
            "Apple MPS profile expected, but torch.backends.mps.is_available() is False"
        )

    from src.utils.config_loader import load_app_config
    from src.utils.gpu_manager import device_info, resolve_torch_device

    config = load_app_config()
    yolo_requested = config["models"]["yolo"].get("device", "auto")
    vlm_requested = config["models"]["vlm"].get("device", "auto")
    yolo_selected = resolve_torch_device(yolo_requested)
    vlm_selected = resolve_torch_device(vlm_requested)
    info = device_info(yolo_requested)
    if args.expected_device == "cuda" and not yolo_selected.startswith("cuda"):
        raise SystemExit(f"YOLO is configured for {yolo_selected}, not CUDA")
    if args.expected_device == "mps" and yolo_selected != "mps":
        raise SystemExit(f"YOLO is configured for {yolo_selected}, not MPS")
    if args.expected_device == "cpu" and yolo_selected != "cpu":
        raise SystemExit(f"YOLO is configured for {yolo_selected}, not CPU")

    accelerator_smoke = "not required"
    if yolo_selected.startswith("cuda"):
        try:
            value = torch.arange(6, device=yolo_selected).sum().item()
            torch.cuda.synchronize()
            accelerator_smoke = f"PASS (tensor sum={int(value)})"
        except Exception as exc:
            raise SystemExit(f"CUDA tensor smoke test failed: {exc}") from exc
    elif yolo_selected == "mps":
        try:
            value = torch.arange(6, device="mps").sum().item()
            accelerator_smoke = f"PASS (tensor sum={int(value)})"
        except Exception as exc:
            raise SystemExit(f"MPS tensor smoke test failed: {exc}") from exc
    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA runtime: {torch.version.cuda}")
    print(f"CUDA available: {cuda_available}")
    print(f"MPS available: {mps_available}")
    print(f"YOLO requested/selected: {yolo_requested}/{yolo_selected}")
    print(f"Florence requested/selected: {vlm_requested}/{vlm_selected}")
    print(f"GPU: {info.get('gpu_name') or 'None'}")
    print(f"Accelerator smoke test: {accelerator_smoke}")
    print("Environment verification: PASS")


if __name__ == "__main__":
    main()
