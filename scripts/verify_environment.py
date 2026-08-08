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
    from src.utils.gpu_manager import device_info

    load_app_config()
    info = device_info("auto")
    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA runtime: {torch.version.cuda}")
    print(f"CUDA available: {cuda_available}")
    print(f"MPS available: {mps_available}")
    print(f"Selected device: {info['selected']}")
    print(f"GPU: {info.get('gpu_name') or 'None'}")
    print("Environment verification: PASS")


if __name__ == "__main__":
    main()
