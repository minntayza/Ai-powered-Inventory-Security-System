# NVIDIA/CUDA Setup Guide

This guide prepares a Windows 11 laptop with an NVIDIA GPU for the exhibition build. The project also runs on CPU when CUDA is unavailable.

## 1. Check the NVIDIA driver

Open PowerShell and run:

```powershell
nvidia-smi
```

Continue when the command displays the GPU and driver. If it is missing or reports a driver error, install the current driver for the exact laptop GPU from NVIDIA or the laptop manufacturer, restart Windows, and run the command again.

The project does not require a separate system-wide CUDA Toolkit. Its setup script installs the matching CUDA-enabled PyTorch wheel inside the virtual environment.

## 2. Install the project environment

Install 64-bit Python 3.11, open PowerShell in the repository root, and run:

```powershell
.\scripts\setup_windows.ps1 -Device CUDA
```

If local scripts are blocked for the current PowerShell session:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\setup_windows.ps1 -Device CUDA
```

Do not copy `.venv` from another computer. CPU and CUDA packages are machine-specific.

## 3. Verify acceleration

```powershell
.\.venv\Scripts\Activate.ps1
python scripts\verify_environment.py
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU fallback')"
```

The second line should print `True` and the NVIDIA GPU name. The dashboard also reports CUDA reserved/total memory and warns when pressure is high.

## 4. Start the dashboard

```powershell
streamlit run src/module_c_ui_dashboard/app.py
```

YOLO starts with `yolov8n.pt`. Florence-2 and the local Whisper voice model load only when their features are first used, so the first request can take longer and may download weights.

## Troubleshooting

- **`torch.cuda.is_available()` is `False`:** restart after the driver installation, confirm `nvidia-smi`, then rerun the setup script with `-Device CUDA`.
- **Out-of-memory error:** stop other GPU applications, reduce camera resolution/FPS in `configs/camera_config.yaml`, and avoid sending visual questions during a busy incident.
- **CUDA model initialization fails:** monitoring can fall back to CPU. Expect lower throughput and keep the default 960×540 at 10 FPS.
- **Thermal throttling:** use AC power, enable the laptop performance profile, keep vents clear, and rehearse for the same duration as the exhibition demo.

Before exhibition day, run a 20-minute dry run and record GPU availability, peak memory shown in the dashboard, temperature reported by `nvidia-smi`, and any fallback messages in `docs/QA_VALIDATION_RESULTS.md`.
