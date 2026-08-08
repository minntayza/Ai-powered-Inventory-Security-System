#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "This installer is only for macOS." >&2
  exit 1
fi

if [[ "$(uname -m)" != "arm64" ]]; then
  echo "This profile requires an Apple Silicon Mac (M1/M2/M3/M4)." >&2
  exit 1
fi

MACOS_MAJOR="$(sw_vers -productVersion | cut -d. -f1)"
if (( MACOS_MAJOR < 14 )); then
  echo "macOS 14 or newer is required for the tested MPS configuration." >&2
  exit 1
fi

if command -v python3.11 >/dev/null 2>&1; then
  PYTHON_COMMAND="$(command -v python3.11)"
elif command -v python3 >/dev/null 2>&1 && \
  [[ "$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')" == "3.11" ]]; then
  PYTHON_COMMAND="$(command -v python3)"
else
  echo "Python 3.11 is required. Install it with: brew install python@3.11" >&2
  exit 1
fi

if [[ ! -x ".venv/bin/python" ]]; then
  "$PYTHON_COMMAND" -m venv .venv
fi

VENV_PYTHON=".venv/bin/python"
VENV_VERSION="$($VENV_PYTHON -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [[ "$VENV_VERSION" != "3.11" ]]; then
  echo "The existing .venv uses Python $VENV_VERSION. Remove it and rerun." >&2
  exit 1
fi

"$VENV_PYTHON" -m pip install --upgrade pip
"$VENV_PYTHON" -m pip install --upgrade --force-reinstall -r requirements-macos.txt
"$VENV_PYTHON" -m pip install -r requirements-base.txt
"$VENV_PYTHON" scripts/verify_environment.py --expected-device mps

echo
echo "Setup complete. Start the dashboard with:"
echo ".venv/bin/python -m streamlit run src/module_c_ui_dashboard/app.py"

