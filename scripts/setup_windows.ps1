[CmdletBinding()]
param(
    [ValidateSet("Auto", "CUDA", "CPU")]
    [string]$Device = "Auto"
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$venvDirectory = Join-Path $projectRoot ".venv"
$venvPython = Join-Path $venvDirectory "Scripts\python.exe"

Push-Location $projectRoot
try {
    if (-not (Test-Path -LiteralPath $venvPython)) {
        $pythonLauncher = Get-Command py -ErrorAction SilentlyContinue
        if ($pythonLauncher) {
            & py -3.11 -m venv $venvDirectory
        }
        else {
            $pythonCommand = Get-Command python -ErrorAction Stop
            $version = & $pythonCommand.Source -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
            if ($version -ne "3.11") {
                throw "Python 3.11 is required. Installed Python is $version."
            }
            & $pythonCommand.Source -m venv $venvDirectory
        }
        if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $venvPython)) {
            throw "Could not create the Python 3.11 virtual environment."
        }
    }

    $venvVersion = & $venvPython -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
    if ($venvVersion -ne "3.11") {
        throw "The existing .venv uses Python $venvVersion. Delete it and rerun with Python 3.11."
    }

    if ($Device -eq "Auto") {
        $nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
        if ($nvidiaSmi) {
            & $nvidiaSmi.Source --query-gpu=name --format=csv,noheader 2>$null | Out-Null
            $Device = if ($LASTEXITCODE -eq 0) { "CUDA" } else { "CPU" }
        }
        else {
            $Device = "CPU"
        }
    }

    Write-Host "Selected installation profile: $Device"
    & $venvPython -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed." }

    $torchRequirements = if ($Device -eq "CUDA") {
        "requirements-cuda.txt"
    }
    else {
        "requirements-cpu.txt"
    }
    & $venvPython -m pip install --upgrade --force-reinstall -r $torchRequirements
    if ($LASTEXITCODE -ne 0) { throw "PyTorch installation failed." }

    & $venvPython -m pip install -r "requirements-base.txt"
    if ($LASTEXITCODE -ne 0) { throw "Application dependency installation failed." }

    & $venvPython "scripts\verify_environment.py" --expected-device $Device.ToLowerInvariant()
    if ($LASTEXITCODE -ne 0) { throw "Environment verification failed." }

    Write-Host ""
    Write-Host "Setup complete. Start the dashboard with:"
    Write-Host ".\.venv\Scripts\python.exe -m streamlit run src/module_c_ui_dashboard/app.py"
}
finally {
    Pop-Location
}

