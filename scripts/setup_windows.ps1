# Windows Setup Script for AI Fine-tuning Project
# Creates virtual environment and installs all dependencies from pyproject.toml

$ErrorActionPreference = "Stop"

function Write-Success { param($Message); Write-Host "✓ $Message" -ForegroundColor Green }
function Write-Info { param($Message); Write-Host "ℹ $Message" -ForegroundColor Cyan }
function Write-Warning { param($Message); Write-Host "⚠ $Message" -ForegroundColor Yellow }
function Write-Error-Custom { param($Message); Write-Host "✗ $Message" -ForegroundColor Red }
function Write-Header { 
    param($Message)
    Write-Host "`n$('=' * 80)" -ForegroundColor Cyan
    Write-Host "  $Message" -ForegroundColor Cyan
    Write-Host "$('=' * 80)" -ForegroundColor Cyan
}

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir

Write-Header "AI Fine-tuning Project - Windows Setup"

Set-Location $ProjectRoot
Write-Info "Project root: $ProjectRoot"

# Step 1: Check pyproject.toml
Write-Header "Step 1: Checking Project Configuration"
if (-not (Test-Path "pyproject.toml")) {
    Write-Error-Custom "pyproject.toml not found"
    exit 1
}
Write-Success "Found pyproject.toml"

# Step 2: Check Python
Write-Header "Step 2: Checking Python Installation"
try {
    $PythonVersion = python --version 2>&1
    Write-Info "Found: $PythonVersion"
    
    if ($PythonVersion -match "Python (\d+)\.(\d+)") {
        $Major = [int]$Matches[1]
        $Minor = [int]$Matches[2]
        
        if ($Major -lt 3 -or ($Major -eq 3 -and $Minor -lt 8)) {
            Write-Error-Custom "Python 3.8+ required. Found: $PythonVersion"
            exit 1
        }
        Write-Success "Python version compatible (>= 3.8)"
    }
} catch {
    Write-Error-Custom "Python not found. Install from https://www.python.org/"
    exit 1
}

# Step 3: Check CUDA
Write-Header "Step 3: Checking CUDA Availability"
$TorchIndexUrl = ""
try {
    $NvidiaSmi = nvidia-smi 2>&1
    if ($LASTEXITCODE -eq 0 -and $NvidiaSmi -match "CUDA Version: (\d+)\.(\d+)") {
        $CudaVersion = $Matches[1] + "." + $Matches[2]
        $CudaMajor = [int]$Matches[1]
        Write-Success "NVIDIA GPU detected (CUDA $CudaVersion)"
        
        if ($CudaMajor -ge 12) {
            $TorchIndexUrl = "https://download.pytorch.org/whl/cu121"
            Write-Info "Will install PyTorch with CUDA 12.1 support"
        } elseif ($CudaMajor -eq 11) {
            $TorchIndexUrl = "https://download.pytorch.org/whl/cu118"
            Write-Info "Will install PyTorch with CUDA 11.8 support"
        }
    } else {
        Write-Warning "No CUDA GPU detected - will use CPU mode"
    }
} catch {
    Write-Warning "No CUDA GPU detected - will use CPU mode"
}

# Step 4: Setup Virtual Environment
Write-Header "Step 4: Setting Up Virtual Environment"
if (Test-Path "venv") {
    $Response = Read-Host "Virtual environment exists. Recreate? (y/N)"
    if ($Response -eq "y" -or $Response -eq "Y") {
        Write-Info "Removing existing venv..."
        Remove-Item -Recurse -Force venv
        Write-Success "Removed existing venv"
    } else {
        Write-Info "Using existing venv"
    }
}

if (-not (Test-Path "venv")) {
    Write-Info "Creating virtual environment..."
    python -m venv venv --clear
    if ($LASTEXITCODE -ne 0) {
        Write-Error-Custom "Failed to create virtual environment"
        exit 1
    }
    Write-Success "Virtual environment created"
}

# Step 5: Activate venv
Write-Info "Activating virtual environment..."
& ".\venv\Scripts\Activate.ps1"
Write-Success "Virtual environment activated"

# Step 6: Create sitecustomize.py to force venv priority
Write-Header "Step 5: Configuring Virtual Environment"
Write-Info "Creating sitecustomize.py to ensure venv package priority..."

$SiteCustomize = @"
import sys
import os

# Force virtual environment site-packages to be first in sys.path
# This ensures venv packages are prioritized over system packages
if hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix:
    # We're in a venv
    venv_site_packages = os.path.join(sys.prefix, 'Lib', 'site-packages')
    
    # Remove venv site-packages from wherever it is
    sys.path = [p for p in sys.path if os.path.normcase(p) != os.path.normcase(venv_site_packages)]
    
    # Insert at position 1 (after script directory)
    if len(sys.path) > 0:
        sys.path.insert(1, venv_site_packages)
    else:
        sys.path.insert(0, venv_site_packages)
"@

$SiteCustomizePath = ".\venv\Lib\site-packages\sitecustomize.py"
$SiteCustomize | Out-File -FilePath $SiteCustomizePath -Encoding UTF8
Write-Success "sitecustomize.py created - venv packages will always be prioritized"

# Step 7: Upgrade pip
Write-Header "Step 6: Upgrading pip"
Write-Info "Upgrading pip, setuptools, and wheel..."
python -m pip install --upgrade pip setuptools wheel --quiet
if ($LASTEXITCODE -ne 0) {
    Write-Error-Custom "Failed to upgrade pip"
    exit 1
}
Write-Success "pip upgraded"

# Step 8: Install project dependencies
Write-Header "Step 7: Installing Project Dependencies"
Write-Info "Installing all dependencies from pyproject.toml..."

if ($TorchIndexUrl) {
    Write-Info "Installing with PyTorch CUDA support..."
    python -m pip install -e . --extra-index-url $TorchIndexUrl
} else {
    Write-Info "Installing with PyTorch CPU support..."
    python -m pip install -e .
}

if ($LASTEXITCODE -ne 0) {
    Write-Error-Custom "Failed to install dependencies"
    exit 1
}
Write-Success "All dependencies installed"

# Step 9: Verify installation
Write-Header "Step 8: Verifying Installation"
$VerifyScript = Join-Path $ScriptDir "verify_setup.py"
if (Test-Path $VerifyScript) {
    python $VerifyScript
    if ($LASTEXITCODE -eq 0) {
        Write-Success "Verification passed!"
    } else {
        Write-Warning "Some verification checks failed"
    }
}

# Final message
Write-Header "Setup Complete!"
Write-Success "Your environment is ready!"
Write-Host ""
Write-Info "To activate the environment:"
Write-Host "    .\venv\Scripts\Activate.ps1" -ForegroundColor Yellow
Write-Host ""
Write-Info "Available commands:"
Write-Host "    ai-train qlora  - Train with QLoRA" -ForegroundColor Yellow
Write-Host "    ai-eval         - Evaluate models" -ForegroundColor Yellow
Write-Host "    ai-infer        - Run inference" -ForegroundColor Yellow
Write-Host ""
Write-Info "Note: sitecustomize.py ensures venv packages are always prioritized"
Write-Host ""
