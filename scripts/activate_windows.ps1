# Activate Virtual Environment Script for Windows
# This script can be run directly with: .\scripts\activate_windows.ps1

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$VenvPath = Join-Path $ProjectRoot "venv"
$ActivateScript = Join-Path $VenvPath "Scripts\Activate.ps1"

if (Test-Path $ActivateScript) {
    Write-Host "Activating virtual environment..." -ForegroundColor Cyan
    
    # On Windows, we need to invoke the activation script in the current scope
    . $ActivateScript
    
    Write-Host "✓ Virtual environment activated" -ForegroundColor Green
    Write-Host ""
    Write-Host "Available commands:" -ForegroundColor Cyan
    Write-Host "  ai-train qlora  - Train with QLoRA" -ForegroundColor Yellow
    Write-Host "  ai-eval         - Evaluate models" -ForegroundColor Yellow
    Write-Host "  ai-infer        - Run inference" -ForegroundColor Yellow
} else {
    Write-Host "✗ Virtual environment not found at: $VenvPath" -ForegroundColor Red
    Write-Host "  Run setup first: .\scripts\setup_windows.ps1" -ForegroundColor Yellow
    exit 1
}
