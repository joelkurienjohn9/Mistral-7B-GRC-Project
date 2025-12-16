# Convenience wrapper for activate_windows.ps1
# Usage: . .\scripts\activate.ps1

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ActivateScript = Join-Path $ScriptDir "activate_windows.ps1"

if (Test-Path $ActivateScript) {
    . $ActivateScript
} else {
    Write-Host "✗ Activation script not found" -ForegroundColor Red
    exit 1
}

