# Verification Script for Windows
# Checks if the environment is properly set up

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir

Set-Location $ProjectRoot

Write-Host "=== Environment Verification ===" -ForegroundColor Cyan
Write-Host ""

# Check for Python
$PythonCmd = ".\venv\Scripts\python.exe"
if (-not (Test-Path $PythonCmd)) {
    $PythonCmd = "python"
}

# Run Python verification script
$VerifyScript = Join-Path $ScriptDir "verify_setup.py"
if (Test-Path $VerifyScript) {
    & $PythonCmd $VerifyScript
    $exitCode = $LASTEXITCODE
    
    Write-Host ""
    if ($exitCode -eq 0) {
        Write-Host "=== All checks passed! ===" -ForegroundColor Green
    } else {
        Write-Host "=== Some checks failed ===" -ForegroundColor Yellow
        Write-Host "Run setup again: .\scripts\setup_windows.ps1" -ForegroundColor Cyan
    }
    
    exit $exitCode
} else {
    Write-Host "✗ Verification script not found" -ForegroundColor Red
    exit 1
}
