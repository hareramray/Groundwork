param([ValidateRange(1, 65535)][int]$Port = 8000)
$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location -LiteralPath $repoRoot
$venvPython = Join-Path $repoRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $venvPython)) { throw 'Run scripts/setup.ps1 first to create the Python environment.' }
if (-not (Test-Path -LiteralPath 'frontend\dist\index.html')) { throw 'Build the frontend first: run scripts/setup.ps1, or npm run build inside frontend.' }
Write-Host "Groundwork: http://127.0.0.1:$Port"
& $venvPython -m uvicorn grounding.api:app --host 127.0.0.1 --port $Port
exit $LASTEXITCODE
