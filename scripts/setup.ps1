param(
    [ValidateSet('cuda', 'cpu')][string]$Device = 'cuda',
    [string]$Python = 'python'
)
$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location -LiteralPath $repoRoot

function Invoke-Checked {
    param([string]$Executable, [string[]]$Arguments)
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Command failed ($LASTEXITCODE): $Executable $($Arguments -join ' ')" }
}

Get-Command $Python -ErrorAction Stop | Out-Null
Get-Command npm -ErrorAction Stop | Out-Null
Invoke-Checked 'node' @('-e', 'const [major,minor]=process.versions.node.split(''.'').map(Number); if(major<22||(major===22&&minor<12)) throw Error(''Use Node.js 22.12 or newer'');')
Invoke-Checked $Python @('-c', 'import sys; assert sys.version_info[:2] == (3, 13), ''Use Python 3.13 for the verified environment''')
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    Invoke-Checked $Python @('-m', 'venv', '.venv')
}
$venvPython = Join-Path $repoRoot '.venv\Scripts\python.exe'
Invoke-Checked $venvPython @('-m', 'pip', 'install', '--upgrade', 'pip')
$wheelIndex = if ($Device -eq 'cpu') { 'https://download.pytorch.org/whl/cpu' } else { 'https://download.pytorch.org/whl/cu128' }
Invoke-Checked $venvPython @('-m', 'pip', 'install', 'torch==2.11.0', '--index-url', $wheelIndex)
$requirements = if ($Device -eq 'cpu') { 'requirements-cpu-lock.txt' } else { 'requirements-lock.txt' }
Invoke-Checked $venvPython @('-m', 'pip', 'install', '-r', $requirements)
Push-Location (Join-Path $repoRoot 'frontend')
try {
    Invoke-Checked 'npm.cmd' @('ci')
    Invoke-Checked 'npm.cmd' @('run', 'build')
} finally { Pop-Location }
Invoke-Checked $venvPython @('-c', 'import torch; print(''PyTorch:'', torch.__version__); print(''CUDA available:'', torch.cuda.is_available()); print(''Device:'', torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''CPU'')')
Write-Host 'Setup complete. Start with: powershell -ExecutionPolicy Bypass -File scripts/start.ps1'
