$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$venvPython = Join-Path $repoRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $venvPython)) {
    throw 'Run scripts/setup.ps1 first to create the Python environment.'
}
Push-Location -LiteralPath $repoRoot
try {
    & $venvPython -m grounding.agent_cli @args
    $agentExitCode = $LASTEXITCODE
} finally {
    Pop-Location
}
exit $agentExitCode
