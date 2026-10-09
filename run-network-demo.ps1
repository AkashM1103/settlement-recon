$ErrorActionPreference = "Stop"

$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    Write-Host "Set up the project first: py -m venv .venv; .\.venv\Scripts\python.exe -m pip install -r requirements.txt"
    exit 1
}

$addresses = Get-NetIPAddress -AddressFamily IPv4 |
    Where-Object { $_.IPAddress -notlike "127.*" -and $_.PrefixOrigin -ne "WellKnown" } |
    Select-Object -ExpandProperty IPAddress -Unique

Write-Host "Settlement Reconciliation demo is starting..."
Write-Host "On this laptop: http://localhost:8501"
foreach ($address in $addresses) {
    Write-Host "From another laptop on the same network: http://${address}:8501"
}
Write-Host "Keep this window open while people use the demo. Press Ctrl+C to stop it."
& $python -m streamlit run (Join-Path $PSScriptRoot "app.py") --server.address=0.0.0.0 --server.port=8501
