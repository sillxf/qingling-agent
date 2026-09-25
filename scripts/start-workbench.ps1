param([int]$Port = 8010)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
if (-not $env:QINGLING_STORE_BACKEND) { $env:QINGLING_STORE_BACKEND = 'sqlite' }
if (-not $env:QINGLING_SQLITE_PATH) { $env:QINGLING_SQLITE_PATH = 'data/workbench.db' }
$projectPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $projectPython)) { throw '请先按 README 创建虚拟环境并安装依赖。' }
& $projectPython -m uvicorn app.main:app --host 127.0.0.1 --port $Port
exit $LASTEXITCODE
