# Starts a local Redis on 127.0.0.1:6379 for the async (Celery) mode, using whatever
# this machine has, in order of preference:
#   1. Docker            docker run redis:7-alpine
#   2. Memurai Developer Redis-compatible Windows service (winget install Memurai.MemuraiDeveloper)
#   3. redis-server.exe  a portable build on PATH (e.g. tporadowski/redis)
#
#     powershell -ExecutionPolicy Bypass -File deploy\run_redis_local.ps1
$ErrorActionPreference = "Stop"

function Test-Redis {
    try { $c = New-Object Net.Sockets.TcpClient("127.0.0.1", 6379); $c.Close(); return $true } catch { return $false }
}
if (Test-Redis) { Write-Host "Redis already listening on 6379"; exit 0 }

if (Get-Command docker -ErrorAction SilentlyContinue) {
    docker run -d --name ocr-redis -p 6379:6379 redis:7-alpine | Out-Null
    Write-Host "started redis:7-alpine in docker (container ocr-redis)"
}
elseif (Get-Service -Name Memurai -ErrorAction SilentlyContinue) {
    Start-Service Memurai; Write-Host "started Memurai service"
}
elseif (Get-Command redis-server -ErrorAction SilentlyContinue) {
    Start-Process redis-server -ArgumentList "--port 6379 --save '' --appendonly no" -WindowStyle Hidden
    Write-Host "started portable redis-server"
}
else {
    Write-Host "No Redis available. Installing Memurai Developer via winget (Redis-compatible, free dev edition)..."
    winget install --id Memurai.MemuraiDeveloper -e --silent --accept-package-agreements --accept-source-agreements
    Start-Service Memurai
    Write-Host "Memurai installed and started"
}
Start-Sleep -Seconds 2
if (Test-Redis) { Write-Host "OK: 127.0.0.1:6379 is up" } else { throw "Redis did not come up" }
