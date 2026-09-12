# ============================================================
# 一键停止本地演示栈（8320 / 8321）
#
# 用法（任意目录均可，幂等可重复执行）：
#   powershell -ExecutionPolicy Bypass -File backend\scripts\stop_local_stack.ps1
#
# 机制：
#   1. 按监听端口定位进程并连子进程树终止（taskkill /T /F）；
#   2. 兜底清理残留包装进程（serve_local_qwen / run_firesage_local）；
#   3. 复核端口已释放。重启用 start_local_stack.ps1。
# ============================================================
$ErrorActionPreference = "Continue"

function Stop-Port([int]$Port, [string]$Name) {
    $conns = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
    if (-not $conns) {
        Write-Host "[跳过] $Name（:$Port）未在监听" -ForegroundColor DarkGray
        return
    }
    $procIds = $conns | Select-Object -ExpandProperty OwningProcess -Unique
    foreach ($procId in $procIds) {
        Write-Host "[停止] $Name（:$Port，PID $procId）"
        & taskkill /PID $procId /T /F 2>$null | Out-Null
    }
}

Stop-Port 8320 "Qwen+LoRA 推理服务"
Stop-Port 8321 "FireSage 本地后端"

# 兜底：清理残留包装/服务进程（run_firesage_local 在 uvicorn 退出后应自行退出，偶有残留）
Get-CimInstance Win32_Process -Filter "Name like 'python%'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -match 'serve_local_qwen\.py|run_firesage_local\.py' } |
    ForEach-Object {
        Write-Host "[清理] 残留进程 PID $($_.ProcessId)：$($_.CommandLine.Substring(0, [Math]::Min(90, $_.CommandLine.Length)))"
        Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
    }

# 复核端口释放
Start-Sleep -Seconds 2
$allReleased = $true
foreach ($port in 8320, 8321) {
    if (Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue) {
        Write-Warning "端口 $port 仍被占用，请手动检查（Get-NetTCPConnection -LocalPort $port）"
        $allReleased = $false
    } else {
        Write-Host "[确认] 端口 $port 已释放" -ForegroundColor Green
    }
}

if ($allReleased) {
    Write-Host ""
    Write-Host "本地演示栈已全部停止。重启：" -ForegroundColor Cyan
    Write-Host "  powershell -ExecutionPolicy Bypass -File backend\scripts\start_local_stack.ps1"
}
