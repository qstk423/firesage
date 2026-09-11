﻿# ============================================================
# 一键启动本地演示栈（桌面端）
#   8320  Qwen2.5-3B + LoRA v2 推理服务（OpenAI 兼容，五层守卫 + 流式）
#   8321  FireSage 独立后端（读 .env.local-model，BGE 走 CPU）
#
# 用法（任意目录均可）：
#   powershell -ExecutionPolicy Bypass -File backend\scripts\start_local_stack.ps1
#
# 幂等：端口已在监听则跳过对应服务；服务异常退出后直接重跑本脚本即可恢复。
# 停止：powershell -ExecutionPolicy Bypass -File backend\scripts\stop_local_stack.ps1
# ============================================================
$ErrorActionPreference = "Stop"

$ScriptsDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$BackendDir = Split-Path -Parent $ScriptsDir
$RepoDir    = Split-Path -Parent $BackendDir
$Python     = Join-Path $RepoDir "venv311\Scripts\python.exe"

if (-not (Test-Path $Python)) {
    Write-Error "找不到 venv311 解释器：$Python"
    exit 1
}

# 子进程继承：无缓冲 + UTF-8，保证日志实时且中文不乱码
$env:PYTHONIOENCODING = "utf-8"

function Test-Port([int]$Port) {
    [bool](Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)
}

function Wait-Url([string]$Url, [int]$TimeoutSec) {
    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    while ((Get-Date) -lt $deadline) {
        try {
            Invoke-RestMethod $Url -TimeoutSec 5 | Out-Null
            return $true
        } catch {
            Start-Sleep -Seconds 5
        }
    }
    return $false
}

function Wait-SystemReady([int]$TimeoutSec) {
    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    while ((Get-Date) -lt $deadline) {
        try {
            $r = Invoke-RestMethod "http://127.0.0.1:8321/api/system" -TimeoutSec 5
            if ($r.ready) { return $true }
        } catch { }
        Start-Sleep -Seconds 5
    }
    return $false
}

# ---- 8320：Qwen+LoRA 推理服务 ----
if (Test-Port 8320) {
    $procId = (Get-NetTCPConnection -LocalPort 8320 -State Listen |
               Select-Object -First 1).OwningProcess
    Write-Host "[跳过] 8320 推理服务已在运行（PID $procId）" -ForegroundColor Yellow
} else {
    Write-Host "[启动] 8320 Qwen+LoRA 推理服务（模型加载约 1-2 分钟）..."
    Start-Process -FilePath $Python `
        -ArgumentList "-u", "scripts\serve_local_qwen.py", "--api-key", "local-firesage" `
        -WorkingDirectory $BackendDir -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $BackendDir "server.local-8320.log") `
        -RedirectStandardError  (Join-Path $BackendDir "server.local-8320.err.log")
    if (Wait-Url "http://127.0.0.1:8320/health" 300) {
        Write-Host "[就绪] 8320 /health OK" -ForegroundColor Green
    } else {
        Write-Error "8320 未在 300 秒内就绪，请查看 backend\server.local-8320.err.log"
        exit 1
    }
}

# ---- 8321：FireSage 独立后端 ----
if (Test-Port 8321) {
    $procId = (Get-NetTCPConnection -LocalPort 8321 -State Listen |
               Select-Object -First 1).OwningProcess
    Write-Host "[跳过] 8321 本地后端已在运行（PID $procId）" -ForegroundColor Yellow
} else {
    Write-Host "[启动] 8321 FireSage 本地后端（BGE CPU 加载约 2-4 分钟）..."
    Start-Process -FilePath $Python `
        -ArgumentList "-u", "scripts\run_firesage_local.py" `
        -WorkingDirectory $BackendDir -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $BackendDir "server.local-8321.log") `
        -RedirectStandardError  (Join-Path $BackendDir "server.local-8321.err.log")
    if (Wait-SystemReady 360) {
        Write-Host "[就绪] 8321 /api/system ready=True" -ForegroundColor Green
    } else {
        Write-Error "8321 未在 360 秒内就绪，请查看 backend\server.local-8321.err.log"
        exit 1
    }
}

# ---- 健康自检：LLM 是否指向 8320 ----
try {
    $sys = Invoke-RestMethod "http://127.0.0.1:8321/api/system" -TimeoutSec 10
    if (-not $sys.runtime.llm_enabled) {
        Write-Warning "8321 LLM 未启用（.env.local-model 未生效？），回答将走抽取式降级"
    }
} catch { }

Write-Host ""
Write-Host "本地演示栈已就绪（桌面端）：" -ForegroundColor Cyan
Write-Host "  前端 / 后端    http://127.0.0.1:8321"
Write-Host "  推理服务健康   http://127.0.0.1:8320/health"
Write-Host "  日志           backend\server.local-8320.log / server.local-8321.log"
Write-Host "  停止           powershell -ExecutionPolicy Bypass -File backend\scripts\stop_local_stack.ps1"
