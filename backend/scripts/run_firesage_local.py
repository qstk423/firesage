#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用 .env.local-model 启动一套独立的 FireSage 后端（不动 DeepSeek 生产配置）。

机制：
1. 读取 backend/.env.local-model 中的 LLM_* / RERANK_* 配置写入子进程环境；
   rag/llm.py 加载 .env.local 用的是 setdefault，进程环境变量优先级更高，
   因此 .env.local（DeepSeek 云端）原样保留、随时可回退（不设这些变量即回退）；
2. CUDA_VISIBLE_DEVICES="-1" 强制 bge-m3 / bge-reranker 走 CPU，
   避免与 8320 端口的 Qwen+LoRA 推理服务抢占显存（faiss 索引本就在 CPU）。
   注意：Windows + CUDA 12.x 上空字符串 "" 会被静默忽略（GPU 仍然可见），
   必须 "-1" 才真正隐藏设备（实测 torch.cuda.is_available()=False）；
3. 启动前用同一环境变量探测子进程的 CUDA 可见性，验收隔离是否生效；
4. 以 uvicorn main:app 方式启动（不执行 main.__main__），
   监听 127.0.0.1:8321，与生产 8319、本地 LLM 8320 互不冲突。

用法（backend 目录）：
    ..\\venv311\\Scripts\\python.exe scripts\\run_firesage_local.py
回退 DeepSeek：直接停掉本进程后，按原方式启动 main.py 即可。
"""
from __future__ import annotations

import os
import subprocess
import sys

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_FILE = os.path.join(BACKEND_DIR, ".env.local-model")
PORT = int(os.getenv("FIRESAGE_LOCAL_PORT", "8321"))


def load_env_file(path: str) -> dict:
    values = {}
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip("\"'")
    return values


def probe_cuda_hidden(env: dict) -> bool:
    """用与 RAG 子进程完全相同的环境变量探测 CUDA 是否被隐藏。

    在独立的探针进程里 import torch，避免本进程提前加载 torch。
    返回 torch.cuda.is_available() 的结果（期望 False）。
    """
    code = "import torch; print(torch.cuda.is_available())"
    try:
        r = subprocess.run([sys.executable, "-c", code], env=env,
                           capture_output=True, text=True, timeout=120)
        return r.stdout.strip().lower().endswith("true")
    except Exception as exc:
        print(f"[警告] CUDA 探针执行失败（{exc}），继续启动但不保证隔离")
        return True  # 保守假设可见，由启动后显存验收兜底


def main() -> None:
    if not os.path.exists(ENV_FILE):
        print(f"[错误] 找不到 {ENV_FILE}")
        raise SystemExit(1)

    env = dict(os.environ)
    # 本地模型配置注入子进程环境（优先级高于 .env.local 的 setdefault）
    for key, value in load_env_file(ENV_FILE).items():
        if key.startswith(("LLM_", "RERANK_")):
            env[key] = value
    # BGE-M3 / reranker 强制 CPU，避免与 Qwen+LoRA 抢显存。
    # 关键：必须在子进程 Python 启动前生效（torch/sentence_transformers 导入即读取）。
    env["CUDA_VISIBLE_DEVICES"] = "-1"

    print(f"[启动] FireSage 独立后端（本地模型） → http://127.0.0.1:{PORT}")
    print("[配置] LLM 指向 127.0.0.1:8320（Qwen2.5-3B + LoRA v2）")
    print("[配置] bge-m3 / bge-reranker 强制 CPU（CUDA_VISIBLE_DEVICES=\"-1\"）")
    print("[回退] 停掉本进程后按原方式启动 main.py 即恢复 DeepSeek 云端")

    # ---- 隔离自检：RAG 子进程 CUDA 是否真正不可见 ----
    cuda_visible = probe_cuda_hidden(env)
    if cuda_visible:
        print("[隔离自检] 失败：RAG 子进程 torch.cuda.is_available()=True，"
              "BGE 可能抢占显存拖慢 Qwen 生成！")
        raise SystemExit(1)
    print("[隔离自检] RAG 子进程 torch.cuda.is_available()=False（GPU 已隐藏，bge 走 CPU）")

    cmd = [sys.executable, "-m", "uvicorn", "main:app",
           "--host", "127.0.0.1", "--port", str(PORT)]
    raise SystemExit(subprocess.call(cmd, cwd=BACKEND_DIR, env=env))


if __name__ == "__main__":
    main()
