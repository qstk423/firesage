#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用 .env.local-model 启动一套独立的 FireSage 后端（不动 DeepSeek 生产配置）。

机制：
1. 读取 backend/.env.local-model 中的 LLM_* 配置写入进程环境变量；
   rag/llm.py 加载 .env.local 用的是 setdefault，进程环境变量优先级更高，
   因此 .env.local（DeepSeek 云端）原样保留、随时可回退（不设这些变量即回退）；
2. CUDA_VISIBLE_DEVICES="" 强制 bge-m3 / bge-reranker 走 CPU，
   避免与 8320 端口的 Qwen+LoRA 推理服务抢占显存（faiss 索引本就在 CPU）；
3. 以 uvicorn main:app 方式启动（不执行 main.__main__），
   监听 127.0.0.1:8321，与生产 8319、本地 LLM 8320 互不冲突。

用法（backend 目录）：
    ..\\venv311\\Scripts\\python.exe scripts\\run_firesage_local.py
回退 DeepSeek：直接停掉本进程，按原方式启动 main.py 即可。
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


def main() -> None:
    if not os.path.exists(ENV_FILE):
        print(f"[错误] 找不到 {ENV_FILE}")
        raise SystemExit(1)

    env = dict(os.environ)
    # 本地模型配置注入进程环境（优先级高于 .env.local 的 setdefault）
    for key, value in load_env_file(ENV_FILE).items():
        if key.startswith("LLM_"):
            env[key] = value
    # BGE-M3 / reranker 强制 CPU，避免与 Qwen+LoRA 抢显存
    env["CUDA_VISIBLE_DEVICES"] = ""
    # 提示 llm.py 不要再落 .env.local 的 DeepSeek 值（环境变量已全部注入）
    print(f"[启动] FireSage 独立后端（本地模型） → http://127.0.0.1:{PORT}")
    print("[配置] LLM 指向 127.0.0.1:8320（Qwen2.5-3B + LoRA v2）")
    print("[配置] bge-m3 / bge-reranker 强制 CPU（CUDA_VISIBLE_DEVICES=\"\"）")
    print("[回退] 停止本进程后按原方式启动 main.py 即恢复 DeepSeek 云端")

    cmd = [sys.executable, "-m", "uvicorn", "main:app",
           "--host", "127.0.0.1", "--port", str(PORT)]
    raise SystemExit(subprocess.call(cmd, cwd=BACKEND_DIR, env=env))


if __name__ == "__main__":
    main()
