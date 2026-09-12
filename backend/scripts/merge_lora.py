#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""一次性合并 LoRA adapter 到基座模型（提升推理吞吐，数学上等价）。

背景：PEFT 未合并的 adapter 在 generate 时逐 token 有额外开销（适配器
前向 + kernel launch），实测长提示下仅 6-9 tok/s；合并后可直接以
普通 CausalLM 加载，免去 PEFT 包装。

用法（backend 目录）：
    ..\\venv311\\Scripts\\python.exe scripts\\merge_lora.py

输出：models/Qwen2.5-3B-Instruct-firesage-v2-merged（约 6GB，已被
.gitignore 的 backend/models/ 覆盖，不入库）。

serve_local_qwen.py 会自动优先加载该目录（存在即用，删除即回退
基座+adapter 原路径，二者可随时切换）。
"""
from __future__ import annotations

import os
import shutil
import sys

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_DIR = os.path.join(BACKEND_DIR, "models", "Qwen2.5-3B-Instruct")
ADAPTER_DIR = os.path.join(BACKEND_DIR, "adapters", "firesage-qwen25-3b-lora-v2")
OUT_DIR = os.path.join(BACKEND_DIR, "models", "Qwen2.5-3B-Instruct-firesage-v2-merged")


def main() -> None:
    for label, path in (("基座模型", BASE_DIR), ("LoRA adapter", ADAPTER_DIR)):
        if not os.path.exists(path):
            print(f"[错误] {label}不存在：{path}")
            raise SystemExit(1)

    if os.path.exists(OUT_DIR):
        print(f"[跳过] 合并模型已存在：{OUT_DIR}（如需重做请先删除该目录）")
        raise SystemExit(0)

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[加载] 基座 {BASE_DIR}（CPU，约 1-2 分钟）")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_DIR, local_files_only=True, dtype=torch.bfloat16, device_map="cpu")
    print(f"[加载] LoRA adapter {ADAPTER_DIR}")
    model = PeftModel.from_pretrained(model, ADAPTER_DIR)

    print("[合并] merge_and_unload（权重级合并，数值等价）")
    model = model.merge_and_unload()

    print(f"[保存] {OUT_DIR}")
    model.save_pretrained(OUT_DIR)

    # tokenizer 与生成配置一并落盘，使合并目录可独立加载
    tok = AutoTokenizer.from_pretrained(BASE_DIR, local_files_only=True)
    tok.save_pretrained(OUT_DIR)
    for fname in ("generation_config.json",):
        src = os.path.join(BASE_DIR, fname)
        if os.path.exists(src) and not os.path.exists(os.path.join(OUT_DIR, fname)):
            shutil.copy2(src, os.path.join(OUT_DIR, fname))

    # 校验：合并目录能独立加载出 CausalLM
    print("[校验] 重新加载合并模型...")
    reloaded = AutoModelForCausalLM.from_pretrained(
        OUT_DIR, local_files_only=True, dtype=torch.bfloat16, device_map="cpu")
    n_params = sum(p.numel() for p in reloaded.parameters())
    del reloaded
    print(f"[完成] 参数量 {n_params/1e9:.2f}B → {OUT_DIR}")


if __name__ == "__main__":
    sys.exit(main())
