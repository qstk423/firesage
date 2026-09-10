# -*- coding: utf-8 -*-
"""从 ModelScope 下载 Qwen/Qwen2.5-3B-Instruct 到 HF 缓存快照结构。"""
import os
import traceback

from modelscope import snapshot_download

hub = os.path.expanduser("~/.cache/huggingface/hub")
target = os.path.join(hub, "models--Qwen--Qwen2.5-3B-Instruct", "snapshots", "main")
os.makedirs(target, exist_ok=True)

print(f"[dl] Qwen/Qwen2.5-3B-Instruct -> {target}", flush=True)
try:
    snapshot_download(
        "Qwen/Qwen2.5-3B-Instruct",
        local_dir=target,
        ignore_patterns=["*.onnx", "onnx*", "*.h5", "*.msgpack", "openvino*", "*.gguf"],
    )
    print("[ok] Qwen/Qwen2.5-3B-Instruct", flush=True)
except Exception as e:
    print(f"[fail] {e}", flush=True)
    traceback.print_exc()

print("ALL DONE", flush=True)
