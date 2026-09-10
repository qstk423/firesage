# -*- coding: utf-8 -*-
"""从 ModelScope（国内源）下载 bge-m3 与 bge-reranker-v2-m3，
放入 HF 缓存快照结构，供 semantic_index.py 离线加载。"""
import os
import traceback

from modelscope import snapshot_download

hub = os.path.expanduser("~/.cache/huggingface/hub")

JOBS = [
    ("BAAI/bge-m3", "models--BAAI--bge-m3"),
    ("BAAI/bge-reranker-v2-m3", "models--BAAI--bge-reranker-v2-m3"),
]

for mid, dirname in JOBS:
    target = os.path.join(hub, dirname, "snapshots", "main")
    os.makedirs(target, exist_ok=True)
    print(f"[dl] {mid} -> {target}", flush=True)
    try:
        snapshot_download(
            mid,
            local_dir=target,
            ignore_patterns=["*.onnx", "onnx*", "*.h5", "*.msgpack", "openvino*"],
        )
        print(f"[ok] {mid}", flush=True)
    except Exception as e:
        print(f"[fail] {mid}: {e}", flush=True)
        traceback.print_exc()

print("ALL DONE", flush=True)
