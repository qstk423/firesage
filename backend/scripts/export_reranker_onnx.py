#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把本地 bge-reranker-v2-m3 快照导出为 ONNX，并做 INT8 动态量化（CPU 推理提速）。

产物：backend/models/bge-reranker-v2-m3-onnx-int8/
  - model_int8.onnx   INT8 动态量化模型（MatMul/Gemm 权重 QInt8 per-channel）
  - tokenizer.json 等 分词器文件（与 fp32 快照同源）
  - meta.json         导出参数、parity 校验与延迟基准结果

无 optimum 依赖：导出用 torch.onnx.export（dynamo），量化用
onnxruntime.quantization.quantize_dynamic（与 optimum avx2 动态量化同底层）。

用法（backend 目录）：
    ..\\venv311\\Scripts\\python.exe scripts\\export_reranker_onnx.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import torch

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_ID = "BAAI/bge-reranker-v2-m3"
OUT_DIR = os.path.join(BACKEND_DIR, "models", "bge-reranker-v2-m3-onnx-int8")
MAX_LENGTH = 512
OPSET = 17

# 校验/基准用的真实分布样本：短问题 × ~480 字条款文本（与检索管线截断一致）
SAMPLE_Q = "楼道堆放杂物违反什么规定，怎么处理？"
SAMPLE_TEXT = (
    "《中华人民共和国消防法》第二十八条：任何单位、个人不得损坏、挪用或者擅自拆除、"
    "停用消防设施、器材，不得埋压、圈占、遮挡消火栓或者占用防火间距，不得占用、堵塞、"
    "封闭疏散通道、安全出口、消防车通道。人员密集场所的门窗不得设置影响逃生和灭火救援的"
    "障碍物。第六十条：单位违反本法规定，有下列行为之一的，责令改正，处五千元以上五万元"
    "以下罚款：（一）消防设施、器材或者消防安全标志的配置、设置不符合国家标准、行业标准，"
    "或者未保持完好有效的；（二）损坏、挪用或者擅自拆除、停用消防设施、器材的；（三）占用、"
    "堵塞、封闭疏散通道、安全出口或者有其他妨碍安全疏散行为的；（四）埋压、圈占、遮挡消火栓"
    "或者占用防火间距的；（五）占用、堵塞、封闭消防车通道，妨碍消防车通行的。"
)


def find_snapshot() -> str:
    hub = os.path.expanduser("~/.cache/huggingface/hub")
    snap_root = os.path.join(hub, "models--" + MODEL_ID.replace("/", "--"), "snapshots")
    if not os.path.isdir(snap_root):
        raise SystemExit(f"[错误] 找不到本地快照 {snap_root}")
    for name in os.listdir(snap_root):
        path = os.path.join(snap_root, name)
        if os.path.isdir(path) and os.path.exists(os.path.join(path, "config.json")):
            return path
    raise SystemExit("[错误] 快照目录里没有含 config.json 的版本")


class _Wrapper(torch.nn.Module):
    """positional (input_ids, attention_mask) → logits，便于 ONNX 导出。"""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return out.logits


def export_fp32(model, tok, out_path: str) -> None:
    enc = tok([(SAMPLE_Q, SAMPLE_TEXT)], return_tensors="pt",
              padding=True, truncation=True, max_length=MAX_LENGTH)
    wrapper = _Wrapper(model).eval()
    # torch 2.11 默认 dynamo 导出；失败则回退 legacy tracing
    try:
        torch.onnx.export(
            wrapper, (enc["input_ids"], enc["attention_mask"]), out_path,
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"], opset_version=OPSET, dynamo=True,
            dynamic_shapes={
                "input_ids": {0: "batch", 1: "seq"},
                "attention_mask": {0: "batch", 1: "seq"},
            },
        )
        print("[导出] dynamo 路径完成")
    except Exception as exc:  # noqa: BLE001
        print(f"[导出] dynamo 失败（{type(exc).__name__}: {exc}），回退 legacy tracing")
        torch.onnx.export(
            wrapper, (enc["input_ids"], enc["attention_mask"]), out_path,
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"], opset_version=OPSET, dynamo=False,
            dynamic_axes={
                "input_ids": {0: "batch", 1: "seq"},
                "attention_mask": {0: "batch", 1: "seq"},
                "logits": {0: "batch"},
            },
        )
        print("[导出] legacy 路径完成")


def ort_session(path: str):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(path, sess_options=opts,
                                providers=["CPUExecutionProvider"])


def run_logits(sess, tok, pairs: list[tuple[str, str]]) -> np.ndarray:
    enc = tok(pairs, return_tensors="np", padding=True,
              truncation=True, max_length=MAX_LENGTH)
    feeds = {
        "input_ids": enc["input_ids"].astype(np.int64),
        "attention_mask": enc["attention_mask"].astype(np.int64),
    }
    return sess.run(["logits"], feeds)[0].reshape(-1)


def torch_logits(model, tok, pairs: list[tuple[str, str]]) -> np.ndarray:
    enc = tok(pairs, return_tensors="pt", padding=True,
              truncation=True, max_length=MAX_LENGTH)
    with torch.no_grad():
        out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
    return out.logits.float().reshape(-1).numpy()


def main() -> None:
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    snapshot = find_snapshot()
    print(f"[加载] {snapshot}（fp32 CPU）")
    tok = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        snapshot, local_files_only=True, dtype=torch.float32).eval()

    os.makedirs(OUT_DIR, exist_ok=True)
    fp32_path = os.path.join(OUT_DIR, "model_fp32.onnx")
    int8_path = os.path.join(OUT_DIR, "model_int8.onnx")

    export_fp32(model, tok, fp32_path)
    print(f"[导出] fp32 ONNX {os.path.getsize(fp32_path) / 1e6:.0f} MB")

    # parity：fp32 ONNX vs PyTorch（必须几乎一致，否则导出有错）
    pairs = [
        (SAMPLE_Q, SAMPLE_TEXT),
        (SAMPLE_Q, SAMPLE_TEXT[:240]),
        ("电动车能在楼道充电吗", SAMPLE_TEXT[:300]),
        ("消防通道被占用怎么处罚", SAMPLE_TEXT[120:400]),
    ]
    ref = torch_logits(model, tok, pairs)
    sess_fp32 = ort_session(fp32_path)
    got = run_logits(sess_fp32, tok, pairs)
    diff_fp32 = float(np.abs(ref - got).max())
    print(f"[校验] fp32-ONNX vs PyTorch logits 最大偏差 {diff_fp32:.4f}（阈值 0.1）")
    if diff_fp32 > 0.1:
        raise SystemExit("[错误] fp32 ONNX 与 PyTorch logits 不一致，导出有误，中止")

    # INT8 动态量化（MatMul/Gemm 权重 per-channel QInt8，激活动态量化）
    quantize_dynamic(
        model_input=fp32_path, model_output=int8_path,
        weight_type=QuantType.QInt8, per_channel=True,
        op_types_to_quantize=["MatMul", "Gemm"],
    )
    print(f"[量化] INT8 ONNX {os.path.getsize(int8_path) / 1e6:.0f} MB")

    # parity：INT8 vs PyTorch（记录漂移；排序质量由后续 e2e 验证）
    sess_int8 = ort_session(int8_path)
    got8 = run_logits(sess_int8, tok, pairs)
    diff_int8 = float(np.abs(ref - got8).max())
    corr = float(np.corrcoef(ref, got8)[0, 1])
    print(f"[校验] INT8 vs PyTorch logits 最大偏差 {diff_int8:.4f}，相关系数 {corr:.4f}")

    # 延迟基准：6 对（CE_CANDIDATES=6），文本 ~480 字符
    bench_pairs = [(SAMPLE_Q, SAMPLE_TEXT)] * 6
    run_logits(sess_int8, tok, bench_pairs)  # 预热
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        run_logits(sess_int8, tok, bench_pairs)
        times.append((time.perf_counter() - t0) * 1000)
    lat = min(times)
    print(f"[基准] INT8 CPU 6 对（~{MAX_LENGTH} 截断）耗时 {lat:.0f} ms（3 次取最小）")

    # 固化产物：tokenizer + meta；删除 fp32 中间文件（~2.3GB）
    tok.save_pretrained(OUT_DIR)
    meta = {
        "source": MODEL_ID,
        "snapshot": snapshot,
        "max_length": MAX_LENGTH,
        "opset": OPSET,
        "quant": "quantize_dynamic QInt8 per_channel (MatMul/Gemm)",
        "parity": {"fp32_vs_torch_max_abs": round(diff_fp32, 5),
                   "int8_vs_torch_max_abs": round(diff_int8, 5),
                   "int8_vs_torch_corr": round(corr, 5)},
        "benchmark_ms_6pairs": round(lat),
        "created": datetime.now(timezone.utc).isoformat(),
    }
    with open(os.path.join(OUT_DIR, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    os.remove(fp32_path)
    print(f"[完成] 产物目录 {OUT_DIR}（fp32 中间文件已删除）")
    print("[提示] 在 .env.local-model 加 RERANK_BACKEND=onnx_int8 启用")


if __name__ == "__main__":
    main()
