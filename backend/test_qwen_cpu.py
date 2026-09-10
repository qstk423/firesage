# -*- coding: utf-8 -*-
"""Qwen2.5-3B-Instruct 冒烟测试（CPU 短生成，验证模型文件完整性）。"""
import os
import time

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from transformers import AutoModelForCausalLM, AutoTokenizer

path = os.path.expanduser(
    "~/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/main"
)
print(f"[load] 从 {path} 加载...", flush=True)

tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    path, local_files_only=True, torch_dtype="auto", low_cpu_mem_usage=True
)
model.eval()
print(f"[ok] 模型加载成功，参数量: {sum(p.numel() for p in model.parameters())/1e9:.2f}B", flush=True)

msgs = [{"role": "user", "content": "用一句话说明什么是消防通道"}]
prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
inputs = tok(prompt, return_tensors="pt")
t0 = time.time()
out = model.generate(
    **inputs, max_new_tokens=30, do_sample=False, pad_token_id=tok.eos_token_id
)
text = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(f"[生成 {time.time()-t0:.1f}s] {text}", flush=True)
print("SMOKE OK")
