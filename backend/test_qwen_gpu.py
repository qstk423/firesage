# -*- coding: utf-8 -*-
"""Qwen2.5-3B-Instruct GPU 冒烟测试：推理速度 + 显存占用。"""
import os
import time

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print(f"torch {torch.__version__} | CUDA: {torch.cuda.is_available()} | {torch.cuda.get_device_name(0)}", flush=True)

path = os.path.expanduser(
    "~/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/main"
)
t0 = time.time()
tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    path, local_files_only=True, torch_dtype=torch.bfloat16, device_map="cuda"
)
model.eval()
print(f"[加载 {time.time()-t0:.1f}s] 显存占用: {torch.cuda.memory_allocated()/1e9:.2f} GB", flush=True)

msgs = [{"role": "user", "content": "占用消防通道会受到什么处罚？请简要回答。"}]
prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
inputs = tok(prompt, return_tensors="pt").to("cuda")

torch.cuda.synchronize()
t0 = time.time()
out = model.generate(
    **inputs, max_new_tokens=120, do_sample=False, pad_token_id=tok.eos_token_id
)
torch.cuda.synchronize()
dt = time.time() - t0
n_new = out.shape[1] - inputs["input_ids"].shape[1]
text = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(f"[生成 {n_new} tokens / {dt:.1f}s = {n_new/dt:.1f} tokens/s]", flush=True)
print(f"[回答] {text}", flush=True)
print(f"[峰值显存] {torch.cuda.max_memory_allocated()/1e9:.2f} GB", flush=True)
print("GPU SMOKE OK")
