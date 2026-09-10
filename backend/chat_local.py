# -*- coding: utf-8 -*-
"""本地 Qwen2.5-3B 最小使用示例：问一句，答一句。"""
import os

os.environ["HF_HUB_OFFLINE"] = "1"  # 强制离线，用本地已下载的模型

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. 模型路径（本地缓存快照）
MODEL_PATH = os.path.expanduser(
    "~/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/main"
)

# 2. 加载 tokenizer（把文字变成数字）和模型（放到 GPU，用 bf16 省显存）
tok = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
    torch_dtype=torch.bfloat16,   # 半精度：3B 模型 6GB 显存放得下
    device_map="cuda",
)
model.eval()

def chat(question: str) -> str:
    """问答一次：问题进，回答出。"""
    messages = [
        {"role": "user", "content": question},
    ]
    # Qwen 的对话格式：<|im_start|>user ... <|im_end|>\n<|im_start|>assistant
    prompt = tok.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    inputs = tok(prompt, return_tensors="pt").to("cuda")
    out = model.generate(
        **inputs,
        max_new_tokens=200,       # 最多生成 200 个字
        do_sample=True,           # 带一点随机性，回答更自然
        temperature=0.7,
        pad_token_id=tok.eos_token_id,
    )
    # 只解码新生成的部分（去掉输入的 prompt）
    return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

if __name__ == "__main__":
    print("输入问题，exit 退出。")
    while True:
        q = input("\n你: ").strip()
        if q.lower() in ("exit", "quit", "q"):
            break
        if q:
            print("Qwen:", chat(q))
