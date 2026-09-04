#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成式 LoRA 训练入口（可选，需 GPU / 云端）。

默认不在本机强制安装大模型训练依赖。当环境具备 transformers+peft+datasets 时，
读取 build_gen_sft.py 产物进行 QLoRA；否则打印数据路径与推荐命令后退出。

用法：
    python3 backend/scripts/train_gen_lora.py --dry-run
    python3 backend/scripts/train_gen_lora.py --model Qwen/Qwen2.5-1.5B-Instruct
"""
from __future__ import annotations

import argparse
import json
import os
import sys

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SFT_DIR = os.path.join(BACKEND_DIR, "data", "models", "sft")
OUT_DIR = os.path.join(BACKEND_DIR, "data", "models", "gen_lora")


def _load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _to_text(row: dict) -> dict:
    return {
        "text": (
            f"### Instruction:\n{row['instruction']}\n\n"
            f"### Input:\n{row['input']}\n\n"
            f"### Response:\n{row['output_text']}"
        )
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="只检查数据与依赖，不训练")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=120)
    args = parser.parse_args()

    train_path = os.path.join(SFT_DIR, "gen_sft_train.jsonl")
    if not os.path.exists(train_path):
        print("缺少 SFT 数据，请先运行：python3 backend/scripts/build_gen_sft.py")
        sys.exit(1)

    train_rows = _load_jsonl(train_path)
    print(f"[data] train={len(train_rows)} from {train_path}")

    if args.dry_run:
        print("[dry-run] 数据就绪。云端/GPU 推荐命令示例：")
        print("  pip install transformers peft datasets accelerate bitsandbytes trl")
        print("  python3 backend/scripts/train_gen_lora.py --model Qwen/Qwen2.5-1.5B-Instruct")
        print(f"  输出目录：{OUT_DIR}")
        return

    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
        from trl import SFTTrainer
    except ImportError as e:
        print(f"[skip] 训练依赖未安装（{e}）。请先 build_gen_sft，再在 GPU 环境安装 peft/trl 后重试。")
        print("本阶段已具备：意图微调 + 重排微调 + 生成 SFT 数据；LoRA 训练可在云端完成。")
        sys.exit(0)

    os.makedirs(OUT_DIR, exist_ok=True)
    ds = Dataset.from_list([_to_text(r) for r in train_rows])
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    lora = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)

    training_args = TrainingArguments(
        output_dir=OUT_DIR,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=10,
        save_steps=60,
        fp16=torch.cuda.is_available(),
        report_to=[],
    )
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        dataset_text_field="text",
        max_seq_length=1024,
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.model.save_pretrained(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)
    print(f"[save] LoRA adapter → {OUT_DIR}")


if __name__ == "__main__":
    main()
