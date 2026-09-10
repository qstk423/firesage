#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""QLoRA 最小可运行训练：Qwen2.5-3B-Instruct → 消防法规 LoRA adapter。

约定：
- 基座模型：backend/models/Qwen2.5-3B-Instruct（4-bit NF4 量化加载）
- 训练数据：backend/data/gen_sft_train.jsonl、gen_sft_dev.jsonl
  （由 scripts/build_gen_sft.py 生成；每条含 instruction / input / output_text）
- 对话格式：system = instruction，user = input，assistant = output_text
- loss 只对 assistant 段计算；system / user 段 labels 置 -100
- 截断策略（max_length=256 装不下全文时的保底）：
  优先保留完整 assistant 段（学习目标），剩余预算给 prompt 开头
  （instruction + 用户问题优先可见）。若直接按头截断，
  中位样本的答案只剩几个 token，学习信号会被截没。
- 产物：backend/adapters/firesage-qwen25-3b-lora（仅保存 adapter）
- 本脚本不修改 FireSage 主服务。

用法（在 backend 目录下执行，先关闭占显存的 chat_local.py 等会话）：
    ..\\venv311\\Scripts\\python.exe scripts\\train_lora_qwen3b.py
    ..\\venv311\\Scripts\\python.exe scripts\\train_lora_qwen3b.py --max-steps 2  # 冒烟测试
"""
from __future__ import annotations

import argparse
import json
import os
import sys

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BACKEND_DIR, "models", "Qwen2.5-3B-Instruct")
TRAIN_PATH = os.path.join(BACKEND_DIR, "data", "gen_sft_train.jsonl")
DEV_PATH = os.path.join(BACKEND_DIR, "data", "gen_sft_dev.jsonl")
DEFAULT_OUT = os.path.join(BACKEND_DIR, "adapters", "firesage-qwen25-3b-lora")


def _fail(msg: str) -> None:
    print(f"[错误] {msg}")
    sys.exit(1)


def check_paths(train_path: str, dev_path: str) -> None:
    """训练前检查模型与数据路径，缺失时给出清晰错误与修复提示。"""
    print("[检查] 模型与数据路径")
    if not os.path.isdir(MODEL_DIR):
        _fail(
            f"基座模型目录不存在：{MODEL_DIR}\n"
            f"       请将 Qwen2.5-3B-Instruct 完整模型放入该目录，"
            f"或建立指向本地 HF 缓存的目录链接（junction）。"
        )
    if not os.path.exists(os.path.join(MODEL_DIR, "config.json")):
        _fail(f"模型目录缺少 config.json：{MODEL_DIR}")
    if not any(
        os.path.exists(os.path.join(MODEL_DIR, f))
        for f in ("model.safetensors", "model.safetensors.index.json", "pytorch_model.bin")
    ):
        _fail(f"模型目录缺少权重文件（*.safetensors / *.bin）：{MODEL_DIR}")
    for label, path in (("训练集", train_path), ("验证集", dev_path)):
        if not os.path.exists(path):
            _fail(
                f"{label}不存在：{path}\n"
                f"       请先运行 python scripts/build_gen_sft.py 生成 SFT 数据，"
                f"并将 gen_sft_train.jsonl / gen_sft_dev.jsonl 复制到 data/ 目录。"
            )
    print(f"  [ok] 模型   {MODEL_DIR}")
    print(f"  [ok] 训练集 {train_path}")
    print(f"  [ok] 验证集 {dev_path}")


def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


class SFTDataset:
    """三段聊天格式数据集：只对 assistant 段计算 loss。

    transformers 5.x 的 apply_chat_template(tokenize=True) 返回字典，
    因此这里先渲染为文本（tokenize=False），再自行编码取 input_ids。
    """

    def __init__(self, rows: list[dict], tok, max_length: int):
        self.items: list[dict] = []
        self.truncated = 0
        for row in rows:
            prompt_msgs = [
                {"role": "system", "content": row["instruction"]},
                {"role": "user", "content": row["input"]},
            ]
            full_msgs = prompt_msgs + [
                {"role": "assistant", "content": row["output_text"]}
            ]
            prompt_text = tok.apply_chat_template(
                prompt_msgs, add_generation_prompt=True, tokenize=False
            )
            full_text = tok.apply_chat_template(
                full_msgs, add_generation_prompt=False, tokenize=False
            )
            prompt_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
            full_ids = tok(full_text, add_special_tokens=False)["input_ids"]
            # 正常模板下 full 以 prompt 为前缀；不一致说明模板异常，放弃该条
            if full_ids[: len(prompt_ids)] != prompt_ids:
                continue
            comp_ids = full_ids[len(prompt_ids):]

            if len(full_ids) > max_length:
                self.truncated += 1
                # 答案优先；答案本身超预算时截断并以 eos 收尾
                comp_budget = max_length - 1
                if len(comp_ids) > comp_budget:
                    comp_ids = comp_ids[: comp_budget - 1] + [tok.eos_token_id]
                prompt_kept = prompt_ids[: max(0, max_length - len(comp_ids))]
            else:
                prompt_kept = prompt_ids

            if not comp_ids:
                continue
            input_ids = prompt_kept + comp_ids
            # system/user 段不计算 loss
            labels = [-100] * len(prompt_kept) + list(comp_ids)
            self.items.append({"input_ids": input_ids, "labels": labels})

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict:
        return self.items[i]


def main() -> None:
    parser = argparse.ArgumentParser(description="QLoRA 训练（最小可运行版）")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--max-steps", type=int, default=-1,
                        help="调试用：只跑 N 步后停止（-1 表示跑完整 epoch）")
    parser.add_argument("--train", default=TRAIN_PATH, help="训练集 jsonl 路径")
    parser.add_argument("--dev", default=DEV_PATH, help="验证集 jsonl 路径")
    args = parser.parse_args()

    check_paths(args.train, args.dev)

    try:
        import torch
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            Trainer,
            TrainingArguments,
        )
    except ImportError as e:
        _fail(
            f"训练依赖未安装（{e}）。\n"
            f"       请在 venv311 中执行："
            f"pip install transformers peft accelerate bitsandbytes"
        )

    if not torch.cuda.is_available():
        _fail("未检测到可用 CUDA 显卡。QLoRA 4-bit 训练需要 NVIDIA GPU。")

    tok = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    train_ds = SFTDataset(load_jsonl(args.train), tok, args.max_length)
    dev_ds = SFTDataset(load_jsonl(args.dev), tok, args.max_length)
    print(
        f"[数据] train {len(train_ds)} 条（超长截断 {train_ds.truncated}），"
        f"dev {len(dev_ds)} 条（超长截断 {dev_ds.truncated}），"
        f"max_length={args.max_length}"
    )
    if len(train_ds) == 0:
        _fail("训练集为空：所有样本都无法构造出有效的 assistant 段。")

    # ---- 4-bit 量化 + QLoRA ----
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        quantization_config=bnb,
        device_map="auto",
        local_files_only=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    lora = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    def collate(batch: list[dict]) -> dict:
        maxlen = max(len(b["input_ids"]) for b in batch)
        input_ids, labels, attn = [], [], []
        for b in batch:
            n = len(b["input_ids"])
            pad = maxlen - n
            input_ids.append(b["input_ids"] + [tok.pad_token_id] * pad)
            labels.append(b["labels"] + [-100] * pad)
            attn.append([1] * n + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }

    targs = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        fp16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="epoch",
        logging_steps=2,
        save_strategy="no",
        report_to=[],
        seed=42,
    )
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        data_collator=collate,
    )
    trainer.train()

    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print(f"[完成] LoRA adapter 已保存 → {args.out}")
    print("       本次未修改 FireSage 主服务；接入验证另行进行。")


if __name__ == "__main__":
    main()
