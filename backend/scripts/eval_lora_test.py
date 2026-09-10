#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""独立评测：gen_sft_test.jsonl 上对比「未挂 LoRA」vs「挂载 LoRA adapter」。

不修改 FireSage 主服务与 .env，仅做只读评测。

评测设定：
- 基座模型：backend/models/Qwen2.5-3B-Instruct（bf16 加载）
- adapter：backend/adapters/firesage-qwen25-3b-lora
- 提示词：与训练格式一致（system=instruction，user=input，来自同一条测试数据）
- 解码：两次完全相同 —— do_sample=False（temperature=0）、max_new_tokens=450

评测维度（每条独立打分）：
1. json_valid     剥离 ```json 围栏 / 截取首尾花括号后可 json.loads
2. fields_complete conclusion/conditions/basis/supplement/confidence 五字段齐全
3. basis_is_array basis 为数组（训练目标格式）
4. basis_correct  金标 expected_articles 的条号（如"高层规定·第三条"的"第三条"）
                  出现在 basis 拼接文本中（仅统计有金标条款的样本；法规名存在
                  简称/全称差异，按条号匹配）
5. refuse_when_no_basis 金标 basis 为空的拒答类样本：生成答案 basis 为空，
                  或 conclusion 含拒答语义（超出/暂不回答/建议咨询/无法）
6. emergency_guide     emergency 类样本：conclusion 含"119"（应急引导不拒答）

产物：
- backend/eval_reports/compare_report.json  机器可读完整报告（含错误案例）
- backend/eval_reports/compare_report.md    人读版汇总 + 错误案例

用法（backend 目录下）：
    ..\\venv311\\Scripts\\python.exe scripts\\eval_lora_test.py
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BACKEND_DIR, "models", "Qwen2.5-3B-Instruct")
ADAPTER_DIR = os.path.join(BACKEND_DIR, "adapters", "firesage-qwen25-3b-lora")
TEST_PATH = os.path.join(BACKEND_DIR, "data", "gen_sft_test.jsonl")
REPORT_DIR = os.path.join(BACKEND_DIR, "eval_reports")

MAX_NEW_TOKENS = 450
FIELDS = ["conclusion", "conditions", "basis", "supplement", "confidence"]
REFUSE_PAT = re.compile(r"超出|暂不回答|建议咨询|无法回答|不属于")
ARTICLE_NUM_PAT = re.compile(r"第[一二三四五六七八九十百零]+条")


def _fail(msg: str) -> None:
    print(f"[错误] {msg}")
    raise SystemExit(1)


def check_paths(adapter_dir: str) -> None:
    for label, path in [("基座模型", MODEL_DIR), ("LoRA adapter", adapter_dir), ("测试集", TEST_PATH)]:
        if not os.path.exists(path):
            _fail(f"{label}不存在：{path}")


def load_jsonl(path: str) -> list[dict]:
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def extract_json(text: str):
    """宽松解析：剥 ``` 围栏 → 直接 loads → 截取首尾花括号重试。"""
    t = text.strip()
    fence = t.startswith("```")
    t = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", t).strip()
    try:
        return json.loads(t), fence
    except Exception:
        pass
    i, j = t.find("{"), t.rfind("}")
    if i >= 0 and j > i:
        try:
            return json.loads(t[i:j + 1]), fence
        except Exception:
            pass
    return None, fence


def expected_article_nums(row: dict) -> list[str]:
    """expected_articles 如 '高层规定·第三条' → 提取条号 '第三条'。"""
    nums = []
    for key in row.get("expected_articles") or []:
        parts = key.split("·")
        if len(parts) == 2 and ARTICLE_NUM_PAT.fullmatch(parts[1].strip()):
            nums.append(parts[1].strip())
    return nums


def score_row(row: dict, raw_output: str) -> dict:
    """对单条生成结果打分。"""
    data, fenced = extract_json(raw_output)
    s = {
        "json_valid": data is not None,
        "fenced": fenced,          # 风格指标：是否带 ``` 围栏
        "fields_complete": False,
        "basis_is_array": False,
        "basis_correct": None,     # None=不适用（无金标条款）
        "refuse_when_no_basis": None,
        "emergency_guide": None,
    }
    gold = row["output"]
    if not isinstance(data, dict):
        return s
    s["fields_complete"] = all(f in data for f in FIELDS)
    basis = data.get("basis")
    s["basis_is_array"] = isinstance(basis, list)
    basis_text = " ".join(str(x) for x in basis) if isinstance(basis, list) else str(basis or "")

    nums = expected_article_nums(row)
    if nums:
        s["basis_correct"] = all(n in basis_text for n in nums)

    gold_basis = gold.get("basis") or []
    if not gold_basis:  # 拒答类：金标无依据
        conclusion = str(data.get("conclusion", ""))
        s["refuse_when_no_basis"] = (not basis) or bool(REFUSE_PAT.search(conclusion))
    if row.get("intent") == "emergency":
        # 语义判定：不强制"119"在 conclusion，检查 conclusion+supplement 全文
        # 三要素：立即安全措施 / 疏散或远离危险 / 必要时报警求助
        full = str(data.get("conclusion", "")) + " " + str(data.get("supplement", ""))
        safety = bool(re.search(r"立即|马上|迅速|尽快|第一时间|切断|关闭|开窗|捂住|低姿|匍匐|断电|停用", full))
        evacuate = bool(re.search(r"疏散|撤离|远离|逃生|离开|避开|不[^，。]{0,8}(电梯|扶梯)", full))
        alarm = bool(re.search(r"119|报警|求助|呼救|救援", full))
        s["emergency_guide"] = safety and evacuate and alarm
    return s


def aggregate(per_rows: list[dict]) -> dict:
    """汇总通过率；None（不适用）不计入分母。"""
    out = {}
    for key in ["json_valid", "fields_complete", "basis_is_array", "basis_correct",
                "refuse_when_no_basis", "emergency_guide"]:
        valid = [r["scores"][key] for r in per_rows if r["scores"][key] is not None]
        out[key] = {"pass": sum(valid), "n": len(valid),
                    "rate": round(sum(valid) / len(valid), 4) if valid else None}
    out["fenced_ratio"] = round(sum(1 for r in per_rows if r["scores"]["fenced"]) / len(per_rows), 4)
    return out


def generate_all(model, tok, rows: list[dict], max_new_tokens: int) -> list[str]:
    outs = []
    for i, row in enumerate(rows):
        msgs = [
            {"role": "system", "content": row["instruction"]},
            {"role": "user", "content": row["input"]},
        ]
        prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        inputs = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
        import torch
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,          # temperature=0
                pad_token_id=tok.eos_token_id,
            )
        text = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        outs.append(text)
        print(f"    [{i + 1}/{len(rows)}] {len(text)} 字符", flush=True)
    return outs


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA 前后对比评测（test 集）")
    parser.add_argument("--adapter", default=ADAPTER_DIR, help="LoRA adapter 目录")
    parser.add_argument("--tag", default="", help="报告文件名后缀（如 _v2）")
    parser.add_argument("--baseline-from", default="",
                        help="复用既有报告的基座输出（JSON 路径），跳过基座生成阶段")
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS,
                        help=f"生成 token 上限（默认 {MAX_NEW_TOKENS}）")
    args = parser.parse_args()

    check_paths(args.adapter)
    rows = load_jsonl(TEST_PATH)
    print(f"[数据] 测试集 {len(rows)} 条 ← {TEST_PATH}")
    print(f"[配置] adapter = {args.adapter}")

    os.environ["HF_HUB_OFFLINE"] = "1"
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, local_files_only=True, dtype=torch.bfloat16, device_map="cuda"
    ).eval()

    if args.baseline_from:
        # 复用既有报告的基座输出：temperature=0 下确定性生成，直接复用保证可比
        with open(args.baseline_from, encoding="utf-8") as f:
            prev = json.load(f)
        prev_base = {r["id"]: r["output"] for r in prev["per_row"]["baseline"]}
        if set(prev_base) != {r["id"] for r in rows}:
            _fail("复用的基座报告与当前测试集样本不一致，拒绝复用")
        base_outs = [prev_base[r["id"]] for r in rows]
        base_secs = prev["summary"]["gen_seconds"]["baseline"]
        print(f"[阶段1] 复用基座输出（{len(base_outs)} 条）← {args.baseline_from}")
    else:
        print("[阶段1] 基座模型生成（未挂 LoRA）…", flush=True)
        t0 = time.time()
        base_outs = generate_all(model, tok, rows, args.max_new_tokens)
        base_secs = time.time() - t0

    print("[阶段2] 挂载 LoRA adapter 生成…", flush=True)
    model = PeftModel.from_pretrained(model, args.adapter).eval()
    t0 = time.time()
    lora_outs = generate_all(model, tok, rows, args.max_new_tokens)
    lora_secs = time.time() - t0

    print("[阶段3] 打分汇总…")
    results = {"baseline": [], "lora": []}
    for row, raw in zip(rows, base_outs):
        results["baseline"].append({"id": row["id"], "intent": row.get("intent"),
                                    "question": row["input"].split("\n")[0].replace("用户问题：", ""),
                                    "scores": score_row(row, raw), "output": raw})
    for row, raw in zip(rows, lora_outs):
        results["lora"].append({"id": row["id"], "intent": row.get("intent"),
                                "question": row["input"].split("\n")[0].replace("用户问题：", ""),
                                "scores": score_row(row, raw), "output": raw})

    summary = {
        "baseline": aggregate(results["baseline"]),
        "lora": aggregate(results["lora"]),
        "gen_seconds": {"baseline": round(base_secs, 1), "lora": round(lora_secs, 1)},
    }

    # ---- 错误案例（LoRA 侧任一适用维度未通过 即收录；同时收录基座失败对比）----
    errors = []
    for b, l in zip(results["baseline"], results["lora"]):
        failed = [k for k in ["json_valid", "fields_complete", "basis_is_array",
                              "basis_correct", "refuse_when_no_basis", "emergency_guide"]
                  if l["scores"][k] is False]
        if failed:
            gold_row = next(r for r in rows if r["id"] == l["id"])
            errors.append({
                "id": l["id"], "question": l["question"], "failed_dims": failed,
                "gold_output_text": gold_row["output_text"][:400],
                "baseline_output": b["output"][:400],
                "lora_output": l["output"][:400],
            })

    os.makedirs(REPORT_DIR, exist_ok=True)
    with open(os.path.join(REPORT_DIR, f"compare_report{args.tag}.json"), "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "errors": errors,
                   "per_row": results, "config": {
                       "model": MODEL_DIR, "adapter": args.adapter,
                       "max_new_tokens": args.max_new_tokens, "temperature": 0}},
                  f, ensure_ascii=False, indent=2)

    # ---- 人读报告 ----
    md = [
        f"# LoRA 微调前后对比评测报告（gen_sft_test.jsonl，n=35）{'（v2）' if args.tag else ''}",
        "",
        f"- adapter：`{args.adapter}`",
        f"- 解码参数：do_sample=False（temperature=0），max_new_tokens={args.max_new_tokens}",
        f"- 生成耗时：基座 {base_secs:.0f}s / LoRA {lora_secs:.0f}s",
        "",
        "## 汇总（通过数/适用样本数）",
        "",
        "| 维度 | 基座(未挂LoRA) | LoRA |",
        "|---|---|---|",
    ]
    keys = ["json_valid", "fields_complete", "basis_is_array", "basis_correct",
            "refuse_when_no_basis", "emergency_guide", "fenced_ratio"]
    for k in keys:
        vb, vl = summary["baseline"][k], summary["lora"][k]
        def cell(v):
            if k == "fenced_ratio":
                return f"{v * 100:.0f}%"
            rate = f"{v['rate'] * 100:.0f}%" if v["rate"] is not None else "—"
            return f"{v['pass']}/{v['n']}（{rate}）"
        md.append(f"| {k} | {cell(vb)} | {cell(vl)} |")
    md += [
        "",
        f"## 错误案例（LoRA 侧，共 {len(errors)} 条）",
        "",
    ]
    for e in errors:
        md.append(f"### {e['id']}：{e['question']}")
        md.append(f"- 未过维度：{', '.join(e['failed_dims'])}")
        md.append(f"- 金标：`{e['gold_output_text'][:200]}…`")
        md.append(f"- LoRA 输出：`{e['lora_output'][:200]}…`")
        md.append("")
    with open(os.path.join(REPORT_DIR, f"compare_report{args.tag}.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n[完成] 报告 → {REPORT_DIR}\\compare_report{args.tag}.md / .json（错误案例 {len(errors)} 条）")


if __name__ == "__main__":
    main()
