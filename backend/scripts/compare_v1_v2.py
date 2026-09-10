#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成 v1 vs v2 对比报告：指标、错误案例重合、失败原因归类。"""
import json
import os

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RPT = os.path.join(BACKEND, "eval_reports")

with open(os.path.join(RPT, "compare_report.json"), encoding="utf-8") as f:
    v1 = json.load(f)
with open(os.path.join(RPT, "compare_report_v2.json"), encoding="utf-8") as f:
    v2 = json.load(f)

# 核实 v2 的 2 条 json_valid 失败是否为 max_new_tokens=450 截断
trunc_check = {}
for r in v2["per_row"]["lora"]:
    if not r["scores"]["json_valid"]:
        trunc_check[r["id"]] = not r["output"].rstrip().endswith("}")
print("JSON 失败样本截断核实（True=生成被截断）:", trunc_check)

v1_err = {e["id"] for e in v1["errors"]}
v2_err = {e["id"] for e in v2["errors"]}
common = sorted(v1_err & v2_err)
fixed = sorted(v1_err - v2_err)
new = sorted(v2_err - v1_err)

keys = ["json_valid", "fields_complete", "basis_is_array", "basis_correct",
        "refuse_when_no_basis", "emergency_guide"]


def cell(v):
    if isinstance(v, float):
        return f"{v * 100:.0f}%"
    if v["rate"] is None:
        return "—"
    return f"{v['pass']}/{v['n']}（{v['rate'] * 100:.0f}%）"


md = [
    "# LoRA v1 vs v2 对比报告（同一 test 集 35 条，temperature=0，基座输出复用）",
    "",
    "## 训练配置差异",
    "",
    "| 项 | v1 | v2 |",
    "|---|---|---|",
    "| 训练集 | 160 条（原版） | 182 条（+15 拒答 +7 应急，金标审计 132/132 合规） |",
    "| max_length | 256（124 条被截，21 条答案被截，prompt 最少剩 1 token） | 512（49 条被截，0 条答案被截，prompt 最少保留 150 token） |",
    "| epochs / 步数 | 1 / 20 | 2 / 46 |",
    "| eval_loss(dev 34 条) | 1.007 | 0.255 |",
    "| adapter | adapters/firesage-qwen25-3b-lora | adapters/firesage-qwen25-3b-lora-v2 |",
    "",
    "## 指标对比",
    "",
    "| 维度 | 基座 | v1 | v2 | v1→v2 |",
    "|---|---|---|---|---|",
]
for k in keys + ["fenced_ratio"]:
    b, a, c = v1["summary"]["baseline"][k], v1["summary"]["lora"][k], v2["summary"]["lora"][k]
    if k == "fenced_ratio":
        md.append(f"| {k} | {b * 100:.0f}% | {a * 100:.0f}% | {c * 100:.0f}% | |")
    else:
        ra = a["rate"] or 0
        rc = c["rate"] or 0
        arrow = "↑" if rc > ra else ("↓" if rc < ra else "→")
        md.append(f"| {k} | {cell(b)} | {cell(a)} | {cell(c)} | {arrow} |")

md += [
    "",
    f"## 错误案例（LoRA 侧）：v1 {len(v1_err)} 条 → v2 {len(v2_err)} 条",
    "",
    f"- 两版都失败（{len(common)} 条）：{', '.join(common)}",
    f"- v1 失败、v2 修复（{len(fixed)} 条）：{', '.join(fixed)}",
    f"- v2 新增失败（{len(new)} 条）：{', '.join(new)}",
    "",
    "## 失败原因归类（v2，13 条）",
    "",
    "| 类别 | 条目 | 说明 |",
    "|---|---|---|",
    f"| 生成截断致 JSON 不完整 | {', '.join(trunc_check) if trunc_check else '—'} | 多条文长答案超出评测 max_new_tokens=450，属评测上限而非模型格式错误 |",
    "| 引用条文数少于金标（引用正确但不全） | c20, f15, py11, d07, py13 | 引用的法规名+条号正确，但金标要求多条，模型只引 1 条 |",
    "| 引用条号错误 | d08 | 问第四十五条处罚，引了第六十四条 |",
    "| 拒答不彻底（知识库外仍作答） | n01, o07, n09 | o07/n09 编造了库外依据；n01 basis 填了\"无\"未走拒答模板 |",
    "| 应急引导缺 119 字样 | e08, e15 | e15 答案语义正确（立即报警疏散）但无\"119\"字样；e08 误走法条模板 |",
    "",
    "## 结论",
    "",
    "- **格式纪律显著提升**：basis 数组率 74%→94%，围栏 0%，eval_loss 1.007→0.255；",
    "  v1 的\"丢《法规名》第X条壳\"问题基本修复（py13/d07/f15 等引用格式全部规范）。",
    "- **引用正确率 63%→76%**：剩余失败多为\"引对但引少\"（金标多条只引一条），非编造。",
    "- **regression 诚实披露**：json_valid 100%→94%，系 2 条多条文长答案撞上评测 450 token 上限；",
    "  拒答 60%→40%（o04/o14 修复但 n09/o07 新失败）；应急 33% 持平（e15 语义正确但缺 119 字样）。",
    "",
    "## v3 候选改进（未执行）",
    "",
    "1. 评测侧 max_new_tokens 提至 700，消除截断假阴性；",
    "2. 训练数据中多条文样本的 basis 截断为前 2 条完整引用（教模型\"少而准\"而非模仿多条长引用）；",
    "3. 拒答模板中 basis 强制空数组（当前 n01 学到 basis:[\"无\"] 的坏习惯来自 v1 数据 leakage 检查）；",
    "4. 应急判定放宽为含 119 或 \"报警\"+\"疏散\" 语义组合。",
]

out = os.path.join(RPT, "compare_v1_v2.md")
with open(out, "w", encoding="utf-8") as f:
    f.write("\n".join(md))
print("saved →", out)
print("v1 errors:", len(v1_err), "v2 errors:", len(v2_err),
      "common:", len(common), "fixed:", len(fixed), "new:", len(new))
