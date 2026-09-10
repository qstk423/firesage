#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成 v1/v2/v3 三版对比报告（统一口径：700 tokens + 语义化应急标准 + 基座复用）。"""
import json
import os

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RPT = os.path.join(BACKEND, "eval_reports")


def load(tag):
    with open(os.path.join(RPT, f"compare_report{tag}.json"), encoding="utf-8") as f:
        return json.load(f)


reps = {
    "v1": load("_v1_r"),
    "v2": load("_v2_r"),
    "v3": load("_v3"),
}

keys = ["json_valid", "fields_complete", "basis_is_array", "basis_correct",
        "refuse_when_no_basis", "emergency_guide"]


def cell(v):
    if v["rate"] is None:
        return "—"
    return f"{v['pass']}/{v['n']}（{v['rate'] * 100:.0f}%）"


errs = {k: {e["id"] for e in r["errors"]} for k, r in reps.items()}
md = [
    "# LoRA v1 vs v2 vs v3 对比报告",
    "",
    "统一评测口径：同一 35 条 test 集、temperature=0、max_new_tokens=700、",
    "语义化应急标准（立即安全措施 + 疏散/远离危险 + 报警求助，119 可在 supplement）、",
    "基座输出复用同一份（450 tokens 生成，仅作参考列）。",
    "",
    "## 训练配置",
    "",
    "| 项 | v1 | v2 | v3 |",
    "|---|---|---|---|",
    "| 训练集 | 160 条 | 182 条（+15 拒答 +7 应急） | 192 条（v2 +10 拒答） |",
    "| max_length / epochs | 256 / 1 | 512 / 2 | 512 / 2 |",
    "| eval_loss(dev 34 条) | 1.007 | 0.255 | 0.236 |",
    "| adapter 目录 | lora | lora-v2 | lora-v3 |",
    "",
    "## 指标对比（700 tokens，新应急标准）",
    "",
    "| 维度 | 基座* | v1 | v2 | v3 |",
    "|---|---|---|---|---|",
]
for k in keys:
    b = reps["v1"]["summary"]["baseline"][k]
    md.append(f"| {k} | {cell(b)} | {cell(reps['v1']['summary']['lora'][k])} | "
              f"{cell(reps['v2']['summary']['lora'][k])} | {cell(reps['v3']['summary']['lora'][k])} |")
md += [
    "",
    f"*基座列为 450 tokens 生成（复用 v1 首次评测），与三版 LoRA 的 700 tokens 不可严格比较。",
    "",
    "## 错误案例（LoRA 侧）",
    "",
    f"- v1：{len(errs['v1'])} 条 → {sorted(errs['v1'])}",
    f"- v2：{len(errs['v2'])} 条 → {sorted(errs['v2'])}",
    f"- v3：{len(errs['v3'])} 条 → {sorted(errs['v3'])}",
    "",
    "## 关键发现",
    "",
    "1. **450→700 tokens 修复了 v2 的截断假阴性**：v2 json_valid 94%→100%，basis_is_array 94%→100%。",
    "   v2 的 json_valid 下降确系输出截断，非模型格式退化。",
    "2. **v3 净退步**：拒答 40%→0%、应急 33%→0%、错误案例 11→14。",
    "   模型自创\"拒答方言\"（conclusion=\"暂无适用条款\"+basis=[\"无\"]+confidence=\"95\"），",
    "   而非金标模板（conclusion=超出范围话术+basis=[]）。",
    "3. **v3 退步根因**：30 条同模板拒答样本（占 16%）+ 2 epochs 过拟合",
    "   （eval_loss 0.236 < v2 0.255，拟合更深但泛化更差）。",
    "   过强的一致性信号让 3B 模型在\"无可用条款\"输入上模式崩塌，",
    "   挤掉了 v2 已学会的应急引导（e15 从通过变失败）。",
    "4. **x15 在 700 tokens 仍截断**：v3 将法条全文堆入 basis，输出超 700 tokens，",
    "   系\"啰嗦\"模式而非截断上限问题。",
    "5. **v2 是当前最优版本**：格式纪律满分 + 引用正确率 78% + 拒答 40% + 应急 33%。",
    "",
    "## v4 建议（未执行）",
    "",
    "1. 拒答金标模板多样化（3~4 种表述轮换），防止单一模板过拟合；",
    "2. 回退到 1 epoch（v1 的配置）或降低学习率至 1e-4；",
    "3. 应急样本扩充至 15+ 条并覆盖\"火势蔓延\"\"电梯困人\"等 test 外场景；",
    "4. basis 中法条原文截断为前 80 字符，抑制 x15 型长答案。",
]

out = os.path.join(RPT, "compare_v1_v2_v3.md")
with open(out, "w", encoding="utf-8") as f:
    f.write("\n".join(md))
print("saved →", out)
for k in reps:
    print(f"{k}: errors={len(errs[k])}")
