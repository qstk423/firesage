# -*- coding: utf-8 -*-
"""分析 35 题正误案例的检索分数分布，计算拒答门槛。

数据源：
- eval_reports/replay_retrieval.json ：回放检索（BGE-M3 CPU + CrossEncoder），
  含每题 top1 fused score 与 cross_encoder 分数；
- eval_reports/e2e_local_test.json   ：上次端到端结果（passed / 失败原因）。

分组（人工复核上次报告）：
- BAD_CONTENT  内容错位（条号错/知识库外硬答）—— 门槛+支撑检查应拦截的目标
- FORMAT_ONLY  仅 basis 格式失败，内容正确 —— 不应误伤
- NO_RETRIEVAL 空检索/误路由（c20/py10 已由意图与 FIRE_WORDS 修复）—— 有分数的看分布
- SHOULD_REFUSE o04/o07/o14/f15：本应拒答的域外/边界题 —— 门槛应拦截
"""
import io
import json
import os
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

with open(os.path.join(BACKEND_DIR, "eval_reports", "replay_retrieval.json"), encoding="utf-8") as f:
    replay = {r["id"]: r for r in json.load(f)}
with open(os.path.join(BACKEND_DIR, "eval_reports", "e2e_local_test.json"), encoding="utf-8") as f:
    rows = {r["id"]: r for r in json.load(f)["rows"]}

BAD_CONTENT = {"f12", "d08", "c11", "d07", "o14"}
FORMAT_ONLY = {"d27", "f04", "p29", "py22", "py23", "p28", "d24", "x15"}
NO_RETRIEVAL = {"c20", "o07", "py10"}          # 误路由（已修复），看分数分布
SHOULD_REFUSE = {"o04", "o07", "o14", "f15"}    # 域外/边界：门槛应拦截


def top1(rid):
    r = replay.get(rid)
    if not r or not r.get("tops"):
        return None, None
    t = r["tops"][0]
    return t["score"], t.get("ce")


def show(ids, label):
    print(f"\n[{label}]")
    for rid in sorted(ids):
        s, ce = top1(rid)
        if s is None:
            print(f"  {rid:6s} 无检索分数")
            continue
        row = rows.get(rid, {})
        mark = "PASS" if row.get("passed") else "FAIL"
        ce_s = f"{ce:.3f}" if ce is not None else " None"
        print(f"  {rid:6s} fused={s:.3f} ce={ce_s} {mark} {row.get('question', '')[:24]}")


law_all = [rid for rid, r in rows.items() if r.get("intent") == "law"]
good = [rid for rid in law_all
        if rid not in BAD_CONTENT | FORMAT_ONLY | NO_RETRIEVAL | SHOULD_REFUSE
        and rows[rid].get("passed")]

print("=== top1 分数分布（回放：BGE-M3 CPU + CrossEncoder）===")
show(sorted(SHOULD_REFUSE), "本应拒答（域外/边界）")
show(sorted(BAD_CONTENT), "内容错位（检索分数正常，错在生成）")
show(sorted(FORMAT_ONLY), "仅格式失败（内容正确，不应误伤）")
show(sorted(NO_RETRIEVAL), "误路由（c20/py10 已修复）")
show(sorted(good), f"正确案例（law 通过，n={len(good)}）")

print("\n=== 联合门槛敏感性（fused<TH_f 且 ce<TH_c → 拒答）===")
print("TH_f  TH_c   拦截应拒答(4)   误伤正确   误伤格式类(8)   误伤内容错位(5)")
for thf, thc in [(0.60, 0.55), (0.62, 0.58), (0.65, 0.58), (0.65, 0.60),
                 (0.68, 0.62), (0.70, 0.65), (0.72, 0.68)]:
    def below(rid):
        s, ce = top1(rid)
        return s is not None and ce is not None and s < thf and ce < thc
    hit_refuse = sum(1 for r in SHOULD_REFUSE if below(r))
    hurt_good = sum(1 for r in good if below(r))
    hurt_fmt = sum(1 for r in FORMAT_ONLY if below(r))
    hurt_bad = sum(1 for r in BAD_CONTENT if below(r))
    print(f"{thf:.2f}  {thc:.2f}   {hit_refuse}/4             {hurt_good}/{len(good)}        "
          f"{hurt_fmt}/8             {hurt_bad}/5")

print("\n说明：内容错位组的检索分数普遍正常（生成错误而非检索错误），")
print("由「原文支撑检查 + 引用守卫」拦截，不在检索门槛职责内。")
