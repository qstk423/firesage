#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""端到端联调测试：用户问题 → RAG 检索 → Qwen+LoRA（8320）→ 五层守卫 → API 响应。

测试集：data/gen_sft_test.jsonl（35 条独立 test 集）。
前置：run_firesage_local.py 已启动（127.0.0.1:8321，BGE 走 CPU）。

每条记录：回答、检索条款（references）、守卫触发（llm_protections）、总耗时。
附加检测「条号匹配但内容不支撑结论」：
  - 守卫介入信号：citation_guard / citation_retry / citation_rejected；
  - 数字无出处：结论中的处罚/量级数字既不在引用条款原文、也不在问题里；
  - 零词重叠：问题关键词与引用条款原文完全无重叠（弱信号）。

产物：backend/eval_reports/e2e_local_test.json

用法（backend 目录，服务就绪后）：
    ..\\venv311\\Scripts\\python.exe scripts\\e2e_local_test.py
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.request

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_PATH = os.path.join(BACKEND_DIR, "data", "gen_sft_test.jsonl")
DATA_DIR = os.path.join(BACKEND_DIR, "data")
OUT_PATH = os.path.join(BACKEND_DIR, "eval_reports", "e2e_local_test.json")

BASE = "http://127.0.0.1:8321"
FIELDS = ["conclusion", "conditions", "basis", "supplement", "confidence"]
CITATION_PAT = re.compile(r"《.+?》第[一二三四五六七八九十百零\d]+条")
ART_NUM_PAT = re.compile(r"第([一二三四五六七八九十百零\d]+)条")
NUM_PAT = re.compile(r"\d+(?:\.\d+)?")
CN_NUM_PAT = re.compile(r"[一二三四五六七八九十百]+(?:万|千|百)?元")
REFUSE_PAT = re.compile(r"超出|暂不回答|无法|不属于|范围外|不能作答|不予作答")
ALARM_PAT = re.compile(r"119|报警")
EVAC_PAT = re.compile(r"疏散|撤离|逃生|远离|电梯")
# 检索相关性弱信号用停用词（问题里高频但条款里未必有的泛化词）
STOPWORDS = set("什么 怎么 如何 哪些 可以 应该 不是 没有 关于 下面 请问 一下"
                " 要求 规定 情况 问题时候 进行 对于 出现 属于 具有".split())


def post_ask(question: str, timeout: int = 300) -> dict:
    payload = json.dumps({"question": question}).encode("utf-8")
    req = urllib.request.Request(
        BASE + "/api/ask", data=payload,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def get(path: str, timeout: int = 15) -> tuple[int, str]:
    with urllib.request.urlopen(BASE + path, timeout=timeout) as resp:
        return resp.status, resp.read().decode("utf-8", errors="replace")


def load_test_rows() -> list[dict]:
    rows = []
    with open(TEST_PATH, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def extract_question(row: dict) -> str:
    first = (row.get("input") or "").split("\n", 1)[0]
    return re.sub(r"^用户问题[：:]", "", first).strip()


def basis_article_nums(basis: list[str]) -> list[str]:
    """basis 引用中的条号（用于与 references 提供条款比对）。"""
    nums = []
    for b in basis:
        m = ART_NUM_PAT.search(b)
        if m:
            nums.append(m.group(0))
    return nums


def find_ref_text_by_num(num: str, references: list[dict]) -> str:
    """在检索提供的条款中找同条号的原文（跨法规按条号匹配，足够检测数字出处）。"""
    for ref in references or []:
        if num in (ref.get("article") or ""):
            return ref.get("text") or ""
    return ""


def detect_unsupported(question: str, conclusion: str, basis: list[str],
                       references: list[dict], guards: list[str]) -> list[str]:
    """「条号匹配但内容不支撑结论」检测，返回标记列表。"""
    marks = []
    if any(g.startswith("citation_") for g in guards):
        marks.append("守卫介入（模型原引用与提供条款不匹配：" +
                     "/".join(g for g in guards if g.startswith("citation_")) + "）")
    cited_text = "".join(find_ref_text_by_num(n, references)
                         for n in basis_article_nums(basis))
    # 数字无出处：结论数字 ⊄（引用原文 ∪ 问题）
    if cited_text:
        q_nums = set(NUM_PAT.findall(question)) | set(CN_NUM_PAT.findall(question))
        c_nums = set(NUM_PAT.findall(conclusion)) | set(CN_NUM_PAT.findall(conclusion))
        stray = {n for n in c_nums - q_nums if n not in cited_text}
        if stray:
            marks.append(f"数字无出处（{sorted(stray)[:5]}）")
        # 零词重叠：问题实词与引用原文完全无交集
        terms = {w for w in re.findall(r"[\u4e00-\u9fa5]{2,}", question)
                 if w not in STOPWORDS}
        if terms and not any(t in cited_text for t in terms):
            marks.append("问题与引用原文零词重叠（弱信号）")
    return marks


def score_row(row: dict, resp: dict) -> tuple[bool, str]:
    """检查结构、引用来源和拒答/应急规则，不把金标命中混入契约检查。"""
    intent = row.get("intent")
    structured = resp.get("structured") or {}
    answer = resp.get("answer") or ""
    if intent == "law":
        # 库外题（expected_articles 为空且金标无条款）：正确拒答为通过，
        # 硬答反而失败（宁可拒答不可编造）
        if not row.get("expected_articles"):
            if resp.get("refused"):
                return True, "OK（库外题，管线拒答）"
            basis = structured.get("basis")
            if basis == [] and REFUSE_PAT.search(answer):
                return True, "OK（库外题，LLM 规范拒答）"
            return False, f"库外题未拒答（refused={resp.get('refused')}，" \
                          f"crag={resp.get('crag')}）"
        basis = structured.get("basis")
        if not isinstance(basis, list) or not basis:
            return False, "basis 为空"
        if not CITATION_PAT.search(" ".join(basis)):
            return False, "basis 缺《法规名》第X条格式"
        refs_nums = [r.get("article", "") for r in resp.get("references") or []]
        stray = [n for n in basis_article_nums(basis)
                 if not any(n in rn for rn in refs_nums)]
        if stray:
            return False, f"basis 条号不在检索提供的条款内：{stray}"
        return True, "OK"
    if intent == "refuse":
        if resp.get("refused"):
            return True, "OK（管线拒答）"
        basis = structured.get("basis")
        if basis == [] and REFUSE_PAT.search(answer):
            return True, "OK（LLM 规范拒答）"
        return False, f"未拒答（refused={resp.get('refused')}，basis={str(basis)[:60]}）"
    if intent == "emergency":
        if not ALARM_PAT.search(answer):
            return False, "缺报警要素"
        if not EVAC_PAT.search(answer):
            return False, "缺疏散要素"
        return True, "OK（应急要素齐全）"
    if intent == "chitchat":
        if resp.get("intent") == "guide" and not resp.get("refused"):
            return True, "OK（引导到可问范围）"
        return False, f"intent={resp.get('intent')}，未按引导处理"
    return False, "未知 intent"


def hits_expected_article(row: dict, resp: dict) -> bool | None:
    """是否至少命中一条期望法规（法规名+条号）；无金标返回 None。

    不能只比“第七条”这类条号，否则《消防法》第七条会被误算成
    《高层民用建筑消防安全管理规定》第七条。
    """
    expected = row.get("expected_articles") or []
    if not expected:
        return None
    basis_items = [str(item) for item in (
        (resp.get("structured") or {}).get("basis") or [])]
    references = resp.get("references") or []

    def compact(text: str) -> str:
        return re.sub(r"[\s《》（）()·]", "", text or "")

    for key in expected:
        article_num = key.split("·")[-1]
        reference = next(
            (ref for ref in references if ref.get("article") == key), None)
        if not reference:
            continue
        law_name = reference.get("law_name") or key.split("·")[0]
        target_law = compact(law_name)
        if any(article_num in item and target_law in compact(item)
               for item in basis_items):
            return True
    return False


def _percentile(values: list[float], ratio: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[round((len(ordered) - 1) * ratio)], 2)


def summarize_timings(rows: list[dict]) -> dict:
    """汇总 API 返回的分段耗时；缺失字段不按 0 计算。"""
    series: dict[str, list[float]] = {}
    for row in rows:
        timing = row.get("timing") or {}
        for key, value in timing.items():
            if key == "retrieval" and isinstance(value, dict):
                for child_key, child_value in value.items():
                    if isinstance(child_value, (int, float)):
                        series.setdefault(f"retrieval.{child_key}", []).append(float(child_value))
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                series.setdefault(key, []).append(float(value))
    return {
        key: {
            "n": len(values),
            "p50": _percentile(values, 0.50),
            "p95": _percentile(values, 0.95),
            "max": round(max(values), 2),
        }
        for key, values in sorted(series.items())
    }


def main() -> None:
    # ---- 前置检查：服务 + 前端静态资源 ----
    status, system = get("/api/system")
    info = json.loads(system)
    print(f"[系统] ready={info.get('ready')} tier={info['runtime']['quality_tier']} "
          f"llm={info['runtime']['llm_enabled']} vector_degraded={info['runtime']['vector_degraded']} "
          f"cross_encoder={info['runtime']['cross_encoder']}")
    assert info["runtime"]["llm_enabled"], "LLM 未启用（.env.local-model 未生效？）"
    assert not info["runtime"]["vector_degraded"], "向量索引降级为 TF-IDF（bge-m3 未加载？）"
    st, html = get("/")
    assert st == 200 and "api/ask" in html, "前端页面异常"
    print(f"[前端] / -> {st}（index.html 含 /api/ask 调用）")

    rows = load_test_rows()
    print(f"[测试集] {len(rows)} 条（law/refuse/emergency/chitchat 分布见结果）")

    results = []
    t_all = time.time()
    for i, row in enumerate(rows, 1):
        q = extract_question(row)
        t0 = time.time()
        try:
            resp = post_ask(q)
        except Exception as e:
            results.append({"id": row.get("id"), "intent": row.get("intent"),
                            "question": q, "error": str(e),
                            "seconds": round(time.time() - t0, 1)})
            print(f"  [{i}/{len(rows)}] {q[:24]} → 调用失败：{e}")
            continue
        secs = round(time.time() - t0, 1)
        structured = resp.get("structured") or {}
        guards = structured.get("llm_protections") or []
        contract_ok, msg = score_row(row, resp)
        gold_hit = hits_expected_article(row, resp)
        # 严格通过：先满足格式/守卫契约；有金标的法规题还必须至少命中一条金标。
        strict_ok = contract_ok and not (
            row.get("intent") == "law" and gold_hit is False)
        if contract_ok and not strict_ok:
            msg = "未命中金标条款"
        unsupported = (detect_unsupported(
            q, structured.get("conclusion") or "", structured.get("basis") or [],
            resp.get("references") or [], guards) if row.get("intent") == "law" else [])
        rec = {
            "id": row.get("id"),
            "intent": row.get("intent"),
            "question": q,
            "passed": strict_ok,
            "contract_passed": contract_ok,
            "gold_hit": gold_hit,
            "message": msg,
            "seconds": secs,
            "latency_ms": resp.get("latency_ms"),
            "trace_id": resp.get("trace_id"),
            "ttft_ms": (resp.get("timing") or {}).get("ttft_ms"),
            "timing": resp.get("timing") or {},
            "api_intent": resp.get("intent"),
            "refused": resp.get("refused"),
            "crag": resp.get("crag"),
            "strategy": resp.get("strategy"),
            "conclusion": structured.get("conclusion"),
            "basis": structured.get("basis"),
            "confidence": structured.get("confidence"),
            "guards": guards,
            "verification_passed": (resp.get("verification") or {}).get("passed"),
            "verification_issues": (resp.get("verification") or {}).get("issues"),
            "retrieved_articles": [r.get("article") for r in resp.get("references") or []][:5],
            "retrieved_scores": [r.get("score") for r in resp.get("references") or []][:5],
            "unsupported_marks": unsupported,
            "expected_articles": row.get("expected_articles"),
        }
        results.append(rec)
        mark = "PASS" if strict_ok else "FAIL"
        extra = f" ⚠{unsupported}" if unsupported else ""
        print(f"  [{i}/{len(rows)}] {mark} {q[:24]} → {msg}（{secs}s，"
              f"守卫：{guards or '无'}）{extra}")

    total = round(time.time() - t_all, 1)
    passed = sum(1 for r in results if r.get("passed"))
    contract_passed = sum(1 for r in results if r.get("contract_passed"))
    # 金标命中只统计有期望条款的 law 题；库外题（expected 空）单独统计正确拒答数
    law_rows = [r for r in results
                if r.get("intent") == "law" and r.get("expected_articles")]
    oov_rows = [r for r in results
                if r.get("intent") == "law" and not r.get("expected_articles")]
    gold_hit = sum(1 for r in law_rows if r.get("gold_hit") is True)
    oov_refused = sum(1 for r in oov_rows if r.get("passed"))
    unsupported_rows = [r for r in results if r.get("unsupported_marks")]

    print(f"\n[严格结果] {passed}/{len(results)} 通过，总耗时 {total}s")
    print(f"[契约通过] {contract_passed}/{len(results)}（格式、引用来源与安全规则）")
    print(f"[金标命中] law（有金标）{len(law_rows)} 条中 basis 含期望条号：{gold_hit}")
    print(f"[库外拒答] law（无金标）{len(oov_rows)} 条中正确拒答：{oov_refused}")
    ttfts = sorted(r["ttft_ms"] for r in results if r.get("ttft_ms"))
    ttft_p50 = ttfts[len(ttfts) // 2] if ttfts else None
    if ttfts:
        print(f"[首字响应] n={len(ttfts)} p50={ttft_p50}ms max={ttfts[-1]}ms")
    timing_summary = summarize_timings(results)
    if timing_summary:
        print("[分段耗时 P50/P95]")
        for key, stats in timing_summary.items():
            print(f"  - {key}: {stats['p50']}/{stats['p95']} ms（n={stats['n']}）")
    print(f"[内容不支撑标注] {len(unsupported_rows)} 条")
    for r in unsupported_rows:
        print(f"  - {r['id']} {r['question'][:30]}：{r['unsupported_marks']}")

    fails = [r for r in results if not r.get("passed") and "error" not in r]
    if fails:
        print("[失败明细]")
        for r in fails:
            print(f"  - {r['id']} [{r['intent']}] {r['question'][:30]}：{r['message']}")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump({"summary": {"total": len(results),
                               "strict_passed": passed,
                               "contract_passed": contract_passed,
                               # 向后兼容；passed 从 v1.0-dev 起代表严格通过数
                               "passed": passed,
                               "seconds_total": total,
                               "gold_hit_law": f"{gold_hit}/{len(law_rows)}",
                               "oov_refused": f"{oov_refused}/{len(oov_rows)}",
                               "ttft_p50_ms": ttft_p50,
                               "timing_ms": timing_summary,
                               "unsupported_marked": len(unsupported_rows)},
                   "rows": results}, f, ensure_ascii=False, indent=2)
    print(f"[保存] {OUT_PATH}")


if __name__ == "__main__":
    main()
