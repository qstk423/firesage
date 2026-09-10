#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""本地 LLM 服务 10 题验证：模拟 FireSage 管线的真实提示词调用本地服务。

覆盖：法规问答 5 题（带真实条款上下文）、知识库外拒答 2 题、
闲聊/能力询问 1 题、应急兜底 2 题。
每题检查：OpenAI 响应结构、JSON 五字段、basis 数组、
拒答 basis=[]、应急含 119+疏散、条款引用格式。

用法（服务启动后，backend 目录下）：
    ..\\venv311\\Scripts\\python.exe scripts\\test_local_service.py
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.request

BASE_URL = "http://127.0.0.1:8320"
API_KEY = "local-firesage"
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BACKEND_DIR, "data")
FIELDS = ["conclusion", "conditions", "basis", "supplement", "confidence"]

# FireSage rag/pipeline.py SYSTEM_TMPL 的副本（模拟真实调用分布）
SYSTEM_TMPL = (
    "你是消防法规领域的专业助手「消安智答」。请严格依据提供的法规条款回答问题，不得编造。"
    "面向物业/网格员等非法律专业读者：结论用大白话，少用生僻术语；能短则短。"
    "若依据中含「报批稿」或非正式施行文本，必须在 supplement 中明确提示其非正式效力，"
    "并优先采信现行有效法规；不得把报批稿写成已生效规章。"
    "回答必须使用以下 JSON 结构（不要输出 JSON 以外的内容）：\n"
    '{"conclusion": "直接回答问题的结论（1句，尽量不超过40字）", '
    '"conditions": "该结论适用的条件（主体/场所/建筑类型，如不适用写 无特殊限制）", '
    '"basis": ["《法规名》第X条：支撑该结论的关键原文片段", ...], '
    '"supplement": "补充说明或注意事项（无则写 无；控制在2句内）", '
    '"confidence": "high|medium|low，依据条款数量与一致性判断"}\n'
    "所有结论必须能在提供的条款原文中找到依据，引用条款编号必须来自提供的条款。"
)

SOURCE_FILES = ["firelaw.json", "regulation61.json", "highrise5.json",
                "responsibility87.json", "entertainment39.json",
                "ebike_charging_draft.json", "assembly_occupancy_draft.json",
                "gd_highrise.json", "tech_service7.json", "construction_fire_review.json"]


def load_article_map() -> dict:
    """与 pipeline._load_article_text 相同的条款索引。"""
    m = {}
    for fname in SOURCE_FILES:
        path = os.path.join(DATA_DIR, fname)
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            law = json.load(f)
        abbr = law.get("law_abbr", "消防法")
        for ch in law["chapters"]:
            for art in ch["articles"]:
                m[f"{abbr}·{art['num']}"] = art
    return m


def build_user(question: str, article_keys: list[str], amap: dict) -> str:
    """与 pipeline._llm_structured 相同的 user 提示词格式。"""
    arts = [amap[k] for k in article_keys if k in amap]
    context = "\n\n".join(f"【{k} {a.get('title', '')}】{a.get('text', '')}"
                          for k, a in zip(article_keys, arts))
    return f"用户问题：{question}\n\n以下为检索到的法规条款：\n{context}\n\n请基于上述条款作答。"


def call_service(system: str, user: str, timeout: int = 180):
    payload = {
        "model": "firesage-qwen25-3b-lora-v2",
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "temperature": 0.2,
        "max_tokens": 700,
    }
    req = urllib.request.Request(
        BASE_URL + "/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def check(resp, kind: str) -> tuple[bool, str]:
    """按题型检查响应。kind: law / refuse / emergency。"""
    choice = (resp.get("choices") or [{}])[0]
    content = (choice.get("message") or {}).get("content", "")
    try:
        d = json.loads(content)
    except Exception:
        return False, "content 不是合法 JSON"
    if not all(f in d for f in FIELDS):
        return False, f"缺字段：{[f for f in FIELDS if f not in d]}"
    basis = d.get("basis")
    if not isinstance(basis, list):
        return False, "basis 不是数组"
    full = str(d.get("conclusion", "")) + str(d.get("supplement", ""))
    if kind == "law":
        if not basis:
            return False, "法规题 basis 为空"
        if not re.search(r"《.+》第[一二三四五六七八九十百零]+条|《.+》\d", " ".join(basis)):
            return False, "basis 缺《法规名》第X条引用格式"
        return True, "OK"
    if kind == "refuse":
        if basis:
            return False, f"拒答题 basis 非空：{basis[:1]}"
        if not re.search(r"超出|暂不回答|无法|消防", str(d.get("conclusion", ""))):
            return False, "结论无拒答语义"
        return True, "OK（basis=[] 规范拒答）"
    if kind == "emergency":
        if not re.search(r"119|报警", full):
            return False, "缺报警要素"
        if not re.search(r"疏散|撤离|逃生|远离|电梯", full):
            return False, "缺疏散要素"
        return True, "OK（应急要素齐全）"
    return False, "未知题型"


def main() -> None:
    amap = load_article_map()
    print(f"[知识库] 可用条款 {len(amap)} 条")

    cases = [
        ("law",      "楼道堆放杂物违反什么规定？",         ["消防法·第二十八条", "消防法·第六十条"]),
        ("law",      "占用消防通道怎么处罚？",             ["消防法·第六十条"]),
        ("law",      "电动车能在楼道充电吗？",             ["高层规定·第三十七条"]),
        ("law",      "物业要履行哪些消防安全职责？",       ["61号令·第十条"]),
        ("law",      "消防控制室值班有什么要求？",         ["高层规定·第二十六条"]),
        ("refuse",   "今天天气怎么样？",                   []),
        ("refuse",   "注册消防工程师的报考条件是什么？",   []),
        ("refuse",   "你能做什么？",                       []),
        ("emergency", "隔壁家着火了，火要烧过来了，怎么办", []),
        ("emergency", "商场着火了怎么逃生",                 []),
    ]

    # 健康检查
    with urllib.request.urlopen(BASE_URL + "/health", timeout=10) as r:
        print(f"[服务] {json.loads(r.read().decode('utf-8'))}")

    results = []
    for i, (kind, q, keys) in enumerate(cases, 1):
        user = build_user(q, keys, amap) if keys else \
            f"用户问题：{q}\n\n以下为检索到的法规条款：\n（无可用条款）\n\n请基于上述条款作答。"
        t0 = time.time()
        try:
            resp = call_service(SYSTEM_TMPL, user)
        except Exception as e:
            results.append((i, kind, q, False, f"调用失败：{e}", 0, []))
            print(f"  [{i}/10] {q} → 调用失败：{e}")
            continue
        secs = round(time.time() - t0, 1)
        ok, msg = check(resp, kind)
        guards = resp.get("firesage_protections") or []
        latency = resp.get("latency_ms", 0)
        results.append((i, kind, q, ok, msg, secs, guards))
        mark = "PASS" if ok else "FAIL"
        print(f"  [{i}/10] {mark} {q} → {msg}（{secs}s，保护：{guards or '无'}）")

    passed = sum(1 for r in results if r[3])
    print(f"\n[结果] {passed}/10 通过")
    fails = [r for r in results if not r[3]]
    if fails:
        print("[失败明细]")
        for r in fails:
            print(f"  - #{r[0]} [{r[1]}] {r[2]}：{r[4]}")
    out = os.path.join(BACKEND_DIR, "eval_reports", "local_service_test.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump([{"id": r[0], "kind": r[1], "question": r[2], "passed": r[3],
                    "message": r[4], "seconds": r[5], "protections": r[6]} for r in results],
                  f, ensure_ascii=False, indent=2)
    print(f"[保存] {out}")


if __name__ == "__main__":
    main()
