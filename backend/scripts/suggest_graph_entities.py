#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""半自动图谱实体建议（T2.1）：从条款正文扫出未入词典的候选主体/行为/对象。

用法：
  python3 backend/scripts/suggest_graph_entities.py
  python3 backend/scripts/suggest_graph_entities.py --law 39号令 --top 30

输出供人工审核后写入 graphrag.py 的 SUBJECTS / BEHAVIORS / OBJECTS。
不自动改词典，避免脏实体入图。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BACKEND)

from rag.graphrag import BEHAVIORS, OBJECTS, SUBJECTS, SUBJECT_ALIASES  # noqa: E402

# 粗粒度中文名词块：2～6 字，避开纯虚词
CAND_RE = re.compile(r"[\u4e00-\u9fff]{2,6}")
STOP = {
    "应当", "必须", "不得", "禁止", "严禁", "可以", "按照", "依照", "有关", "下列",
    "规定", "本办法", "本法", "本章", "本规定", "以下", "以上", "以及", "或者",
    "进行", "组织", "落实", "加强", "做好", "发生", "造成", "其他", "相关",
    "第一", "第二", "第三", "第四", "第五", "第六", "第七", "第八", "第九", "第十",
}


def _known_lexicon() -> set[str]:
    known = set(SUBJECTS)
    for aliases in SUBJECT_ALIASES.values():
        known.update(aliases)
    for name, spec in BEHAVIORS.items():
        known.add(name)
        known.update(spec.get("aliases") or [])
        known.update(spec.get("verbs") or [])
        known.update(spec.get("objects") or [])
    for name, aliases in OBJECTS.items():
        known.add(name)
        known.update(aliases)
    return known


def _iter_articles(law_filter: str | None):
    data_dir = os.path.join(BACKEND, "data")
    files = [
        "firelaw.json", "regulation61.json", "highrise5.json", "responsibility87.json",
        "entertainment39.json", "ebike_charging_draft.json",
        "assembly_occupancy_draft.json", "gd_highrise.json",
    ]
    for fname in files:
        path = os.path.join(data_dir, fname)
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            law = json.load(f)
        abbr = law.get("law_abbr") or fname
        if law_filter and law_filter not in (abbr, law.get("law_name", "")):
            continue
        for ch in law.get("chapters") or []:
            for art in ch.get("articles") or []:
                yield abbr, art.get("num", ""), art.get("text", "")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--law", default="", help="只看某一 law_abbr，如 39号令")
    parser.add_argument("--top", type=int, default=40)
    args = parser.parse_args()

    known = _known_lexicon()
    counts: Counter[str] = Counter()
    samples: dict[str, str] = {}
    for abbr, num, text in _iter_articles(args.law or None):
        for tok in CAND_RE.findall(text or ""):
            if tok in STOP or tok in known:
                continue
            if any(k in tok for k in ("第一百", "第二章", "第三章")):
                continue
            counts[tok] += 1
            samples.setdefault(tok, f"{abbr}·{num}")

    print(f"# 候选未登录词 Top {args.top}" + (f" · {args.law}" if args.law else ""))
    print("# 人工审核后写入 SUBJECTS / BEHAVIORS / OBJECTS，再 rebuild 图谱")
    print()
    for tok, n in counts.most_common(args.top):
        print(f"{n:4d}  {tok}  ← {samples.get(tok, '')}")


if __name__ == "__main__":
    main()
