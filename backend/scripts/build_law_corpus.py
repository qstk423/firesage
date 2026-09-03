#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""将法规 Markdown 转为 FireSage 层级 JSON，并重建 chunks/graph。

消防法优先使用 LawRefBook 公有领域整理稿（与官方 2021 修正文本一致）；
61 号令使用同目录 raw markdown。官方链接与生效信息写入元数据。
"""
from __future__ import annotations

import json
import os
import re
import sys
from datetime import date

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")

NUM = "一二三四五六七八九十百零〇"
CHAPTER_RE = re.compile(rf"^#{{1,3}}\s*(第[{NUM}]+章\s*.*)$")
CHAPTER_PLAIN_RE = re.compile(rf"^(第[{NUM}]+章\s*.*)$")
ARTICLE_RE = re.compile(rf"^(第[{NUM}]+条)\s*(.*)$")


def normalize(text: str) -> str:
    text = text.replace("\u3000", " ").replace("　", " ")
    return re.sub(r"\s+", " ", text).strip()


def article_title(text: str) -> str:
    first = re.split(r"[：；。（]", text, maxsplit=1)[0]
    return (first[:22] + "…") if len(first) > 22 else first


def parse_markdown(md_text: str) -> list[dict]:
    chapters: list[dict] = []
    chapter = None
    article = None

    def flush_article():
        nonlocal article
        if article and chapter is not None:
            article["text"] = normalize(article["text"])
            if not article["title"]:
                article["title"] = article_title(article["text"])
            chapter["articles"].append(article)
        article = None

    for raw in md_text.splitlines():
        line = raw.strip()
        if not line or line.startswith("<!--") or line.startswith("---"):
            continue
        if line.startswith("# ") and "消防" in line:
            continue

        chapter_match = CHAPTER_RE.match(line) or (
            CHAPTER_PLAIN_RE.match(line) if line.startswith("第") and "章" in line[:6] else None
        )
        if chapter_match:
            flush_article()
            title = normalize(chapter_match.group(1).replace("总 则", "总则").replace("附 则", "附则"))
            chapter = {"chapter_title": title, "articles": []}
            chapters.append(chapter)
            continue

        article_match = ARTICLE_RE.match(line)
        if article_match:
            flush_article()
            if chapter is None:
                chapter = {"chapter_title": "正文", "articles": []}
                chapters.append(chapter)
            article = {
                "num": article_match.group(1),
                "title": "",
                "text": article_match.group(2).strip(),
            }
            continue

        if article is not None:
            # 合并续行（含枚举项）
            article["text"] += line if article["text"].endswith(("：", "，", "；", "、")) else (" " + line)

    flush_article()
    return chapters


def write_law(filename: str, meta: dict, chapters: list[dict]) -> tuple[str, int]:
    if not chapters or sum(len(c["articles"]) for c in chapters) < 10:
        raise ValueError(f"{filename} 解析条款过少")
    payload = dict(meta)
    payload["retrieved_date"] = date.today().isoformat()
    payload["chapters"] = chapters
    path = os.path.join(DATA_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path, sum(len(c["articles"]) for c in chapters)


def build_firelaw() -> tuple[str, int]:
    raw_path = os.path.join(RAW_DIR, "firelaw_2021.md")
    with open(raw_path, encoding="utf-8") as f:
        chapters = parse_markdown(f.read())
    meta = {
        "law_name": "中华人民共和国消防法（2021修正）",
        "law_abbr": "消防法",
        "authority": "全国人民代表大会常务委员会",
        "document_no": "主席令第81号（2021修正）",
        "issued_date": "2021-04-29",
        "effective_date": "2021-04-29",
        "source_url": "https://www.gov.cn/xinwen/2021-04/29/content_5603930.htm",
        "mirror_url": "https://github.com/LawRefBook/Laws/blob/master/行政法/消防法(2021-04-29).md",
        "notes": "正文来自 LawRefBook 公有领域整理稿，与 2021 年修正文本对齐；问答仍以检索条款为准。",
    }
    return write_law("firelaw.json", meta, chapters)


def build_regulation61() -> tuple[str, int]:
    raw_path = os.path.join(RAW_DIR, "regulation61.md")
    with open(raw_path, encoding="utf-8") as f:
        chapters = parse_markdown(f.read())
    meta = {
        "law_name": "机关、团体、企业、事业单位消防安全管理规定（公安部61号令）",
        "law_abbr": "61号令",
        "authority": "中华人民共和国公安部",
        "document_no": "公安部令第61号",
        "issued_date": "2001-11-14",
        "effective_date": "2002-05-01",
        "source_url": "https://www.119.gov.cn/gk/flfg/bmgz/2022/29134.shtml",
        "mirror_url": "https://zh.wikisource.org/wiki/中华人民共和国国务院公报/2002年/第25号",
        "notes": "正文依据公安部令第61号公开文本整理；官方页面保留为权威来源链接。",
    }
    return write_law("regulation61.json", meta, chapters)


def rebuild_graph() -> None:
    sys.path.insert(0, BASE_DIR)
    from rag.graphrag import GraphBuilder

    builder = GraphBuilder()
    stats = builder.stats()
    print(
        f"[rebuild] chunks={stats['semantic_sents']} entities={stats['entities']} "
        f"edges={stats['edges']} articles={len(stats['articles'])}"
    )


if __name__ == "__main__":
    os.makedirs(RAW_DIR, exist_ok=True)
    for builder in (build_firelaw, build_regulation61):
        path, count = builder()
        print(f"已写入 {os.path.basename(path)}：{count} 条")
    rebuild_graph()
