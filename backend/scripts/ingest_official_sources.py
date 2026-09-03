#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从权威网页抓取消防法规，转换为 FireSage 的层级 JSON。

仅收录配置中明确列出的政府站点；每份数据保留来源、发布机关与生效时间。

消防法与 61 号令优先使用 ``data/raw/*.md`` 本地全文（见 ``build_law_corpus.py``）；
本脚本继续负责高层规定与责任制办法的在线抓取。
"""
import html
import json
import os
import re
import subprocess
import sys
from html.parser import HTMLParser

import requests

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

SOURCES = [
    {
        "filename": "highrise5.json",
        "law_name": "高层民用建筑消防安全管理规定",
        "law_abbr": "高层规定",
        "authority": "中华人民共和国应急管理部",
        "document_no": "应急管理部令第5号",
        "issued_date": "2021-06-21",
        "effective_date": "2021-08-01",
        "source_url": "https://www.mem.gov.cn/gk/zfxxgkpt/fdzdgknr/202106/t20210625_389980.shtml",
        "stop": "相关链接：",
    },
    {
        "filename": "responsibility87.json",
        "law_name": "消防安全责任制实施办法",
        "law_abbr": "责任制办法",
        "authority": "国务院办公厅",
        "document_no": "国办发〔2017〕87号",
        "issued_date": "2017-10-29",
        "effective_date": "2017-10-29",
        "source_url": "https://app.www.gov.cn/govdata/gov/201711/09/414736/article.html",
        "stop": "相关新闻",
    },
]

NUM = "一二三四五六七八九十百零〇"
CHAPTER_RE = re.compile(rf"^(第[{NUM}]+章)\s*(.*)$")
ARTICLE_RE = re.compile(rf"^(第[{NUM}]+条)\s*(.*)$")


class TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.lines = []
        self.skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "noscript"}:
            self.skip_depth += 1

    def handle_endtag(self, tag):
        if tag in {"script", "style", "noscript"} and self.skip_depth:
            self.skip_depth -= 1

    def handle_data(self, data):
        if self.skip_depth:
            return
        text = re.sub(r"\s+", " ", html.unescape(data)).strip()
        if text:
            self.lines.append(text)


def normalize(text):
    return re.sub(r"\s+", " ", text).strip()


def extract_body(raw_html, config):
    parser = TextExtractor()
    parser.feed(raw_html)
    lines = [normalize(line) for line in parser.lines]
    start_at = next((i for i, line in enumerate(lines) if line.startswith("第一章")), -1)
    end_at = next((i for i, line in enumerate(lines[start_at:], start_at) if line == config["stop"]), -1)
    if start_at < 0 or end_at < 0:
        raise ValueError(f"无法定位 {config['law_name']} 的正文边界")
    return lines[start_at:end_at]


def article_title(text):
    first = re.split(r"[：；。（]", text, maxsplit=1)[0]
    return (first[:22] + "…") if len(first) > 22 else first


def parse_law(lines, config):
    chapters, chapter, article = [], None, None

    def flush_article():
        nonlocal article
        if article and chapter:
            article["text"] = normalize(article["text"])
            article["title"] = article_title(article["text"])
            chapter["articles"].append(article)
        article = None

    skip_next = False
    for index, line in enumerate(lines):
        if skip_next:
            skip_next = False
            continue
        chapter_match = CHAPTER_RE.match(line)
        if chapter_match:
            flush_article()
            title = chapter_match.group(2).strip()
            if not title and index + 1 < len(lines):
                title = lines[index + 1]
                skip_next = True
            chapter = {"chapter_title": f"{chapter_match.group(1)} {title}", "articles": []}
            chapters.append(chapter)
            continue
        article_match = ARTICLE_RE.match(line)
        if article_match:
            flush_article()
            if chapter is None:
                chapter = {"chapter_title": "正文", "articles": []}
                chapters.append(chapter)
            article = {"num": article_match.group(1), "title": "", "text": article_match.group(2)}
            continue
        if article:
            article["text"] += line
    flush_article()
    if not chapters or sum(len(c["articles"]) for c in chapters) < 10:
        raise ValueError(f"{config['law_name']} 解析结果异常")
    return chapters


def ingest(config):
    response = requests.get(config["source_url"], timeout=30, headers={"User-Agent": "FireSage/0.4 (+educational prototype)"})
    response.raise_for_status()
    response.encoding = response.apparent_encoding or "utf-8"
    chapters = parse_law(extract_body(response.text, config), config)
    payload = {key: value for key, value in config.items() if key not in {"filename", "stop"}}
    payload["retrieved_date"] = "2026-09-02"
    payload["chapters"] = chapters
    path = os.path.join(DATA_DIR, config["filename"])
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path, sum(len(chapter["articles"]) for chapter in chapters)


if __name__ == "__main__":
    corpus_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "build_law_corpus.py")
    if os.path.exists(corpus_script):
        print("先构建消防法 / 61 号令本地全文…")
        subprocess.check_call([sys.executable, corpus_script])
    for source in SOURCES:
        path, count = ingest(source)
        print(f"已写入 {os.path.basename(path)}：{count} 条")
