#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""扩展语料：解析 raw markdown → JSON，跨库正文去重，重建 chunks/graph。

新增来源（均带官方/政府镜像 URL，报批稿明确标注）：
- 公安部39号令《公共娱乐场所消防安全管理规定》
- XF/T 报批稿《电动自行车充电及停放场所消防安全管理》
- 人员密集场所消防安全管理（报批稿，OCR 整理）
- 《广东省高层建筑消防安全管理规定》
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from datetime import date
from difflib import SequenceMatcher

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")

NUM = "一二三四五六七八九十百零〇"
CHAPTER_RE = re.compile(rf"^#{{0,3}}\s*(第[{NUM}]+章\s*.*)$")
CHAPTER_PLAIN_RE = re.compile(rf"^(第[{NUM}]+章\s*.+)$")
ARTICLE_RE = re.compile(rf"^(第[{NUM}]+条)\s*(.*)$")
# 标准条款号：4.1 / 5.1.1 / 7.10 （排除单独的章标题如 "4 一般要求"）
SECTION_RE = re.compile(r"^(\d+(?:\.\d+)+)\s+(.*)$")
SECTION_HEAD_RE = re.compile(r"^(\d+)\s+([^\d].{0,40})$")

EXISTING_FILES = [
    "firelaw.json",
    "regulation61.json",
    "highrise5.json",
    "responsibility87.json",
]

NEW_SPECS = [
    {
        "raw": "entertainment39.md",
        "filename": "entertainment39.json",
        "law_name": "公共娱乐场所消防安全管理规定",
        "law_abbr": "39号令",
        "authority": "中华人民共和国公安部",
        "document_no": "公安部令第39号",
        "issued_date": "1999-05-25",
        "effective_date": "1999-05-25",
        "status": "现行",
        "source_url": "https://www.mem.gov.cn/gk/zfxxgkpt/fdzdgknr/gz11/199905/t19990525_405691.shtml",
        "mirror_url": "https://zh.wikisource.org/wiki/公共娱乐场所消防安全管理规定",
        "notes": "部门规章现行文本；正文据应急管理部公开文本与维基文库公有领域整理稿核对。",
        "parser": "article",
    },
    {
        "raw": "ebike_charging_draft.md",
        "filename": "ebike_charging_draft.json",
        "law_name": "电动自行车充电及停放场所消防安全管理（报批稿）",
        "law_abbr": "电动车充电",
        "authority": "国家消防救援局",
        "document_no": "XF/T XXXX—XXXX（报批稿）",
        "issued_date": "2025-01",
        "effective_date": "",
        "status": "报批稿",
        "source_url": "https://www.119.gov.cn/images/zfxxgk/fdzdgknr/zqyj/2025/02/21/1740101557778051251.pdf",
        "notes": "消防救援行业标准报批稿，非正式施行；问答时须提示非正式效力。",
        "parser": "section",
    },
    {
        "raw": "assembly_occupancy_draft.md",
        "filename": "assembly_occupancy_draft.json",
        "law_name": "人员密集场所消防安全管理（报批稿）",
        "law_abbr": "密集场所",
        "authority": "国家消防救援局",
        "document_no": "拟替代 GB/T 40248—2021（报批稿）",
        "issued_date": "",
        "effective_date": "",
        "status": "报批稿",
        "source_url": "https://www.119.gov.cn/",
        "notes": "报批稿 OCR 整理，非正式施行；与 61 号令职责表述重叠处已按正文指纹去重。",
        "parser": "section",
        "min_text_len": 40,
    },
    {
        "raw": "gd_highrise.md",
        "filename": "gd_highrise.json",
        "law_name": "广东省高层建筑消防安全管理规定",
        "law_abbr": "广东高层",
        "authority": "广东省人民政府",
        "document_no": "广东省人民政府令",
        "issued_date": "2025",
        "effective_date": "2026-04-01",
        "status": "现行（地方）",
        "source_url": "http://www.gd.gov.cn/zwgk/wjk/qbwj/yfl/content/post_4862264.html",
        "notes": "省级政府规章；与应急部《高层民用建筑消防安全管理规定》主题相近，正文不同则保留。",
        "parser": "article",
    },
]


def normalize(text: str) -> str:
    text = text.replace("\u3000", " ").replace("　", " ")
    text = re.sub(r"报\s*批\s*稿", "", text)
    return re.sub(r"\s+", " ", text).strip()


def fingerprint(text: str) -> str:
    core = re.sub(r"[^\u4e00-\u9fff0-9a-zA-Z]", "", normalize(text))
    return hashlib.sha1(core.encode("utf-8")).hexdigest()


def article_title(text: str) -> str:
    first = re.split(r"[：；。（]", text, maxsplit=1)[0]
    return (first[:22] + "…") if len(first) > 22 else first


def parse_article_markdown(md_text: str) -> list[dict]:
    chapters: list[dict] = []
    chapter = None
    article = None

    def flush():
        nonlocal article
        if article and chapter is not None:
            article["text"] = normalize(article["text"])
            if not article["title"]:
                article["title"] = article_title(article["text"])
            if article["text"]:
                chapter["articles"].append(article)
        article = None

    for raw in md_text.splitlines():
        line = raw.strip()
        if not line or line.startswith("<!--") or line.startswith("---"):
            continue
        if line.startswith("# ") and ("消防" in line or "规定" in line or "管理" in line):
            continue
        if line.startswith("（") and ("发布" in line or "施行" in line or "报批" in line):
            continue

        ch_m = CHAPTER_RE.match(line) or (
            CHAPTER_PLAIN_RE.match(line) if line.startswith("第") and "章" in line[:8] else None
        )
        if ch_m:
            flush()
            title = normalize(ch_m.group(1).replace("总 则", "总则").replace("附 则", "附则"))
            chapter = {"chapter_title": title, "articles": []}
            chapters.append(chapter)
            continue

        art_m = ARTICLE_RE.match(line)
        if art_m:
            flush()
            if chapter is None:
                chapter = {"chapter_title": "正文", "articles": []}
                chapters.append(chapter)
            article = {"num": art_m.group(1), "title": "", "text": art_m.group(2).strip()}
            continue

        if article is not None:
            article["text"] += line if article["text"].endswith(("：", "，", "；", "、")) else (" " + line)

    flush()
    return chapters


def parse_section_markdown(md_text: str, min_text_len: int = 20) -> list[dict]:
    """把标准编号条款切成章节。支持一行多条款连写。"""
    text = normalize(md_text)
    # 去掉标题行
    text = re.sub(r"^#.*?(?=<!-- INFO END -->|1\s+范围|4\s+)", "", text)
    text = text.replace("<!-- INFO END -->", " ")

    # 在条款号前插入分隔，便于切分
    text = re.sub(r"(?<!\d)(\d+(?:\.\d+)+)\s+", r"\n\1 ", text)
    text = re.sub(r"(?<!\d)(\d+)\s+(范围|规范性引用文件|术语和定义|一般要求|总则|"
                  r"场地消防技术要求|场地防火技术要求|充电设施要求|消防安全管理|"
                  r"消防安全责任|消防组织|消防安全制度和管理|消防措施|"
                  r"灭火和应急疏散预案编制和演练|档案)\b", r"\n\1 \2", text)

    chapters: list[dict] = []
    chapter = None
    current_chapter_no = None

    for raw in text.splitlines():
        line = normalize(raw)
        if not line:
            continue
        head = SECTION_HEAD_RE.match(line)
        if head and "." not in head.group(1):
            # 章标题：4 一般要求
            title = f"{head.group(1)} {head.group(2)}"
            current_chapter_no = head.group(1)
            chapter = {"chapter_title": title, "articles": []}
            chapters.append(chapter)
            continue

        sec = SECTION_RE.match(line)
        if not sec:
            continue
        num, body = sec.group(1), normalize(sec.group(2))
        if len(body) < min_text_len:
            continue
        # 跳过纯引用列表 / 国标编号误切（如 13495.1）
        top = num.split(".", 1)[0]
        if not top.isdigit() or int(top) > 20:
            continue
        if body.startswith("下列文件") or body.startswith("下列术语"):
            continue
        if re.match(r"^(GB|GB/T|XF|XF/T|DB)\b", body):
            continue
        if current_chapter_no in {"2"}:  # 规范性引用文件章
            continue
        if chapter is None or (current_chapter_no and top != current_chapter_no):
            current_chapter_no = top
            chapter = {"chapter_title": f"{top} 条款", "articles": []}
            chapters.append(chapter)
        chapter["articles"].append(
            {"num": num, "title": article_title(body), "text": body}
        )

    return chapters


def iter_articles(law: dict):
    for ch in law.get("chapters", []):
        for art in ch.get("articles", []):
            yield ch, art


def load_existing_fingerprints() -> dict[str, tuple[str, str]]:
    """fp -> (law_abbr, num)"""
    fps: dict[str, tuple[str, str]] = {}
    for name in EXISTING_FILES:
        path = os.path.join(DATA_DIR, name)
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            law = json.load(f)
        abbr = law.get("law_abbr", name)
        for _, art in iter_articles(law):
            fp = fingerprint(art.get("text", ""))
            if fp and fp not in fps:
                fps[fp] = (abbr, art.get("num", ""))
    return fps


def near_duplicate(text: str, corpus_texts: list[str], threshold: float = 0.92) -> bool:
    t = normalize(text)
    if len(t) < 40:
        return False
    for other in corpus_texts:
        if abs(len(other) - len(t)) > max(80, int(0.35 * len(t))):
            continue
        if SequenceMatcher(None, t[:400], other[:400]).ratio() >= threshold:
            return True
        # 短条款被长条款完整包含
        if len(t) < 180 and t in other:
            return True
    return False


def dedupe_chapters(
    chapters: list[dict],
    fps: dict[str, tuple[str, str]],
    corpus_texts: list[str],
    law_abbr: str,
) -> tuple[list[dict], int]:
    kept_chapters = []
    skipped = 0
    for ch in chapters:
        arts = []
        for art in ch.get("articles", []):
            text = art.get("text", "")
            fp = fingerprint(text)
            if fp in fps:
                skipped += 1
                continue
            if near_duplicate(text, corpus_texts):
                skipped += 1
                continue
            arts.append(art)
            fps[fp] = (law_abbr, art.get("num", ""))
            corpus_texts.append(normalize(text))
        if arts:
            kept_chapters.append({"chapter_title": ch["chapter_title"], "articles": arts})
    return kept_chapters, skipped


def write_law(filename: str, meta: dict, chapters: list[dict]) -> tuple[str, int]:
    n = sum(len(c["articles"]) for c in chapters)
    if n < 8:
        raise ValueError(f"{filename} 去重后条款过少：{n}")
    payload = dict(meta)
    payload["retrieved_date"] = date.today().isoformat()
    payload["chapters"] = chapters
    path = os.path.join(DATA_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path, n


def update_source_files(new_filenames: list[str]) -> None:
    path = os.path.join(BASE_DIR, "rag", "graphrag.py")
    with open(path, encoding="utf-8") as f:
        src = f.read()
    files = EXISTING_FILES + new_filenames
    literal = "[" + ", ".join(f'"{x}"' for x in files) + "]"
    new_src, n = re.subn(
        r"SOURCE_FILES\s*=\s*\[[^\]]*\]",
        f"SOURCE_FILES = {literal}",
        src,
        count=1,
    )
    if n != 1:
        raise RuntimeError("未能更新 SOURCE_FILES")
    with open(path, "w", encoding="utf-8") as f:
        f.write(new_src)


def rebuild_graph() -> dict:
    sys.path.insert(0, BASE_DIR)
    # 强制重新加载，避免旧 SOURCE_FILES 缓存
    for mod in list(sys.modules):
        if mod == "rag" or mod.startswith("rag."):
            del sys.modules[mod]
    from rag.graphrag import GraphBuilder

    builder = GraphBuilder()
    return builder.stats()


def main() -> None:
    fps = load_existing_fingerprints()
    corpus_texts = []
    for name in EXISTING_FILES:
        path = os.path.join(DATA_DIR, name)
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            law = json.load(f)
        for _, art in iter_articles(law):
            corpus_texts.append(normalize(art.get("text", "")))

    written: list[str] = []
    summary = []
    for spec in NEW_SPECS:
        raw_path = os.path.join(RAW_DIR, spec["raw"])
        if not os.path.exists(raw_path):
            print(f"[skip] missing raw {spec['raw']}")
            continue
        with open(raw_path, encoding="utf-8") as f:
            md = f.read()
        if spec["parser"] == "article":
            chapters = parse_article_markdown(md)
        else:
            chapters = parse_section_markdown(md, min_text_len=spec.get("min_text_len", 20))
        before = sum(len(c["articles"]) for c in chapters)
        chapters, skipped = dedupe_chapters(chapters, fps, corpus_texts, spec["law_abbr"])
        meta = {k: v for k, v in spec.items() if k not in {"raw", "filename", "parser", "min_text_len"}}
        path, after = write_law(spec["filename"], meta, chapters)
        written.append(spec["filename"])
        summary.append(
            f"{spec['law_abbr']}: {before}→{after} (去重跳过 {skipped}) → {os.path.basename(path)}"
        )
        print(summary[-1])

    update_source_files(written)
    print("[SOURCE_FILES] updated:", EXISTING_FILES + written)
    stats = rebuild_graph()
    print(
        f"[rebuild] chunks={stats['semantic_sents']} entities={stats['entities']} "
        f"edges={stats['edges']} articles={len(stats['articles'])}"
    )


if __name__ == "__main__":
    main()
