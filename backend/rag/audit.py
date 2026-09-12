# -*- coding: utf-8 -*-
"""问答审计留痕：问题 / 意图 / 依据条款 / 是否拒答 / 耗时。

默认写入 backend/data/audit/ask_YYYYMMDD.jsonl（可配置 AUDIT_DIR）。
"""
from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timezone
from typing import Any, Optional

_lock = threading.Lock()

DEFAULT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "audit"
)


def _dir() -> str:
    return os.getenv("AUDIT_DIR", DEFAULT_DIR)


def enabled() -> bool:
    return os.getenv("AUDIT_DISABLED", "").strip() not in ("1", "true", "True")


def write(event: dict[str, Any]) -> Optional[str]:
    if not enabled():
        return None
    path_dir = _dir()
    os.makedirs(path_dir, exist_ok=True)
    day = datetime.now(timezone.utc).astimezone().strftime("%Y%m%d")
    path = os.path.join(path_dir, f"ask_{day}.jsonl")
    row = {
        "ts": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        **event,
    }
    line = json.dumps(row, ensure_ascii=False)
    with _lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    return path


def from_response(question: str, previous: Optional[str], response: dict, client: str = "") -> Optional[str]:
    refs = response.get("references") or []
    return write({
        "client": client or "-",
        "question": question,
        "previous_question": previous or "",
        "intent": response.get("intent"),
        "refused": bool(response.get("refused")),
        "crag": response.get("crag"),
        "strategy": response.get("strategy"),
        "articles": [r.get("article") for r in refs if r.get("article")],
        "latency_ms": response.get("latency_ms"),
        "timing": response.get("timing") or {},
        "trace_id": response.get("trace_id"),
        "llm_protections": (
            (response.get("structured") or {}).get("llm_protections") or []),
        "cached": bool(response.get("cached")),
    })


def list_days(limit: int = 31) -> list[str]:
    """返回已有审计日期（YYYYMMDD），新→旧。"""
    path_dir = _dir()
    if not os.path.isdir(path_dir):
        return []
    days = []
    for name in os.listdir(path_dir):
        if name.startswith("ask_") and name.endswith(".jsonl"):
            days.append(name[4:-6])
    days.sort(reverse=True)
    return days[: max(1, limit)]


def export_day(day: str) -> tuple[str, list[dict]]:
    """读取某日审计；day 形如 20260905。返回 (path, rows)。"""
    day = "".join(ch for ch in (day or "") if ch.isdigit())
    if len(day) != 8:
        raise ValueError("day 须为 YYYYMMDD")
    path = os.path.join(_dir(), f"ask_{day}.jsonl")
    if not os.path.isfile(path):
        return path, []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return path, rows


def write_feedback(event: dict[str, Any]) -> Optional[str]:
    """用户反馈落盘：backend/data/feedback/fb_YYYYMMDD.jsonl"""
    if not enabled():
        return None
    root = os.path.dirname(_dir())
    path_dir = os.path.join(root, "feedback")
    os.makedirs(path_dir, exist_ok=True)
    day = datetime.now(timezone.utc).astimezone().strftime("%Y%m%d")
    path = os.path.join(path_dir, f"fb_{day}.jsonl")
    row = {
        "ts": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        **event,
    }
    with _lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path
