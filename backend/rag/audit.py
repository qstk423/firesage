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
        "cached": bool(response.get("cached")),
    })
