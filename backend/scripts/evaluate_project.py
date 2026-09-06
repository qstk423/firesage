#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""消安智答 · 项目级评估脚本。

聚合静态检查与（可选）运行时评测，按「命题对齐 / 评测闭环 / 知识库 /
申报材料 / 工程成熟度」五维打分，并映射到申报 Demo · 企业试点 · 正式产品三档。

用法：
    python3 backend/scripts/evaluate_project.py
    python3 backend/scripts/evaluate_project.py --mode quick
    python3 backend/scripts/evaluate_project.py --mode full
    python3 backend/scripts/evaluate_project.py --report docs/项目评估报告.md
    python3 backend/scripts/evaluate_project.py --json-out /tmp/firesage_eval.json

模式：
    quick  静态资产 + 单元烟雾 + 金标漂移 + 图谱健康度（默认，约 1–2 分钟）
    full   quick + FireEval v1 dev + 检索-only 门禁（可能数十分钟，依赖模型）
    static 仅静态资产与文档检查（秒级）
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"
DOCS = ROOT / "docs"
EVAL_DIR = BACKEND / "eval"
DATA = BACKEND / "data"
MODELS = DATA / "models"

# 对外口径基线（docs/校赛明确要求.md）
EXPECTED = {
    "laws": 10,
    "articles": 524,
    "semantic_sents": 1675,
    "entities": 112,
    "edges": 952,
    "intent_hybrid_test": 0.9688,
    "version": "0.8.4",
}

DOC_REQUIRED = [
    "项目概述-消安智答FireSage.md",
    "解决方案-消安智答FireSage.md",
    "校赛明确要求.md",
    "校赛演示材料包.md",
    "evaluation-report-v1.md",
    "消融表-retrieval.md",
    "产品级差距评估.md",
    "图谱健康度.md",
    "演示截图清单.md",
    "完整项目差距清单.md",
    "部署与配置清单.md",
]

# 发布门槛（与 evaluate_fireeval.py 对齐）
FIREEVAL_THRESHOLDS = {
    "hit_at_1": 0.85,
    "hit_at_3": 0.95,
    "citation_accuracy": 0.95,
    "refusal_accuracy": 0.95,
    "severe_error_rate": 0.01,
    "avg_latency_s": 8.0,
}


@dataclass
class Check:
    id: str
    dimension: str
    title: str
    passed: bool
    weight: float
    score: float  # 0–100 对本检查项
    detail: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)


def _run(cmd: list[str], timeout: int = 600) -> tuple[int, str]:
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    p = subprocess.run(
        cmd,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def _parse_last_json(text: str) -> dict:
    """从混合 stdout 中提取最后一个完整 JSON 对象。"""
    end = text.rfind("}")
    if end < 0:
        return {}
    depth = 0
    start = None
    for i in range(end, -1, -1):
        ch = text[i]
        if ch == "}":
            depth += 1
        elif ch == "{":
            depth -= 1
            if depth == 0:
                start = i
                break
    if start is None:
        return {}
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {}


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# 静态检查
# ---------------------------------------------------------------------------

def check_version() -> Check:
    main_py = (BACKEND / "main.py").read_text(encoding="utf-8")
    m = re.search(r'APP_VERSION\s*=\s*"([^"]+)"', main_py)
    got = m.group(1) if m else ""
    ok = got == EXPECTED["version"]
    return Check(
        "version",
        "工程成熟度",
        "产品版本与对外口径一致",
        ok,
        4,
        100 if ok else 40,
        f"main.py={got} · 口径={EXPECTED['version']}",
        {"version": got},
    )


def check_corpus_scale() -> Check:
    chunks = _read_json(DATA / "chunks.json")
    graph = _read_json(DATA / "graph.json")
    arts = {c["article"] for c in chunks}
    laws = {a.split("·", 1)[0] for a in arts if "·" in a}
    got = {
        "laws": len(laws),
        "articles": len(arts),
        "semantic_sents": len(chunks),
        "entities": len(graph.get("nodes", [])),
        "edges": len(graph.get("edges", [])),
        "law_names": sorted(laws),
    }
    # 允许略高于基线（扩库），低于基线扣分
    ratios = []
    for key in ("laws", "articles", "semantic_sents", "entities", "edges"):
        ratios.append(min(1.0, got[key] / max(1, EXPECTED[key])))
    score = round(100 * sum(ratios) / len(ratios), 1)
    ok = got["laws"] >= EXPECTED["laws"] and got["articles"] >= EXPECTED["articles"] * 0.95
    return Check(
        "corpus",
        "知识库",
        "语料规模达到对外口径",
        ok,
        12,
        score,
        f"{got['laws']} 部法规 · {got['articles']} 条款 · "
        f"{got['semantic_sents']} 语义句 · 图谱 {got['entities']}/{got['edges']}",
        got,
    )


def check_docs() -> Check:
    missing = [n for n in DOC_REQUIRED if not (DOCS / n).exists()]
    present = len(DOC_REQUIRED) - len(missing)
    score = round(100 * present / len(DOC_REQUIRED), 1)
    return Check(
        "docs",
        "申报材料",
        "关键申报/评测文档齐全",
        not missing,
        10,
        score,
        f"{present}/{len(DOC_REQUIRED)}" + (f" · 缺: {', '.join(missing)}" if missing else ""),
        {"missing": missing, "required": DOC_REQUIRED},
    )


def check_fireeval_assets() -> Check:
    needed = [
        EVAL_DIR / "fireeval_v1.json",
        EVAL_DIR / "fireeval_v1_train.json",
        EVAL_DIR / "fireeval_v1_dev.json",
        EVAL_DIR / "fireeval_v1_test.json",
        BACKEND / "scripts" / "evaluate_fireeval.py",
        ROOT / ".github" / "workflows" / "fireeval.yml",
    ]
    missing = [str(p.relative_to(ROOT)) for p in needed if not p.exists()]
    n = 0
    if (EVAL_DIR / "fireeval_v1.json").exists():
        n = len(_read_json(EVAL_DIR / "fireeval_v1.json"))
    ok = not missing and n >= 180
    score = 100 if ok else (60 if not missing else 20)
    return Check(
        "fireeval_assets",
        "评测闭环",
        "FireEval 数据集与 CI 门禁就绪",
        ok,
        12,
        score,
        f"全集 {n} 题" + (f" · 缺: {', '.join(missing)}" if missing else " · CI 已配置"),
        {"cases": n, "missing": missing},
    )


def check_finetune_evidence() -> Check:
    """命题硬要求 1：大模型微调可复现证据。"""
    pieces = {
        "intent_clf": (MODELS / "intent_clf.joblib").exists(),
        "intent_metrics": (MODELS / "intent_clf_metrics.json").exists(),
        "intent_compare": (MODELS / "intent_compare.json").exists(),
        "train_intent_script": (BACKEND / "scripts" / "train_intent.py").exists(),
        "rerank_script": (BACKEND / "scripts" / "train_rerank.py").exists(),
        "sft_train": (MODELS / "sft" / "gen_sft_train.jsonl").exists(),
        "sft_meta": (MODELS / "sft" / "gen_sft_meta.json").exists(),
    }
    hybrid = None
    if pieces["intent_compare"]:
        compare = _read_json(MODELS / "intent_compare.json")
        for row in compare.get("compare", []):
            if row.get("split") == "test":
                hybrid = row.get("hybrid")
    hit = sum(1 for v in pieces.values() if v)
    score = round(100 * hit / len(pieces), 1)
    if hybrid is not None and hybrid >= 0.9:
        score = min(100.0, score + 5)
    ok = pieces["intent_clf"] and pieces["intent_compare"] and pieces["train_intent_script"]
    detail = f"意图 hybrid(test)={hybrid} · 证据项 {hit}/{len(pieces)}"
    return Check(
        "finetune",
        "命题对齐",
        "大模型微调可复现（意图已落地；重排/SFT 有入口）",
        ok,
        14,
        score,
        detail,
        {"pieces": pieces, "intent_hybrid_test": hybrid},
    )


def check_rag_pipeline() -> Check:
    """命题硬要求 2：RAG 融合链路文件存在。"""
    mods = [
        "rag/pipeline.py",
        "rag/retriever.py",
        "rag/semantic_index.py",
        "rag/graphrag.py",
        "rag/reranker.py",
        "rag/verifier.py",
        "rag/intent.py",
        "rag/scene.py",
    ]
    missing = [m for m in mods if not (BACKEND / m).exists()]
    ok = not missing
    return Check(
        "rag",
        "命题对齐",
        "混合 RAG + GraphRAG + 核验链路完整",
        ok,
        12,
        100 if ok else round(100 * (len(mods) - len(missing)) / len(mods), 1),
        "缺失: " + ", ".join(missing) if missing else "pipeline/retriever/graphrag/verifier 齐全",
        {"missing": missing},
    )


def check_ops_signals() -> Check:
    main_py = (BACKEND / "main.py").read_text(encoding="utf-8")
    signals = {
        "audit_module": (BACKEND / "rag" / "audit.py").exists(),
        "api_key_hook": "API_KEY" in main_py,
        "rate_limit": "_check_rate_limit" in main_py,
        "env_example": (BACKEND / ".env.example").exists(),
        "kb_reload": "/api/kb/reload" in main_py,
        "feedback": "/api/feedback" in main_py,
        "audit_export": "/api/audit/export" in main_py,
        "deploy_doc": (DOCS / "部署与配置清单.md").exists(),
        "docker": (ROOT / "Dockerfile").exists() or (ROOT / "docker-compose.yml").exists(),
        "auth_tenant": False,  # 正式产品项：当前未实现
    }
    core = ["audit_module", "api_key_hook", "rate_limit", "env_example", "kb_reload",
            "feedback", "audit_export", "deploy_doc"]
    core_hit = sum(1 for k in core if signals[k])
    score = round(70 * core_hit / len(core), 1)
    if signals["docker"]:
        score += 15
    if signals["auth_tenant"]:
        score += 15
    ok = core_hit >= 6
    missing_ops = [k for k in ("docker", "auth_tenant") if not signals[k]]
    return Check(
        "ops",
        "工程成熟度",
        "试点运维信号（审计/反馈/导出/鉴权/限流/热加载）",
        ok,
        10,
        min(100.0, score),
        f"核心 {core_hit}/{len(core)}"
        + (f" · 未具备: {', '.join(missing_ops)}" if missing_ops else ""),
        signals,
    )


# ---------------------------------------------------------------------------
# 运行时检查
# ---------------------------------------------------------------------------

def check_smoke() -> Check:
    code, out = _run([sys.executable, "-m", "unittest", "tests.test_smoke", "-q"], timeout=180)
    ok = code == 0
    # unittest -q 失败时 stderr 有 FAIL
    fail_n = len(re.findall(r"^FAIL:", out, re.M)) + len(re.findall(r"^ERROR:", out, re.M))
    return Check(
        "smoke",
        "评测闭环",
        "单元烟雾测试通过",
        ok,
        10,
        100 if ok else max(0, 100 - 25 * max(1, fail_n)),
        "全部通过" if ok else f"失败 exit={code} · 摘录: {out[-400:].strip()}",
        {"exit": code, "failures": fail_n},
    )


def check_gold_drift() -> Check:
    code, out = _run(
        [sys.executable, "backend/scripts/check_gold_drift.py", "--split", "all", "--strict"],
        timeout=120,
    )
    payload = _parse_last_json(out)
    ok = code == 0
    return Check(
        "gold_drift",
        "评测闭环",
        "金标条款未漂移出语料",
        ok,
        8,
        100 if ok else 30,
        "strict 通过" if ok else f"存在漂移 · {payload or out[-300:].strip()}",
        payload or {"exit": code},
    )


def check_graph_health() -> Check:
    import importlib.util

    path = BACKEND / "scripts" / "graph_health.py"
    spec = importlib.util.spec_from_file_location("graph_health", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    stats = mod.compute()
    coverage = float(stats.get("edge_coverage") or 0)
    orphans = int(stats.get("orphan_articles") or 0)
    # 当前语料边覆盖约 50%+ 为常态（大量程序性条款无实体边）；≥0.45 及格，≥0.70 满分
    score = round(max(0, min(100, (coverage - 0.35) / 0.35 * 100)), 1)
    ok = coverage >= 0.45 and orphans < max(250, int(stats.get("articles", 0) * 0.6))
    return Check(
        "graph_health",
        "知识库",
        "图谱边覆盖与孤儿条款健康",
        ok,
        8,
        score,
        f"edge_coverage={coverage:.2%} · 孤儿条款 {orphans}",
        {
            "edge_coverage": coverage,
            "orphan_articles": orphans,
            "entities": stats.get("entities"),
            "edges": stats.get("edges"),
        },
    )


def check_demo_questions() -> Check:
    sys.path.insert(0, str(BACKEND))
    from rag.pipeline import Pipeline

    chunks = _read_json(DATA / "chunks.json")
    pl = Pipeline(chunks)
    cases = [
        ("楼道堆放杂物违反什么规定", {"refused": False}, "law"),
        ("家里着火了现在怎么办", {"intent": "emergency"}, None),
        ("推荐几部好看的电影", {"refused": True}, None),
    ]
    rows = []
    all_ok = True
    for q, expect, _ in cases:
        r = pl.ask(q, use_cache=False)
        ok = all(r.get(k) == v for k, v in expect.items())
        rows.append({"q": q, "ok": ok, "intent": r.get("intent"), "refused": r.get("refused")})
        if not ok:
            all_ok = False
    score = round(100 * sum(1 for x in rows if x["ok"]) / len(rows), 1)
    return Check(
        "demo_questions",
        "命题对齐",
        "演示三问分流正确（法规 / 应急 / 拒答）",
        all_ok,
        10,
        score,
        "；".join(f"{'✓' if r['ok'] else '✗'}{r['intent']}" for r in rows),
        {"demos": rows},
    )


def check_fireeval_dev() -> Check:
    code, out = _run(
        [sys.executable, "backend/scripts/evaluate_fireeval.py", "--split", "dev"],
        timeout=2400,
    )
    metrics = _parse_last_json(out)
    if not metrics:
        return Check(
            "fireeval_dev",
            "评测闭环",
            "FireEval v1 dev 发布门槛",
            False,
            16,
            0,
            f"未能解析指标 · exit={code}",
            {"exit": code, "tail": out[-500:]},
        )

    checks = []
    ok = True
    for key, bound in FIREEVAL_THRESHOLDS.items():
        if key not in metrics:
            continue
        value = metrics[key]
        if key in ("severe_error_rate", "avg_latency_s"):
            passed = value <= bound
        else:
            passed = value >= bound
        checks.append({"metric": key, "value": value, "bound": bound, "passed": passed})
        ok = ok and passed

    # 连续分：按门槛达成比例
    score = round(100 * sum(1 for c in checks if c["passed"]) / max(1, len(checks)), 1)
    # Hit@1 额外加权观感
    hit1 = metrics.get("hit_at_1")
    detail = (
        f"Hit@1={metrics.get('hit_at_1')} Hit@3={metrics.get('hit_at_3')} "
        f"拒答={metrics.get('refusal_accuracy')} 严重错误={metrics.get('severe_error_rate')} "
        f"延迟={metrics.get('avg_latency_s')}s"
    )
    return Check(
        "fireeval_dev",
        "评测闭环",
        "FireEval v1 dev 发布门槛",
        ok and code == 0,
        16,
        score if hit1 is not None else 0,
        detail,
        {"metrics": metrics, "threshold_checks": checks, "exit": code},
    )


def check_retrieval_gate() -> Check:
    code, out = _run(
        [sys.executable, "backend/scripts/gate_retrieval.py", "--split", "dev", "--strict"],
        timeout=1200,
    )
    metrics = _parse_last_json(out)
    ok = code == 0
    hit1 = metrics.get("hit_at_1")
    hit3 = metrics.get("hit_at_3")
    score = 100 if ok else 40
    if hit1 is not None:
        score = round(min(100, max(0, (hit1 - 0.6) / 0.25 * 100)), 1)
    return Check(
        "retrieval_gate",
        "评测闭环",
        "检索-only Hit 门禁（dev）",
        ok,
        10,
        score,
        f"Hit@1={hit1} Hit@3={hit3} · degraded={metrics.get('degraded_vector')}",
        metrics or {"exit": code},
    )


# ---------------------------------------------------------------------------
# 打分汇总
# ---------------------------------------------------------------------------

def weighted_dimension_scores(checks: list[Check]) -> dict[str, float]:
    buckets: dict[str, list[Check]] = {}
    for c in checks:
        buckets.setdefault(c.dimension, []).append(c)
    out = {}
    for dim, items in buckets.items():
        tw = sum(i.weight for i in items) or 1
        out[dim] = round(sum(i.score * i.weight for i in items) / tw, 1)
    return out


def tier_scores(dims: dict[str, float], checks: list[Check]) -> dict[str, Any]:
    """映射到三档成熟度（口径对齐 docs/产品级差距评估.md）。"""
    by_id = {c.id: c for c in checks}
    demo = (
        0.30 * dims.get("命题对齐", 0)
        + 0.30 * dims.get("评测闭环", 0)
        + 0.20 * dims.get("知识库", 0)
        + 0.15 * dims.get("申报材料", 0)
        + 0.05 * dims.get("工程成熟度", 0)
    )
    pilot = (
        0.25 * dims.get("命题对齐", 0)
        + 0.25 * dims.get("评测闭环", 0)
        + 0.20 * dims.get("知识库", 0)
        + 0.05 * dims.get("申报材料", 0)
        + 0.25 * dims.get("工程成熟度", 0)
    )
    # 正式产品：缺 Docker/租户时硬性封顶
    product = pilot * 0.55
    ops = by_id.get("ops")
    if ops and ops.evidence.get("docker") and ops.evidence.get("auth_tenant"):
        product = pilot * 0.85
    elif ops and ops.evidence.get("docker"):
        product = pilot * 0.7

    def band(score: float) -> str:
        if score >= 80:
            return "达标"
        if score >= 60:
            return "基本可用"
        if score >= 40:
            return "差距明显"
        return "远未就绪"

    return {
        "申报_Demo": {"score": round(demo, 1), "band": band(demo)},
        "企业试点": {"score": round(pilot, 1), "band": band(pilot)},
        "正式产品": {"score": round(product, 1), "band": band(product)},
    }


def verdict(tiers: dict[str, Any], checks: list[Check]) -> str:
    fails = [c for c in checks if not c.passed]
    demo_s = tiers["申报_Demo"]["score"]
    if demo_s >= 80 and not any(c.id in ("finetune", "rag", "demo_questions") for c in fails):
        base = "申报 / Demo 档位可对外演示与答辩；"
    elif demo_s >= 60:
        base = "申报材料与核心链路基本齐，但仍有阻塞项需先修；"
    else:
        base = "尚未达到稳定申报演示门槛；"
    pilot_s = tiers["企业试点"]["score"]
    if pilot_s < 60:
        base += "企业试点仍需补运维与生成稳定性。"
    else:
        base += "可考虑封闭联调试点。"
    if fails:
        base += " 未通过: " + ", ".join(c.id for c in fails[:8])
    return base


def render_markdown(report: dict) -> str:
    lines = [
        "# 消安智答 · 项目评估报告",
        "",
        f"> 自动生成于 `{report['timestamp']}` · 模式 `{report['mode']}` · 版本 `{report.get('app_version', '?')}`",
        "",
        "## 总评",
        "",
        report["verdict"],
        "",
        "## 三档成熟度",
        "",
        "| 档位 | 得分 | 判断 |",
        "|------|------|------|",
    ]
    for name, row in report["tiers"].items():
        lines.append(f"| {name.replace('_', ' / ')} | **{row['score']}** | {row['band']} |")
    lines += ["", "## 五维得分", "", "| 维度 | 得分 |", "|------|------|"]
    for dim, score in report["dimensions"].items():
        lines.append(f"| {dim} | {score} |")
    lines += ["", "## 检查明细", "", "| ID | 维度 | 项 | 结果 | 分 | 说明 |",
              "|----|------|----|------|----|------|"]
    for c in report["checks"]:
        mark = "PASS" if c["passed"] else "FAIL"
        detail = c["detail"].replace("|", "/")
        lines.append(
            f"| `{c['id']}` | {c['dimension']} | {c['title']} | {mark} | {c['score']} | {detail} |"
        )
    lines += [
        "",
        "## 建议下一步",
        "",
    ]
    fails = [c for c in report["checks"] if not c["passed"]]
    if not fails:
        lines.append("- 全部门禁通过：可跑 `--mode full` 刷新 FireEval 数字，并更新答辩 PPT 指标页。")
    else:
        for c in fails:
            lines.append(f"- 修复 `{c['id']}`：{c['title']}（{c['detail']}）")
    if not any(c["id"] == "ops" and c.get("evidence", {}).get("docker") for c in report["checks"]):
        lines.append("- 补 Docker 一键部署，抬高「企业试点」档位。")
    lines.append("")
    return "\n".join(lines)


def run(mode: str) -> dict:
    t0 = time.time()
    checks: list[Check] = []

    # 始终跑静态
    print("→ 静态检查…")
    for fn in (
        check_version,
        check_corpus_scale,
        check_docs,
        check_fireeval_assets,
        check_finetune_evidence,
        check_rag_pipeline,
        check_ops_signals,
    ):
        c = fn()
        checks.append(c)
        print(f"  [{'PASS' if c.passed else 'FAIL'}] {c.id}: {c.detail}")

    if mode in ("quick", "full"):
        print("→ 运行时检查（烟雾 / 金标 / 图谱 / 演示三问）…")
        for fn in (check_smoke, check_gold_drift, check_graph_health, check_demo_questions):
            c = fn()
            checks.append(c)
            print(f"  [{'PASS' if c.passed else 'FAIL'}] {c.id}: {c.detail}")

    if mode == "full":
        print("→ FireEval + 检索门禁（可能较久）…")
        for fn in (check_retrieval_gate, check_fireeval_dev):
            c = fn()
            checks.append(c)
            print(f"  [{'PASS' if c.passed else 'FAIL'}] {c.id}: {c.detail}")

    dims = weighted_dimension_scores(checks)
    tiers = tier_scores(dims, checks)
    app_version = next((c.evidence.get("version") for c in checks if c.id == "version"), None)

    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "mode": mode,
        "app_version": app_version,
        "elapsed_s": round(time.time() - t0, 1),
        "dimensions": dims,
        "tiers": tiers,
        "verdict": verdict(tiers, checks),
        "passed": all(c.passed for c in checks),
        "checks": [asdict(c) for c in checks],
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="消安智答项目级评估")
    parser.add_argument(
        "--mode",
        choices=["static", "quick", "full"],
        default="quick",
        help="static=仅资产; quick=含烟雾/金标/图谱; full=+FireEval/检索门禁",
    )
    parser.add_argument("--report", default=None, help="写出 Markdown 报告路径")
    parser.add_argument("--json-out", default=None, help="写出 JSON 报告路径")
    args = parser.parse_args()

    print("=" * 64)
    print("消安智答 FireSage · 项目评估")
    print(f"模式: {args.mode} · 根目录: {ROOT}")
    print("=" * 64)

    report = run(args.mode)

    print("\n===== 五维得分 =====")
    for k, v in report["dimensions"].items():
        print(f"  {k}: {v}")
    print("\n===== 三档成熟度 =====")
    for k, v in report["tiers"].items():
        print(f"  {k}: {v['score']} ({v['band']})")
    print(f"\n总评: {report['verdict']}")
    print(f"耗时: {report['elapsed_s']}s")

    json_path = Path(args.json_out) if args.json_out else ROOT / "project_eval_report.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nJSON: {json_path}")

    if args.report:
        md_path = Path(args.report)
        if not md_path.is_absolute():
            md_path = ROOT / md_path
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(render_markdown(report), encoding="utf-8")
        print(f"Markdown: {md_path}")

    sys.exit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
