# -*- coding: utf-8 -*-
"""执行路径显微镜：看清一个问题在系统里到底走了哪些函数。

用法：
    cd backend
    python3 scripts/trace.py "楼道堆放杂物违反什么规定"
    python3 scripts/trace.py "家里着火了现在怎么办"
    python3 scripts/trace.py            # 不带参数 = 跑内置的 5 个对照样例

输出的文件名和行号是从函数对象里实时读出来的（__code__.co_filename），
不是手写的，所以代码改了它也不会说谎。想跳到某一行：编辑器里 Cmd+P 输入 文件:行号。
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 终端颜色（不支持时自动退化为空串）
_TTY = sys.stdout.isatty()
def _c(code, s):
    return f"\033[{code}m{s}\033[0m" if _TTY else s
DIM, BOLD, CYAN, YEL, GRN, RED = "2", "1", "36", "33", "32", "31"

_log = []
_depth = [0]


def _short_path(filename):
    try:
        return os.path.relpath(filename, BASE)
    except ValueError:
        return filename


def _brief(name, value):
    """把返回值压缩成一行人话。看不懂的类型就截断 repr。"""
    try:
        if name == "structure" and isinstance(value, dict):
            bits = []
            for k in ("subjects", "behaviors", "venues", "objects"):
                if value.get(k):
                    bits.append(f"{k}={value[k]}")
            if value.get("ambiguity"):
                bits.append(f"缺要素={value['ambiguity']}")
            if value.get("rewrite"):
                bits.append(f"改写='{value['rewrite']}'")
            return " ".join(bits) or "(什么要素都没识别出来)"
        if name == "assess_risk" and isinstance(value, dict):
            return f"risk_level={value.get('risk_level')}"
        if name == "retrieve" and isinstance(value, tuple) and len(value) == 2:
            fused = value[0] or []
            top = [f"{f['article']}({f['score']:.2f})" for f in fused[:3]]
            return f"召回 {len(fused)} 条，top3 = " + ", ".join(top) if top else "召回 0 条"
        if name == "verify" and isinstance(value, dict):
            flag = "通过" if value.get("passed") else "未通过"
            issues = value.get("issues") or []
            return flag + (f"，问题：{issues}" if issues else "")
        if name == "_build_refs" and isinstance(value, list):
            return f"{len(value)} 条依据：" + ", ".join(r["article"] for r in value[:3])
        if name == "decompose_queries" and isinstance(value, list):
            return f"{len(value)} 路子查询：{value[:3]}"
        if name == "_crag_judge":
            return f"置信判定 = {value}"
        if isinstance(value, bool):
            return str(value)
        if isinstance(value, str):
            v = value.replace("\n", " ⏎ ")
            return f"'{v[:70]}…'" if len(v) > 70 else f"'{v}'"
        if isinstance(value, dict):
            return f"dict(keys={list(value)[:6]})"
        r = repr(value)
        return r[:90] + "…" if len(r) > 90 else r
    except Exception:
        return "<无法概括>"


def _mk_wrapper(orig, label, note):
    def wrapper(*a, **kw):
        d = _depth[0]
        _depth[0] += 1
        t0 = time.time()
        try:
            out = orig(*a, **kw)
        finally:
            _depth[0] = d
        _log.append({
            "depth": d,
            "label": label,
            "file": _short_path(orig.__code__.co_filename),
            "line": orig.__code__.co_firstlineno,
            "note": note,
            "ms": (time.time() - t0) * 1000,
            "brief": _brief(label.split(".")[-1], out),
        })
        return out
    wrapper.__name__ = getattr(orig, "__name__", label)
    return wrapper


def _patch_module_func(mod, name, note):
    orig = getattr(mod, name, None)
    if orig is None or not hasattr(orig, "__code__"):
        return
    setattr(mod, name, _mk_wrapper(orig, name, note))


def _patch_method(cls, name, note):
    raw = cls.__dict__.get(name)
    if raw is None:
        return
    is_static = isinstance(raw, staticmethod)
    orig = raw.__func__ if is_static else raw
    if not hasattr(orig, "__code__"):
        return
    w = _mk_wrapper(orig, f"{cls.__name__}.{name}", note)
    setattr(cls, name, staticmethod(w) if is_static else w)


def install_hooks():
    """在 pipeline 的命名空间里挂钩子。

    注意 pipeline.py 用的是 `from .intent import route`，
    所以 patch rag.intent.route 是没用的 —— 必须 patch pipeline 模块里的那个名字。
    这本身就是一个值得记住的 Python 知识点。
    """
    import rag.pipeline as P
    from rag.retriever import HybridRetriever

    # 检索前的分流层
    _patch_module_func(P, "structure", "把口语问题拆成 主体/行为/场所/对象")
    _patch_module_func(P, "route", "判意图：law / emergency / refuse / chitchat")
    _patch_module_func(P, "assess_risk", "按场景定风险等级（规则，检索前完成）")
    _patch_module_func(P, "clarify_question", "缺关键要素时生成澄清问句")
    _patch_module_func(P, "detect_query_mode", "local=具体场景 / global=职责总述")
    _patch_module_func(P, "decompose_queries", "复合问拆成多路子查询")
    _patch_method(P.Pipeline, "_is_gibberish", "乱码/无效输入护栏")
    _patch_method(P.Pipeline, "_mentioned_law_not_in_kb", "点名了库外法规就拦下")

    # 检索与判定层
    _patch_method(HybridRetriever, "retrieve", "三路召回 + RRF 融合 + 精排")
    _patch_method(P.Pipeline, "_crag_judge", "证据够不够：Correct/Ambiguous/Incorrect")
    _patch_method(P.Pipeline, "_needs_guide", "问法太宽 → 先引导")

    # 生成与核验层
    _patch_method(P.Pipeline, "_llm_structured", "调大模型，要结构化 JSON")
    _patch_method(P.Pipeline, "_extractive_structured", "降级：直接摘录原文，不靠模型")
    _patch_method(P.Pipeline, "_build_refs", "把命中条款整理成引用列表")
    _patch_module_func(P, "verify", "核验：回答里的条款/金额/主体对不对得上依据")
    _patch_module_func(P, "suggest_followups", "生成相关追问")


def run(question, previous=None):
    import json as _json
    import rag.pipeline as P

    _log.clear()
    _depth[0] = 0

    chunks_path = os.path.join(BASE, "data", "chunks.json")
    with open(chunks_path, encoding="utf-8") as f:
        chunks = _json.load(f)

    global _PL
    if _PL is None:
        sys.stderr.write("（首次加载模型，约 10–30 秒…）\n")
        _PL = P.Pipeline(chunks)

    t0 = time.time()
    result = _PL.ask(question, previous_question=previous, use_cache=False)
    total = (time.time() - t0) * 1000

    print()
    print(_c(BOLD, f"问题：{question}"))
    if previous:
        print(_c(DIM, f"上一轮：{previous}"))
    print(_c(DIM, "─" * 78))

    for i, e in enumerate(_log, 1):
        pad = "  " * e["depth"]
        loc = f'{e["file"]}:{e["line"]}'
        print(f'{_c(DIM, f"{i:2d}.")} {pad}{_c(CYAN, e["label"])}  {_c(DIM, loc)}')
        print(f'{pad}      {_c(DIM, "作用：")}{e["note"]}')
        ms = _c(DIM, "[%.0fms]" % e["ms"])
        print(f'{pad}      {_c(GRN, "结果：")}{e["brief"]}  {ms}')

    print(_c(DIM, "─" * 78))
    color = RED if result.get("refused") else YEL
    print(
        f'{_c(BOLD, "最终")}  intent={_c(color, result.get("intent"))}  '
        f'crag={result.get("crag")}  refused={result.get("refused")}'
    )
    print(f'      stages = {result.get("stages")}')
    print(f'      策略   = {result.get("strategy")}')
    refs = result.get("references") or []
    if refs:
        print(f'      依据   = {", ".join(r["article"] for r in refs)}')
    print(f'      耗时   = {total:.0f}ms')
    ans = (result.get("answer") or "").replace("\n", "\n               ")
    print(f'      回答   = {ans[:400]}')
    print()


_PL = None

SAMPLES = [
    ("楼道堆放杂物违反什么规定", None),      # 正常法规问答，走完整链路
    ("家里着火了现在怎么办", None),          # 应急：根本不该进检索
    ("网络安全法第二十一条是什么", None),    # 库外法规：检索前就拦
    ("你好呀", None),                        # 闲聊：引导回可问范围
    ("消防", None),                          # 问法太宽：先引导
]


if __name__ == "__main__":
    install_hooks()
    args = sys.argv[1:]
    if args:
        run(args[0], args[1] if len(args) > 1 else None)
    else:
        print(_c(BOLD, "\n=== 5 个对照样例：注意它们在哪一步分道扬镳 ==="))
        for q, prev in SAMPLES:
            run(q, prev)
