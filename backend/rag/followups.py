"""回答后的相关追问推荐（规则生成，不额外调 LLM）。"""

from __future__ import annotations

_FALLBACK = (
    "楼道堆放杂物违反什么规定",
    "占用消防通道怎么处罚",
    "电动车能不能在楼道充电",
    "消防设施坏了不修违法吗",
    "消防控制室可以没人值班吗",
    "高层住宅物业有哪些消防职责",
)

_PENALTY_MARKERS = ("怎么处罚", "怎么罚", "罚多少", "罚款", "拘留", "什么后果", "会怎么罚", "处罚标准")
_DUTY_MARKERS = ("谁负责", "责任人", "责任主体", "有哪些职责", "义务")


def _norm(s: str) -> str:
    return "".join(ch for ch in (s or "").strip().lower() if ch not in "？?。.!！、，,；;：: ")


def suggest_followups(
    *,
    question: str = "",
    intent: str | None = None,
    scene: dict | None = None,
    refused: bool = False,
    references: list | None = None,
    limit: int = 4,
) -> list[str]:
    """生成 3–4 条可点击的相关问题；拒答/引导时给可问示例。"""
    q = (question or "").strip()
    qn = _norm(q)
    out: list[str] = []
    seen = {qn} if qn else set()

    def add(text: str) -> None:
        text = (text or "").strip().rstrip("？?。")
        if not text or len(text) < 4:
            return
        key = _norm(text)
        if not key or key in seen or key == qn:
            return
        # 避免与当前问句高度重叠
        if qn and (key in qn or qn in key) and abs(len(key) - len(qn)) <= 2:
            return
        seen.add(key)
        out.append(text)

    scene = scene or {}
    behaviors = list(scene.get("behaviors") or [])
    objects = list(scene.get("objects") or [])
    venues = list(scene.get("venues") or [])
    subjects = [s for s in (scene.get("subjects") or []) if s not in ("个人", "我")]
    asked_penalty = any(m in q for m in _PENALTY_MARKERS)
    asked_duty = any(m in q for m in _DUTY_MARKERS)

    if intent == "emergency":
        add("疏散通道被杂物堵住违法吗")
        add("火场逃生时不能做什么")
        add("日常怎样检查楼道消防通道")
        add("物业发现占用通道应如何处理")
        return out[:limit]

    if intent == "clarify":
        # 澄清回合不抢话，给轻量备选
        for b in behaviors[:1]:
            add(f"如果是个人{b}怎么处罚")
            add(f"如果是单位{b}怎么处罚")
        add("占用的是疏散通道还是消防车通道")
        for t in _FALLBACK:
            add(t)
            if len(out) >= limit:
                break
        return out[:limit]

    if refused or intent in ("guide", "chitchat"):
        for t in _FALLBACK:
            add(t)
            if len(out) >= limit:
                break
        return out[:limit]

    # ---- 正常法规回答：围绕场景要素发散 ----
    for b in behaviors[:2]:
        if not asked_penalty:
            add(f"{b}怎么处罚")
        if not asked_duty:
            add(f"{b}谁负责")
        add(f"{b}违反什么规定")
        if len(out) >= limit:
            return out[:limit]

    for o in objects[:2]:
        if "电动" in o or "电瓶" in o:
            add(f"{o}能在楼道里充电吗")
            add(f"{o}违规停放怎么处罚")
        else:
            add(f"关于{o}有哪些消防规定")
            add(f"{o}损坏不修违法吗")
        if len(out) >= limit:
            return out[:limit]

    for v in venues[:1]:
        add(f"{v}占用消防通道怎么处理")
        add(f"{v}物业有哪些消防职责")
        if len(out) >= limit:
            return out[:limit]

    for s in subjects[:1]:
        add(f"{s}的消防安全责任有哪些")
        if not asked_penalty:
            add(f"{s}违反消防规定怎么处罚")
        if len(out) >= limit:
            return out[:limit]

    if behaviors and not asked_penalty:
        add("个人和单位的处罚标准有什么不同")
    if behaviors and asked_penalty and not asked_duty:
        add(f"{behaviors[0]}应当由谁整改")

    for ref in (references or [])[:3]:
        title = (ref.get("title") or "").strip()
        art = (ref.get("article") or "").strip()
        if title and title not in q:
            add(f"{title}具体规定了什么")
        elif art and art not in q:
            add(f"{art}说了什么")
        if len(out) >= limit:
            return out[:limit]

    for t in _FALLBACK:
        add(t)
        if len(out) >= limit:
            break

    return out[:limit]
