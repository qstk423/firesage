# -*- coding: utf-8 -*-
"""意图路由：规则引擎 + 可选领域微调分类器。

INTENT_BACKEND:
  rules  — 仅规则（默认无模型时）
  ml     — 仅微调分类器
  hybrid — 规则硬约束 + 微调分类器（有模型时默认）
"""
import os

from .graphrag import PUNISH_KEYWORDS

FIRE_DOMAIN_WORDS = [
    "消防", "法规", "规定", "违反", "条款", "火灾", "疏散", "安全出口", "消火栓",
    "灭火器", "电动车", "电动自行车", "高层住宅", "高层建筑", "物业", "防火", "动火",
    "消防控制室", "消控室", "应急预案", "火警",
    "通道", "楼道", "堆放", "杂物", "消防车", "着火点", "逃生",
    "高层", "民用建筑", "公共建筑", "建筑",
    "违规", "电焊", "气焊", "明火", "装修", "施工",
    "演练", "巡查", "检查记录", "职责", "灭火", "报警",
    "烟感", "探测器", "灭火系统", "喷淋", "验收", "避难层", "防火间距",
]

CHAT_WORDS = ["你好", "您好", "谢谢", "你是谁", "介绍一下", "能做什么", "再见", "hello", "hi ", "在吗", "加油"]

LAW_PRIORITY_MARKERS = [
    "隐患", "整改", "规定", "条款", "违反", "违法", "处罚", "罚款", "职责", "责任",
    "监督检查", "演练", "培训", "着火点", "怎么罚", "负什么责", "有什么要求",
    "如何整改", "应当怎么", "应当做", "追究", "谁负责", "责任在谁", "可以不报警",
]

LAW_SCENE_MARKERS = ["广告牌", "挡住", "堆放", "占用", "堵塞", "障碍"]

ACTIVE_EMERGENCY_SIGNALS = [
    "着火了", "起火了", "冒烟", "烟很大", "全是烟", "浓烟", "烧起来",
    "火很大", "火势很大", "火要烧过来", "被困", "困住了", "救命",
    "出不去", "爆炸声", "火情", "烧着", "快烧到", "火势",
]
WEAK_EMERGENCY_FIRE = ["着火", "起火"]
EMERGENCY_CONTEXT = ["怎么办", "现在", "现场", "家里", "楼里", "发生", "逃生", "救命", "怎么处理"]

REFUSE_WORDS = ["你多大", "你女朋友", "作弊", "帮我犯罪", "赚钱方法"]

# 明显非消防闲聊 / 娱乐请求：规则硬拒，避免微调分类器误判为 law
OFF_TOPIC_MARKERS = [
    "电影", "电视剧", "综艺", "游戏", "音乐", "明星", "八卦",
    "笑话", "段子", "天气", "旅游", "美食", "菜谱", "股票", "基金",
    "恋爱", "相亲", "星座", "运势", "足球", "篮球", "世界杯",
]


def _is_off_topic(q: str) -> bool:
    if any(w in q for w in FIRE_DOMAIN_WORDS):
        return False
    return any(w in q for w in OFF_TOPIC_MARKERS)


def _is_law_consultation(q: str) -> bool:
    if any(w in q for w in LAW_PRIORITY_MARKERS):
        return True
    if "火灾" in q and any(w in q for w in LAW_SCENE_MARKERS):
        return True
    return False


def _has_active_emergency(q: str) -> bool:
    q_fire = q.replace("着火点", "【点火位置】")
    if any(w in q_fire for w in ACTIVE_EMERGENCY_SIGNALS):
        return True
    if any(w in q_fire for w in WEAK_EMERGENCY_FIRE):
        return True
    if any(w in q for w in ["火灾", "119", "报警"]) and any(w in q for w in EMERGENCY_CONTEXT):
        return True
    return False


def route_rules(question):
    """纯规则路由：law / emergency / chitchat / refuse"""
    q = question.lower()
    if any(w in q for w in REFUSE_WORDS) and "消防" not in q:
        return "refuse"
    if _is_off_topic(q):
        return "refuse"
    if _is_law_consultation(q):
        return "law"
    if _has_active_emergency(q):
        return "emergency"
    if any(k in q for k in PUNISH_KEYWORDS) or any(k in q for k in FIRE_DOMAIN_WORDS) and "闲聊" not in q:
        return "law"
    if any(w in q for w in CHAT_WORDS) or len(q) <= 2:
        return "chitchat"
    return "refuse"


def _backend():
    mode = os.getenv("INTENT_BACKEND", "").strip().lower()
    if mode in ("rules", "ml", "hybrid"):
        return mode
    try:
        from .intent_ml import available
        return "hybrid" if available() else "rules"
    except Exception:
        return "rules"


def route(question):
    """对外路由入口（支持规则 / 微调 / 混合）。"""
    backend = _backend()
    if backend == "rules":
        return route_rules(question)

    from . import intent_ml

    if backend == "ml":
        pred = intent_ml.predict(question, min_confidence=0.0)
        return pred or route_rules(question)

    # hybrid：安全硬约束优先，其余交给微调分类器
    q = question.lower()
    if any(w in q for w in REFUSE_WORDS) and "消防" not in q:
        return "refuse"
    if _is_off_topic(q):
        return "refuse"
    if _is_law_consultation(q):
        return "law"
    if _has_active_emergency(q):
        return "emergency"
    pred = intent_ml.predict(question, min_confidence=0.42)
    if pred:
        return pred
    return route_rules(question)
