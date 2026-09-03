# -*- coding: utf-8 -*-
"""意图路由：4路规则引擎（法规/应急/拒答/闲聊）
比赛版可升级为微调分类器。
"""
from .graphrag import PUNISH_KEYWORDS

FIRE_DOMAIN_WORDS = [
    "消防", "法规", "规定", "违反", "条款", "火灾", "疏散", "安全出口", "消火栓",
    "灭火器", "电动车", "电动自行车", "高层住宅", "高层建筑", "物业", "防火", "动火",
    "消防控制室", "消控室", "应急预案", "火警",
    # 口语场景常用词：让"占用通道罚多少钱"这类问题进入法规流程而非误拒答
    "通道", "楼道", "堆放", "杂物", "消防车", "着火点", "逃生",
    # 泛化建筑与作业词（"高层民用建筑""电焊气焊"等口语组合常不含完整词组）
    "高层", "民用建筑", "公共建筑", "建筑",
    "违规", "电焊", "气焊", "明火", "装修", "施工",
    "演练", "巡查", "检查记录", "职责", "灭火", "报警",
    "烟感", "探测器", "灭火系统", "喷淋", "验收", "避难层", "防火间距",
]

# 闲聊识别
CHAT_WORDS = ["你好", "您好", "谢谢", "你是谁", "介绍一下", "能做什么", "再见", "hello", "hi ", "在吗", "加油"]

# 明确描述正在发生危险的信号。不能把“报警有什么规定”一类法规问题误判为求救。
EMERGENCY_SIGNALS = ["着火", "起火", "火情", "冒烟", "烟很大", "困住了", "被困", "爆炸",
                     "救命", "出不去", "全是烟", "浓烟", "烧起来", "火势", "烧着", "快烧到"]
EMERGENCY_CONTEXT = ["怎么办", "现在", "现场", "家里", "楼里", "发生", "逃生", "救命"]

# 拒答词（隐私 / 与消防无关）
REFUSE_WORDS = ["你多大", "你女朋友", "作弊", "帮我犯罪", "赚钱方法"]


def route(question):
    """返回 intent: law / emergency / chitchat / refuse"""
    q = question.lower()
    # 拒答优先
    if any(w in q for w in REFUSE_WORDS) and "消防" not in q:
        return "refuse"
    # 应急求救：强信号可直接命中；“火灾/119”需要同时有处置语境。
    if any(w in q for w in EMERGENCY_SIGNALS) or (
        any(w in q for w in ["火灾", "119", "报警"])
        and any(w in q for w in EMERGENCY_CONTEXT)
    ):
        return "emergency"
    # 法规询问（含处罚词或消防相关对象关键词）
    if any(k in q for k in PUNISH_KEYWORDS) or any(k in q for k in FIRE_DOMAIN_WORDS) and "闲聊" not in q:
        return "law"
    # 闲聊兜底
    if any(w in q for w in CHAT_WORDS) or len(q) <= 2:
        return "chitchat"
    # 没有消防语义的开放问题直接拒答，避免偶然词项造成误检索。
    return "refuse"
