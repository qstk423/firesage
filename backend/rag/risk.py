# -*- coding: utf-8 -*-
"""风险等级评估（规则版，第一版不上模型）：

在场景结构化之后、检索之前打标签，纯规则、稳定可复现。

  emergency — 着火/冒烟/爆炸/被困等正在发生的险情（红条 + 强制 119 指引）
  high      — 疏散受阻、飞线充电、占用消防车通道等高危害行为（高风险隐患）
  medium    — 一般违规、职责不清、处罚咨询
  low       — 概念问、职责清单、制度咨询

输出字段：
  risk_level:  low | medium | high | emergency
  risk_reasons: 1~3 条短因（答辩可讲，如「疏散通道受阻」）

设计原则：风险看场景严重度；可信度看证据够不够——两个维度互相独立、并排展示。
"""
from .intent import _has_active_emergency

# 高危行为：人员生命 / 灭火救援直接受威胁（对应 BEHAVIOR_MAP 的 canonical 词）
HIGH_BEHAVIORS = {
    "占用疏散通道", "占用消防车通道", "埋压圈占消火栓", "损坏挪用消防设施",
    "电动自行车违规停放充电", "违规明火作业", "违规储存危险品", "占用避难层",
    "设置障碍物影响逃生", "违规使用燃气", "高风险作业指使",
}

# 行为 → 短因文案（1~3 条，面向答辩与前端展示）
RISK_REASON_TEXT = {
    "占用疏散通道": "疏散通道受阻",
    "占用消防车通道": "占用消防车通道",
    "埋压圈占消火栓": "消火栓被埋压圈占",
    "损坏挪用消防设施": "消防设施损坏停用",
    "电动自行车违规停放充电": "电动自行车违规充电",
    "违规明火作业": "违规动火作业",
    "违规储存危险品": "违规储存易燃易爆品",
    "占用避难层": "避难层被占用",
    "设置障碍物影响逃生": "门窗设障碍影响逃生",
    "违规使用燃气": "违规使用燃气",
    "高风险作业指使": "违章指挥冒险作业",
    "占用电缆井管道井": "电缆井/管道井堆物",
    "占用防火间距": "占用防火间距",
    "谎报火警": "谎报火警",
    "无人值班": "消防控制室无人值班",
    "值班人员不足": "消防控制室值班不足",
    "未保持完好有效": "消防设施未保持完好有效",
    "未组织防火检查": "未开展防火检查",
    "未组织消防演练": "未组织消防演练",
    "未开展消防培训": "未开展消防培训",
}

# 高风险场所：命中时作为补充理由（第一版只加 reason 不升级等级）
HIGH_VENUES = ("人员密集场所", "高层民用建筑", "高层住宅建筑", "高层公共建筑")
_VENUE_TEXT = "涉及人员密集/高层场所"

RISK_NOTICE_HIGH = (
    "【风险提示】您描述的场景存在现实消防隐患，建议立即通知管理方整改；"
    "若现场已出现险情，请直接拨打 119。"
)


def assess(question, scene=None, intent=None):
    """风险分级主入口。规则短路、先到先得，检索前完成（零延迟成本）。"""
    scene = scene or {}
    # 1) 紧急：正在发生的险情（复用 intent 的应急信号词，保证口径一致）
    if intent == "emergency" or _has_active_emergency(question or ""):
        return {"risk_level": "emergency", "risk_reasons": ["疑似正在发生火情/险情"]}

    behaviors = scene.get("behaviors") or []
    venues = scene.get("venues") or []
    venue_hit = any(v in HIGH_VENUES for v in venues)

    # 2) 高危行为 → high
    high_hits = [b for b in behaviors if b in HIGH_BEHAVIORS]
    if high_hits:
        reasons = [RISK_REASON_TEXT.get(b, b) for b in high_hits[:2]]
        if venue_hit:
            reasons.append(_VENUE_TEXT)
        return {"risk_level": "high", "risk_reasons": reasons[:3]}

    # 3) 一般违规 / 处罚咨询 → medium
    if behaviors:
        reasons = [RISK_REASON_TEXT.get(b, b) for b in behaviors[:2]]
        if venue_hit:
            reasons.append(_VENUE_TEXT)
        return {"risk_level": "medium", "risk_reasons": reasons[:3]}
    if "处罚" in (scene.get("question_intents") or []):
        return {"risk_level": "medium",
                "risk_reasons": ["处罚咨询：主体/场景不明时结论分歧大"]}

    # 4) 概念问 / 职责清单 / 制度咨询 → low
    return {"risk_level": "low", "risk_reasons": []}


def notice(risk):
    """高风险场景的引导话术前缀；低/中风险返回 None 不打扰。"""
    if risk and risk.get("risk_level") == "high":
        return RISK_NOTICE_HIGH
    return None
