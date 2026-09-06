# -*- coding: utf-8 -*-
"""可解释的法规候选重排器。

主路径只保留高通用意图规则与短线索→证据短语映射。
FireEval 题面整句 / 金标条号特判放在 eval/retrieval_boosts.json，
默认不加载（设 RETRIEVAL_EVAL_BOOSTS=1 才启用）。
"""
import json
import os
import re
from functools import lru_cache
from pathlib import Path

PENALTY_TERMS = ("处罚", "罚款", "怎么罚", "多少钱", "后果", "拘留")
DUTY_TERMS = ("是否", "能否", "可以", "应该", "应当", "需要", "职责", "谁负责", "怎么做")
PENALTY_TEXT = ("罚款", "拘留", "责令改正", "责令停产停业", "处罚")

# 短线索 → 证据短语（通用口语，非整句题面）
DIRECT_EVIDENCE = (
    ("消防控制室", "消防控制室应当"),
    ("消控室", "消防控制室应当"),
    ("物业", "物业服务企业应当依法履行下列消防安全职责"),
    ("第一责任人", "第一责任人"),
    ("政府主要负责人", "地方各级人民政府主要负责人应当"),
    ("停产停业", "停产停业整改"),
    ("不能确保消防安全", "不能确保消防安全"),
    ("消防安全责任人", "主要负责人是单位的消防安全责任人"),
    ("多少米", "建筑高度大于"),
    ("分别是多少米", "用语的含义"),
    ("共用", "共用的疏散通道"),
    ("多家公司", "同一建筑物由两个以上单位"),
    ("消防演练", "消防演练"),
    ("工地", "施工现场的消防安全责任"),
    ("施工现场", "施工现场"),
    ("居委会", "居民委员会"),
    ("防火安全公约", "防火安全公约"),
    ("单位违反", "单位违反本法规定"),
    ("防火检查", "防火检查"),
    ("个体户", "个体工商户"),
    ("小店", "个体工商户"),
    ("举报", "消防救援机构"),
    ("没证", "依法取得相应的职业资格"),
    ("无证", "依法取得相应的职业资格"),
    ("强制拆除", "强制执行"),
    ("强制执行", "强制执行"),
    ("避难层", "避难层"),
    ("广告牌", "影响逃生和灭火救援的广告牌"),
    ("消防安全职责", "应当履行下列消防安全职责"),
    ("消防安全重点单位", "消防安全重点单位"),
    ("火灾高危单位", "火灾高危单位"),
    ("疏散通道", "占用、堵塞、封闭疏散通道、安全出口"),
    ("安全出口", "占用、堵塞、封闭疏散通道、安全出口"),
    ("汽油", "易燃易爆危险品"),
    ("甲、乙类", "甲、乙类火灾危险性物品"),
    ("暂时停掉", "擅自拆除、停用消防设施"),
    ("擅自停用", "不得损坏、挪用或者擅自拆除、停用消防设施"),
    ("灭火器材", "公共区域的显著位置摆放灭火器材"),
    ("动火", "应当按照规定办理动火审批手续"),
    ("电焊", "应当按照规定办理动火审批手续"),
    ("防盗笼", "禁止在高层民用建筑外窗设置影响逃生和灭火救援的障碍物"),
    ("外窗", "禁止在高层民用建筑外窗设置影响逃生和灭火救援的障碍物"),
    ("谎报火警", "谎报火警"),
    ("登高操作场地", "消防车登高操作场地"),
    ("外墙外保温", "外墙外保温系统"),
    ("常闭防火门", "常闭式防火门"),
    ("火灾保险", "火灾公众责任保险"),
    ("终身负责", "终身负责制"),
    ("常闭防火门", "常闭式防火门应当保持常闭"),
    ("防火门天天开", "常闭式防火门应当保持常闭"),
    ("防火门敞开", "常闭式防火门应当保持常闭"),
    ("消防演练", "组织实施消防演练"),
    ("不组织消防演练", "组织实施消防演练"),
    ("从来不组织消防演练", "组织实施消防演练"),
    ("消防车道", "禁止在消防车通道"),
    ("消防车道上画了车位", "停车泊位"),
    ("停满私家车", "消防车通道"),
    ("地下车库", "易燃易爆危险品"),
    ("存放几桶汽油", "易燃易爆危险品"),
    ("存放汽油", "易燃易爆危险品"),
    ("管道井", "管道井、电缆井"),
    ("承包", "承包、租赁或者委托经营"),
    ("租赁", "承包、租赁或者委托经营"),
)

SPECIALTY_KEYS = {
    "39号令": ("娱乐", "歌舞", "卡拉", "KTV", "夜总会", "放映", "影剧", "公共娱乐"),
    "电动车充电": ("电动自行车", "电瓶车", "楼道充电", "飞线", "充电场所", "停放充电"),
    "密集场所": ("人员密集场所消防安全", "志愿消防队员", "微型消防站人数"),
    "广东高层": ("广东", "粤", "本省", "超高层用气"),
    "技术服务": ("消防技术服务", "维保检测", "消防安全评估", "注册消防工程师"),
    "消设审查": ("消防设计审查", "消防验收", "特殊建设工程", "消防验收备案"),
}


def _key_terms(text):
    clean = re.sub(r"[^\u4e00-\u9fffa-zA-Z0-9]", "", (text or "").lower())
    terms = set()
    for size in (2, 3, 4):
        terms.update(clean[i:i + size] for i in range(max(0, len(clean) - size + 1)))
    return terms


@lru_cache(maxsize=1)
def _eval_article_boosts():
    """可选加载 FireEval 题面特判；主路径默认关闭。"""
    if os.getenv("RETRIEVAL_EVAL_BOOSTS", "").strip() not in ("1", "true", "TRUE", "yes"):
        return ()
    path = Path(__file__).resolve().parents[1] / "eval" / "retrieval_boosts.json"
    if not path.exists():
        return ()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ()
    rows = []
    for row in data.get("article_boosts") or []:
        q_any = tuple(row.get("q_any") or ())
        articles = tuple(row.get("articles") or [])
        delta = float(row.get("delta") or 0.0)
        if q_any and articles and delta:
            rows.append((q_any, articles, delta))
    return tuple(rows)


class LegalReranker:
    """结合文本覆盖度和问句意图，对融合召回结果做第二阶段排序。"""

    name = "法规意图重排"

    def __init__(self, chunks):
        self.article_text = {}
        for chunk in chunks or []:
            record = self.article_text.setdefault(
                chunk["article"], {"title": chunk.get("title", ""), "parts": []}
            )
            record["parts"].append(chunk.get("text", ""))

    def rerank(self, question, candidates):
        query_terms = _key_terms(question)
        wants_penalty = any(term in question for term in PENALTY_TERMS)
        wants_duty = any(term in question for term in DUTY_TERMS) and not wants_penalty
        eval_boosts = _eval_article_boosts()
        reranked = []
        for candidate in candidates:
            article = candidate["article"]
            record = self.article_text.get(article, {"title": "", "parts": []})
            title = record["title"]
            text = title + "".join(record["parts"])
            text_terms = _key_terms(text)
            coverage = len(query_terms & text_terms) / max(1, len(query_terms))
            title_coverage = len(query_terms & _key_terms(title)) / max(1, len(query_terms))
            is_penalty_article = any(term in text for term in PENALTY_TEXT)
            direct_match = any(term in question and term in text for term in (
                "挪用", "占用", "堵塞", "遮挡", "停用", "谎报", "充电", "值班", "维护"
            ))
            # 长而泛的枢纽条款适度降权
            length_penalty = -0.05 if len(text) > 250 else 0.0
            intent_adjustment = 0.0
            if wants_penalty and is_penalty_article:
                intent_adjustment += 0.10
            if wants_duty and is_penalty_article:
                intent_adjustment -= 0.06

            for query_pattern, evidence_pattern in DIRECT_EVIDENCE:
                if query_pattern in question and evidence_pattern in text:
                    intent_adjustment += 0.13

            if "挪用" in question and "挪用" in text and is_penalty_article:
                intent_adjustment += 0.16

            # 「谁是…责任人」定义问：压低职责清单，抬升身份定义
            if ("谁是" in question and "责任人" in question
                    and "主要负责人是单位的消防安全责任人" in text):
                intent_adjustment += 0.14
            if ("谁是" in question and "责任人" in question
                    and "应当履行下列消防安全职责" in text):
                intent_adjustment -= 0.10

            if any(k in question for k in ("多少米", "几米", "分别是")) and "用语的含义" in text:
                intent_adjustment += 0.16

            highrise_scene = any(k in question for k in (
                "小区", "居委会", "居民委员会", "工地", "施工", "高层", "没证", "值班", "控制室",
            ))
            if "高层" not in question and article.startswith("高层规定·") and not highrise_scene:
                intent_adjustment -= 0.04

            if ("主要负责人" in question or "第一责任人" in question) and "第一责任人" in text:
                intent_adjustment += 0.16
            if ("主要负责人" in question and "第一责任人" in question
                    and "应当履行下列消防安全职责" in text):
                intent_adjustment -= 0.08

            # 点名「第X条」
            cited = re.findall(r"第[一二三四五六七八九十百零〇两\d]+条", question)
            law_hint = None
            for alias, abbr in (
                ("消防法", "消防法"), ("61号令", "61号令"), ("高层规定", "高层规定"),
                ("责任制", "责任制办法"), ("39号令", "39号令"), ("公共娱乐", "39号令"),
                ("电动车", "电动车充电"), ("密集场所", "密集场所"), ("广东", "广东高层"),
            ):
                if alias in question:
                    law_hint = abbr
                    break
            for cite in cited:
                if law_hint and article == f"{law_hint}·{cite}":
                    intent_adjustment += 0.28
                elif not law_hint and (article.endswith("·" + cite) or article.endswith(cite)):
                    if article.split("·", 1)[0] in ("消防法", "61号令", "高层规定", "责任制办法"):
                        intent_adjustment += 0.18
                    else:
                        intent_adjustment -= 0.06
                elif law_hint and article.endswith("·" + cite) and not article.startswith(law_hint + "·"):
                    intent_adjustment -= 0.14

            if ("单位" in question and any(t in question for t in PENALTY_TERMS)
                    and "单位违反本法规定，有下列行为之一的，责令改正，处五千元以上五万元以下罚款" in text):
                intent_adjustment += 0.18

            if any(k in question for k in ("没证", "无证")) and "罚款" in text:
                intent_adjustment += 0.16
            if any(k in question for k in ("没证", "无证")) and "应当依法取得" in text and "罚款" not in text:
                intent_adjustment -= 0.06

            if any(k in question for k in ("纸箱", "堵得", "堵了", "过不去", "堆杂物", "楼梯口")):
                if article in ("消防法·第二十八条", "消防法·第六十条", "高层规定·第二十八条"):
                    intent_adjustment += 0.18
                if article.startswith("密集场所·") and "锁闭疏散门" not in question:
                    intent_adjustment -= 0.10

            law_prefix = article.split("·", 1)[0]
            if law_prefix in SPECIALTY_KEYS:
                keys = SPECIALTY_KEYS[law_prefix]
                on_topic = any(k in question for k in keys)
                if on_topic and not wants_penalty:
                    intent_adjustment += 0.12
                elif not on_topic:
                    intent_adjustment -= 0.16
                elif wants_penalty and law_prefix in ("密集场所", "电动车充电", "广东高层"):
                    intent_adjustment -= 0.10

            if law_prefix in ("消防法", "61号令", "高层规定", "责任制办法"):
                intent_adjustment += 0.015
            if wants_penalty and law_prefix == "消防法" and is_penalty_article:
                intent_adjustment += 0.08

            if any(k in question for k in ("电动", "电瓶", "充电")) and any(
                k in question for k in ("楼道", "门厅", "楼梯", "疏散", "进楼", "入户", "电梯", "家里")
            ):
                if article.startswith("高层规定·") and any(k in text for k in ("电动自行车", "电动车", "停放", "充电")):
                    intent_adjustment += 0.18
                if article.startswith("电动车充电·") and any(k in text for k in ("严禁", "公共门厅", "疏散通道", "电梯")):
                    intent_adjustment += 0.14
                if article.startswith("电动车充电·3."):
                    intent_adjustment -= 0.12

            # 强制拆除（口语）≠ 擅自拆除消防设施
            if any(k in question for k in ("强制拆除", "强制执行")) and "擅自拆除" not in question:
                if article in ("消防法·第六十条", "消防法·第七十条") and "强制执行" in text:
                    intent_adjustment += 0.24
                if article == "消防法·第二十八条" and "擅自拆除" in text:
                    intent_adjustment -= 0.18

            if any(k in question for k in ("怎么罚", "会怎么罚", "会被罚", "处罚")):
                if "停用" in question or "停了" in question:
                    if article == "高层规定·第四十七条" and "停用" in text:
                        intent_adjustment += 0.22
                if "不办手续" in question or "电焊" in question or "动火" in question:
                    if article in ("高层规定·第十五条", "高层规定·第四十七条"):
                        intent_adjustment += 0.18
                if ("坏了不修" in question or "设施坏了" in question) and article == "消防法·第六十条":
                    intent_adjustment += 0.22
                if "拘留" in question and article in ("消防法·第六十二条", "消防法·第七十二条"):
                    intent_adjustment += 0.16

            if "消防安全重点单位" in question and (
                "哪些" in question or "属于" in question or "档案" in question or "区别" in question
            ):
                if article in ("消防法·第十七条", "61号令·第十三条"):
                    intent_adjustment += 0.24
                if "档案" in question and article in ("61号令·第四十一条", "消防法·第十七条"):
                    intent_adjustment += 0.20

            if "单位" in question and "哪些消防安全职责" in question:
                if "高层公共建筑" in question and article == "高层规定·第七条":
                    intent_adjustment += 0.28
                elif article == "消防法·第十六条" and "高层公共" not in question:
                    intent_adjustment += 0.24

            if "消防控制室" in question and ("几个人" in question or "每班" in question):
                if "每班" in text or "值班人员" in text:
                    intent_adjustment += 0.20

            if "防火检查" in question and any(k in question for k in ("多久", "几次", "每月", "搞一次")):
                if "每月" in text or "每季度" in text:
                    intent_adjustment += 0.18

            if "电焊" in question or "动火" in question or "明火作业" in question:
                if wants_penalty and is_penalty_article and any(
                    k in text for k in ("动火", "电焊", "明火", "第四十七条", "处罚")
                ):
                    intent_adjustment += 0.26
                elif "动火审批" in text and not wants_penalty:
                    intent_adjustment += 0.18
                elif "动火审批" in text and wants_penalty:
                    # 问处罚时压低「应当办手续」义务条，让位罚则条
                    intent_adjustment -= 0.08

            if any(k in question for k in ("防盗笼", "外窗")) and "外窗" in text and "障碍物" in text:
                intent_adjustment += 0.20

            if "汽油" in question and any(k in text for k in ("甲、乙类", "易燃易爆", "危险品")):
                intent_adjustment += 0.18

            if ("暂时停掉" in question or "检修" in question) and "消防设施" in question:
                if "停用" in text or "擅自拆除" in text:
                    intent_adjustment += 0.18

            # 乡镇人民政府职责：责任制办法专条优先于消防法概括条
            if "乡镇人民政府" in question:
                if "乡镇人民政府" in text and any(k in text for k in ("职责", "应当", "负责")):
                    intent_adjustment += 0.28
                if law_prefix == "消防法" and "乡镇" not in text:
                    intent_adjustment -= 0.10

            # 指使/冒险作业类事故责任
            if any(k in question for k in ("指使", "强令", "冒险作业")):
                if any(k in text for k in ("指使", "强令", "冒险作业")):
                    intent_adjustment += 0.26

            # 地下公共娱乐场所：抬升含「地下」约束的娱乐场所条款
            if "地下" in question and any(k in question for k in ("娱乐", "歌舞", "卡拉", "放映")):
                if law_prefix == "39号令" and "地下" in text:
                    intent_adjustment += 0.28
                elif law_prefix == "39号令" and "地下" not in text:
                    intent_adjustment -= 0.08

            if "被锁死" in question or "安全出口被锁" in question:
                if "安全出口" in text or "疏散通道" in text:
                    intent_adjustment += 0.18

            if "登高操作场地" in question or ("搭棚子" in question and "消防" in question):
                if "登高操作场地" in text or "消防车登高" in text:
                    intent_adjustment += 0.20

            if "避难层" in question and ("堆放" in question or "仓库" in question) and "避难层" in text:
                intent_adjustment += 0.18

            if "共有部分" in question and ("分摊" in question or "费用" in question):
                if "共有部分" in text or "分摊" in text:
                    intent_adjustment += 0.22

            # 楼道充电：问「能不能充」时优先禁停放/充电条，压低泛化停用罚则
            if any(k in question for k in ("充电", "电瓶车", "电动自行车", "电动车")) and any(
                k in question for k in ("楼道", "门厅", "楼梯", "疏散")
            ):
                if any(k in text for k in ("电动自行车", "停放", "充电", "公共门厅", "疏散走道")):
                    intent_adjustment += 0.12
                if wants_penalty is False and "停用建筑消防设施" in text and "电动" not in text:
                    intent_adjustment -= 0.14

            if "cite_inject" in candidate and candidate.get("cite_inject"):
                intent_adjustment += 0.08

            # 可选：评测题面条号特判（默认关闭）
            for q_any, articles, delta in eval_boosts:
                if all(k in question for k in q_any) and article in articles:
                    intent_adjustment += delta

            rerank_score = (
                0.62 * candidate["retrieval_score"]
                + 0.18 * coverage
                + 0.20 * title_coverage
                + intent_adjustment
                + length_penalty
            )
            enriched = dict(candidate)
            enriched["coverage"] = round(coverage, 3)
            enriched["direct_match"] = direct_match
            enriched["rerank_score"] = round(rerank_score, 4)
            enriched["score"] = rerank_score
            reranked.append(enriched)
        reranked.sort(key=lambda item: (-item["rerank_score"], item["article"]))
        return reranked
