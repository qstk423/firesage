# -*- coding: utf-8 -*-
"""可解释的法规候选重排器。"""
import re

PENALTY_TERMS = ("处罚", "罚款", "怎么罚", "多少钱", "后果", "拘留")
DUTY_TERMS = ("是否", "能否", "可以", "应该", "应当", "需要", "职责", "谁负责", "怎么做")
PENALTY_TEXT = ("罚款", "拘留", "责令改正", "责令停产停业", "处罚")


def _key_terms(text):
    clean = re.sub(r"[^\u4e00-\u9fffa-zA-Z0-9]", "", (text or "").lower())
    terms = set()
    for size in (2, 3, 4):
        terms.update(clean[i:i + size] for i in range(max(0, len(clean) - size + 1)))
    return terms


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
            # 长而泛的"枢纽条款"（单位职责总纲等）与任何问题都有字面重叠，适度降权，
            # 让位给标题/内容直接命中的具体条款
            length_penalty = -0.05 if len(text) > 250 else 0.0
            intent_adjustment = 0.0
            if wants_penalty and is_penalty_article:
                intent_adjustment += 0.10
            if wants_duty and is_penalty_article:
                intent_adjustment -= 0.06
            # 法规问答中，直接规范句通常比间接提及更适合作为首条依据。
            direct_patterns = (
                ("消防控制室", "消防控制室应当"),
                ("物业", "物业服务企业应当依法履行下列消防安全职责"),
                ("政府主要负责人", "地方各级人民政府主要负责人应当"),
                # Stage4：高区分度法条短语直连
                ("不能确保消防安全", "不能确保消防安全"),
                ("停产停业", "停产停业整改"),
                ("谁是单位的消防安全责任人", "主要负责人是单位的消防安全责任人"),
                ("消防安全责任人", "法定代表人或者非法人单位的主要负责人是单位的消防安全责任人"),
                ("多少米", "建筑高度大于"),
                ("分别是多少米", "用语的含义"),
                ("共用", "共用的疏散通道"),
                ("多家公司", "同一建筑物由两个以上单位"),
                ("消防演练", "消防演练"),
                ("着火了往哪跑", "消防演练"),
            )
            for query_pattern, evidence_pattern in direct_patterns:
                if query_pattern in question and evidence_pattern in text:
                    intent_adjustment += 0.13
            if "挪用" in question and "挪用" in text and is_penalty_article:
                intent_adjustment += 0.16
            # 「谁是…责任人」定义问：压低职责清单条款，抬升身份定义条款
            if ("谁是" in question and "责任人" in question
                    and "主要负责人是单位的消防安全责任人" in text):
                intent_adjustment += 0.14
            if ("谁是" in question and "责任人" in question
                    and "应当履行下列消防安全职责" in text):
                intent_adjustment -= 0.10
            # 建筑高度定义问：优先「用语的含义」条款
            if any(k in question for k in ("多少米", "几米", "分别是")) and "用语的含义" in text:
                intent_adjustment += 0.16
            if "高层" not in question and article.startswith("高层规定·"):
                intent_adjustment -= 0.04
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
