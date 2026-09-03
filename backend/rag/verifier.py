# -*- coding: utf-8 -*-
"""答案核验器：LLM 生成回答后，独立检查结论与引用条款的一致性。

核验规则（任一失败则判定不通过，触发重生成或降级）：
1. 条款存在性：回答引用的条款必须来自本次检索到的依据，防止编造；
2. 处罚金额一致：回答中的罚款金额必须能在引用条款原文中找到；
3. 处罚主体匹配：对单位的处罚不得引用只有个人处罚的条款（反之亦然）；
4. 建筑适用性：未提及高层的问题不得以高层专项规定作为唯一处罚依据；
5. 处罚意图覆盖：问处罚时，依据中必须包含真正的处罚条款，而非仅义务条款。
"""
import re

# 中文数字 → 数值（覆盖法规罚则中的常见写法）
_CN_NUM = {"零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
           "六": 6, "七": 7, "八": 8, "九": 9, "十": 10, "百": 100, "千": 1000, "万": 10000}

PENALTY_INTENT_WORDS = ("罚", "拘留", "处罚", "处罚款", "责令", "后果", "责任")
FINE_PATTERN = re.compile(
    r"([零一二两三四五六七八九十百千万\d]+)\s*(?:元|万?元)(?:以上)?(?:至|到|—|~|-)?"
    r"(?:([零一二两三四五六七八九十百千万\d]+)\s*万?元)?(?:以下)?")
PENALTY_ARTICLE_WORDS = ("罚款", "拘留", "责令改正", "责令停产停业", "处罚", "警告")


def _cn_to_int(text):
    """中文数字/阿拉伯数字转 int；无法解析返回 None。支持 一万/二十万/五千/一百二十三 等。"""
    text = text.strip()
    if re.fullmatch(r"\d+", text):
        return int(text)
    total, section, num = 0, 0, 0
    for ch in text:
        if ch not in _CN_NUM:
            return None
        v = _CN_NUM[ch]
        if v >= 10000:
            total += (section + num or 1) * v
            section, num = 0, 0
        elif v >= 10:
            section += (num or 1) * v
            num = 0
        else:
            num = num * 10 + v
    return total + section + num


def _extract_fine_amounts(text):
    """从文本提取罚款金额区间（元），返回 [(low, high)]。"""
    amounts = []
    for m in FINE_PATTERN.finditer(text or ""):
        low = _cn_to_int(m.group(1))
        high = _cn_to_int(m.group(2)) if m.group(2) else low
        if low is not None and high is not None:
            if high < low:  # "五万元以上二十万元以下" 中 second 是"二十万"
                high = high if high > low else low
            amounts.append((low, high))
    return amounts


# 法规简称 → 回答中可能出现的各种写法
LAW_ALIASES = {
    "消防法": ["消防法", "中华人民共和国消防法"],
    "61号令": ["61号令", "机关、团体、企业、事业单位消防安全管理规定", "六十一号令"],
    "高层规定": ["高层民用建筑消防安全管理规定", "高层规定"],
    "责任制办法": ["消防安全责任制实施办法", "责任制办法"],
}


def _extract_cited_articles(answer, known_articles):
    """从回答中提取引用的条款 key（需在 known_articles 中）。"""
    text = (answer or "").replace(" ", "")
    hits = set()
    for key in known_articles:
        law, num = key.split("·", 1)
        aliases = LAW_ALIASES.get(law, [law])
        # (?![一二三四五六七八九十百]) 防止「第六条」误匹配「第六十条」
        num_hit = re.search(num + "(?![一二两三四五六七八九十百])", text)
        if num_hit and any(alias.replace(" ", "") in text for alias in aliases):
            hits.add(key)
    return hits


def verify(answer, references, question, scene=None):
    """核验回答与引用的一致性。

    返回 {"passed": bool, "checks": [...], "issues": [...], "cited": [...]}
    """
    ref_articles = {r["article"] for r in references}
    ref_text = " ".join(r.get("text", "") for r in references)
    checks, issues = [], []

    # 1. 条款存在性：回答引用的条款必须在检索依据内
    cited = _extract_cited_articles(answer, ref_articles)
    # 编造检测：回答引用了知识库中存在、但本次未检索到的条款
    fabricated = set()
    if _ALL_ARTICLE_KEYS:
        mentioned = _extract_cited_articles(answer, _ALL_ARTICLE_KEYS)
        fabricated = mentioned - ref_articles
    checks.append({"rule": "条款存在性", "passed": not fabricated})
    if fabricated:
        checks[-1]["detail"] = f"引用了未检索到的条款: {sorted(fabricated)}"
        issues.append(f"引用条款 {sorted(fabricated)} 不在检索依据中，疑似编造")

    # 2. 处罚金额一致：回答中的罚款金额需在依据条款原文中出现
    ans_amounts = _extract_fine_amounts(answer)
    ref_amounts = set(_extract_fine_amounts(ref_text))
    amount_ok = True
    if ans_amounts:
        for low, high in ans_amounts:
            if high >= 10000 and (low, high) not in ref_amounts:
                # 允许 answer 把"五千元以上五万元以下"写成 (5000, 50000)
                if not any(abs(r_low - low) <= 1 and abs(r_high - high) <= 1
                           for r_low, r_high in ref_amounts):
                    amount_ok = False
                    issues.append(f"罚款金额 {low}~{high} 元未在引用条款中找到")
    checks.append({"rule": "处罚金额一致", "passed": amount_ok})

    # 3. 处罚主体匹配
    subject_ok = True
    mentions_unit_penalty = bool(re.search(r"对单位|单位处|单位.{0,6}(?:罚款|处罚)", answer or ""))
    mentions_person_penalty = bool(re.search(r"对个人|个人处|个人.{0,6}(?:罚款|处罚)", answer or ""))
    if mentions_unit_penalty and not mentions_person_penalty:
        # 仅当依据明确指向另一主体（只提个人不提单位）时才判失败；
        # 依据未提主体时无法证伪，保持中性
        refs_person = any(re.search(r"个人", r.get("text", "")) for r in references)
        refs_unit = any(re.search(r"单位", r.get("text", "")) for r in references)
        if refs_person and not refs_unit:
            subject_ok = False
            issues.append("回答称对单位处罚，但依据条款仅涉及个人处罚")
    if mentions_person_penalty and not mentions_unit_penalty:
        refs_person = any(re.search(r"个人", r.get("text", "")) for r in references)
        refs_unit = any(re.search(r"单位", r.get("text", "")) for r in references)
        if refs_unit and not refs_person:
            subject_ok = False
            issues.append("回答称对个人处罚，但依据条款仅涉及单位处罚")
    checks.append({"rule": "处罚主体匹配", "passed": subject_ok})

    # 4. 建筑适用性：非高层问题不得仅以高层规定作为处罚依据
    building_ok = True
    is_highrise_question = bool("高层" in (question or "")) or bool(
        scene and (scene.get("building_hint") or any(v.startswith("高层") for v in scene.get("venues", []))))
    if not is_highrise_question and mentions_penalty_answer(answer):
        highrise_only = ref_articles and all(a.startswith("高层规定·") for a in ref_articles)
        if highrise_only:
            building_ok = False
            issues.append("非高层场景仅引用了高层专项规定作为处罚依据")
    checks.append({"rule": "建筑适用性", "passed": building_ok})

    # 5. 处罚意图覆盖：问处罚时依据必须含处罚条款
    coverage_ok = True
    if any(w in (question or "") for w in PENALTY_INTENT_WORDS) and mentions_penalty_answer(answer):
        if not any(any(w in r.get("text", "") for w in PENALTY_ARTICLE_WORDS) for r in references):
            coverage_ok = False
            issues.append("处罚类回答的依据条款中未包含任何处罚条款")
    checks.append({"rule": "处罚意图覆盖", "passed": coverage_ok})

    return {"passed": not issues, "checks": checks, "issues": issues,
            "cited": sorted(cited)}


def mentions_penalty_answer(answer):
    return any(w in (answer or "") for w in PENALTY_ARTICLE_WORDS)


# 全局条款键集合（由 pipeline 注入，用于编造检测）
_ALL_ARTICLE_KEYS = None


def set_article_keys(keys):
    global _ALL_ARTICLE_KEYS
    _ALL_ARTICLE_KEYS = set(keys)
