# -*- coding: utf-8 -*-
"""问答主流程：意图路由 → 混合检索 → CRAG自纠错 → 生成（可溯源）
「宁可拒答不可答错」：检索质量低于阈值时触发拒答。
"""
import json
import os

from .intent import route
from .retriever import HybridRetriever
from .llm import LLMClient

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# CRAG 阈值
CRAG_AMBIGUOUS = 0.25   # 疑似明确但质量低 → 拒答
CRAG_INCORRECT = 0.15   # 完全未见相关 → 拒答

FOLLOW_UP_MARKERS = ["那", "那么", "这个", "这种", "上述", "还会", "具体", "罚多少", "多少钱", "怎么处理"]

# 抽取式回答模板（无LLM时的降级）
SYSTEM_TMPL = (
    "你是消防法规领域的专业助手「消安智答」。请严格依据提供的法规条款回答问题，"
    "不得编造。如条款不足以支撑，明确说明存疑。回答引用条款时带上条款编号。"
)


class Pipeline:
    def __init__(self, chunks=None):
        self.retriever = HybridRetriever(chunks)
        self.llm = LLMClient()
        self.chunk_index = {c["id"]: c for c in chunks or []}
        self.article_text = self._load_article_text()

    def _load_article_text(self):
        """条款号 -> {"num","title","text","chapter"}（多部法规，编号带法规前缀）"""
        import json as j
        from .graphrag import ARCHIVE_PATHS
        m = {}
        for path, law_abbr in ARCHIVE_PATHS:
            with open(path, encoding="utf-8") as f:
                law = j.load(f)
            if "law_abbr" in law:
                law_abbr = law["law_abbr"]
            for ch in law["chapters"]:
                for art in ch["articles"]:
                    key = f"{law_abbr}·{art['num']}"
                    m[key] = {"num": key, "title": art["title"],
                              "text": art["text"], "chapter": ch["chapter_title"],
                              "law": law_abbr, "law_name": law.get("law_name", law_abbr),
                              "authority": law.get("authority", ""),
                              "effective_date": law.get("effective_date", ""),
                              "source_url": law.get("source_url", "")}
        return m

    def _crag_judge(self, fused, question):
        """三分支判定：Correct / Ambiguous / Incorrect"""
        top = fused[0] if fused else None
        if not top or top["score"] <= CRAG_INCORRECT:
            return "Incorrect"
        # 只有泛化词命中、没有图谱实体且向量相似度很低时，不能因为排名第一就通过。
        if (not top.get("graph_hit") and not top.get("direct_match")
                and top.get("vector", 0) < 0.08
                and top.get("coverage", 0) < 0.5):
            return "Incorrect"
        if top["score"] <= CRAG_AMBIGUOUS:
            return "Ambiguous"
        return "Correct"

    def _extractive_answer(self, fused):
        """无LLM降级：从命中的最高条款抽取要点"""
        if not fused:
            return "抱歉，现有知识库中没有检索到相关内容，无法提供可靠的回答。", []
        top = fused[0]
        art = self.article_text.get(top["article"], {})
        law_name = art.get("law_name", art.get("law", "消防法规"))
        summary = f"根据《{law_name}》{art.get('num','')}「{art.get('title','')}」：{art.get('text','')[:120]}…"
        return summary, []

    def _llm_answer(self, question, fused):
        context = "\n\n".join(
            f"【{c['num']} {c['title']}】{c['text']}"
            for c in (self.article_text.get(f["article"], {}) for f in fused)
            if c
        )
        user = f"用户问题：{question}\n\n以下为检索到的法规条款：\n{context}\n\n请基于上述条款作答，逐句标注引用的条款编号。"
        resp = self.llm.complete(SYSTEM_TMPL, user)
        if resp:
            return resp, []
        return None, None

    def ask(self, question, previous_question=None):
        question = question.strip()
        previous_question = (previous_question or "").strip()
        context_used = bool(
            previous_question
            and len(question) <= 24
            and any(marker in question for marker in FOLLOW_UP_MARKERS)
        )
        retrieval_question = f"{previous_question}；追问：{question}" if context_used else question
        intent = route(retrieval_question if context_used else question)

        # ---- 应急模式：直接给逃生指引，不走RAG ----
        if intent == "emergency":
            return {
                "intent": intent,
                "answer": EMERGENCY_GUIDE,
                "refused": False,
                "crag": "Correct",
                "references": [],
                "graph_trace": {"matched": False},
                "strategy": "应急流程引导（内置）",
                "context_used": context_used,
                "citations": [{
                    "title": "国家消防救援局：公共场所火灾应急处理",
                    "url": "https://www.119.gov.cn/kp/hzyf/2022/622.shtml",
                }],
            }

        if intent == "refuse":
            return {
                "intent": intent,
                "answer": "抱歉，这是一个消防法规专业问答系统，超出范围的问题我无法回答。",
                "refused": True, "crag": "Refuse", "references": [], "citations": [],
                "graph_trace": {"matched": False}, "strategy": "拒答护栏",
                "context_used": context_used,
            }

        if intent == "chitchat":
            return {
                "intent": intent,
                "answer": "我是消安智答，可回答消防法规与应急处置问题，例如「楼道堆放杂物违反什么规定」。",
                "refused": False, "crag": "Chat", "references": [], "citations": [],
                "graph_trace": {"matched": False}, "strategy": "闲聊分流",
                "context_used": context_used,
            }

        # ---- 法规问答：混合检索 + CRAG ----
        # 三条证据足以覆盖主依据与交叉依据，避免弱相关条款稀释结论。
        fused, summary = self.retriever.retrieve(retrieval_question, top_k=3)
        crag = self._crag_judge(fused, retrieval_question)

        graph_matched = bool(summary["graph_articles"])
        graph_ratio = 0.0
        vector_ratio = 0.0
        if fused:
            graph_ratio = sum(f["graph_share"] for f in fused) / len(fused)
            vector_ratio = sum(f.get("vector_share", 0) for f in fused) / len(fused)

        # CRAG 低质量 → 拒答
        if crag in ("Ambiguous", "Incorrect"):
            return {
                "intent": intent,
                "answer": "很抱歉，针对您的问题，现有知识库中的检索结果置信度不足，为避免误导，暂不回答。建议换个说法，或咨询属地消防救援机构。",
                "refused": True, "crag": crag, "references": [], "citations": [],
                "graph_trace": {"matched": graph_matched, "graph_ratio": round(graph_ratio, 2)},
                "vector_trace": {"matched": bool(summary.get("vector_articles")),
                                 "vector_ratio": round(vector_ratio, 2)},
                "strategy": f"CRAG自纠错·{crag}（宁可拒答不可答错）",
                "context_used": context_used,
            }

        # 生成
        llm_ans, _ = self._llm_answer(retrieval_question, fused)
        if llm_ans:
            answer = llm_ans
            strategy = "LLM生成（引用溯源）"
        else:
            answer, _ = self._extractive_answer(fused)
            strategy = "抽取式回答（未配置LLM降级）"

        # 引用溯源
        refs = []
        for f in fused:
            art = self.article_text.get(f["article"], {})
            refs.append({
                "article": f["article"],
                "title": art.get("title", ""),
                "chapter": art.get("chapter", ""),
                "text": art.get("text", ""),
                "bm25": f["bm25"],
                "vector": f.get("vector", 0),
                "graph": f["graph"],
                "bm25_share": f["bm25_share"],
                "vector_share": f.get("vector_share", 0),
                "graph_share": f["graph_share"],
                "rerank_score": f.get("rerank_score", f["score"]),
                "score": round(f["score"], 3),
                "graph_hit": f["graph_hit"],
                "path": f["graph_path"],
                "law_name": art.get("law_name", ""),
                "authority": art.get("authority", ""),
                "effective_date": art.get("effective_date", ""),
                "source_url": art.get("source_url", ""),
            })

        return {
            "intent": intent,
            "answer": answer,
            "refused": False,
            "crag": crag,
            "strategy": strategy,
            "references": refs,
            "graph_trace": {
                "matched": graph_matched,
                "related_articles": sorted(summary["graph_articles"]),
                "graph_ratio": round(graph_ratio, 2),
            },
            "vector_trace": {
                "matched": bool(summary.get("vector_articles")),
                "vector_ratio": round(vector_ratio, 2),
            },
            "fused": fused,
            "context_used": context_used,
        }


EMERGENCY_GUIDE = (
    "【火灾应急指引】\n"
    "1. 立即拨打 119，说明详细地址、起火位置、燃烧物、火势和被困情况；\n"
    "2. 先判断疏散通道是否受到烟火威胁；通道安全时沿安全出口撤离，切勿乘坐电梯；\n"
    "3. 遇到烟气要降低身体姿势，尽量减少吸入；不要为取物或制作湿毛巾延误逃生；\n"
    "4. 不要贪恋财物，撤离后不要返回火场；\n"
    "5. 若房门发烫或通道已被浓烟封堵，不要贸然开门。关门堵烟，到无烟窗口发出求救信号并等待救援。\n"
    "※ 这是通用提示，现场情况瞬息万变，请优先听从消防救援人员指挥。"
)
