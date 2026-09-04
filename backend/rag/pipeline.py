# -*- coding: utf-8 -*-
"""问答主流程（v0.6 可信检索版）：

用户自由提问
  → 场景结构化（主体/行为/场所/对象/意图）
  → 查询改写与同义扩展
  → BM25+语义向量+知识图谱三路召回
  → 法规意图重排 + CrossEncoder 精排
  → LLM 依据条款结构化生成（无 LLM 时抽取式降级）
  → 独立核验器逐结论检查引用一致性（失败重生成一次，再失败降级/拒答）
  → 可信回答 / 澄清追问 / 安全拒答

对外仅暴露 stages（正在检索/重排/生成/核验）过程状态，不暴露模型内部思维链。
"""
import json
import os
import re
import time

from .intent import route
from .retriever import HybridRetriever
from .llm import LLMClient
from .scene import clarify_question, structure
from .verifier import set_article_keys, verify

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# CRAG 阈值
CRAG_AMBIGUOUS = 0.25   # 疑似明确但质量低 → 拒答
CRAG_INCORRECT = 0.15   # 完全未见相关 → 拒答
CRAG_CE_FLOOR = 0.53    # CrossEncoder 低于此分 = 与问题基本无关（库外主题地板分约 0.50）

# 库外主题护栏：知识库为建筑消防管理类法规，以下行政/产品/其他领域主题明确不在范围内
OUT_OF_KB_TOPICS = ("报考", "技能鉴定", "考试科目", "职业资格", "CCC", "认证目录", "森林火灾")

FOLLOW_UP_MARKERS = ["那", "那么", "这个", "这种", "上述", "还会", "具体", "罚多少", "多少钱", "怎么处理"]

SYSTEM_TMPL = (
    "你是消防法规领域的专业助手「消安智答」。请严格依据提供的法规条款回答问题，不得编造。"
    "回答必须使用以下 JSON 结构（不要输出 JSON 以外的内容）：\n"
    '{"conclusion": "直接回答问题的结论（1-2句）", '
    '"conditions": "该结论适用的条件（主体/场所/建筑类型，如不适用写 无特殊限制）", '
    '"basis": ["《法规名》第X条：支撑该结论的关键原文片段", ...], '
    '"supplement": "补充说明或注意事项（无则写 无）", '
    '"confidence": "high|medium|low，依据条款数量与一致性判断"}\n'
    "所有结论必须能在提供的条款原文中找到依据，引用条款编号必须来自提供的条款。"
)


class Pipeline:
    # 回答缓存上限（FIFO 淘汰），减少重复调用的 LLM 费用
    CACHE_LIMIT = 256

    def __init__(self, chunks=None):
        self.retriever = HybridRetriever(chunks)
        self.llm = LLMClient()
        self.chunk_index = {c["id"]: c for c in chunks or []}
        self.article_text = self._load_article_text()
        self._cache = {}
        set_article_keys(self.article_text.keys())

    def _load_article_text(self):
        """条款号 -> {"num","title","text","chapter"}（多部法规，编号带法规前缀）"""
        from .graphrag import ARCHIVE_PATHS
        m = {}
        for path, law_abbr in ARCHIVE_PATHS:
            with open(path, encoding="utf-8") as f:
                law = json.load(f)
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

    # ---------- CRAG ----------
    def _crag_judge(self, fused, question):
        top = fused[0] if fused else None
        if not top or top["score"] <= CRAG_INCORRECT:
            return "Incorrect"
        ce = top.get("cross_encoder")
        if ce is not None and ce < CRAG_CE_FLOOR:
            # CrossEncoder 判定最高分条款也与问题基本无关 → 知识库覆盖不了
            return "Incorrect"
        if (not top.get("graph_hit") and not top.get("direct_match")
                and top.get("vector", 0) < 0.08
                and top.get("coverage", 0) < 0.5):
            return "Incorrect"
        if top["score"] <= CRAG_AMBIGUOUS:
            return "Ambiguous"
        return "Correct"

    # ---------- 生成 ----------
    def _llm_structured(self, question, fused, feedback=None):
        context = "\n\n".join(
            f"【{c['num']} {c['title']}】{c['text']}"
            for c in (self.article_text.get(f["article"], {}) for f in fused)
            if c
        )
        user = f"用户问题：{question}\n\n以下为检索到的法规条款：\n{context}\n\n请基于上述条款作答。"
        if feedback:
            user += f"\n\n注意：上一次回答未通过核验，问题如下，请修正：{feedback}"
        resp = self.llm.complete(SYSTEM_TMPL, user)
        if not resp:
            return None
        try:
            start = resp.find("{")
            end = resp.rfind("}")
            data = json.loads(resp[start:end + 1])
            return {
                "conclusion": str(data.get("conclusion", "")).strip(),
                "conditions": str(data.get("conditions", "无特殊限制")).strip(),
                "basis": [str(b) for b in data.get("basis", []) if b],
                "supplement": str(data.get("supplement", "无")).strip(),
                "confidence": data.get("confidence", "medium"),
            }
        except Exception:
            return {"conclusion": resp.strip(), "conditions": "无特殊限制",
                    "basis": [], "supplement": "无", "confidence": "low"}

    def _extractive_structured(self, question, fused, scene):
        """无 LLM 降级：从最高置信条款抽取结构化回答（原文直接来自条款，天然可信）。"""
        if not fused:
            return None
        top = fused[0]
        art = self.article_text.get(top["article"], {})
        law_name = art.get("law_name", art.get("law", "消防法规"))
        conditions = "、".join(scene.get("venues") or []) or "无特殊限制"
        if scene.get("subjects"):
            conditions = f"主体：{'、'.join(scene['subjects'])}；场所：{conditions}"
        # 多条款交叉时可信度更高
        laws = {f["article"].split("·")[0] for f in fused}
        confidence = "high" if len(laws) >= 2 or fused[0].get("graph_hit") else "medium"
        text = art.get("text", "")
        # 问个人处罚时抽取"个人有前款…"专属罚则段，避免把单位罚款金额误答给个人
        if "个人" in (scene.get("subjects") or []) and "个人有" in text:
            seg = text[text.index("个人有"):]
            seg = seg[:seg.index("。") + 1] if "。" in seg else seg[:200]
            return {
                "conclusion": f"依据《{law_name}》{art.get('num', '')}「{art.get('title', '')}」：{seg}",
                "conditions": conditions,
                "basis": [f"《{art.get('law_name', '')}》{art.get('num', '')}：{seg}"],
                "supplement": "以上内容直接摘自法规条款原文。" if len(fused) > 1 else "无",
                "confidence": confidence,
            }
        return {
            "conclusion": f"依据《{law_name}》{art.get('num', '')}「{art.get('title', '')}」：{text[:150]}",
            "conditions": conditions,
            "basis": [f"《{art.get('law_name', '')}》{art.get('num', '')}：{text[:200]}"],
            "supplement": "以上内容直接摘自法规条款原文。" if len(fused) > 1 else "无",
            "confidence": confidence,
        }

    def _mentioned_law_not_in_kb(self, question):
        """问题中以书名号点名的法规若不在知识库，返回其名称；否则 None。
        例：《消防设施通用规范》→ 不在库 → 拒答；《消防法》→ 在库 → None。"""
        for title in re.findall(r"《([^《》]{4,30})》", question):
            for info in self.article_text.values():
                kb_names = {info.get("law_name", ""), info.get("law", "")}
                if any(title in name or name in title for name in kb_names if name):
                    break
            else:
                return title
        return None

    # ---------- 主入口 ----------
    def ask(self, question, previous_question=None, use_cache=True):
        """带缓存的入口：相同（问题, 上轮问题）直接返回上次结果，省去 LLM 重复调用。"""
        key = ((question or "").strip(), (previous_question or "").strip())
        if use_cache:
            hit = self._cache.get(key)
            if hit is not None:
                cached = dict(hit)
                cached["cached"] = True
                return cached
        result = self._ask(question, previous_question)
        if len(self._cache) >= self.CACHE_LIMIT:
            self._cache.pop(next(iter(self._cache)))
        self._cache[key] = result
        return result

    def _ask(self, question, previous_question=None):
        t0 = time.time()
        question = question.strip()
        previous_question = (previous_question or "").strip()
        # 短追问合并上一轮：含追问标记词，或自身不含任何消防语义（"每班至少几个人"）
        context_used = bool(
            previous_question
            and len(question) <= 24
            and (any(marker in question for marker in FOLLOW_UP_MARKERS)
                 or route(question) not in ("law", "emergency"))
        )
        full_question = f"{previous_question}；追问：{question}" if context_used else question

        # ---- 第 1 层：场景结构化 ----
        scene = structure(full_question)
        intent = route(full_question)
        # 词表未命中但场景结构化识别出消防行为/对象（如"楼梯口让纸箱堵得过不去了"）
        # → 说明是生活化口语描述，按法规流程走，交由后续 CRAG 判断证据是否充分
        if intent == "refuse" and (scene.get("behaviors") or scene.get("objects")):
            intent = "law"

        # ---- 应急模式 ----
        if intent == "emergency":
            return self._pack({
                "intent": intent,
                "answer": EMERGENCY_GUIDE,
                "refused": False, "crag": "Correct",
                "citations": [{
                    "title": "国家消防救援局：公共场所火灾应急处理",
                    "url": "https://www.119.gov.cn/kp/hzyf/2022/622.shtml",
                }],
                "strategy": "应急流程引导（内置）",
                "context_used": context_used, "scene": scene,
            }, t0, ["分流", "应急引导"])

        if intent == "refuse":
            return self._pack({
                "intent": intent,
                "answer": "抱歉，这是一个消防法规专业问答系统，超出范围的问题我无法回答。",
                "refused": True, "crag": "Refuse",
                "strategy": "拒答护栏",
                "context_used": context_used, "scene": scene,
            }, t0, ["分流", "拒答"])

        if intent == "chitchat":
            return self._pack({
                "intent": intent,
                "answer": "我是消安智答，可回答消防法规与应急处置问题，例如「楼道堆放杂物违反什么规定」。",
                "refused": False, "crag": "Chat",
                "strategy": "闲聊分流",
                "context_used": context_used, "scene": scene,
            }, t0, ["分流", "引导"])

        # ---- 知识库外法规拒答：问题点名了某部法规但不在知识库中 ----
        out_kb_law = self._mentioned_law_not_in_kb(full_question)
        out_kb_topic = next((w for w in OUT_OF_KB_TOPICS if w in full_question), None)
        if out_kb_law or out_kb_topic:
            reason = (f"《{out_kb_law}》暂未收录在本系统知识库中"
                      if out_kb_law else f"「{out_kb_topic}」属于本知识库范围外的主题")
            return self._pack({
                "intent": intent,
                "answer": (f"很抱歉，{reason}"
                           "（当前收录：消防法、61号令、高层规定、责任制办法），"
                           "无法提供准确内容，为避免误导暂不回答。"),
                "refused": True, "crag": "OutOfKB",
                "strategy": "知识库范围拒答（宁可拒答不可答错）",
                "context_used": context_used, "scene": scene,
            }, t0, ["分流", "范围判定", "拒答"])

        # ---- 澄清追问：关键要素缺失且会影响处罚结论时，先问清再答 ----
        if scene["ambiguity"] and not context_used:
            ask_text = clarify_question(scene["ambiguity"], scene)
            if ask_text:
                return self._pack({
                    "intent": "clarify",
                    "answer": ask_text,
                    "refused": False, "crag": "Clarify",
                    "scene": scene,
                    "strategy": "要素澄清（先问清再答，避免套错条款）",
                    "context_used": context_used,
                }, t0, ["场景解析", "澄清追问"])

        # ---- 第 2 层：查询改写（口语 → 法规术语；与原问题拼接，两者关键词都保留） ----
        if scene["rewrite"] and scene["rewrite"] != question:
            retrieval_question = f"{scene['rewrite']} {question}"
        else:
            retrieval_question = full_question

        # ---- 第 3、4 层：三路召回 + 精排 ----
        fused, summary = self.retriever.retrieve(retrieval_question, top_k=3)
        crag = self._crag_judge(fused, retrieval_question)

        graph_matched = bool(summary["graph_articles"])
        graph_ratio = sum(f["graph_share"] for f in fused) / len(fused) if fused else 0.0
        vector_ratio = sum(f.get("vector_share", 0) for f in fused) / len(fused) if fused else 0.0

        # ---- CRAG 低质量 → 拒答 ----
        if crag in ("Ambiguous", "Incorrect"):
            return self._pack({
                "intent": intent,
                "answer": "很抱歉，针对您的问题，现有知识库中的检索结果置信度不足，为避免误导，暂不回答。建议换个说法，或咨询属地消防救援机构。",
                "refused": True, "crag": crag,
                "graph_trace": {"matched": graph_matched, "graph_ratio": round(graph_ratio, 2)},
                "vector_trace": {"matched": bool(summary.get("vector_articles")),
                                 "vector_ratio": round(vector_ratio, 2)},
                "strategy": f"CRAG自纠错·{crag}（宁可拒答不可答错）",
                "context_used": context_used, "scene": scene,
            }, t0, ["检索", "重排", "置信判定", "拒答"])

        # ---- 第 5 层：生成 ----
        refs = self._build_refs(fused)
        stages = ["检索", "重排", "生成", "核验"]
        verification = None
        if self.llm.enabled:
            structured = self._llm_structured(retrieval_question, fused)
            answer_text = self._render(structured)
            if structured:
                verification = verify(answer_text, refs, question, scene)
                if not verification["passed"]:
                    # 核验失败 → 带反馈重生成一次
                    structured = self._llm_structured(
                        retrieval_question, fused, feedback="；".join(verification["issues"]))
                    answer_text = self._render(structured)
                    verification = verify(answer_text, refs, question, scene)
                    if not verification["passed"]:
                        # 仍失败 → 弃用 LLM 回答，降级为抽取式（原文摘录天然可信）
                        structured = self._extractive_structured(retrieval_question, fused, scene)
                        answer_text = self._render(structured)
                        verification["degraded"] = True
                        stages.append("降级")
            else:
                # LLM 未返回结构化 JSON（例如网络/鉴权失败）→ 必须降级，保证 answer/reference 评测可用。
                structured = self._extractive_structured(retrieval_question, fused, scene)
                answer_text = self._render(structured)
                stages.append("降级（LLM无结构输出）")
            strategy = "LLM生成（结构化+引用核验）"
        else:
            structured = self._extractive_structured(retrieval_question, fused, scene)
            answer_text = self._render(structured)
            strategy = "抽取式回答（未配置LLM降级）"

        return self._pack({
            "intent": intent,
            "answer": answer_text,
            "structured": structured,
            "refused": False, "crag": crag,
            "strategy": strategy,
            "references": refs,
            "verification": verification,
            "graph_trace": {
                "matched": graph_matched,
                "related_articles": sorted(summary["graph_articles"]),
                "graph_ratio": round(graph_ratio, 2),
            },
            "vector_trace": {
                "matched": bool(summary.get("vector_articles")),
                "vector_ratio": round(vector_ratio, 2),
            },
            "context_used": context_used, "scene": scene,
        }, t0, stages)

    def _build_refs(self, fused):
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
                "cross_encoder": f.get("cross_encoder"),
                "rerank_score": f.get("rerank_score", f["score"]),
                "score": round(f["score"], 3),
                "graph_hit": f["graph_hit"],
                "path": f["graph_path"],
                "law_name": art.get("law_name", ""),
                "authority": art.get("authority", ""),
                "effective_date": art.get("effective_date", ""),
                "source_url": art.get("source_url", ""),
            })
        return refs

    @staticmethod
    def _render(s):
        """结构化回答 → 固定格式文本（结论/适用条件/法规依据/补充说明/可信度）。"""
        if not s:
            return ""
        lines = [
            f"【结论】{s['conclusion']}",
            f"【适用条件】{s['conditions']}",
            "【法规依据】",
        ]
        lines.extend(f"- {b}" for b in s.get("basis") or ["（见引用条款）"])
        lines.append(f"【补充说明】{s['supplement']}")
        conf = {"high": "高", "medium": "中", "low": "低"}.get(s.get("confidence"), "中")
        lines.append(f"【可信度】{conf}")
        return "\n".join(lines)

    @staticmethod
    def _pack(payload, t0, stages):
        payload.setdefault("references", [])
        payload.setdefault("citations", [])
        payload.setdefault("verification", None)
        payload.setdefault("structured", None)
        payload.setdefault("scene", None)
        payload["stages"] = stages
        payload["latency_ms"] = int((time.time() - t0) * 1000)
        return payload


EMERGENCY_GUIDE = (
    "【火灾应急指引】\n"
    "1. 立即拨打 119，说明详细地址、起火位置、燃烧物、火势和被困情况；\n"
    "2. 先判断疏散通道是否受到烟火威胁；通道安全时沿安全出口撤离，切勿乘坐电梯；\n"
    "3. 遇到烟气要降低身体姿势，尽量减少吸入；不要为取物或制作湿毛巾延误逃生；\n"
    "4. 不要贪恋财物，撤离后不要返回火场；\n"
    "5. 若房门发烫或通道已被浓烟封堵，不要贸然开门。关门堵烟，到无烟窗口发出求救信号并等待救援。\n"
    "※ 这是通用提示，现场情况瞬息万变，请优先听从消防救援人员指挥。"
)
