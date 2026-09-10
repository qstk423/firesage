#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""本地 OpenAI 兼容推理服务：Qwen2.5-3B-Instruct + FireSage LoRA v2。

协议兼容 FireSage 的 rag/llm.py LLMClient（POST {base}/v1/chat/completions，
Bearer 鉴权，读取 choices[0].message.content），切换只需改配置指向本服务。

设计：
1. 提示词桥接：FireSage 管线的 system/user 提示词会被重写为微调训练格式
   （训练分布 = 最佳表现分布），非 FireSage 格式的请求原样透传；
2. 五层输出保护：
   a. JSON 校验：剥围栏/截取花括号/截断修复，全部失败则降级为合法 JSON（低置信）；
   b. 应急保护：问题含强火情信号 + 求助语义，而回答缺报警/疏散要素 →
      替换为标准应急引导 JSON（安全兜底，正常情况 FireSage 分流层已拦截）；
   c. 无条款守卫：桥接请求未检索到任何条款 → 强制标准拒答
      （严格依据条款的助手无依据不作答，杜绝无条款时的编造）；
   d. 引用守卫：basis 逐条验证条款号——匹配的用法规全名+原文规范化引用；
      全部无效时带强化指令重试一次（citation_retry），二次仍无效则输出
      标准拒答（citation_rejected，basis=[]），绝不强行重建引用；
   e. 拒答保护：basis 为占位符（["无"] 等）或结论含拒答话术 → 规范化为
      标准拒答 JSON（basis 严格为 []）。

启动（backend 目录）：
    ..\\venv311\\Scripts\\python.exe scripts\\serve_local_qwen.py
默认监听 http://127.0.0.1:8320，不占用 FireSage 的 8319。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
import uuid

from fastapi import FastAPI, HTTPException, Request

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BACKEND_DIR, "models", "Qwen2.5-3B-Instruct")
DEFAULT_ADAPTER = os.path.join(BACKEND_DIR, "adapters", "firesage-qwen25-3b-lora-v2")
DATA_DIR = os.path.join(BACKEND_DIR, "data")

# 法规索引：'abbr·num' -> {law_name, num, title, text}（main 启动时加载）
LAW_INDEX: dict = {}

# 训练时的 system 指令（与 build_gen_sft.py 完全一致）
TRAIN_SYSTEM = (
    "你是消防法规助手。请严格依据给定条款，输出 JSON："
    '{"conclusion","conditions","basis","supplement","confidence"}。'
    "不得编造未提供的条文。"
)
FIELDS = ["conclusion", "conditions", "basis", "supplement", "confidence"]

# ---- FireSage 管线提示词识别与桥接 ----
PIPELINE_MARK = "以下为检索到的法规条款"
FEEDBACK_PAT = re.compile(r"注意：上一次回答未通过核验[\s\S]*$")
ART_PAT = re.compile(r"【([^】]+)】([^【]*)")

# ---- 保护规则 ----
PLACEHOLDER_PAT = re.compile(r"^\s*(无|未知|暂无|N/?A|不适用)\s*$")
REFUSE_MARK = re.compile(
    r"超出.{0,12}(范围|知识库)|暂不回答|暂无.{0,4}(适用|条款)|无法回答|知识库外|非消防主题")
# 仅保留「正在发生的火情」强信号：不含「明火」——
# 「违规明火作业/动火证」是法规术语（如"能硬焊吗"类咨询），不是险情
FIRE_WORDS = re.compile(r"着火|起火|烧起来|烧着|冒烟|火势|被困|浓烟|呛|火灾现场|烧焦")
ACTION_WORDS = re.compile(r"怎么办|怎么做|怎么处理|如何|能.{0,8}吗|要不要|还能|紧急|救命|逃生|下楼|撤离|跑不|快烧")
SAFE_ALARM = re.compile(r"119|报警|呼救|求助")
SAFE_EVAC = re.compile(r"疏散|撤离|逃生|远离|离开|避开|不[^，。]{0,8}(电梯|扶梯)")
# 条款号提取（先剥括号内"第N号"等文号，避免误匹配）
PAREN_PAT = re.compile(r"[（(][^)）]*[)）]|第\d+号")
ART_NUM_PAT = re.compile(r"第[一二三四五六七八九十百零]+条|\d+(?:\.\d+)*")
CN_DIGIT = {"零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
            "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}


def cn_to_int(s: str):
    """中文/阿拉伯条款号数值化：'二十八'→28，'十'→10，'10'→10。非法返回 None。"""
    if not s:
        return None
    if s.isdigit():
        return int(s)
    if not all(c in CN_DIGIT or c == "十" for c in s):
        return None
    if "十" in s:
        left, _, right = s.partition("十")
        l = CN_DIGIT.get(left, 1) if left else 1
        r = CN_DIGIT.get(right, 0) if right else 0
        return l * 10 + r
    v = 0
    for c in s:
        v = v * 10 + CN_DIGIT[c]
    return v


def extract_art_num(text: str):
    """从引用文本提取条款号数值（支持'第十条'/'第10条'/'》28'）。失败返回 None。"""
    m = re.search(r"第\s*([一二三四五六七八九十零两\d]+)\s*条", text)
    if m:
        return cn_to_int(m.group(1))
    m = re.search(r"》\s*(\d+)", text)
    return int(m.group(1)) if m else None


def load_law_index() -> dict:
    """扫描 data/*.json 构建 'abbr·num' -> 条款信息（与 pipeline._load_article_text 同源）。"""
    index = {}
    if not os.path.isdir(DATA_DIR):
        return index
    for fname in sorted(os.listdir(DATA_DIR)):
        if not fname.endswith(".json"):
            continue
        try:
            with open(os.path.join(DATA_DIR, fname), encoding="utf-8") as f:
                law = json.load(f)
            abbr = law.get("law_abbr")
            if not abbr or not law.get("chapters"):
                continue
            law_name = law.get("law_name") or abbr
            for ch in law["chapters"]:
                for art in ch.get("articles") or []:
                    num = str(art.get("num") or "").strip()
                    if num:
                        index[f"{abbr}·{num}"] = {
                            "law_name": law_name, "num": num,
                            "title": art.get("title", ""),
                            "text": art.get("text", ""),
                        }
        except Exception:
            continue
    return index


def build_citation(info: dict) -> str:
    """按训练格式重建规范引用：《法规全名》第X条：原文摘录。"""
    snippet = (info.get("text") or "")[:120].rstrip("。") + "。"
    return f"《{info['law_name']}》{info['num']}：{snippet}"


# 引用重试的强化指令（附加到 user 提示词末尾，二次生成用）
RETRY_SUFFIX = (
    "\n\n注意：上一次回答的 basis 引用未通过核验。basis 只能引用上述给定条款，"
    "格式为《法规名》第X条：原文摘录；不得引用未提供的法规、不得编造条款号。"
    "若给定条款确实不适用，则输出 basis 为空数组并说明无法依据给定条款回答。"
)


def validate_citations(basis: list, provided_keys: list):
    """验证 basis 引用是否来自提供的条款。

    返回 (规范化后的有效引用列表, 无效引用数)：条款号匹配提供条款的
    引用用法规全名+原文重建（build_citation）；不匹配的（编造/张冠李戴）
    丢弃并计数。
    """
    prov = {num: k for k in provided_keys
            if (num := extract_art_num(k)) is not None}
    kept, dropped = [], 0
    for b in basis:
        key = prov.get(extract_art_num(b))
        if key and key in LAW_INDEX:
            kept.append(build_citation(LAW_INDEX[key]))
        elif key:  # 提供条款不在索引（罕见）：保留原文
            kept.append(b)
        else:      # 条款号不在提供列表 = 编造/张冠李戴
            dropped += 1
    return kept, dropped

REFUSE_JSON = {
    "conclusion": "该问题超出当前消防法规知识库范围，暂不回答以免误导。",
    "conditions": "知识库外 / 非消防主题",
    "basis": [],
    "supplement": "请改换消防合规或应急相关问题，或咨询属地消防救援机构；紧急情况请直接拨打119。",
    "confidence": "high",
}
EMERGENCY_JSON = {
    "conclusion": "紧急情况：请立即拨打119报警，说明详细地址与火势；沿安全出口疏散撤离，切勿乘坐电梯。",
    "conditions": "真火情 / 现场危险",
    "basis": [],
    "supplement": "低姿避烟、不贪恋财物、撤离后不返回火场；现场请听从消防救援人员指挥。",
    "confidence": "high",
}


def bridge_prompt(system: str, user: str):
    """FireSage 管线提示词 → 训练格式。

    返回 (system, user, question, provided_keys) 或 None（透传）。
    provided_keys：提供条款的 'abbr·num' key 列表（空列表 = 无可用条款），
    供引用守卫验证 basis。
    """
    if PIPELINE_MARK not in user:
        return None
    q_match = re.match(r"用户问题:([\s\S]+?)(?:\n\n|$)", user) or \
        re.match(r"用户问题：([\s\S]+?)(?:\n\n|$)", user)
    question = q_match.group(1).strip() if q_match else user.strip()
    arts = ART_PAT.findall(user)
    keys = [head.strip().split()[0] for head, _ in arts if head.strip()]
    parts = [f"【{head.strip()}】{text.strip()}" for head, text in arts if head.strip()]
    body = "\n\n".join(parts) if parts else "（无可用条款）"
    tail = ""
    fb = FEEDBACK_PAT.search(user)
    if fb:
        tail = "\n\n" + fb.group(0).strip()
    return TRAIN_SYSTEM, f"用户问题：{question}\n\n依据条款：\n{body}{tail}", question, keys


def _escape_inner_quotes(seg: str) -> str:
    """把字符串值内部未转义的直引号转义。

    模型偶发输出 basis 形如 《法》第X条："原文"，摘录内直引号未转义导致
    JSON 解析失败。判定规则：闭引号后必跟 , } ] : （跳过空白），
    其余情况的引号视为字符串内部字符，转义处理。
    """
    out = []
    in_str = esc = False
    i, n = 0, len(seg)
    while i < n:
        ch = seg[i]
        if esc:
            out.append(ch)
            esc = False
        elif ch == "\\":
            out.append(ch)
            esc = True
        elif ch == '"':
            if not in_str:
                in_str = True
                out.append(ch)
            else:
                j = i + 1
                while j < n and seg[j] in " \t\r\n":
                    j += 1
                if j >= n or seg[j] in ",}]:":
                    in_str = False
                    out.append(ch)
                else:
                    out.append('\\"')
        else:
            out.append(ch)
        i += 1
    return "".join(out)


def parse_and_repair(text: str):
    """JSON 校验与修复：返回 (dict, 修复方式）。"""
    t = text.strip()
    t = re.sub(r"^```[a-zA-Z]*\s*", "", t)
    t = re.sub(r"\s*```$", "", t).strip()
    try:
        return json.loads(t), "direct"
    except Exception:
        pass
    i, j = t.find("{"), t.rfind("}")
    if i >= 0 and j > i:
        try:
            return json.loads(t[i:j + 1]), "brace_extract"
        except Exception:
            pass
        # 内引号转义修复：basis 摘录含未转义直引号（第X条："原文"）
        try:
            return json.loads(_escape_inner_quotes(t[i:j + 1])), "quote_escape_repair"
        except Exception:
            pass
    # 截断修复：闭合未结束的字符串与括号
    if i >= 0:
        seg = t[i:].rstrip().rstrip(",")
        in_str = esc = False
        stack = []
        for ch in seg:
            if esc:
                esc = False
                continue
            if ch == "\\":
                if in_str:
                    esc = True
                continue
            if ch == '"':
                in_str = not in_str
                continue
            if not in_str:
                if ch in "{[":
                    stack.append(ch)
                elif ch in "}]":
                    if stack:
                        stack.pop()
        fix = ('"' if in_str else "") + "".join("}" if c == "{" else "]" for c in reversed(stack))
        try:
            return json.loads(seg + fix), "truncation_repair"
        except Exception:
            pass
    return None, "fallback"


def normalize_fields(data: dict):
    """字段归一：五字段齐全、basis 为 list[str]。"""
    out = dict(REFUSE_JSON)  # 兜底结构
    out["conclusion"] = str(data.get("conclusion") or "").strip() or out["conclusion"]
    out["conditions"] = str(data.get("conditions") or "无特殊限制").strip()
    basis = data.get("basis")
    if isinstance(basis, list):
        out["basis"] = [str(b).strip() for b in basis if str(b).strip()]
    elif basis:
        out["basis"] = [str(basis).strip()]
    else:
        out["basis"] = []
    out["supplement"] = str(data.get("supplement") or "无").strip()
    conf = str(data.get("confidence") or "medium").strip().lower()
    out["confidence"] = conf if conf in ("high", "medium", "low") else "medium"
    return out


def apply_protections(question: str, data: dict | None, raw: str,
                      provided_keys: list[str] | None = None,
                      guards: list[str] | None = None):
    """输出保护：返回 (最终 dict, 触发的保护列表)。

    引用守卫（citation_retry / citation_rejected / citation_guard）在生成侧
    完成后经 guards 传入；本函数负责 JSON 兜底、应急、无条款拒答与拒答规范化。
    provided_keys：None = 非桥接请求；[] = 桥接但无可用条款。
    """
    guards = list(guards) if guards else []
    if data is None:
        data = {"conclusion": raw.strip()[:200], "basis": [],
                "supplement": "（输出未通过 JSON 校验，已降级为原文摘录）",
                "confidence": "low"}
        guards.append("json_fallback")
    d = normalize_fields(data)
    q = question or ""

    # 1) 应急保护（最高优先级：紧急问题的第一要务是安全指引）
    emergency_hit = False
    if (FIRE_WORDS.search(q) and ACTION_WORDS.search(q)
            and not (SAFE_ALARM.search(d["conclusion"] + d["supplement"])
                     and SAFE_EVAC.search(d["conclusion"] + d["supplement"]))):
        d = dict(EMERGENCY_JSON)
        guards.append("emergency_guard")
        emergency_hit = True
    # 2) 无条款守卫：严格依据条款的助手，无依据只能拒答（防编造）
    if not emergency_hit and provided_keys is not None and not provided_keys:
        d = dict(REFUSE_JSON)
        guards.append("no_articles_guard")
    # 3) 拒答保护：占位符 basis 或拒答话术 → 规范拒答（basis 严格 []）
    #    no_articles / citation_rejected 已输出标准拒答，跳过以免重复标记
    if not any(g in guards for g in ("no_articles_guard", "citation_rejected")):
        placeholder_basis = bool(d["basis"]) and all(PLACEHOLDER_PAT.match(b) for b in d["basis"])
        if placeholder_basis or (REFUSE_MARK.search(d["conclusion"]) and not d["basis"]):
            d = dict(REFUSE_JSON)
            guards.append("refuse_guard")

    return d, guards


# ---------------- FastAPI 服务 ----------------
def build_app(model, tok, adapter_dir: str, api_key: str):
    # 注意：Request 必须在模块顶层导入——本文件启用了 from __future__ import annotations，
    # 闭包内局部导入的注解无法被 FastAPI 解析，会被误判为 query 参数。
    app = FastAPI(title="FireSage Local LLM (OpenAI-compatible)", version="1.0")
    gen_lock = threading.Lock()
    model_name = os.path.basename(adapter_dir.rstrip("\\/"))
    # 服务端再设一道默认输出上限：即使旧的 .env.local-model 仍写着 700，
    # 拉取新版代码并重启后也会立即生效。长答案评测可显式设置为 700。
    server_token_cap = max(
        64, min(int(os.getenv("LOCAL_LLM_MAX_TOKENS", "480") or "480"), 900))

    @app.get("/health")
    def health():
        return {"ok": True, "model": model_name, "adapter": adapter_dir,
                "backend": "qwen2.5-3b-instruct + lora",
                "max_output_tokens": server_token_cap}

    @app.get("/v1/models")
    def models():
        return {"object": "list", "data": [{"id": model_name, "object": "model", "owned_by": "local"}]}

    async def chat_completions(request: Request):
        if api_key:
            auth = request.headers.get("authorization", "")
            if auth != f"Bearer {api_key}":
                raise HTTPException(status_code=401, detail="invalid api key")
        body = await request.json()
        messages = body.get("messages") or []
        if not messages:
            raise HTTPException(status_code=400, detail="messages is empty")
        system = next((m["content"] for m in messages if m.get("role") == "system"), TRAIN_SYSTEM)
        user = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
        temperature = float(body.get("temperature", 0) or 0)
        requested_tokens = int(body.get("max_tokens", 480) or 480)
        max_tokens = max(64, min(requested_tokens, server_token_cap, 900))
        stream = bool(body.get("stream"))

        bridged = bridge_prompt(system, user)
        if bridged:
            sys_text, user_text, question, provided_keys = bridged
        else:
            sys_text, user_text, question, provided_keys = system, user, user[:200], None

        import torch

        def _gen_kwargs(inputs, streamer=None):
            kw = dict(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=temperature > 0.05,
                pad_token_id=tok.eos_token_id,
            )
            # 贪心解码不传 temperature/top_p，避免 transformers 的无效参数警告；
            # 显式要求采样时才加入对应参数。
            if temperature > 0.05:
                kw["temperature"] = temperature
                kw["top_p"] = 0.8
            if streamer is not None:
                kw["streamer"] = streamer
            return kw

        def _build_inputs(u_text: str):
            msgs = [{"role": "system", "content": sys_text},
                    {"role": "user", "content": u_text}]
            prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
            return tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)

        def do_generate(u_text: str) -> dict:
            inputs = _build_inputs(u_text)
            t0 = time.time()
            with gen_lock, torch.no_grad():
                out = model.generate(**_gen_kwargs(inputs))
            raw = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            return {"raw": raw, "latency_ms": int((time.time() - t0) * 1000),
                    "ttft_ms": None,
                    "prompt_tokens": int(inputs["input_ids"].shape[1]),
                    "completion_tokens": len(out[0]) - int(inputs["input_ids"].shape[1]),
                    "total_tokens": len(out[0])}

        def post_process(raw: str, provided_keys, gen: dict):
            """生成后守卫管线：JSON 修复 → 引用守卫（可重试）→ 输出保护。"""
            data, repair = parse_and_repair(raw)
            guards = []
            if repair in ("truncation_repair", "fallback"):
                guards.append(f"json_{repair}")

            # ---- 引用守卫（桥接 + 有条款 + 非应急 + 模型未拒答 + JSON 有效）----
            # 匹配的引用规范化为法规全名+原文；全部无效 → 带强化指令重试一次
            # （citation_retry）；二次仍无效 → 标准拒答（citation_rejected），
            # 绝不用提供条款强行重建引用。
            if provided_keys and data is not None:
                q = question or ""
                emergency_q = bool(FIRE_WORDS.search(q) and ACTION_WORDS.search(q))
                probe = normalize_fields(data)
                refuse_intended = bool(REFUSE_MARK.search(probe["conclusion"])) and not probe["basis"]
                if not emergency_q and not refuse_intended:
                    kept, dropped = validate_citations(probe["basis"], provided_keys)
                    if kept:
                        data["basis"] = kept  # 匹配的引用规范化
                        if dropped:
                            guards.append("citation_guard")
                    else:
                        guards.append("citation_retry")
                        gen = do_generate(user_text + RETRY_SUFFIX)
                        data2, repair2 = parse_and_repair(gen["raw"])
                        if repair2 in ("truncation_repair", "fallback"):
                            guards.append(f"json_{repair2}")
                        probe2 = normalize_fields(data2) if data2 is not None else None
                        if probe2 is None:
                            guards.append("citation_rejected")
                            data = dict(REFUSE_JSON)
                        elif REFUSE_MARK.search(probe2["conclusion"]) and not probe2["basis"]:
                            data = probe2  # 模型重试后自拒答 → 交拒答保护规范化
                        else:
                            kept2, _ = validate_citations(probe2["basis"], provided_keys)
                            if kept2:
                                data = probe2
                                data["basis"] = kept2
                            else:
                                guards.append("citation_rejected")
                                data = dict(REFUSE_JSON)

            final, guards = apply_protections(question, data, raw, provided_keys, guards)
            return final, guards, gen

        # ---------------- 流式输出（stream=true）----------------
        if stream:
            from fastapi.responses import StreamingResponse
            from transformers import TextIteratorStreamer

            def sse():
                cid = f"chatcmpl-{uuid.uuid4().hex[:12]}"
                created = int(time.time())

                def chunk(delta, **extra):
                    d = {"id": cid, "object": "chat.completion.chunk", "created": created,
                         "model": body.get("model", model_name),
                         "choices": [{"index": 0, "delta": delta}]}
                    d.update(extra)
                    return f"data: {json.dumps(d, ensure_ascii=False)}\n\n"

                yield chunk({"role": "assistant"})
                inputs = _build_inputs(user_text)
                streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
                t0 = time.time()
                ttft_ms = None

                def _run():
                    with gen_lock, torch.no_grad():
                        model.generate(**_gen_kwargs(inputs, streamer=streamer))

                th = threading.Thread(target=_run)
                th.start()
                parts = []
                for text in streamer:  # 逐段产出，阻塞等待生成
                    if not text:
                        continue
                    if ttft_ms is None:
                        ttft_ms = int((time.time() - t0) * 1000)
                    parts.append(text)
                    yield chunk({"content": text})
                th.join()
                raw = "".join(parts)
                n_out = len(tok(raw, add_special_tokens=False)["input_ids"]) if raw else 0
                gen = {"raw": raw, "latency_ms": int((time.time() - t0) * 1000),
                       "ttft_ms": ttft_ms,
                       "prompt_tokens": int(inputs["input_ids"].shape[1]),
                       "completion_tokens": n_out,
                       "total_tokens": int(inputs["input_ids"].shape[1]) + n_out}
                # 守卫在完整输出上执行；最终内容以 final_content 为准（增量仅为预览）
                final, guards, gen = post_process(raw, provided_keys, gen)
                yield chunk({}, finish_reason="stop",
                            firesage_protections=guards,
                            final_content=json.dumps(final, ensure_ascii=False),
                            ttft_ms=ttft_ms,
                            latency_ms=gen["latency_ms"],
                            usage={"prompt_tokens": gen["prompt_tokens"],
                                   "completion_tokens": gen["completion_tokens"],
                                   "total_tokens": gen["total_tokens"]})
                yield "data: [DONE]\n\n"

            return StreamingResponse(
                sse(), media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

        # ---------------- 非流式（默认，兼容原协议）----------------
        gen = do_generate(user_text)
        final, guards, gen = post_process(gen["raw"], provided_keys, gen)
        content = json.dumps(final, ensure_ascii=False)

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": body.get("model", model_name),
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": content}}],
            "usage": {"prompt_tokens": gen["prompt_tokens"],
                      "completion_tokens": gen["completion_tokens"],
                      "total_tokens": gen["total_tokens"]},
            "firesage_protections": guards,
            "latency_ms": gen["latency_ms"],
            "ttft_ms": gen.get("ttft_ms"),
        }

    # 双路径注册（叠装饰器会让第二路由把 Request 误判为 query 参数）
    app.add_api_route("/v1/chat/completions", chat_completions, methods=["POST"])
    app.add_api_route("/chat/completions", chat_completions, methods=["POST"])
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="本地 OpenAI 兼容 Qwen+LoRA 服务")
    parser.add_argument("--adapter", default=DEFAULT_ADAPTER)
    parser.add_argument("--port", type=int, default=8320)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--api-key", default=os.getenv("LOCAL_LLM_API_KEY", ""),
                        help="为空则不校验（仅本机监听）")
    parser.add_argument("--skip-warmup", action="store_true",
                        help="跳过启动时 GPU 热身（默认热身，减少第一次问答等待）")
    args = parser.parse_args()

    for label, path in (("基座模型", MODEL_DIR), ("LoRA adapter", args.adapter)):
        if not os.path.exists(path):
            print(f"[错误] {label}不存在：{path}")
            raise SystemExit(1)

    global LAW_INDEX
    LAW_INDEX = load_law_index()
    print(f"[法规索引] 加载 {len(LAW_INDEX)} 条条款（引用守卫用）")

    os.environ["HF_HUB_OFFLINE"] = "1"
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[加载] 基座 {MODEL_DIR}")
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, local_files_only=True, dtype=torch.bfloat16, device_map="cuda"
    )
    print(f"[加载] LoRA adapter {args.adapter}")
    model = PeftModel.from_pretrained(model, args.adapter).eval()
    if not args.skip_warmup:
        print("[热身] 正在预热 GPU（只在启动时执行一次）")
        warm_messages = [
            {"role": "system", "content": TRAIN_SYSTEM},
            {"role": "user", "content": "只输出一个左花括号"},
        ]
        warm_prompt = tok.apply_chat_template(
            warm_messages, add_generation_prompt=True, tokenize=False)
        warm_inputs = tok(
            warm_prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
        with torch.no_grad():
            model.generate(
                **warm_inputs,
                max_new_tokens=1,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        print("[热身] 完成")
    print(f"[就绪] Qwen2.5-3B + {os.path.basename(args.adapter)} → http://{args.host}:{args.port}")

    import uvicorn
    uvicorn.run(build_app(model, tok, args.adapter, args.api_key),
                host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
