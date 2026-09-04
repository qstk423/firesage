# -*- coding: utf-8 -*-
"""消安智答 FireSage · FastAPI 后端服务
启动：python3 main.py  →  http://localhost:8319
"""
import os
import json
import time
from typing import Optional

from fastapi import FastAPI, Query, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from rag.graphrag import ARCHIVE_PATHS, build_if_missing
from rag.retriever import HybridRetriever
import rag.pipeline as pipeline_mod

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

APP_VERSION = "0.7.0"
app = FastAPI(title="消安智答 FireSage", version=APP_VERSION)

# 可选鉴权：设置 API_KEY 后，/api/*（除 /api/system）需带 Header: X-API-Key
API_KEY = os.getenv("API_KEY", "").strip()
# 简易限流：每 IP 每分钟最大请求数（0=关闭）
RATE_LIMIT_PER_MIN = int(os.getenv("RATE_LIMIT_PER_MIN", "60") or "60")
_rate_bucket: dict[str, list[float]] = {}


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _check_rate_limit(ip: str) -> None:
    if RATE_LIMIT_PER_MIN <= 0:
        return
    now = time.time()
    window = _rate_bucket.setdefault(ip, [])
    _rate_bucket[ip] = [t for t in window if now - t < 60]
    if len(_rate_bucket[ip]) >= RATE_LIMIT_PER_MIN:
        raise HTTPException(status_code=429, detail="请求过于频繁，请稍后再试")
    _rate_bucket[ip].append(now)


@app.middleware("http")
async def security_and_log(request: Request, call_next):
    """鉴权（可选）+ 限流 + 请求耗时日志。"""
    path = request.url.path
    if path.startswith("/api/") and path not in ("/api/system",):
        if API_KEY:
            key = request.headers.get("x-api-key", "")
            if key != API_KEY:
                from fastapi.responses import JSONResponse
                return JSONResponse(status_code=401, content={"detail": "未授权：需要有效的 X-API-Key"})
        try:
            _check_rate_limit(_client_ip(request))
        except HTTPException as exc:
            from fastapi.responses import JSONResponse
            return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

    start = time.time()
    response = await call_next(request)
    elapsed_ms = int((time.time() - start) * 1000)
    if path.startswith("/api/"):
        print(f"[api] {request.method} {path} -> {response.status_code} "
              f"({elapsed_ms}ms)", flush=True)
    response.headers["X-Process-Time-Ms"] = str(elapsed_ms)
    return response


graph = build_if_missing()

# 加载语义句 chunks
with open(os.path.join(DATA_DIR, "chunks.json"), encoding="utf-8") as f:
    CHUNKS = json.load(f)

pl = pipeline_mod.Pipeline(chunks=CHUNKS)


class AskBody(BaseModel):
    question: str = Field(..., min_length=1, max_length=500)
    previous_question: Optional[str] = Field(default=None, max_length=500)


def _entity_stats():
    stats = {}
    for n in graph["nodes"]:
        stats[n["type"]] = stats.get(n["type"], 0) + 1
    return stats


def source_catalog():
    sources = []
    for path, fallback_abbr in ARCHIVE_PATHS:
        with open(path, encoding="utf-8") as source_file:
            source = json.load(source_file)
        sources.append({
            "law": source.get("law_abbr", fallback_abbr),
            "name": source.get("law_name", fallback_abbr),
            "authority": source.get("authority", ""),
            "effective_date": source.get("effective_date", ""),
            "source_url": source.get("source_url", ""),
        })
    return sources


@app.get("/api/graph/stats")
def graph_stats():
    arts = set()
    for e in graph["edges"]:
        if e.get("article"):
            arts.add(e["article"])
    for c in CHUNKS:
        arts.add(c["article"])
    law_dist = {}
    for article in arts:
        law = article.split("·", 1)[0]
        law_dist[law] = law_dist.get(law, 0) + 1
    return {
        "semantic_sents": len(CHUNKS),
        "entities": len(graph["nodes"]),
        "edges": len(graph["edges"]),
        "articles": sorted(arts),
        "type_dist": _entity_stats(),
        "law_dist": law_dist,
        "relation_dist": {rel: sum(1 for edge in graph["edges"] if edge["relation"] == rel)
                          for rel in sorted({edge["relation"] for edge in graph["edges"]})},
        "sources": source_catalog(),
    }


def _dedupe_edges(edges):
    """可视化去重：同一对实体+关系只保留一条，附带涉及条款列表。"""
    merged = {}
    for edge in edges:
        key = (edge["source"], edge["target"], edge["relation"])
        if key not in merged:
            item = dict(edge)
            item["articles"] = [edge["article"]] if edge.get("article") else []
            merged[key] = item
        else:
            art = edge.get("article")
            if art and art not in merged[key]["articles"]:
                merged[key]["articles"].append(art)
    return list(merged.values())


@app.get("/api/graph/data")
def graph_data(types: str = Query("", description="逗号分隔的类型过滤，如 行为,主体"),
               laws: str = Query("", description="逗号分隔的法规简称"),
               relations: str = Query("", description="逗号分隔的关系类型"),
               query: str = Query("", description="关键词搜索")):
    type_set = set(t for t in types.split(",") if t)
    law_set = set(law for law in laws.split(",") if law)
    relation_set = set(rel for rel in relations.split(",") if rel)
    all_nodes, all_edges = graph["nodes"], graph["edges"]
    edge_pool = all_edges
    if law_set:
        edge_pool = [edge for edge in edge_pool if edge.get("law", edge["article"].split("·", 1)[0]) in law_set]
    if relation_set:
        edge_pool = [edge for edge in edge_pool if edge["relation"] in relation_set]
    nodes = all_nodes
    if type_set:
        nodes = [n for n in nodes if n["type"] in type_set]
    if query:
        q = query.strip()
        nodes = [n for n in nodes if q in n["name"] or any(q in alias for alias in n.get("aliases", []))]
    matched_ids = {n["id"] for n in nodes}
    if type_set or query:
        edges = [e for e in edge_pool if e["source"] in matched_ids or e["target"] in matched_ids]
        context_ids = {e["source"] for e in edges} | {e["target"] for e in edges}
        nodes = [dict(n, contextual=n["id"] not in matched_ids) for n in all_nodes if n["id"] in context_ids]
    elif law_set or relation_set:
        edges = edge_pool
        context_ids = {e["source"] for e in edges} | {e["target"] for e in edges}
        nodes = [n for n in all_nodes if n["id"] in context_ids]
    else:
        edges = edge_pool
    visible_ids = {n["id"] for n in nodes}
    edges = [e for e in edges if e["source"] in visible_ids and e["target"] in visible_ids]
    # 全图默认去重，避免同一关系因多条款重复画线导致“一团乱”
    edges = _dedupe_edges(edges)
    return {"nodes": nodes, "edges": edges, "matched": len(matched_ids)}


@app.get("/api/graph/ego")
def graph_ego(id: str = Query(..., description="中心实体 id，如 违规行为:占用疏散通道"),
              hops: int = Query(2, ge=1, le=2, description="展开跳数：1=仅邻居，2=邻居+二级关联"),
              max_nodes: int = Query(56, ge=8, le=120, description="子图节点上限，防止二级爆炸")):
    """点击节点后的关系网：中心 + 一跳邻居（+ 可选二跳关联）。"""
    center = next((n for n in graph["nodes"] if n["id"] == id), None)
    if not center:
        return {"error": "not found", "nodes": [], "edges": []}

    # 邻接表
    adj = {}
    for e in graph["edges"]:
        adj.setdefault(e["source"], set()).add(e["target"])
        adj.setdefault(e["target"], set()).add(e["source"])

    hop_of = {id: 0}
    hop1 = set(adj.get(id, ()))
    for nid in hop1:
        hop_of[nid] = 1

    hop2 = set()
    if hops >= 2 and hop1:
        # 二级：一级邻居的邻居，排除中心与一级
        candidates = []
        for n1 in hop1:
            for n2 in adj.get(n1, ()):
                if n2 == id or n2 in hop1:
                    continue
                # 与一级的连接数越高越优先（更“相关”）
                bridge = sum(1 for x in hop1 if n2 in adj.get(x, ()))
                candidates.append((bridge, n2))
        candidates.sort(key=lambda x: (-x[0], x[1]))
        remain = max(0, max_nodes - 1 - len(hop1))
        for _, n2 in candidates:
            if len(hop2) >= remain:
                break
            hop2.add(n2)
            hop_of[n2] = 2

    node_ids = {id} | hop1 | hop2
    # 只保留子图内部边；二级节点仅保留连到一级/中心的边，避免二级之间乱成一团
    raw_edges = []
    for e in graph["edges"]:
        s, t = e["source"], e["target"]
        if s not in node_ids or t not in node_ids:
            continue
        hs, ht = hop_of.get(s, 99), hop_of.get(t, 99)
        if min(hs, ht) >= 2:
            continue  # 丢掉纯二级↔二级边
        raw_edges.append(e)
    edges = _dedupe_edges(raw_edges)

    nodes = []
    for n in graph["nodes"]:
        if n["id"] not in node_ids:
            continue
        hop = hop_of[n["id"]]
        nodes.append(dict(
            n,
            highlight=(n["id"] == id),
            hop=hop,
            degree=sum(1 for e in edges if e["source"] == n["id"] or e["target"] == n["id"]),
        ))

    return {
        "center": id,
        "center_name": center.get("name", id),
        "nodes": nodes,
        "edges": edges,
        "neighbor_count": len(hop1),
        "second_count": len(hop2),
        "edge_count": len(edges),
        "hops": hops,
    }


@app.get("/api/graph/subgraph")
def graph_subgraph(articles: str = Query(..., description="逗号分隔的条款 key，如 消防法·第二十八条,消防法·第六十条"),
                   highlight: str = Query("", description="逗号分隔的高亮实体 id")):
    """回答-图谱联动：只返回本次回答引用条款相关的局部子图。

    比 /api/graph/data 返回全量 435 条关系更易读：
    节点 = 引用条款涉及的实体；边 = 这些实体之间的关系；
    与问题场景直接匹配的实体标记 highlight=true。
    """
    article_set = set(a.strip() for a in articles.split(",") if a.strip())
    highlight_set = set(h.strip() for h in highlight.split(",") if h.strip())
    # 1) 引用条款直接涉及的边
    edge_index = {}
    for i, e in enumerate(graph["edges"]):
        if e.get("article") in article_set:
            edge_index[i] = e
    # 2) 高亮实体的一跳邻居边，保证主体-行为-处罚链完整
    for i, e in enumerate(graph["edges"]):
        if i in edge_index:
            continue
        if e["source"] in highlight_set or e["target"] in highlight_set:
            edge_index[i] = e
    edges = _dedupe_edges(list(edge_index.values()))
    node_ids = {e["source"] for e in edges} | {e["target"] for e in edges}
    nodes = [dict(n, highlight=n["id"] in highlight_set)
             for n in graph["nodes"] if n["id"] in node_ids]
    return {"nodes": nodes, "edges": edges, "articles": sorted(article_set)}


@app.get("/api/graph/entity")
def entity_detail(id: str = Query(...),
                  laws: str = Query("", description="逗号分隔的法规简称"),
                  relations: str = Query("", description="逗号分隔的关系类型")):
    node = next((n for n in graph["nodes"] if n["id"] == id), None)
    if not node:
        return {"error": "not found"}
    triples = [e for e in graph["edges"] if e["source"] == id or e["target"] == id]
    law_set = set(law for law in laws.split(",") if law)
    relation_set = set(rel for rel in relations.split(",") if rel)
    if law_set:
        triples = [edge for edge in triples if edge.get("law", edge["article"].split("·", 1)[0]) in law_set]
    if relation_set:
        triples = [edge for edge in triples if edge["relation"] in relation_set]
    articles = set()
    for e in triples:
        if e.get("article"):
            articles.add(e["article"])
    article_infos = []
    for a in sorted(articles):
        if a in pl.article_text:
            info = pl.article_text[a]
            article_infos.append({"article": info["num"], "title": info["title"],
                                  "chapter": info["chapter"], "text": info["text"],
                                  "law_name": info.get("law_name", ""),
                                  "authority": info.get("authority", ""),
                                  "effective_date": info.get("effective_date", ""),
                                  "source_url": info.get("source_url", "")})
    return {"entity": node, "triples": triples, "articles": article_infos,
            "triple_count": len(triples), "article_count": len(article_infos)}


@app.post("/api/ask")
def ask(body: AskBody, request: Request):
    question = body.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="问题不能为空")
    result = pl.ask(question, body.previous_question)
    try:
        from rag import audit
        audit.from_response(question, body.previous_question, result, client=_client_ip(request))
    except Exception as exc:
        print(f"[audit] write failed: {exc}", flush=True)
    return result


@app.get("/api/system")
def system_status():
    sources = source_catalog()
    return {
        "name": "消安智答 FireSage",
        "version": APP_VERSION,
        "generation_mode": "LLM 生成（结构化+引用核验）" if pl.llm.enabled else "本地抽取式回答",
        "retrieval": "BM25 + bge-m3语义向量 + GraphRAG 三路召回，RRF 融合 + 法规意图重排 + CrossEncoder 精排",
        "retrieval_channels": pl.retriever.channels,
        "knowledge_bases": [source["name"] for source in sources],
        "sources": sources,
        "security": {
            "api_key_required": bool(API_KEY),
            "rate_limit_per_min": RATE_LIMIT_PER_MIN,
            "audit_enabled": True,
        },
    }


@app.get("/")
def index():
    return FileResponse(os.path.join(BASE_DIR, "..", "frontend", "index.html"))


app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "..", "frontend")), name="static")


if __name__ == "__main__":
    import uvicorn
    print("消安智答 FireSage 启动中 → http://localhost:8319")
    uvicorn.run(app, host="0.0.0.0", port=8319)
