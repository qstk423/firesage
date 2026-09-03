# -*- coding: utf-8 -*-
"""消安智答 FireSage · FastAPI 后端服务
启动：python3 main.py  →  http://localhost:8319
"""
import os
import json
from typing import Optional

from fastapi import FastAPI, Query, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from rag.graphrag import ARCHIVE_PATHS, build_if_missing
from rag.retriever import HybridRetriever
import rag.pipeline as pipeline_mod

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

APP_VERSION = "0.5.0"
app = FastAPI(title="消安智答 FireSage", version=APP_VERSION)
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
    return {"nodes": nodes, "edges": edges, "matched": len(matched_ids)}


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
def ask(body: AskBody):
    question = body.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="问题不能为空")
    return pl.ask(question, body.previous_question)


@app.get("/api/system")
def system_status():
    sources = source_catalog()
    return {
        "name": "消安智答 FireSage",
        "version": APP_VERSION,
        "generation_mode": "LLM 生成" if pl.llm.enabled else "本地抽取式回答",
        "retrieval": "BM25 + TF-IDF向量 + GraphRAG + 法规重排",
        "retrieval_channels": pl.retriever.channels,
        "knowledge_bases": [source["name"] for source in sources],
        "sources": sources,
    }


@app.get("/")
def index():
    return FileResponse(os.path.join(BASE_DIR, "..", "frontend", "index.html"))


app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "..", "frontend")), name="static")


if __name__ == "__main__":
    import uvicorn
    print("消安智答 FireSage 启动中 → http://localhost:8319")
    uvicorn.run(app, host="0.0.0.0", port=8319)
