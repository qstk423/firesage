# -*- coding: utf-8 -*-
"""轻量级法规向量索引。

使用字符 n-gram TF-IDF 向量，保证离线和无模型环境也能稳定运行。
句级建索引，召回后按条款取 max 分（与 bge 句级通道对齐）。
"""
import math
import re
from collections import Counter


def _features(text):
    text = re.sub(r"\s+", "", (text or "").lower())
    blocks = re.findall(r"[\u4e00-\u9fff]+|[a-z0-9]+", text)
    out = []
    for block in blocks:
        if re.fullmatch(r"[a-z0-9]+", block):
            out.append(block)
            continue
        out.extend(block[i:i + 2] for i in range(max(0, len(block) - 1)))
        out.extend(block[i:i + 3] for i in range(max(0, len(block) - 2)))
    return out


class TfidfVectorIndex:
    """句级 TF-IDF 余弦向量；search 返回条款级 max 分。"""

    name = "TF-IDF 字符向量（句级）"

    def __init__(self, chunks):
        self.unit_articles = []
        self.vectors = []
        docs_tokens = []
        for chunk in chunks or []:
            article = chunk.get("article") or ""
            if not article:
                continue
            title = chunk.get("title", "") or ""
            text = chunk.get("text", "") or ""
            tokens = _features(f"{title} {text}")
            self.unit_articles.append(article)
            docs_tokens.append(tokens)

        document_frequency = Counter()
        for tokens in docs_tokens:
            document_frequency.update(set(tokens))
        total = max(1, len(docs_tokens))
        self.idf = {
            token: math.log((total + 1) / (frequency + 1)) + 1
            for token, frequency in document_frequency.items()
        }
        self.vectors = [self._vector(tokens) for tokens in docs_tokens]
        self.articles = sorted(set(self.unit_articles))

    def _vector(self, tokens):
        counts = Counter(tokens)
        weighted = {token: (1 + math.log(count)) * self.idf.get(token, 0.0)
                    for token, count in counts.items() if token in self.idf}
        norm = math.sqrt(sum(value * value for value in weighted.values())) or 1.0
        return {token: value / norm for token, value in weighted.items()}

    def search(self, query, top_k=10):
        query_vector = self._vector(_features(query))
        if not query_vector or not self.vectors:
            return []
        best = {}
        for article, vector in zip(self.unit_articles, self.vectors):
            score = sum(value * vector.get(token, 0.0) for token, value in query_vector.items())
            if score > best.get(article, 0.0):
                best[article] = score
        ranked = sorted(best.items(), key=lambda x: -x[1])[:top_k]
        return ranked
