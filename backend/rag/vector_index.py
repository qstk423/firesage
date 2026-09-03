# -*- coding: utf-8 -*-
"""轻量级法规向量索引。

使用字符 n-gram TF-IDF 向量，保证离线和无模型环境也能稳定运行。
它不是语义模型的替代品，而是 FireSage 三路召回中的可复现基线。
"""
import math
import re
from collections import Counter, defaultdict


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
    """按条款聚合的 TF-IDF 余弦向量索引。"""

    name = "TF-IDF 字符向量"

    def __init__(self, chunks):
        article_parts = defaultdict(list)
        article_titles = {}
        for chunk in chunks or []:
            article_parts[chunk["article"]].append(chunk.get("text", ""))
            article_titles[chunk["article"]] = chunk.get("title", "")
        self.articles = sorted(article_parts)
        docs = {
            article: _features(article_titles.get(article, "") + " " + " ".join(article_parts[article]))
            for article in self.articles
        }
        document_frequency = Counter()
        for tokens in docs.values():
            document_frequency.update(set(tokens))
        total = max(1, len(docs))
        self.idf = {
            token: math.log((total + 1) / (frequency + 1)) + 1
            for token, frequency in document_frequency.items()
        }
        self.vectors = {article: self._vector(tokens) for article, tokens in docs.items()}

    def _vector(self, tokens):
        counts = Counter(tokens)
        weighted = {token: (1 + math.log(count)) * self.idf.get(token, 0.0)
                    for token, count in counts.items() if token in self.idf}
        norm = math.sqrt(sum(value * value for value in weighted.values())) or 1.0
        return {token: value / norm for token, value in weighted.items()}

    def search(self, query, top_k=10):
        query_vector = self._vector(_features(query))
        if not query_vector:
            return []
        scored = []
        for article, vector in self.vectors.items():
            score = sum(value * vector.get(token, 0.0) for token, value in query_vector.items())
            if score > 0:
                scored.append((article, score))
        scored.sort(key=lambda item: (-item[1], item[0]))
        return scored[:top_k]
