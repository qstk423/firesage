# FireSage 检索消融表

- split: `test`
- vector_channel: `bge-m3 语义向量（句级）`
- degraded_vector: `False`
- channels: `BM25, bge-m3 语义向量（句级）, GraphRAG, 法规意图重排, CrossEncoder 语义精排（ONNX INT8）`
- generated_at: `2026-09-12 22:15:14`

| 模式 | 说明 | Hit@1 | Hit@3 | Hit@5 | MRR | n | P50(ms) | P95(ms) |
|------|------|------:|------:|------:|----:|--:|--------:|--------:|
| `bm25` | 仅 BM25 | 0.5926 | 0.7407 | 0.8148 | 0.6790 | 27 | 1.65 | 5.14 |
| `bm25_vector` | BM25 + 向量 RRF | 0.5556 | 0.8889 | 0.9630 | 0.7204 | 27 | 90.04 | 195.31 |
| `hybrid` | 三路 RRF（无重排） | 0.4074 | 0.7778 | 0.9630 | 0.6228 | 27 | 81.76 | 118.64 |
| `full` | 三路 RRF + 法规重排 + CE | 0.7778 | 0.9259 | 0.9630 | 0.8611 | 27 | 1726.91 | 2224.28 |

## 读表提示

- `bm25` → `bm25_vector`：向量通道贡献
- `bm25_vector` → `hybrid`：GraphRAG 通道贡献
- `hybrid` → `full`：法规意图重排 + CrossEncoder 贡献
- 若 `degraded_vector=true`，本表为 TF-IDF 降级环境，答辩需同时标注「全量模型待复跑」
