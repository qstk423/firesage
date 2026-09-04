# FireSage 检索消融表

- split: `all`
- vector_channel: `bge-m3 语义向量`
- degraded_vector: `False`
- channels: `BM25, bge-m3 语义向量, GraphRAG, 法规意图重排, CrossEncoder 语义精排`
- generated_at: `2026-09-04 17:20:07`

| 模式 | 说明 | Hit@1 | Hit@3 | MRR | n | 耗时(s) |
|------|------|------:|------:|----:|--:|--------:|
| `bm25` | 仅 BM25 | 0.6098 | 0.8659 | 0.7415 | 164 | 0.63 |
| `bm25_vector` | BM25 + 向量 RRF | 0.7256 | 0.9268 | 0.8254 | 164 | 6.44 |
| `hybrid` | 三路 RRF（无重排） | 0.5732 | 0.8902 | 0.7308 | 164 | 5.75 |
| `full` | 三路 RRF + 法规重排 + CE | 0.9024 | 0.9756 | 0.9394 | 164 | 395.37 |

## 读表提示

- `bm25` → `bm25_vector`：向量通道贡献
- `bm25_vector` → `hybrid`：GraphRAG 通道贡献
- `hybrid` → `full`：法规意图重排 + CrossEncoder 贡献
- 若 `degraded_vector=true`，本表为 TF-IDF 降级环境，答辩需同时标注「全量模型待复跑」
