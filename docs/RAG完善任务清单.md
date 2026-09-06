# FireSage RAG 完善任务清单

> 用途：后续完善检索 / GraphRAG / 可信闭环时的对照清单。  
> 来源：当前自研架构评估 + GitHub 标杆项目参照（LightRAG、MS GraphRAG、RAGFlow、ChatLaw 等）。  
> 原则：**窄域高可信消防法规 RAG**，不整仓换成通用框架套壳。  
> 更新日期：2026-09-04

---

## 使用说明

- `[ ]` 未做 · `[~]` 进行中 · `[x]` 完成  
- 优先级：**P0** 演示/评测阻塞 · **P1** 国赛/指标明显增益 · **P2** 增强项 · **P3** 可选调研  
- 每条尽量写清：**目标 / 参照 / 落点文件 / 验收**

---

## 0. 基线（已完成，勿回退）

- [x] 核心四法 + 专题库分库 BM25，专题门控 ×0.72
- [x] BM25 + 向量 + GraphRAG → RRF → 法规意图重排 →（可选 CE）
- [x] CRAG 低置信拒答 + Out-of-KB + verifier
- [x] 追问合并上一轮；改写保留 full_question
- [x] 10 源语料入库；图谱重建（约 112 实体 / 952 边；「单位」枢纽已压缩）
- [x] FireEval Hit@1 / Hit@3 过门槛（降级环境下快照）
- [x] 图谱页法规来源与 README 规模同步

---

## P0 · 演示与环境（先稳住）

### T0.1 语义通道常开
- [x] 本地接通 bge-m3 / bge-reranker（或等价中文 embedding + cross-encoder）
- [x] 确认非降级权重生效：`CHANNEL_WEIGHTS`（bm25 0.42 / vector 0.33 / graph 0.25）
- **参照**：LlamaIndex / RAGFlow 默认「向量 + 重排」主路径  
- **落点**：`semantic_index.py`（优先本地 HF snapshot + `local_files_only`）  
- **验收**：启动日志为 bge-m3 / CrossEncoder；消融表 `degraded_vector=false`

### T0.2 进程加载新图谱
- [x] 改 `graph.json` / `chunks.json` 后：`POST /api/kb/reload` 或 ask 时按 mtime 自动热加载；启动日志打印规模
- **验收**：`/api/graph/stats` 显示 10 源、实体/边与 README 一致

### T0.3 消融表可一键复跑
- [x] 固定脚本输出：`bm25` / `bm25_vector` / `hybrid` / `full` 对照表
- [x] 降级 vs 全量模型各跑一份，写入 `docs/` 或答辩附录
- **参照**：RAGAS / 各开源 README 的 ablation  
- **落点**：`scripts/ablation_retrieval.py`、`docs/消融表-retrieval.md`  
- **验收**：一张表能讲清「图谱 / 重排各自贡献」
- **备注（2026-09-04 全量模型）**：bm25 0.61 → +向量 0.73 → +图谱 hybrid 0.57（图谱需重排托住）→ full 0.90 / Hit@3 0.98

---

## P1 · 召回质量（指标与可维护性）

### T1.1 收敛手调规则，防过拟合
- [x] 盘点 `FORCE_RECALL`、`direct_patterns`、同义词表：只留高通用规则
- [x] 评测集专属硬编码改为「数据驱动」或移入 eval fixture，不进主路径
- **参照**：LlamaIndex 混合检索配置化，少写死题面  
- **落点**：`retriever.py`、`reranker.py`、`eval/retrieval_boosts.json`（默认不加载，需 `RETRIEVAL_EVAL_BOOSTS=1`）
- **验收**：删掉一批题面特判后，dev Hit 跌幅 &lt; 约定阈值（建议 &lt; 3pt）
- **备注（2026-09-04）**：同义词去掉整句键；`DIRECT_EVIDENCE` 改为短线索；题面条号特判外置。收敛后 dev Hit@1 **0.9655** / Hit@3 **1.0**（与收敛前持平）

### T1.2 查询侧增强（选 1～2 个落地）
- [x] 子问题分解（复合问：「谁负责 + 怎么罚」拆两路召回再合并）
- [ ] 可选 HyDE / 伪文档扩展（仅 law 意图、有模型时）
- [x] 处罚意图 vs 职责意图路由加强（已有雏形，做成显式通道）
- **参照**：LlamaIndex query engines；LangChain multi-query  
- **落点**：`scene.py`、`pipeline.py`、`retriever.py`  
- **验收**：跨条款题（FireEval `x*`）Hit@1 提升
- **备注（2026-09-05）**：`decompose_queries` / `detect_query_mode`；retriever 多查询取 max 融合；global 压专题抬核心枢纽句

### T1.3 专题门控精细化
- [x] 复核 `SPECIALTY_GATES`：避免该开不开 / 不该开乱开
- [x] 「人员密集场所」类通用表述是否误伤核心库
- **落点**：`retriever.py`  
- **验收**：核心题不被专题挤掉；专题题（`tx*` / 电动车等）仍进专题库
- **备注**：密集场所仅高区分度词；电动车/广东门控补口语触发词

### T1.4 重排可学习化
- [ ] 打开并固化 `RERANK_ML` 训练→推理路径（若已有脚本则补文档）
- [ ] 规则分与模型分融合权重做成配置，避免再堆 if
- **参照**：RAGFlow / LlamaIndex rerank 节点  
- **落点**：`rerank_ml.py`、`retriever.py`  
- **验收**：`--compare` 显示 ML 重排相对规则有增益

---

## P1 · GraphRAG（对照 LightRAG / MS GraphRAG）

### T2.1 半自动实体建议（推荐路径，不要上完整 MS GraphRAG）
- [x] 新法规入库时：脚本建议「主体 / 行为 / 对象」候选（`scripts/suggest_graph_entities.py`）
- [ ] 人工审核后写入 `SUBJECTS` / `BEHAVIORS` / `OBJECTS`
- **参照**：LightRAG 自动抽实体；ChatLaw 图谱+人工筛选  
- **落点**：`scripts/suggest_graph_entities.py` + `graphrag.py` 词典
- **验收**：密集场所 / 39号令 孤儿条款占比下降；重建后边数上升且 Hit 不降

### T2.2 查询模式对齐
- [x] 区分 **局部**（实体→条款）与 **主题/汇总**（多实体社区式）查询入口
- [x] 全局类问题（「单位消防职责有哪些」）优先核心枢纽条款，避免被专题噪声淹没
- **参照**：LightRAG local/global；MS GraphRAG community summary  
- **落点**：`graphrag.py` `GraphRetriever`、`pipeline.py`  
- **验收**：职责总述类题 Hit@1 稳定；图谱路径可解释
- **备注**：`query_mode=global|local` 经 scene → pipeline → retriever；global 关闭专题门控并抬核心职责句

### T2.3 图谱覆盖与质量
- [x] 定期统计：有边条款占比 / 各来源边数 / 枢纽节点度（`scripts/graph_health.py` → `docs/图谱健康度.md`）
- [x] 压缩「单位」万能枢纽边（无明确主体时不再一律回落「单位」；度 99→43）
- [x] 39号令边偏少：补娱乐场所专用行为词典（边 13→29，有边条款 5→11）
- **验收**：文档中维护一张「图谱健康度」表
- **备注**：半自动候选脚本 `scripts/suggest_graph_entities.py`（人工审核后再写入词典）

### T2.4 明确不做（防摊薄）
- [ ] ~~整仓引入 microsoft/graphrag 重索引~~（成本高、与窄域词典冲突）
- [ ] ~~为答辩改成通用 Dify/LangChain 套壳~~

---

## P1 · 可信与领域（对照 ChatLaw）

### T3.1 报批稿效力明示
- [x] 回答 / 依据区对「电动车充电」「密集场所」标注**非正式施行 / 报批稿**
- [x] 与现行法冲突时优先现行法，并说明
- **落点**：`pipeline.py` SYSTEM_TMPL + `_with_draft_notice`（抽取式/LLM 共用）  
- **验收**：相关题回答含效力提示；无误导为已生效规章

### T3.2 多步 SOP（轻量多 Agent，可选）
- [ ] 法规研究员（检索）→ 结论起草 → 核验员（已有 verifier）流程显式化
- [ ] 答辩材料画一页 SOP，对齐 ChatLaw「减幻觉」叙事
- **参照**：ChatLaw 法律研究员 + SOP  
- **验收**：trace/stages 能演示三步；非必须拆独立服务

### T3.3 引用核验覆盖扩库法规
- [x] `verifier` 的法规别名补全：39号令 / 电动车充电 / 密集场所 / 广东高层
- **落点**：`verifier.py` `LAW_ALIASES`  
- **验收**：扩库来源条款引用准确率不掉

### T3.4 金标与语料对齐机制
- [x] 定期脚本：`expected_articles` 与语料存在性检查（`scripts/check_gold_drift.py`）
- [ ] 旧条号（如重点单位条号变更）纳入 checklist；must 字面与正文一致性（`--strict-must` 偏严）
- **落点**：`scripts/check_gold_drift.py` + `eval/fireeval_*.json`  
- **验收**：CI 或本地一键报告「金标漂移」

---

## P2 · 数据预处理与工程化

### T4.1 入库流水线文档化
- [x] 统一：`raw → json → dedupe → SOURCE_FILES → GraphBuilder → eval smoke`
- [x] README / 本清单交叉链接命令
- **参照**：RAGFlow 文档理解流水线（学流程清晰，不学重 PDF）  
- **落点**：`expand_corpus.py`、`build_law_corpus.py`、README

### T4.2 切片策略复查
- [ ] 超长条款 / 列举项是否切过碎或过粗
- [ ] 罚则项与禁止项同条时，是否利于 Hit@1
- **验收**：抽样 20 条难例，人工看 chunk 边界

### T4.3 元数据强制字段
- [ ] 所有源具备：`authority` / `effective_date` / `source_url` / `status`
- [ ] 前端条款卡片必显
- **验收**：图谱实体详情与问答引用均可点开原文

---

## P2 · 评测体系（对照 RAGAS / FireEval）

### T5.1 FireEval 扩展
- [ ] 专题题、追问题、跨法冲突题按比例补强
- [ ] train/dev/test 同步，避免只改 v1 全集
- **验收**：split 与全集指标叙事一致

### T5.2 忠实度指标可解释
- [ ] citation / must_conclusion / refusal 与 Hit 分列报告
- [ ] 可选接入 RAGAS 子集做对照（非替换 FireEval）
- **验收**：答辩有「检索命中 ≠ 忠实回答」两张数

### T5.3 回归门禁
- [x] CI：金标漂移 + `gate_retrieval.py --strict`；全链路仍跑 `evaluate_fireeval.py`
- [x] 无模型时至少跑检索-only Hit 门禁（`gate_retrieval`，阈值 Hit@1≥0.80 / Hit@3≥0.93）
- **验收**：PR 不能默默打烂 Hit

---

## P3 · 调研备忘（先读后做）

| 仓库 | 建议阅读重点 | 是否引入代码 |
|------|--------------|--------------|
| [HKUDS/LightRAG](https://github.com/HKUDS/LightRAG) | 双层图、增量更新、local/global | 学方法；可抄抽实体思路 |
| [microsoft/graphrag](https://github.com/microsoft/graphrag) | 社区摘要、全局问答 | 只读论文/文档，不整仓 |
| [infiniflow/ragflow](https://github.com/infiniflow/ragflow) | 引用展示、解析流水线 | UI/引用体验参考 |
| [PKU-YuanGroup/ChatLaw](https://github.com/PKU-YuanGroup/ChatLaw) | 防幻觉 SOP、法律 RAG Agent | 叙事 + 轻量 SOP |
| [run-llama/llama_index](https://github.com/run-llama/llama_index) | Hybrid / rerank / query transform | API 设计参考 |
| [explodinggradients/ragas](https://github.com/explodinggradients/ragas) | 忠实度/相关性指标 | 可选对照评测 |
| ChatLaw / LaWGPT / Lawyer-LLaMA | 法律微调与数据 | 仅阶段5微调叙事时再看 |

---

## 建议迭代顺序（后面开干时按此）

```text
1) T0.1 模型接通 + T0.3 消融表
2) T1.1 收敛手调  + T1.3 门控复核
3) T2.1 半自动 NER + T2.3 图谱健康度
4) T3.1 报批稿效力 + T3.3 verifier 别名
5) T5.* 评测门禁固化
6) 有余力再做 T1.2 查询分解 / T3.2 轻量 SOP
```

---

## 完成定义（RAG「内容部分」可宣告阶段完成）

同时满足：

1. 非降级环境下 test：**Hit@1 ≥ 0.85，Hit@3 ≥ 0.95**，且有消融表  
2. 图谱 10 源边覆盖可解释；孤儿条款有统计与改进记录  
3. 报批稿效力在回答中可见；verifier 覆盖全部 law_abbr  
4. 主路径手调规则已收敛，新增法规主要靠「入库脚本 + 词典增量」而非堆题面 if  
5. 答辩材料能用 1 页图画清：分库召回 + GraphRAG + CRAG/核验（对照 ChatLaw/LightRAG 话术，但不宣称已达到其规模）

---

*关联文档：桌面 `FireSage-RAG现状评估.md` / `.pdf`；画布 `rag-stack-assessment.canvas.tsx`。*
