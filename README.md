# 消安智答 FireSage

**面向消防法规咨询与火灾应急提示的可信知识问答原型。**

用自然语言描述真实场景（例如「楼梯口让纸箱堵得过不去了」），系统会检索可溯源的法规条款，说明责任主体、禁止行为与处罚依据；问法过宽时引导把问题问具体，证据不足或主题无关时拒答，不编造条文。

> 本系统仅供辅助参考，**不替代**执法解释、专业咨询或现场消防指挥。紧急情况请直接拨打 **119**。

---

## 申报材料

- 项目概述：[`docs/项目概述-消安智答FireSage.md`](docs/项目概述-消安智答FireSage.md)
- 解决方案文档（企业命题组）：[`docs/解决方案-消安智答FireSage.md`](docs/解决方案-消安智答FireSage.md)
- 产品级差距评估：[`docs/产品级差距评估.md`](docs/产品级差距评估.md)
- RAG 完善任务清单：[`docs/RAG完善任务清单.md`](docs/RAG完善任务清单.md)
- 图谱健康度：[`docs/图谱健康度.md`](docs/图谱健康度.md)
- 基层试用反馈问卷：[`docs/基层用户调查问卷-消安智答.md`](docs/基层用户调查问卷-消安智答.md)
- 演示截图清单：[`docs/演示截图清单.md`](docs/演示截图清单.md)
- 意图微调（可复现）：`python3 backend/scripts/train_intent.py --compare`

---

## 作品定位

| 维度 | 说明 |
|------|------|
| 赛道叙事 | 大模型微调与 RAG 技术融合的智慧消防应急智能问答 |
| 当前阶段 | 可演示、可评测的混合 RAG + GraphRAG 可信问答原型（立项 / 展示版） |
| 设计原则 | **法规事实来自检索，不靠模型背条文**；宽问引导、库外拒答 |

---

## 核心能力

### 1. 可信法规问答
- 口语场景理解：楼道堆物、占用通道、电动车充电、无证动火等
- 结构化回答：结论 / 适用条件 / 法规依据 / 补充说明 / 可信度
- 多轮追问：可结合上一轮问题理解「那具体罚多少钱」
- 要素不足时先澄清（如通道类型、单位/个人），避免套错条款

### 2. 混合检索（Hybrid RAG）
```text
用户提问
  → 场景结构化 + 查询改写
  → BM25 + 语义向量 + GraphRAG 三路召回
  → RRF 融合 + 法规意图重排 + CrossEncoder 精排
  → LLM 依据条款生成（可选）
  → 引用核验；失败则降级原文摘录或拒答
```

- **BM25**：关键词精确命中
- **语义向量**：BAAI/bge-m3（不可用时自动降级 TF-IDF）
- **GraphRAG**：主体–行为–对象–处罚关系补充口语召回
- **RRF**：消除不同检索器分数量纲差异

### 3. 知识图谱可视化
- 全图浏览：按法规来源、实体类型、关系筛选与搜索（**分簇漂浮，全览不画边**）
- **点击节点展开关系网**：以该实体为中心展示一级/二级邻居与连边，可继续点邻居切换中心
- 回答一键跳转「本次证据链」局部子图
- 节点详情展示支撑条款、发布机关、生效日期与官方原文链接
- 改完 `chunks.json` / `graph.json` 后可用 `POST /api/kb/reload` 热加载，无需重启进程

### 4. 安全分流、引导与拒答
- **法规问答** / **火灾应急指引** / **闲聊引导** / **引导提问** / **安全拒答**
- 问法过宽（如「了解一下消防知识」）：给出示例问法，引导补全场景要素
- 知识库外法规/标准或无关主题：明确说明未收录 / 超出范围，避免误导
- CRAG：低置信时优先引导或拒答，不硬编条款

### 5. FireEval 评测闭环
- **200** 道人工核验题，覆盖口语、处罚、跨条款、多轮、应急、拒答等 8 类
- 已划分 **train / dev / test**，便于后续微调与回归
- 指标：Hit@1 / Hit@3 / MRR、路由准确率、拒答准确率、引用准确率、严重错误率
- CI 含金标漂移检查与检索-only Hit 门禁（见 `.github/workflows/fireeval.yml`）

### 6. 前端体验
- 单页问答 + 图谱；**手机端响应式**（底部 Tab、图谱上下布局），桌面布局保持不变

---

## 知识库规模

| 法规 | 规模 | 来源元数据 |
|------|------|------------|
| 《中华人民共和国消防法（2021 修正）》 | **74 条全文** | 发布机关、生效日期、官方链接 |
| 《机关、团体、企业、事业单位消防安全管理规定》（公安部 61 号令） | **48 条全文** | 同上 |
| 《高层民用建筑消防安全管理规定》 | 全文 | 同上 |
| 《消防安全责任制实施办法》 | 全文 | 同上 |
| 《公共娱乐场所消防安全管理规定》（公安部 39 号令） | **23 条全文** | 同上 |
| 《电动自行车充电及停放场所消防安全管理》（报批稿） | 条款级入库 | 标注非正式施行 |
| 《人员密集场所消防安全管理》（报批稿） | 条款级入库 | 标注非正式施行；与既有库正文去重 |
| 《广东省高层建筑消防安全管理规定》 | **34 条全文** | 地方规章 |

当前索引约 **442** 个条款、**1421** 个语义句、**101** 个实体、**826** 条关系。
实体类型：**违规行为 / 管理义务 / 政府职责 / 主体 / 消防对象 / 处罚**（行为已三分；已覆盖 39 号令 / 电动车充电 / 密集场所 / 广东高层等扩库来源）。
可视化展示时会按实体对去重，界面更清晰。图谱页「法规来源」筛选对应上述 8 部法规。

---

## 快速开始

```bash
git clone https://github.com/qstk423/firesage.git
cd firesage
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
cd backend
python3 main.py
```

浏览器打开：**http://localhost:8319**（手机可用同一地址；窄屏自动切底部导航）。

### 可选：接入大模型

未配置时系统使用**抽取式回答**（直接摘录条款，天然可溯源）。配置兼容 OpenAI Chat Completions 的服务后，启用结构化生成 + 引用核验：

```bash
export LLM_BASE_URL="https://api.deepseek.com"
export LLM_API_KEY="your-key"
export LLM_MODEL="deepseek-v4-flash"
```

也可复制 `backend/.env.example` → `backend/.env.local`（该文件已被 Git 忽略）。

### 可选：中文 Embedding / 精排模型

首次启动可能下载 `bge-m3` 与 `bge-reranker-v2-m3`（体积较大）。国内可加速：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

模型不可用时**不影响启动**，自动降级为 TF-IDF + 规则重排。

---

## 领域微调（可复现）

当前微调分三层，请按实际完成度表述：

| 层次 | 状态 | 命令 |
|------|------|------|
| 意图路由微调 | **已落地**（默认 hybrid） | `python3 backend/scripts/train_intent.py --compare` |
| 条款重排微调 | **可训练**；默认关闭（`RERANK_ML=1` 开启） | `python3 backend/scripts/train_rerank.py --compare` |
| 生成式 LoRA | **SFT 数据已就绪**；训练需 GPU/云端 | `python3 backend/scripts/build_gen_sft.py` → `python3 backend/scripts/train_gen_lora.py --dry-run` |

生成 SFT 数据目录：`backend/data/models/sft/gen_sft_*.jsonl`（问题 + 给定条款 → 结构化 JSON，禁止无依据扩写）。

---

## 演示建议（3 分钟）

1. **口语法规**：「楼道堆放杂物违反什么规定」→ 看条款依据 + 点「在知识图谱中查看证据链」
2. **应急分流**：「家里着火了现在怎么办」→ 应急指引，提示拨打 119
3. **引导提问**：「我想了解一下消防方面的知识」→ 不硬答，给出可跟问的示例
4. **安全拒答**：询问电影/天气或未收录标准（如 GB51309）→ 明确拒答、不编造
5. **图谱探索**：进入「知识图谱」页，实体按类型分簇漂浮；点选后关系线长出，再点「返回全图」回到分簇视图
6. **强制执行口语**：「我想了解强制拆除相关知识」→ 锚定《消防法》第六十条「强制执行」条款

---

## 评测

```bash
cd backend

# 全量 v1（200 题）
python3 scripts/evaluate_fireeval.py

# 按划分评测（推荐日常看 dev）
python3 scripts/evaluate_fireeval.py --split dev
python3 scripts/evaluate_fireeval.py --split test

# 一键验收（意图对比 + FireEval + 烟雾 + 演示三问）
python3 scripts/demo_check.py --quick

# 金标漂移（expected_articles 是否仍在语料）
python3 scripts/check_gold_drift.py --split all --strict

# 检索-only Hit 门禁（不依赖 LLM）
python3 scripts/gate_retrieval.py --split dev --strict

# 图谱健康度
python3 scripts/graph_health.py --markdown docs/图谱健康度.md

# 旧版 18 题轻量基线
python3 scripts/evaluate_fireeval.py --v0 --strict

# 重新划分 train/dev/test
python3 scripts/split_fireeval.py
```

冒烟测试：

```bash
python3 -m unittest discover -s tests -v
```

CI：推送到 `main` 时自动跑单元测试与 FireEval（见 `.github/workflows/fireeval.yml`）。

---

## 主要接口

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/api/ask` | 问答；`question`，可选 `previous_question` |
| `GET` | `/api/system` | 版本、检索通道、生成模式、知识来源 |
| `POST` | `/api/kb/reload` | 热加载 chunks/graph（改语料后免重启） |
| `GET` | `/api/graph/stats` | 图谱统计 |
| `GET` | `/api/graph/data` | 图谱数据（`types` / `laws` / `relations` / `query`） |
| `GET` | `/api/graph/ego?id=` | **节点关系网**（中心 + 一跳邻居） |
| `GET` | `/api/graph/subgraph?articles=` | 回答相关局部子图 |
| `GET` | `/api/graph/entity?id=` | 实体详情与支撑条款 |

---

## 项目结构

```text
firesage/
├── backend/
│   ├── main.py                 # FastAPI 入口
│   ├── data/                   # 法规 JSON、chunks、graph、raw 全文
│   ├── eval/
│   │   ├── fireeval_v1.json           # 200 题全集
│   │   ├── fireeval_v1_train.json     # 训练集
│   │   ├── fireeval_v1_dev.json       # 开发集
│   │   └── fireeval_v1_test.json      # 测试集
│   ├── rag/
│   │   ├── pipeline.py         # 问答主管线
│   │   ├── scene.py            # 场景结构化 / 改写
│   │   ├── intent.py           # 意图路由
│   │   ├── retriever.py        # 混合检索 + RRF
│   │   ├── semantic_index.py   # Embedding + CrossEncoder
│   │   ├── reranker.py         # 法规意图重排
│   │   ├── verifier.py         # 引用 / 结论核验
│   │   └── graphrag.py         # 图谱构建与检索
│   └── scripts/
│       ├── evaluate_fireeval.py
│       ├── ablation_retrieval.py
│       ├── gate_retrieval.py       # 检索-only Hit 门禁
│       ├── check_gold_drift.py     # 金标条款存在性
│       ├── graph_health.py         # 图谱健康度
│       ├── expand_corpus.py
│       ├── split_fireeval.py
│       ├── build_law_corpus.py
│       └── ingest_official_sources.py
├── frontend/
│   ├── index.html              # 问答 + 图谱单页（含手机适配）
│   └── echarts.min.js
├── docs/                       # 方案、评测、问卷、健康度等
├── tests/
└── requirements.txt
```

---

## 更新法规语料

统一流水线（扩库后务必重建索引；运行中的后端可用热加载）：

```text
raw/*.md
  → expand_corpus.py / build_law_corpus.py / ingest_official_sources.py
  → data/*法规*.json（含 authority / status / source_url）
  → python3 rag/graphrag.py          # 重建 chunks.json + graph.json
  → python3 scripts/graph_health.py  # 健康度
  → python3 scripts/check_gold_drift.py --strict
  → python3 scripts/gate_retrieval.py --split dev --strict
  → POST /api/kb/reload 或重启 main.py
```

```bash
cd backend
# 消防法 / 61 号令：从 data/raw 生成 JSON 并重建图谱
python3 scripts/build_law_corpus.py

# 高层规定 / 责任制办法：从登记的官方地址抓取
python3 scripts/ingest_official_sources.py
# 扩展语料（39号令 / 电动车与密集场所报批稿 / 广东高层等，含去重）
python3 scripts/expand_corpus.py
python3 rag/graphrag.py

# 不重启进程时热加载
curl -X POST http://localhost:8319/api/kb/reload
```

---

## 当前局限（展示时请如实说明）

- Hit@1 / Hit@3：全量模型下 retrieval gate（dev）约 **0.96 / 1.00**；降级环境以本机消融表为准
- 意图路由以规则 + 可选微调为主；生成式 LoRA 仍需 GPU/云端训练
- Embedding / CrossEncoder 依赖本地下载；网络受限时会自动降级，并加大 BM25 权重
- 知识库覆盖有限，不构成完整消防法规汇编；报批稿须明示非正式效力

---

## 检索消融

```bash
cd backend
python3 scripts/ablation_retrieval.py --split dev
```

对比：`bm25` / `bm25_vector` / `hybrid`（三路无重排）/ `full`（三路+重排）。降级环境下常见结论：BM25 很强，弱向量/图谱会稀释 Hit@1，法规意图重排能部分挽回。书面表见 [`docs/消融表-retrieval.md`](docs/消融表-retrieval.md)。

---

## 后续路线（简）

1. ~~修复应急误路由边界~~  
2. ~~检索消融表 + 同义扩展/短语重排~~  
3. ~~意图分类与重排微调，形成「微调 + RAG」可对比演示~~（意图微调已落地，见 `train_intent.py`）  
4. ~~图谱健康度 / 金标漂移 / 检索门禁 / 宽问引导~~  
5. 收敛手调规则；半自动 NER 补词典；生成式 LoRA 迭代；基层问卷回收  

---

## License / 声明

原型代码用于学习、课程与创新竞赛展示。法规文本请以官方发布版本为准；使用本系统产生的任何决策后果由使用者自行承担。
