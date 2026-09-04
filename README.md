# 消安智答 FireSage

**面向消防法规咨询与火灾应急提示的可信知识问答原型。**

用自然语言描述真实场景（例如「楼梯口让纸箱堵得过不去了」），系统会检索可溯源的法规条款，说明责任主体、禁止行为与处罚依据；证据不足时主动拒答，不编造条文。

> 本系统仅供辅助参考，**不替代**执法解释、专业咨询或现场消防指挥。紧急情况请直接拨打 **119**。

---

## 申报材料

- 项目概述：[`docs/项目概述-消安智答FireSage.md`](docs/项目概述-消安智答FireSage.md)
- 解决方案文档（企业命题组）：[`docs/解决方案-消安智答FireSage.md`](docs/解决方案-消安智答FireSage.md)
- 演示截图清单：[`docs/演示截图清单.md`](docs/演示截图清单.md)
- 意图微调（可复现）：`python3 backend/scripts/train_intent.py --compare`

---

## 作品定位

| 维度 | 说明 |
|------|------|
| 赛道叙事 | 大模型微调与 RAG 技术融合的智慧消防应急智能问答 |
| 当前阶段 | 可演示、可评测的混合 RAG + GraphRAG 可信问答原型（立项 / 展示版） |
| 设计原则 | **法规事实来自检索，不靠模型背条文**；证据不足则拒答 |

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
- 全图浏览：按法规来源、实体类型、关系筛选与搜索
- **点击节点展开关系网**：以该实体为中心展示邻居与连边，可继续点击邻居切换中心
- 回答一键跳转「本次证据链」局部子图
- 节点详情展示支撑条款、发布机关、生效日期与官方原文链接

### 4. 安全分流与拒答
- **法规问答** / **火灾应急指引** / **闲聊引导** / **安全拒答**
- 知识库外法规或主题：明确说明未收录，避免误导
- CRAG 低置信检索结果：拒答而非硬答

### 5. FireEval 评测闭环
- **200** 道人工核验题，覆盖口语、处罚、跨条款、多轮、应急、拒答等 8 类
- 已划分 **train / dev / test**，便于后续微调与回归
- 指标：Hit@1 / Hit@3 / MRR、路由准确率、拒答准确率、引用准确率、严重错误率

---

## 知识库规模

| 法规 | 规模 | 来源元数据 |
|------|------|------------|
| 《中华人民共和国消防法（2021 修正）》 | **74 条全文** | 发布机关、生效日期、官方链接 |
| 《机关、团体、企业、事业单位消防安全管理规定》（公安部 61 号令） | **48 条全文** | 同上 |
| 《高层民用建筑消防安全管理规定》 | 全文 | 同上 |
| 《消防安全责任制实施办法》 | 全文 | 同上 |

当前索引约 **204** 个条款、**828** 个语义句、**82** 个实体、**357** 条关系。
实体类型：**违规行为 / 管理义务 / 政府职责 / 主体 / 消防对象 / 处罚**（行为已三分，去掉「依法处罚」万能兜底；「单位」枢纽边占比约降至 19%）。
可视化展示时会按实体对去重，界面更清晰。

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

浏览器打开：**http://localhost:8319**

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

## 演示建议（3 分钟）

1. **口语法规**：「楼道堆放杂物违反什么规定」→ 看条款依据 + 点「在知识图谱中查看证据链」
2. **应急分流**：「家里着火了现在怎么办」→ 应急指引，提示拨打 119
3. **安全拒答**：询问知识库外标准或无关问题 → 明确拒答、不编造
4. **图谱探索**：进入「知识图谱」页，实体按类型分簇漂浮；点选后关系线长出，再点「返回全图」回到分簇视图

---

## 评测

```bash
cd backend

# 全量 v1（200 题）
python3 scripts/evaluate_fireeval.py

# 按划分评测（推荐日常看 dev）
python3 scripts/evaluate_fireeval.py --split dev
python3 scripts/evaluate_fireeval.py --split test

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
│       ├── split_fireeval.py
│       ├── build_law_corpus.py
│       └── ingest_official_sources.py
├── frontend/
│   ├── index.html              # 问答 + 图谱单页
│   └── echarts.min.js
├── tests/
└── requirements.txt
```

---

## 更新法规语料

```bash
cd backend
# 消防法 / 61 号令：从 data/raw 生成 JSON 并重建图谱
python3 scripts/build_law_corpus.py

# 高层规定 / 责任制办法：从登记的官方地址抓取
python3 scripts/ingest_official_sources.py
python3 rag/graphrag.py
```

---

## 当前局限（展示时请如实说明）

- Hit@1 / Hit@3 仍在提升中（dev 约 0.74 / 0.91，降级模式下）；语料扩充后竞争条款变多
- 意图路由仍以规则为主；赛道规划中的「意图 / 重排 / 生成」微调尚未全部落地
- Embedding / CrossEncoder 依赖本地下载；网络受限时会自动降级，并加大 BM25 权重
- 知识库覆盖有限，不构成完整消防法规汇编

---

## 检索消融

```bash
cd backend
python3 scripts/ablation_retrieval.py --split dev
```

对比：`bm25` / `bm25_vector` / `hybrid`（三路无重排）/ `full`（三路+重排）。降级环境下常见结论：BM25 很强，弱向量/图谱会稀释 Hit@1，法规意图重排能部分挽回。

---

## 后续路线（简）

1. ~~修复应急误路由边界~~  
2. ~~检索消融表 + 同义扩展/短语重排~~  
3. ~~意图分类与重排微调，形成「微调 + RAG」可对比演示~~（意图微调已落地，见 `train_intent.py`）  
4. 一键演示脚本与答辩材料；生成式 LoRA 迭代  

---

## License / 声明

原型代码用于学习、课程与创新竞赛展示。法规文本请以官方发布版本为准；使用本系统产生的任何决策后果由使用者自行承担。
