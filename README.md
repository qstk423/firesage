# 消安智答 FireSage

面向真实消防场景的可信法规问答系统。自由描述你遇到的情况（如「楼梯口让纸箱堵得过不去了」），系统回答：谁的责任、什么行为、违反哪条、如何处罚，每个结论都落到条款；证据不足时明确拒答。

## v0.6 可信检索版

```text
用户自由提问
   ↓ 场景结构化（主体＋行为＋场所＋对象＋意图）
   ↓ 查询改写与同义表达扩展
   ↓ BM25＋bge-m3 语义向量＋知识图谱 三路召回
   ↓ CrossEncoder 精确重排
   ↓ LLM 依据条款生成
   ↓ 逐项核验结论与引用一致性
   ↓ 可信回答 / 澄清追问 / 安全拒答
```

### 可信度体系

- **语义理解**：BAAI/bge-m3 中文语义向量（FAISS 存储）＋ BAAI/bge-reranker-v2-m3 精排，口语与法规术语即使无相同字词也能建立联系；模型缺失时自动降级 TF-IDF。
- **答案核验器**：LLM 生成后逐结论独立核验——引用条款是否支持该结论、处罚金额与主体是否匹配、高层规定是否被误用于普通建筑。核验未过则降级为条款原文摘录，绝不编造。
- **三道拒答护栏**：书名号识别知识库外法规（如《消防设施通用规范》）、CrossEncoder 地板分（无关问题集中在 0.50 附近，库内命中 p25=0.716）、领域主题护栏（报考、森林火灾等）。
- **澄清追问**：「占用通道罚多少钱」不直接套条款，先确认通道类型（疏散/安全出口/消防车通道）与行为人（单位/个人）——不同组合适用不同处罚条款。

### 回答形态

固定五段式结构：**结论 / 适用条件 / 法规依据 / 补充说明 / 可信度**，附过程状态条（检索→重排→生成→核验）与各检索通道占比。回答底部一键跳转知识图谱，仅展示本次回答相关的局部子图。

### 知识库

| 法规 | 条款 |
|------|------|
| 中华人民共和国消防法（2021 修正） | 74 条全文 |
| 机关、团体、企业、事业单位消防安全管理规定（61 号令） | 48 条全文 |
| 高层民用建筑消防安全管理规定 | 全文 |
| 消防安全责任制实施办法 | 全文 |

合计约 828 个语义句、63 个实体、612 条关系，覆盖 204 个法规条款。仍不是完整消防法规库，不替代执法解释、专业咨询或现场消防指挥。

## 启动

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cd backend
python3 main.py
```

浏览器访问 `http://localhost:8319`。

首次启动会自动下载 bge-m3 与 bge-reranker-v2-m3 模型（约 4.4GB，国内可 `export HF_ENDPOINT=https://hf-mirror.com` 加速）；模型缺失时不影响启动，自动降级 TF-IDF 检索。

### 接入 LLM

配置兼容 OpenAI Chat Completions 的服务（未配置时使用抽取式回答）：

```bash
export LLM_BASE_URL="https://api.deepseek.com"
export LLM_API_KEY="your-key"
export LLM_MODEL="deepseek-v4-flash"
```

也可复制 `backend/.env.example` 为 `backend/.env.local` 填写，本地配置已被 Git 忽略。回答带缓存与超时重试。

## 主要接口

- `POST /api/ask`：问答；参数 `question`，可选 `previous_question`（多轮追问）。
- `GET /api/system`：版本、检索方式、生成模式与知识来源。
- `GET /api/graph/stats` `/data` `/entity`：图谱统计、数据（支持 `types`/`query` 过滤）与实体详情。
- `GET /api/graph/subgraph?articles=...`：按条款提取回答相关局部子图。

## FireEval v1 评测

200 道人工核验题，覆盖 8 类题型：直接法规查询 40、生活化口语 40、处罚与责任主体 35、跨条款综合 25、多轮追问 20、应急 15、知识库外消防 15、非消防 10。每题含标准意图、可接受条款、必须/禁止出现的结论、是否拒答。

```bash
cd backend
python3 scripts/evaluate_fireeval.py           # 输出全量指标
python3 scripts/evaluate_fireeval.py --strict  # 回归门禁（未达标退出码 1）
```

| 指标 | 当前 | 发布门槛 |
|------|------|----------|
| Hit@3 | 85% | ≥ 95% |
| Hit@1 | 72.5% | ≥ 85% |
| 引用准确率 | 100% | ≥ 95% ✓ |
| 拒答准确率 | 97.5% | ≥ 95% ✓ |
| 严重错误率 | 0.5% | ≤ 1% ✓ |
| 平均响应 | 6.5s | ≤ 8s ✓ |

Hit@1/Hit@3 尚未达标，已定位到召回阶段（部分条款如消防法第五十八条未进候选池），计划通过同义词词典扩充与查询意图分类器解决。GitHub Actions 会在每次推送时自动运行评测（`.github/workflows/fireeval.yml`）。

## 测试

```bash
python3 -m unittest discover -s tests -v   # 14 项冒烟测试
```

覆盖场景结构化（口语识别、澄清判定）、核验器（中文数字解析、处罚一致性、伪造引用拦截）与端到端问答（检索命中、澄清、多轮、应急、拒答）。

## 更新官方资料

```bash
cd backend
# 消防法 / 61 号令：由 data/raw 全文生成 JSON 并重建图谱
python3 scripts/build_law_corpus.py
# 高层规定 / 责任制办法：在线抓取（可与上一命令分开跑）
python3 scripts/ingest_official_sources.py
python3 rag/graphrag.py
```

`build_law_corpus.py` 从 `data/raw/*.md` 生成层级 JSON（保留官方链接、生效日期等元数据）并重建语义句与知识图谱。在线采集器只访问脚本中明确登记的政府地址。

## 目录

```text
firesage/
├── backend/
│   ├── main.py               # FastAPI 服务
│   ├── data/                 # 法规 JSON / 原始语料 / 图谱
│   ├── eval/fireeval_v1.json # 200 题评测集
│   ├── rag/
│   │   ├── pipeline.py       # 问答主管线（CRAG + 核验 + 澄清）
│   │   ├── scene.py          # 场景结构化与查询改写
│   │   ├── retriever.py      # 三路召回 + RRF 融合
│   │   ├── semantic_index.py # bge-m3 + FAISS + CrossEncoder
│   │   ├── reranker.py       # 法规意图重排
│   │   ├── verifier.py       # 答案核验器
│   │   └── graphrag.py       # 知识图谱构建与检索
│   └── scripts/              # 评测 / 语料构建 / 采集
├── frontend/
│   ├── index.html            # 单页应用（问答 + 图谱）
│   └── echarts.min.js
└── tests/test_smoke.py       # 冒烟测试
```

## 路线图

- **v0.7 知识工程版**：新消防 Schema（多并列行为、单位/个人处罚分离、义务-处罚条款关联）、法规版本与生效状态管理、扩充官方资料（建设工程消防审验、消防监督检查、电动自行车规范等）。
- **v1.0 可交付版本**：Docker 部署、管理后台、运行监控、安全与费用控制、正式评测报告。
