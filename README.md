# 消安智答 FireSage

**面向消防法规咨询与火灾应急提示的可信知识问答原型。**

用自然语言描述真实场景（例如「楼梯口让纸箱堵得过不去了」），系统会检索可溯源的法规条款，说明责任主体、禁止行为与处罚依据；问法过宽时引导把问题问具体，证据不足或主题无关时拒答，不编造条文。

> 本系统仅供辅助参考，**不替代**执法解释、专业咨询或现场消防指挥。紧急情况请直接拨打 **119**。

---

## 申报材料

- 项目概述：[`docs/项目概述-消安智答FireSage.md`](docs/项目概述-消安智答FireSage.md)
- 解决方案文档（企业命题组）：[`docs/解决方案-消安智答FireSage.md`](docs/解决方案-消安智答FireSage.md)
- **答辩前检查清单（校赛收口）**：[`docs/答辩前检查清单.md`](docs/答辩前检查清单.md)
- 校赛演示材料包：[`docs/校赛演示材料包.md`](docs/校赛演示材料包.md)
- 产品级差距评估：[`docs/产品级差距评估.md`](docs/产品级差距评估.md)
- 完整项目差距清单：[`docs/完整项目差距清单.md`](docs/完整项目差距清单.md)
- 部署与配置清单：[`docs/部署与配置清单.md`](docs/部署与配置清单.md)
- RAG 完善任务清单：[`docs/RAG完善任务清单.md`](docs/RAG完善任务清单.md)
- 图谱健康度：[`docs/图谱健康度.md`](docs/图谱健康度.md)
- 演示截图清单：[`docs/演示截图清单.md`](docs/演示截图清单.md)
- 端到端联调报告：[`docs/end_to_end_report.md`](docs/end_to_end_report.md)
- 意图微调（可复现）：`python3 backend/scripts/train_intent.py --compare`
- 答辩日预检：`python3 backend/scripts/demo_check.py --preflight`

---

## 作品定位

| 维度 | 说明 |
|------|------|
| 赛道叙事 | 大模型微调与 RAG 技术融合的智慧消防应急智能问答 |
| 当前阶段 | 可演示、可评测的混合 RAG + GraphRAG + **本地微调 LoRA 生成**可信问答原型 |
| 设计原则 | **法规事实来自检索，不靠模型背条文**；宽问引导、库外拒答、无支撑不作答 |

---

## 核心能力

### 1. 可信法规问答
- 口语场景理解：楼道堆物、占用通道、电动车充电、无证动火、无证值班等
- 结构化回答：结论 / 适用条件 / 法规依据 / 补充说明 / 可信度
- 多轮追问：可结合上一轮问题理解「那具体罚多少钱」
- 要素不足时先澄清（如通道类型、单位/个人），避免套错条款

### 2. 混合检索（Hybrid RAG）
```text
用户提问
  → 场景结构化 + 查询改写
  → BM25 + 语义向量 + GraphRAG 三路召回
  → RRF 融合 + 法规意图重排 + CrossEncoder 精排
  → 检索门槛判定（top1 双低 → 直接标准拒答）
  → LLM 依据条款生成（流式输出 + 首字响应计时）
  → 引用核验 + 原文支撑核验；失败则重试一次，仍失败拒答
```

- **BM25**：关键词精确命中（句级索引，按条款取 max）
- **语义向量**：BAAI/bge-m3 **句级**检索后聚到条款（不可用时自动降级 TF-IDF）
- **GraphRAG**：主体–行为–对象–处罚 + **主题层**宽问下钻
- **RRF**：消除不同检索器分数量纲差异；报批稿条款检索降权
- **检索拒答门槛**：top1 fused<0.63 且 CrossEncoder<0.60 → 无可靠依据，直接标准拒答（门槛由 35 题正误案例的检索分数分布回放计算，见 `scripts/analyze_threshold.py`）

### 3. 本地微调 LoRA 生成（v0.8.5 新增）

- **基座**：Qwen2.5-3B-Instruct + LoRA（PEFT），单卡可训练、可推理
- **数据**：`backend/data/gen_sft_train_v2.jsonl`（问题 + 给定条款 → 结构化 JSON，禁止无依据扩写；训练/测试严格隔离，测试题不入训练集）
- **版本对比**：v1 / v2 / v3 已完成训练与横向评测，**v2 冻结为当前最佳**（输出格式纪律 100%、basis 引用正确率 78%）；v3 因拒答模板过拟合净回退，保留用于消融
- **推理服务**：`scripts/serve_local_qwen.py`，OpenAI Chat Completions 兼容协议，任何支持该协议的客户端可直接接入

### 4. 五层输出守卫（本地推理服务内置）

| 层 | 守卫 | 行为 |
|----|------|------|
| 1 | JSON 校验 | 剥围栏 / 截取花括号 / 截断修复 / 内引号转义；全部失败降级合法 JSON |
| 2 | 应急保护 | 强火情信号 + 求助语义但回答缺报警/疏散要素 → 替换标准应急引导 |
| 3 | 无条款守卫 | 桥接请求未检索到条款 → 强制标准拒答，杜绝无条款编造 |
| 4 | 引用守卫 | basis 逐条验证条款号；短键（`高层规定·第七条`）规范化为《法规全名》第N条 + 120 字摘录；全部无效时带强化指令重试一次（`citation_retry`），二次仍无效标准拒答（`citation_rejected`），**绝不强行重建引用** |
| 5 | 拒答规范化 | basis 占位符 / 拒答话术 → 规范化为标准拒答 JSON（basis 严格为 `[]`） |

### 5. 原文支撑核验（金额 / 期限 / 主体 / 处罚）

- 结论中的**罚款金额、期限、责任主体、处罚措施**必须能在引用条款原文中找到支撑（中文/阿拉伯数字等价转换）
- 引用提取**按行匹配**：法规名与条号须同现于同一行，杜绝跨法规虚假组合
- 无支撑 → 带反馈重试一次（要求删除无支撑表述）→ 仍失败 → 标准拒答（宁可拒答，不可编造）

### 6. 安全分流、引导与拒答
- **法规问答** / **火灾应急指引** / **闲聊引导** / **引导提问** / **安全拒答**
- **义务缺位吐槽 ≠ 应急**：「公司从来不组织消防演练，着火了都不知道往哪跑」是法规咨询不是险情
- 问法过宽（如「了解一下消防知识」）：给出示例问法，引导补全场景要素
- 知识库外法规/标准或无关主题：明确说明未收录 / 超出范围，避免误导
- CRAG：低置信时优先引导或拒答，不硬编条款

### 7. 流式输出与响应可观测
- 本地推理服务支持 SSE 流式生成，**首字响应时间（TTFT）** 全链路记录并透出到 `/api/ask` 响应的 `timing` 字段
- 端到端实测：TTFT p50 ≈ **3.4s**（CPU 检索 + 消费级 GPU 生成）

### 8. 知识图谱可视化
- 全图浏览：按法规来源、实体类型、关系筛选与搜索（**分簇漂浮，全览不画边**）
- **点击节点展开关系网**：以该实体为中心展示一级/二级邻居与连边，可继续点邻居切换中心
- 回答一键跳转「本次证据链」局部子图
- 节点详情展示支撑条款、发布机关、生效日期与官方原文链接
- 改完 `chunks.json` / `graph.json` 后可用 `POST /api/kb/reload` 热加载，无需重启进程

### 9. FireEval 评测闭环
- **229** 道人工核验题，覆盖口语、处罚、跨条款、多轮、应急、拒答及物业高频场景等
- 已划分 **train / dev / test**，便于后续微调与回归
- 指标：Hit@1 / Hit@3 / MRR、路由准确率、拒答准确率、引用准确率、严重错误率
- CI 含金标漂移检查与检索-only Hit 门禁（见 `.github/workflows/fireeval.yml`）

### 10. 前端体验
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
| 《社会消防技术服务管理规定》（应急管理部令第 7 号） | **39 条全文** | 维保检测 / 安全评估机构 |
| 《建设工程消防设计审查验收管理暂行规定》（住建部令第 51 号） | **43 条全文** | 特殊工程审查与验收备案 |

当前索引约 **524** 个条款、**1675** 个语义句、**118** 个实体、**972** 条关系（含高层「主题」节点）。
实体类型：**主题 / 违规行为 / 管理义务 / 政府职责 / 主体 / 消防对象 / 处罚**。
可视化展示时会按实体对去重，界面更清晰。图谱页「法规来源」筛选与侧栏规模数字均来自 `/api/graph/stats`、`/api/system`（当前 **10** 部）。

---

## 快速开始

### 方式一：云端 LLM（DeepSeek 等 OpenAI 兼容服务）

```bash
git clone https://github.com/qstk423/firesage.git
cd firesage
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
export LLM_BASE_URL="https://api.deepseek.com"
export LLM_API_KEY="your-key"
export LLM_MODEL="deepseek-v4-flash"
cd backend
python3 main.py
```

浏览器打开：**http://localhost:8319**（手机可用同一地址；窄屏自动切底部导航）。

### 方式二：本地微调 LoRA 栈（不依赖云端，不动生产配置）

三步，分别三个终端（均在 `backend` 目录）：

```bash
# 1. 本地推理服务：Qwen2.5-3B + LoRA v2，OpenAI 兼容，监听 8320
..\venv311\Scripts\python.exe scripts\serve_local_qwen.py --api-key local-firesage

# 2. 独立 FireSage 后端：读 .env.local-model，BGE 走 CPU 避让显存，监听 8321
..\venv311\Scripts\python.exe scripts\run_firesage_local.py

# 3. 端到端 35 题测试（法规 30 + 拒答/应急/闲聊 5）
..\venv311\Scripts\python.exe scripts\e2e_local_test.py
```

- `.env.local-model` 为独立配置文件，**不修改** DeepSeek 生产配置（`.env.local`），停掉本地进程即回退云端
- 显存不足时 `run_firesage_local.py` 已强制 `CUDA_VISIBLE_DEVICES=""`，检索模型走 CPU，与 8320 推理服务互不抢占
- 需要 `LLM_STREAM=1`（默认已写入 `.env.local-model`）开启流式生成与 TTFT 记录
- 演示默认采用确定性生成并把输出限制为 480 tokens；8320 启动时会自动预热 GPU，减少第一次问答额外等待
- 返回的 `timing` 包含检索、总生成、纯模型生成、TTFT 与 token 用量，可据此区分 CPU 检索和 GPU 生成瓶颈
- 若要复现 700-token 长答案评测，可在启动 8320 前设置 `LOCAL_LLM_MAX_TOKENS=700`，并同步调整 `LLM_MAX_TOKENS=700`

### Docker 一键启动（试点 / 演示）

默认轻量镜像（`FIRESAGE_LITE=1`）：不下载 bge，BM25 + TF-IDF + GraphRAG，适合现场演示。

```bash
docker compose up --build -d
# 浏览器 http://localhost:8319
curl -s http://localhost:8319/api/system | python3 -c "import sys,json;print(json.load(sys.stdin).get('runtime'))"
docker compose logs -f firesage
docker compose down
```

全量质量档（需 LLM 密钥 + 较大磁盘/时间）：

```bash
docker compose -f docker-compose.full.yml --env-file backend/.env.local up --build -d
python3 backend/scripts/check_runtime.py --strict
```

公网反代示例见 `docs/nginx-firesage.conf.example`。更多变量说明见 [`docs/部署与配置清单.md`](docs/部署与配置清单.md)。

### 可选：中文 Embedding / 精排模型

首次启动可能下载 `bge-m3` 与 `bge-reranker-v2-m3`（体积较大）。国内可加速：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

模型不可用时**不影响启动**，自动降级为 TF-IDF + 规则重排。

---

## 领域微调（可复现）

三层微调全部落地：

| 层次 | 状态 | 命令 |
|------|------|------|
| 意图路由微调 | **已落地**（默认 hybrid） | `python3 backend/scripts/train_intent.py --compare` |
| 条款重排微调 | **可训练**；默认关闭（`RERANK_ML=1` 开启） | `python3 backend/scripts/train_rerank.py --compare` |
| 生成式 LoRA | **已落地**（v1/v2/v3 已训练，v2 冻结） | 数据：`build_gen_sft_v2.py` → 训练：`train_lora_qwen3b.py` → 评测：`eval_lora_test.py` / `compare_v123.py` |

生成 SFT 数据：`backend/data/gen_sft_train_v2.jsonl`（v2 分布；v1/v3 保留用于消融对比）。
**评测纪律**：`gen_sft_test.jsonl`（35 题）为独立测试集，**严禁加入任何版本训练集**。

### LoRA 版本对比结论

| 版本 | 训练要点 | 表现 | 结论 |
|------|----------|------|------|
| v1 | 基础 SFT | 格式可，引用正确率一般 | 基线 |
| **v2** | 修正拒答样本 basis=[]、评测口径 | **格式纪律 100%、basis 正确率 78%** | **冻结为当前最佳** |
| v3 | 30 条同模板拒答 + 2 epochs | 拒答/应急双 0%（过拟合净回退） | 教训：模板需多样、epochs 需收敛 |

v4 候选方向（未实施）：拒答模板多样化、epochs 降至 1、basis 法条文本截断、应急场景扩写。

---

## 演示建议（3 分钟）

1. **口语法规**：「楼道堆放杂物违反什么规定」→ 看条款依据 + 点「在知识图谱中查看证据链」
2. **口语化金标**：「安排没证的人值班会怎么样」→ 检索门槛放行 + 引用《高层民用建筑消防安全管理规定》第四十七条
3. **应急分流**：「家里着火了现在怎么办」→ 应急指引，提示拨打 119
4. **义务吐槽 ≠ 应急**：「公司从来不组织消防演练，着火了都不知道往哪跑」→ 法规咨询（不误入应急）
5. **安全拒答**：「飞机库的防火分区面积要求」→ 检索门槛拦截，标准拒答不编造
6. **图谱探索**：进入「知识图谱」页，实体按类型分簇漂浮；点选后关系线长出，再点「返回全图」回到分簇视图

---

## 评测

### 端到端（本地 LoRA 栈，35 题独立测试集）

```bash
cd backend
..\venv311\Scripts\python.exe scripts\e2e_local_test.py
```

最新结果（v0.8.5，报告：`backend/eval_reports/e2e_local_test.json`）：

| 指标 | 修复前 | 修复后 |
|------|--------|--------|
| 通过率 | 22/35（62.9%） | **31/35（88.6%）** |
| 金标条款命中 | — | 22/30 |
| 首字响应 TTFT | 未记录 | **p50 3.4s / max 4.7s** |

4 条未通过中：3 条为库外题**正确拒答**（评分脚本未区分库外正确拒答，实际行为正确 34/35）；1 条为模型错引法规名被引用守卫正确拦截后拒答（宁可拒答不可编造）。

### 检索分数分布与拒答门槛

```bash
python3 scripts\replay_retrieval.py    # 回放 35 题检索分数（不进 LLM）
python3 scripts\analyze_threshold.py   # 门槛敏感性分析
```

由正确/错误案例的 top1 fused + CrossEncoder 联合分布计算得 `fused<0.63 且 CE<0.60`：拦截全部库外题、零误伤口语化金标（「安排没证的人值班」top1=0.633）。

### FireEval 与项目级评估

```bash
# 项目级评估：五维打分 + 申报/试点/产品三档
python3 backend/scripts/evaluate_project.py --mode quick \
  --report docs/项目评估报告.md

cd backend
# 按划分评测（推荐日常看 dev）
python3 scripts/evaluate_fireeval.py --split dev
python3 scripts/evaluate_fireeval.py --split test

# 一键验收（意图对比 + FireEval + 烟雾 + 演示三问）
python3 scripts/demo_check.py --quick

# 金标漂移 / 检索-only Hit 门禁 / 图谱健康度
python3 scripts/check_gold_drift.py --split all --strict
python3 scripts/gate_retrieval.py --split dev --strict
python3 scripts/graph_health.py --markdown docs/图谱健康度.md
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
| `POST` | `/api/ask` | 问答；`question`，可选 `previous_question`；响应含 `timing.ttft_ms`（流式开启时） |
| `POST` | `/api/feedback` | 用户反馈（答偏/条款不准等） |
| `GET` | `/api/audit/days` | 审计日期列表 |
| `GET` | `/api/audit/export` | 按日导出审计（`day=YYYYMMDD`） |
| `GET` | `/api/system` | 版本、检索通道、生成模式、知识来源与规模 |
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
├── Dockerfile              # 演示轻量镜像
├── docker-compose.yml      # docker compose up --build
├── requirements.txt
├── requirements-docker.txt # 无 torch 的轻量依赖
├── backend/
│   ├── main.py                 # FastAPI 入口（8319，云端 LLM）
│   ├── data/                   # 法规 JSON、chunks、graph、SFT 数据
│   │   ├── gen_sft_train_v2.jsonl   # LoRA v2 训练集（当前最佳）
│   │   └── gen_sft_test.jsonl       # 35 题独立测试集（严禁入训练）
│   ├── rag/
│   │   ├── pipeline.py         # 问答主管线（检索门槛/支撑核验重试/TTFT）
│   │   ├── scene.py            # 场景结构化 / 改写
│   │   ├── intent.py           # 意图路由（义务吐槽≠应急、值班/持证词表）
│   │   ├── retriever.py        # 混合检索 + RRF
│   │   ├── semantic_index.py   # Embedding + CrossEncoder
│   │   ├── reranker.py         # 法规意图重排
│   │   ├── verifier.py         # 引用/金额/期限/主体/处罚核验（按行匹配）
│   │   └── graphrag.py         # 图谱构建与检索
│   └── scripts/
│       ├── serve_local_qwen.py     # 本地推理服务（8320，五层守卫+流式）
│       ├── run_firesage_local.py   # 独立本地栈启动器（8321，CPU 检索）
│       ├── e2e_local_test.py       # 端到端 35 题测试
│       ├── replay_retrieval.py     # 检索分数回放
│       ├── analyze_threshold.py    # 拒答门槛敏感性分析
│       ├── train_lora_qwen3b.py    # LoRA 训练（Qwen2.5-3B）
│       ├── build_gen_sft_v2.py     # SFT v2 数据构建
│       ├── eval_lora_test.py       # LoRA 版本评测
│       ├── compare_v123.py         # v1/v2/v3 横向对比
│       ├── evaluate_project.py     # 项目级评估（五维 + 三档）
│       ├── evaluate_fireeval.py
│       ├── gate_retrieval.py       # 检索-only Hit 门禁
│       ├── check_gold_drift.py     # 金标条款存在性
│       └── graph_health.py         # 图谱健康度
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

---

## 版本与迭代记录

| 版本 | 要点 |
|------|------|
| **v0.8.5** | **本地微调 LoRA 生成落地**：v1/v2/v3 训练对比、v2 冻结；本地推理服务（五层守卫 + SSE 流式 + TTFT）；检索拒答门槛（fused+CE 双低）；金额/期限/主体/处罚原文支撑核验（重试一次仍失败拒答）；引用短键规范化 + basis 120 字精简；意图加固（义务吐槽≠应急、值班/持证词表）；端到端 22/35 → **31/35** |
| **v0.8.4** | 响应加速：CrossEncoder 只精排前 N 条；LLM 上下文限 3 条；默认不二次生成；回答展示检索/生成耗时拆分 |
| **v0.8.3** | 学成熟 RAG：parent–child 切块、`status` 进索引并降权报批稿、句级向量、图谱「主题」双层召回 |
| **v0.8.2** | 质量可观测（`/api/system.runtime`）、`check_runtime.py`、全量/轻量 Docker、nginx 示例 |
| **v0.8.x** | 语料扩至 10 部、FireEval 229 题、反馈/审计导出、隐患≠应急分流加固 |

### 用户问卷反馈 → 产品改动（2026-09）

公网小样本试用结论：

1. **出处可信**受认可 → 保留引用；前端改为**结论先行 + 法规依据折叠**（已落地）
2. **太专业/不够直接** → 主卡片突出结论，依据默认折叠；生成提示约束大白话短结论
3. **反应慢** → v0.8.4 砍精排宽度、限制生成上下文、关掉默认二次 LLM；v0.8.5 流式输出 + TTFT 可观测（p50 3.4s）；可用 `CE_CANDIDATES` / `LLM_CONTEXT_ARTICLES` / `VERIFY_REGENERATE` 调参（见 `backend/.env.example`）

### 本版发布前验收（2026-09-10）

```bash
# 端到端 35 题（本地 LoRA 栈）
cd backend && ..\venv311\Scripts\python.exe scripts\e2e_local_test.py
# → 31/35 通过；金标命中 22/30；TTFT p50 3424ms

PYTHONPATH=backend python3 -m unittest tests.test_smoke -q
# → 19 tests OK
```

---

## 当前局限（展示时请如实说明）

- Hit@1 / Hit@3：全量模型下 retrieval gate（dev）约 **0.96 / 1.00**；降级环境以本机消融表为准
- LoRA 生成在 3B 基座上偶发错引法规名（如「消防法·第七条」实为高层规定第七条），由引用守卫拦截后拒答，未构成错误输出；v4 待改进
- Embedding / CrossEncoder 依赖本地下载；网络受限时会自动降级，并加大 BM25 权重
- 知识库覆盖有限，不构成完整消防法规汇编；报批稿须明示非正式效力
- 检索拒答门槛（fused<0.63 且 CE<0.60）基于 35 题分布回放，样本有限，扩库后建议重跑 `analyze_threshold.py` 校准

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
3. ~~意图分类与重排微调，形成「微调 + RAG」可对比演示~~
4. ~~图谱健康度 / 金标漂移 / 检索门禁 / 宽问引导~~
5. ~~生成式 LoRA 训练落地（v2 冻结）+ 端到端可信增强（门槛/核验/流式）~~
6. LoRA v4：拒答模板多样化、epochs 收敛、basis 截断、应急扩写；半自动 NER 补词典；基层问卷回收

---

## License / 声明

原型代码用于学习、课程与创新竞赛展示。法规文本请以官方发布版本为准；使用本系统产生的任何决策后果由使用者自行承担。
