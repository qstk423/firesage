# 消安智答 FireSage

面向消防法规咨询与火灾应急提示的可信知识服务原型。系统以 BM25、TF-IDF 字符向量和领域知识图谱进行三路召回，通过 RRF 融合与法规意图重排输出可核验条款；证据不足时主动拒答。

## 当前能力

- 消防法规问答：覆盖《中华人民共和国消防法（2021 修正）》《机关、团体、企业、事业单位消防安全管理规定》《高层民用建筑消防安全管理规定》和《消防安全责任制实施办法》。
- 混合检索：BM25、离线向量和知识图谱三路召回，使用 RRF 消除分数量纲差异。
- 法规重排：根据问题意图、条款文本覆盖度和标题相关性重新排列候选条款。
- 知识图谱检索：从主体、行为、消防对象、处罚关系辅助召回口语化问题。
- 轻量多轮追问：可结合上一轮问题理解“那具体罚多少钱”等短追问。
- 应急分流：识别明确的正在发生的火情，并展示通用逃生提示与官方来源。
- 安全拒答：对越界问题或低置信度结果不强行生成答案。

当前内置数据为原型语料，共 452 个语义句、62 个实体、435 条关系，覆盖 106 个法规条款。它不是完整消防法规库，不应替代执法解释、专业咨询或现场消防指挥。

知识图谱页支持按法规来源、实体类型和关系筛选；节点详情会展示支撑条款、发布机关、生效日期和官方原文链接。

## 启动

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cd backend
python3 main.py
```

浏览器访问 `http://localhost:8319`。

若需要接入兼容 OpenAI Chat Completions 的模型服务，可配置：

```bash
export LLM_BASE_URL="https://example.com/v1"
export LLM_API_KEY="your-key"
export LLM_MODEL="your-model"
```

也可以将 `backend/.env.example` 复制为 `backend/.env.local` 后填写。本地配置已被 Git 忽略。DeepSeek 当前可使用 `LLM_BASE_URL=https://api.deepseek.com` 与 `LLM_MODEL=deepseek-v4-flash`。

未配置时系统自动使用抽取式回答，页面右上角会如实显示当前模式。

## 主要接口

- `POST /api/ask`：问答；参数为 `question`，可选 `previous_question`。
- `GET /api/system`：版本、检索方式、生成模式与知识来源。
- `GET /api/graph/stats`：当前图谱统计。
- `GET /api/graph/data`：图谱数据，支持 `types` 与 `query` 过滤。
- `GET /api/graph/entity?id=...`：实体、关系与关联条款详情。

## 运行评测

FireEval v0 包含法规检索、意图路由、应急分流和安全拒答样例。运行：

```bash
cd backend
python3 scripts/evaluate_fireeval.py
python3 scripts/evaluate_fireeval.py --strict
```

当前评测输出 Hit@1、Hit@3、MRR、路由准确率和拒答准确率。`--strict` 可用于持续集成中的回归门禁。
当前 18 条小规模基线结果为 Hit@1 92.86%、Hit@3 100%、MRR 96.43%；样本量较小，不能替代后续人工扩充后的正式结论。

## 更新官方资料

```bash
cd backend
python3 scripts/ingest_official_sources.py
python3 rag/graphrag.py
```

第一步从配置的政府网站更新法规 JSON，第二步重新生成语义句和知识图谱。采集器只访问脚本中明确登记的官方地址，并为每份资料保留来源元数据。

## 目录

```text
firesage/
├── backend/
│   ├── main.py
│   ├── data/
│   └── rag/
└── frontend/
    ├── index.html
    └── echarts.min.js
```

## 下一步建议

1. 将 FireEval 扩展到至少 100 道人工核验题，并加入答案忠实度与引用完整性指标。
2. 在本地向量基线之上接入中文法律语义 Embedding 和 CrossEncoder，完成消融实验。
3. 将知识来源扩充到更多有效法规、标准与地方公开材料，并记录版本与生效状态。
4. 在数据和评测闭环成熟后，再进行领域微调与偏好对齐实验。
