# 消安智答 FireSage

面向消防法规咨询与火灾应急提示的可信知识服务原型。系统以 BM25 和领域知识图谱进行混合检索，在回答中展示支撑条款与检索贡献；证据不足时主动拒答。

## 当前能力

- 消防法规问答：覆盖《中华人民共和国消防法（2021 修正）》《机关、团体、企业、事业单位消防安全管理规定》《高层民用建筑消防安全管理规定》和《消防安全责任制实施办法》。
- 知识图谱检索：从主体、行为、消防对象、处罚关系辅助召回口语化问题。
- 轻量多轮追问：可结合上一轮问题理解“那具体罚多少钱”等短追问。
- 应急分流：识别明确的正在发生的火情，并展示通用逃生提示与官方来源。
- 安全拒答：对越界问题或低置信度结果不强行生成答案。

当前内置数据为原型语料，共 452 个语义句、62 个实体、435 条关系，覆盖 106 个法规条款。它不是完整消防法规库，不应替代执法解释、专业咨询或现场消防指挥。

知识图谱页支持按法规来源、实体类型和关系筛选；节点详情会展示支撑条款、发布机关、生效日期和官方原文链接。

## 启动

```bash
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

未配置时系统自动使用抽取式回答，页面右上角会如实显示当前模式。

## 主要接口

- `POST /api/ask`：问答；参数为 `question`，可选 `previous_question`。
- `GET /api/system`：版本、检索方式、生成模式与知识来源。
- `GET /api/graph/stats`：当前图谱统计。
- `GET /api/graph/data`：图谱数据，支持 `types` 与 `query` 过滤。
- `GET /api/graph/entity?id=...`：实体、关系与关联条款详情。

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

1. 建立带标准答案和引用依据的 FireEval 评测集，先证明检索准确率、忠实度与拒答表现。
2. 引入向量召回并做 BM25 / GraphRAG / 三路融合消融实验，而不是只展示架构名词。
3. 将知识来源扩充到更多有效法规、标准与地方公开材料，并记录版本与生效状态。
4. 在数据和评测闭环成熟后，再进行领域微调与偏好对齐实验。
