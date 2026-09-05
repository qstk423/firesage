# -*- coding: utf-8 -*-
"""场景结构化：把用户自由提问解析为 主体+行为+场所+对象+意图 五要素。

这是下一代处理流程的第二层：用户自由提问 → 场景结构化 → 查询改写与扩展。
结构化结果用于：
1. 查询改写（把口语场景转写为法规术语查询）；
2. 澄清判定（缺少关键要素且候选条款分歧大时，向用户追问）；
3. 回答的"适用条件"段落（明确主体/场所是否与条款匹配）。
"""
import re

# ---------- 主体词典 ----------
SUBJECT_MAP = {
    "单位": ["单位", "公司", "企业", "工厂", "商场", "宾馆", "酒店", "学校", "医院",
            "机关", "团体", "事业单位", "店铺", "门店", "写字楼", "业主", "使用人", "建设单位", "施工单位"],
    "个人": ["个人", "我", "居民", "住户", "业主个人", "员工", "租户", "承租人", "本人", "家里人", "邻居"],
    "物业服务企业": ["物业", "物业公司", "物业服务企业", "物业管理单位", "统一管理人"],
    "消防安全责任人": ["消防安全责任人", "法人", "法定代表人", "主要负责人", "实际控制人"],
    "消防安全管理人": ["消防安全管理人"],
    "消防救援机构": ["消防救援机构", "消防部门", "消防大队", "消防队"],
    "政府部门": ["政府", "人民政府", "乡镇政府", "公安机关", "住建部门", "住房和城乡建设主管部门"],
    "村民委员会/居民委员会": ["居委会", "居民委员会", "村委会", "村民委员会", "社区"],
    "消防技术服务机构": ["消防技术服务机构", "维保单位", "检测机构", "消防设施施工安装企业"],
}

# ---------- 行为词典：口语 → 法规表述 ----------
BEHAVIOR_MAP = {
    "占用疏散通道": ["楼道堆放", "楼道堆杂物", "堆放杂物", "堆东西", "过道堆", "走廊堆放",
                    "占用通道", "堵塞通道", "堵通道", "占用疏散通道", "堵塞疏散通道", "封闭疏散通道",
                    "堆纸箱", "纸箱堵", "杂物堵", "占道", "通道堆物", "楼梯口堆", "楼梯间堆放",
                    "安全出口堆放", "堵安全出口", "锁闭安全出口", "封安全出口", "锁安全出口"],
    "占用消防车通道": ["占用消防通道", "堵消防通道", "消防通道停车", "占用消防车通道", "堵消防车通道",
                       "消防车道停车", "私家车堵消防通道", "设置构筑物", "固定隔离桩"],
    "埋压圈占消火栓": ["消火栓被埋", "埋压消火栓", "圈占消火栓", "遮挡消火栓", "消火栓挡住", "消火栓被挡"],
    "损坏挪用消防设施": ["消防设施坏了不修", "损坏消防设施", "挪用消防器材", "擅自拆除", "停用消防设施",
                        "灭火器过期", "拿走灭火器", "消防设施故障", "设施损坏", "器材挪用", "拆除喷淋"],
    "电动自行车违规停放充电": ["电动车楼道充电", "楼道充电", "电动车进楼", "电瓶车入户", "电动自行车充电",
                              "飞线充电", "电动车停楼道", "电瓶车楼道", "推电动车进电梯", "入户充电",
                              "公共门厅充电", "楼梯间充电"],
    "违规明火作业": ["电焊", "气焊", "动火", "明火作业", "焊接", "没有动火证"],
    "谎报火警": ["谎报火警", "谎报", "报假警", "假火警", "打假119"],
    "无人值班": ["消防控制室无人", "无人值班", "控制室没人", "消控室无人", "没有值班", "减配值班人员"],
    "值班人员不足": ["只有一名值班", "值班人员不足", "单人值班"],
    "未保持完好有效": ["消防设施不符合标准", "配置不符合", "标志不符合", "设施不完好", "没有保持完好"],
    "占用避难层": ["避难层堆放", "占用避难层", "避难层锁闭", "占用避难走道"],
    "占用电缆井管道井": ["电缆井堆放", "管道井堆物", "占用管井", "井道堆杂物"],
    "违规储存危险品": ["储存危险品", "存放易燃易爆", "储存甲类物品", "违规储存", "地下车库存放汽油"],
    "占用防火间距": ["占用防火间距", "搭棚占间距", "防火间距堆物"],
    "未组织防火检查": ["不搞防火检查", "没有防火检查", "从不巡查", "未开展防火检查"],
    "未组织消防演练": ["不搞演练", "没有消防演练", "未组织演练", "从不演练"],
    "未开展消防培训": ["不培训员工", "没有消防培训", "未开展教育培训"],
    "设置障碍物影响逃生": ["装防盗窗", "设置障碍物", "门窗设置障碍", "广告牌挡窗", "外窗设置障碍"],
    "违规使用燃气": ["擅自改装燃气", "违规用气", "瓶装液化气", "擅自拆除燃气"],
    "高风险作业指使": ["强令冒险作业", "指使冒险作业", "违章指挥"],
}

# ---------- 场所词典 ----------
VENUE_MAP = {
    "高层住宅建筑": ["高层住宅", "高层小区", "高层住宅建筑", "高层居民楼", "30层的住宅", "高层住宅楼"],
    "高层公共建筑": ["高层公共建筑", "高层写字楼", "高层办公楼", "高层商业建筑", "高层公寓", "高层宿舍"],
    "高层民用建筑": ["高层民用建筑", "高层建筑", "超高层", "100米以上建筑", "摩天楼"],
    "人员密集场所": ["人员密集场所", "商场", "超市", "影院", "网吧", "KTV", "歌舞厅", "体育馆", "候车厅",
                    "公众聚集场所", "养老院", "幼儿园", "学校", "医院"],
    "人员密集场所门窗": ["人员密集场所的门窗"],
    "一般单位": ["单位", "公司", "企业", "工厂", "仓库"],
    "住宅区": ["住宅区", "小区", "居民楼", "居民区"],
    "公共门厅/疏散走道/楼梯间": ["公共门厅", "疏散走道", "楼梯间", "安全出口"],
}

# ---------- 对象词典（法规中的消防对象） ----------
OBJECT_MAP = {
    "疏散通道": ["疏散通道", "楼道", "走廊", "过道", "楼梯间", "楼梯口", "安全出口", "逃生通道"],
    "消防车通道": ["消防车通道", "消防通道", "消防车道"],
    "消火栓": ["消火栓", "消防栓"],
    "消防设施器材": ["消防设施", "消防器材", "灭火器", "喷淋", "报警系统", "应急照明", "疏散指示标志", "防火门"],
    "消防控制室": ["消防控制室", "消控室"],
    "避难层": ["避难层", "避难间", "避难走道"],
    "防火间距": ["防火间距"],
    "电动自行车": ["电动自行车", "电动车", "电瓶车"],
    "电缆井/管道井": ["电缆井", "管道井", "管井", "桥架"],
}

# ---------- 问题意图 ----------
QUESTION_INTENT_MAP = {
    "责任主体": ["谁负责", "责任主体", "谁来管", "归谁管", "谁的职责", "谁的责任", "谁来整改", "追究谁"],
    "处罚": ["怎么处罚", "罚多少", "罚款", "处罚", "什么后果", "拘留", "怎么罚", "要罚", "责令", "承担什么责任"],
    "是否违规": ["违反什么", "违规吗", "违法吗", "算不算", "可以吗", "能不能", "允许吗", "犯法吗", "违反了什么规定"],
    "义务要求": ["有什么职责", "要求", "应当", "需要做什么", "怎么做", "有什么义务", "标准是什么", "多久一次"],
    "应急处理": ["着火", "起火", "冒烟", "烧起来", "火势", "逃生", "被困", "怎么逃", "紧急"],
}


def _match_multi(question, mapping):
    hits = []
    for canonical, phrases in mapping.items():
        for phrase in phrases:
            if phrase in question:
                hits.append(canonical)
                break
    return hits


def _detect_building_height(question):
    """从口语中提取建筑高度线索（高层定义：住宅>27m，公共>24m）。"""
    m = re.search(r"(\d+)\s*(?:层|米|m)", question)
    if not m:
        return None
    value = int(m.group(1))
    if "米" in question or "m" in question.lower():
        return {"height_m": value}
    # 经验换算：住宅层高约3米
    return {"floors": value, "height_m_est": value * 3}


def structure(question):
    """把自由提问解析为五要素场景。

    返回 dict：subject / behaviors / venues / objects / question_intent /
    building_hint（高层线索）/ ambiguity（待澄清要素列表）/ rewrite（查询改写）。
    """
    subjects = _match_multi(question, SUBJECT_MAP)
    behaviors = _match_multi(question, BEHAVIOR_MAP)
    venues = _match_multi(question, VENUE_MAP)
    objects = _match_multi(question, OBJECT_MAP)
    intents = _match_multi(question, QUESTION_INTENT_MAP)
    building_hint = _detect_building_height(question)

    # --- 澄清判定 ---
    # 处罚条款按"疏散通道/消防车通道"和"单位/个人"分叉，缺失时处罚依据完全不同。
    # 澄清必须保守：仅当"通道"完全未指明类型时才追问（此时单位/个人处罚分叉最大）；
    # 已明确通道类型或专用条款（如电动自行车）的问题一律直接回答。
    ambiguity = []
    if intents and "处罚" in intents:
        specific_channel = bool(set(objects) & {"疏散通道", "消防车通道"}) or any(
            w in question for w in ("消防车", "疏散", "安全出口", "楼道", "楼梯"))
        mentions_channel = specific_channel or "通道" in question or bool(
            set(behaviors) & {"占用疏散通道", "占用消防车通道"})
        if mentions_channel and not specific_channel:
            # 只说了"通道"等泛称，未指明类型
            ambiguity.append("channel_type")
            if not subjects and not re.search(r"[个我居邻本][^楼区]*?(?:占用|堵|堆|放)", question):
                # 处罚对象未明：单位(消防法第六十条)与个人(第六十四条)处罚完全不同
                ambiguity.append("subject")

    # --- 查询改写：把口语要素组合成法规术语查询 ---
    parts = []
    if subjects:
        parts.append(subjects[0])
    if behaviors:
        parts.extend(behaviors)
    if objects and objects[0] not in parts:
        parts.append(objects[0])
    if venues:
        parts.append(venues[0])
    if intents:
        parts.append(intents[0])
    if not behaviors and not objects:
        # 未识别出任何消防行为/对象时，结构化改写会丢失问题核心词
        #（如"单位多长时间进行一次防火检查"只剩"单位"），保留原问题
        rewrite = question
    else:
        rewrite = " ".join(parts) if parts else question

    return {
        "subjects": subjects,
        "behaviors": behaviors,
        "venues": venues,
        "objects": objects,
        "question_intents": intents,
        "building_hint": building_hint,
        "ambiguity": ambiguity,
        "rewrite": rewrite,
    }


def merge_scenes(current, context):
    """多轮追问合并：追问自身的要素优先（如「个人呢」明确切换了主体），
    上一轮的要素作为背景补充。返回合并后的要素 dict（含 rewrite）。"""
    merged = {}
    for key in ("subjects", "behaviors", "venues", "objects", "question_intents"):
        head = list(current.get(key) or [])
        merged[key] = head + [x for x in (context.get(key) or []) if x not in head]
    merged["building_hint"] = current.get("building_hint") or context.get("building_hint")
    merged["ambiguity"] = []
    parts = []
    if merged["subjects"]:
        parts.append(merged["subjects"][0])
    parts.extend(merged["behaviors"])
    if merged["objects"] and merged["objects"][0] not in parts:
        parts.append(merged["objects"][0])
    if merged["venues"]:
        parts.append(merged["venues"][0])
    if merged["question_intents"]:
        parts.append(merged["question_intents"][0])
    merged["rewrite"] = " ".join(parts) if parts else None
    return merged


def clarify_question(ambiguity, scene):
    """根据缺失要素生成澄清问题。"""
    asks = []
    if "channel_type" in ambiguity:
        asks.append("请问是疏散通道、安全出口，还是消防车通道？")
    if "subject" in ambiguity:
        asks.append("行为人是单位还是个人？")
    if "building_type" in ambiguity:
        asks.append("是否属于高层民用建筑（住宅高度大于27米、公共建筑大于24米）？")
    if not asks:
        return None
    return "为了给出准确的处罚依据，请补充两个信息：" + " ".join(asks) if len(asks) > 1 else "为了给出准确的依据，请补充：" + asks[0]


# 总述/清单类：走「全局」检索，压专题噪声、抬核心枢纽条款
_GLOBAL_DUTY_MARKERS = (
    "有哪些职责", "应当履行哪些", "消防安全职责有哪些", "单位应当履行",
    "职责有哪些", "需要履行哪些", "消防工作职责",
)
_PENALTY_MARKERS = ("怎么处罚", "怎么罚", "罚多少", "罚款", "拘留", "什么后果", "会怎么罚", "处罚标准")
_DUTY_MARKERS = ("谁负责", "责任主体", "归谁管", "谁的职责", "谁来管", "责任怎么划分", "怎么划分")


def detect_query_mode(question, scene=None):
    """local=实体/场景条款；global=职责总述/清单类。"""
    q = question or ""
    if any(m in q for m in _GLOBAL_DUTY_MARKERS):
        return "global"
    intents = (scene or {}).get("question_intents") or []
    if "义务要求" in intents and not any(m in q for m in _PENALTY_MARKERS):
        if any(k in q for k in ("哪些", "有哪些", "应当履行")):
            return "global"
    return "local"


def decompose_queries(question, scene=None):
    """复合问拆子查询：责任划分 + 处罚 等拆两路召回再合并。

    返回去重后的查询列表；单意图时仍为 [原问]。
    """
    q = (question or "").strip()
    if not q:
        return []
    scene = scene or structure(q)
    intents = set(scene.get("question_intents") or [])
    has_penalty = "处罚" in intents or any(m in q for m in _PENALTY_MARKERS)
    has_duty = (
        "责任主体" in intents
        or "义务要求" in intents
        or any(m in q for m in _DUTY_MARKERS)
    )
    # 显式复合：谁负责……怎么罚 / 责任……处罚
    compound = has_penalty and has_duty
    if not compound and ("责任" in q and any(m in q for m in _PENALTY_MARKERS)):
        compound = True
    if not compound:
        base = scene.get("rewrite") or q
        return [q] if base == q else [base, q]

    # 场景骨架（去掉意图词）再分别挂「责任」「处罚」
    skeleton_parts = []
    for key in ("subjects", "behaviors", "objects", "venues"):
        vals = scene.get(key) or []
        if vals:
            skeleton_parts.append(vals[0] if key == "subjects" else " ".join(vals[:2]))
    skeleton = " ".join(skeleton_parts).strip() or q
    duty_q = f"{skeleton} 责任主体 职责划分"
    penalty_q = f"{skeleton} 处罚 罚款"
    out = []
    for item in (duty_q, penalty_q, q):
        if item and item not in out:
            out.append(item)
    return out
