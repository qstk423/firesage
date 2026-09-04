# -*- coding: utf-8 -*-
"""GraphRAG 录入管线 + 图谱检索（核心模块）

录入管线：法规层级解析 → 语义断句 → 领域模板NER → 关系构建 → 知识图谱
图谱检索：实体链接（精确/模糊/意图增强）→ 邻边扩展 → 多跳打分

v0.2.0-graphrag
"""
import json
import os
import re

try:
    import jieba
    _HAS_JIEBA = True
except ImportError:
    jieba = None
    _HAS_JIEBA = False

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
FIRELAW_PATH = os.path.join(DATA_DIR, "firelaw.json")
# 只加载经过核验的法规文件，避免把缓存与构建产物误当成语料。
SOURCE_FILES = ["firelaw.json", "regulation61.json", "highrise5.json", "responsibility87.json", "entertainment39.json", "ebike_charging_draft.json", "assembly_occupancy_draft.json", "gd_highrise.json"]
ARCHIVE_PATHS = []
for _f in SOURCE_FILES:
    p = os.path.join(DATA_DIR, _f)
    if os.path.exists(p):
        with open(p, encoding="utf-8") as _source_file:
            _source = json.load(_source_file)
        ARCHIVE_PATHS.append((p, _source.get("law_abbr", "消防法")))
CHUNKS_PATH = os.path.join(DATA_DIR, "chunks.json")
GRAPH_PATH = os.path.join(DATA_DIR, "graph.json")

# ============================================================
# 一、领域知识库（实体词典 / 模板）
# ============================================================

# 行为三分：节点 type 直接用子类，便于筛选与配色
TYPE_VIOLATION = "违规行为"
TYPE_DUTY = "管理义务"
TYPE_GOV = "政府职责"
BEHAVIOR_TYPES = {TYPE_VIOLATION, TYPE_DUTY, TYPE_GOV}


def _is_behavior_type(ntype: str) -> bool:
    return ntype in BEHAVIOR_TYPES


SUBJECTS = [
    "地方各级人民政府", "乡镇人民政府", "街道办事处", "村民委员会", "居民委员会",
    "消防技术服务机构", "消防服务单位", "物业服务企业", "物业管理单位", "统一管理人",
    "消防安全责任人", "消防安全管理人", "单位主要负责人", "行业主管部门",
    "消防救援机构", "人员密集场所", "公共娱乐场所", "公众聚集场所",
    "总承包单位", "建设单位", "施工单位", "产权单位", "产权人",
    "事业单位", "机关", "团体", "企业", "业主", "使用人", "个人", "负责人", "单位",
    "经营单位", "物业服务人",
]
SUBJECT_ALIASES = {
    "物业服务企业": ["物业公司", "物业", "物业管理单位"],
    "物业管理单位": ["物业管理单位"],
    "物业服务人": ["物业服务人", "物业服务企业"],
    "消防安全责任人": ["责任人", "法定代表人", "主要负责人"],
    "消防安全管理人": ["管理人"],
    "单位": ["本单位"],
    "建设单位": ["甲方"],
    "施工单位": ["施工方", "承包商", "工地"],
    "总承包单位": ["总包"],
    "业主": ["房东", "产权人", "业主单位"],
    "产权人": ["产权人", "所有权人"],
    "使用人": ["住户", "承租人", "租户"],
    "统一管理人": ["统一管理单位"],
    "消防技术服务机构": ["消防维保公司", "消防检测机构", "维保单位"],
    "居民委员会": ["居委会", "社区居委会"],
    "村民委员会": ["村委会"],
    "人员密集场所": ["人员密集场所"],
    "公共娱乐场所": ["公共娱乐场所", "娱乐场所", "歌舞厅", "卡拉OK", "夜总会"],
    "公众聚集场所": ["公众聚集场所"],
    "经营单位": ["经营单位", "经营者"],
}

# kind: 违规行为 | 管理义务 | 政府职责
# aliases 只用口语动作短语，禁止与消防对象原名撞车
BEHAVIORS = {
    # ---- 违规行为 ----
    "占用疏散通道": {
        "kind": TYPE_VIOLATION,
        "verbs": ["占用", "堵塞", "堆积", "堆放", "锁闭"],
        "objects": ["疏散通道", "安全出口"],
        "aliases": ["堆杂物", "堆放杂物", "占通道", "堵住通道", "楼道堆", "堆东西", "纸箱堵"],
    },
    "堵塞安全出口": {
        "kind": TYPE_VIOLATION,
        "verbs": ["堵塞", "封锁", "锁住", "封闭"],
        "objects": ["安全出口", "出口"],
        "aliases": ["锁门", "封死出口", "堵住出口", "锁闭安全出口"],
    },
    "封闭消防车通道": {
        "kind": TYPE_VIOLATION,
        "verbs": ["占用", "堵塞", "封闭", "堵住", "圈占"],
        "objects": ["消防车通道", "消防通道"],
        "aliases": ["占用消防通道", "挡消防车", "停车挡道", "占消防通道"],
    },
    "损坏消防设施": {
        "kind": TYPE_VIOLATION,
        "verbs": ["损坏", "破坏", "损毁"],
        "objects": ["消防设施", "灭火器", "器材", "灭火系统"],
        "aliases": ["破坏消防设施", "损坏灭火器"],
    },
    "挪用消防设施": {
        "kind": TYPE_VIOLATION,
        "verbs": ["挪用"],
        "objects": ["消防设施", "灭火器", "器材"],
        "aliases": ["挪用灭火器", "挪用消防器材"],
    },
    "擅自拆除停用消防设施": {
        "kind": TYPE_VIOLATION,
        "verbs": ["拆除", "停用", "自行拆"],
        "objects": ["消防设施", "报警", "喷淋"],
        "aliases": ["拆消防设施", "关消防", "停用消防", "擅自停用"],
    },
    "埋压圈占遮挡消火栓": {
        "kind": TYPE_VIOLATION,
        "verbs": ["埋压", "圈占", "遮挡", "埋", "压", "盖"],
        "objects": ["消火栓", "消防栓"],
        "aliases": ["消火栓被埋", "盖住消火栓", "压住消火栓", "遮挡消火栓"],
    },
    "占用防火间距": {
        "kind": TYPE_VIOLATION,
        "verbs": ["占用"],
        "objects": ["防火间距", "间距"],
        "aliases": ["占防火间距", "搭棚占间距"],
    },
    "设置影响逃生障碍物": {
        "kind": TYPE_VIOLATION,
        "verbs": ["设置", "安装", "封"],
        "objects": ["障碍物", "防盗窗", "铁栅栏", "护栏"],
        "aliases": ["装防盗窗", "封窗户", "装铁栅栏", "广告牌挡窗"],
    },
    "谎报火警": {
        "kind": TYPE_VIOLATION,
        "verbs": ["谎报", "假报", "虚报"],
        "objects": ["火警", "火情"],
        "aliases": ["报假警", "假报警"],
    },
    "违规动火作业": {
        "kind": TYPE_VIOLATION,
        "verbs": ["动火", "电焊", "气焊", "明火作业", "冒险"],
        "objects": ["动火", "电焊", "气焊", "明火", "作业"],
        "aliases": ["违规动火", "无证电焊", "明火施工", "无证动火", "冒险作业"],
    },
    "非法携带危险品": {
        "kind": TYPE_VIOLATION,
        "verbs": ["携带"],
        "objects": ["易燃易爆危险品", "危险品"],
        "aliases": ["带危险品上车", "带危险品"],
    },
    "违法生产储存经营危险品": {
        "kind": TYPE_VIOLATION,
        "verbs": ["生产", "储存", "经营", "存放"],
        "objects": ["易燃易爆危险品", "危险品", "汽油", "液化气", "煤油"],
        "aliases": ["私存汽油", "违规存放汽油", "违规储存危险品"],
    },
    "不及时消除隐患": {
        "kind": TYPE_VIOLATION,
        "verbs": ["不及时", "拒不", "未"],
        "objects": ["消除隐患", "整改", "消除火灾隐患"],
        "aliases": ["拒不整改", "不整改", "不消除隐患"],
    },
    "违规停放充电电动自行车": {
        "kind": TYPE_VIOLATION,
        "verbs": ["停放", "充电", "放置"],
        "objects": ["电动自行车", "电动车"],
        "aliases": ["电动车进楼", "电动车上楼", "楼道充电", "飞线充电", "电瓶车充电"],
    },
    "改变防火防烟分区": {
        "kind": TYPE_VIOLATION,
        "verbs": ["改变", "变更", "拆改"],
        "objects": ["防火分区", "防烟分区"],
        "aliases": ["拆改防火分区", "改变防火分区"],
    },
    "使用易燃可燃装修材料": {
        "kind": TYPE_VIOLATION,
        "verbs": ["使用", "装修", "采用"],
        "objects": ["易燃材料", "可燃材料", "装修装饰材料"],
        "aliases": ["易燃装修", "可燃装修材料"],
    },
    "停用建筑消防设施": {
        "kind": TYPE_VIOLATION,
        "verbs": ["停用", "关闭"],
        "objects": ["建筑消防设施", "消防设施"],
        "aliases": ["关闭消防设施", "停用消防系统"],
    },
    "地下设置公共娱乐场所": {
        "kind": TYPE_VIOLATION,
        "verbs": ["设置在地下", "设在地下", "设于地下"],
        "objects": ["公共娱乐场所", "娱乐场所", "歌舞"],
        "aliases": ["地下娱乐场所", "地下室开娱乐", "地下设歌舞厅", "不得设置在地下"],
    },
    "营业期间违规施工动火": {
        "kind": TYPE_VIOLATION,
        "verbs": ["动火", "施工", "电焊"],
        "objects": ["营业期间", "营业时间"],
        "aliases": ["营业期间动火", "营业期间施工", "营业时电焊"],
    },
    "飞线充电": {
        "kind": TYPE_VIOLATION,
        "verbs": ["飞线", "私拉电线", "乱拉电线"],
        "objects": ["充电", "插座", "电动"],
        "aliases": ["飞线充电", "私拉电线充电", "乱拉电线充电"],
    },
    "电动车进入电梯楼道": {
        "kind": TYPE_VIOLATION,
        "verbs": ["推进", "带进", "抬入", "进入电梯"],
        "objects": ["电梯", "楼道", "门厅", "疏散走道"],
        "aliases": ["电动车进电梯", "电池进电梯", "电瓶车进楼道", "推进电梯"],
    },
    "未申报检查擅自开业": {
        "kind": TYPE_VIOLATION,
        "verbs": ["擅自开业", "擅自投入使用", "未申报"],
        "objects": ["开业", "使用", "营业"],
        "aliases": ["擅自开业", "未经验收投入使用", "未经消防检查开业"],
    },
    # ---- 管理义务 ----
    "开展防火检查": {
        "kind": TYPE_DUTY,
        "verbs": ["组织", "开展", "进行", "实施"],
        "objects": ["防火检查"],
        "aliases": ["做防火检查", "每月检查", "季度检查", "组织防火检查"],
    },
    "组织防火巡查": {
        "kind": TYPE_DUTY,
        "verbs": ["巡查", "组织巡查", "进行巡查"],
        "objects": ["防火巡查", "用火", "电气"],
        "aliases": ["做防火巡查", "日常巡查"],
    },
    "制定灭火应急疏散预案": {
        "kind": TYPE_DUTY,
        "verbs": ["制定", "编制", "组织制定"],
        "objects": ["应急预案", "应急疏散预案", "疏散预案", "灭火和应急疏散预案"],
        "aliases": ["编应急预案", "制定逃生预案", "灭火应急预案"],
    },
    "组织消防培训演练": {
        "kind": TYPE_DUTY,
        "verbs": ["组织", "开展", "实施", "进行"],
        "objects": ["消防演练", "演练", "培训", "宣传教育"],
        "aliases": ["组织演练", "消防演练", "消防培训", "消防知识教育", "着火了往哪跑"],
    },
    "维护保养消防设施": {
        "kind": TYPE_DUTY,
        "verbs": ["维护", "保养", "维护保养", "检测", "检验", "维修"],
        "objects": ["消防设施", "消防器材", "灭火器材", "消防安全标志"],
        "aliases": ["维保消防设施", "检测消防设施", "保养灭火器材"],
    },
    "保障通道出口畅通": {
        "kind": TYPE_DUTY,
        "verbs": ["保障", "保持", "确保", "保证"],
        "objects": ["疏散通道", "安全出口", "消防车通道"],
        "aliases": ["保持通道畅通", "保障出口畅通", "确保疏散畅通"],
    },
    "建立消防队": {
        "kind": TYPE_DUTY,
        "verbs": ["建立", "组建"],
        "objects": ["消防队", "义务消防队", "专职消防队", "志愿消防队"],
        "aliases": ["建义务消防队", "组建专职消防队", "志愿消防队"],
    },
    "设置微型消防站": {
        "kind": TYPE_DUTY,
        "verbs": ["设置", "建立", "组建"],
        "objects": ["微型消防站"],
        "aliases": ["建微型消防站", "设置微型消防站", "微型消防站人数"],
    },
    "配备志愿消防队员": {
        "kind": TYPE_DUTY,
        "verbs": ["配备", "配置", "不少于"],
        "objects": ["志愿消防队员", "志愿消防员"],
        "aliases": ["志愿消防队员", "配备志愿消防员"],
    },
    "建立消防档案": {
        "kind": TYPE_DUTY,
        "verbs": ["建立", "健全", "完善"],
        "objects": ["消防档案", "消防安全档案"],
        "aliases": ["建消防档案", "建立健全消防档案", "消防安全档案"],
    },
    "充电场所消防管理": {
        "kind": TYPE_DUTY,
        "verbs": ["设置", "管理", "配备"],
        "objects": ["充电场所", "停放场所", "充电设施", "停车库"],
        "aliases": ["充电场所管理", "停放充电场所", "集中充电"],
    },
    "用气用电安全管理": {
        "kind": TYPE_DUTY,
        "verbs": ["用气安全", "用电安全", "燃气安全"],
        "objects": ["用气", "燃气", "用电", "电气"],
        "aliases": ["用气安全", "超高层用气", "燃气安全管理", "用电安全管理"],
    },
    "及时消除火灾隐患": {
        "kind": TYPE_DUTY,
        "verbs": ["消除", "整改", "及时消除"],
        "objects": ["火灾隐患", "隐患"],
        "aliases": ["消除隐患", "整改火灾隐患", "及时整改隐患"],
    },
    "落实消防安全责任": {
        "kind": TYPE_DUTY,
        "verbs": ["落实", "履行"],
        "objects": ["消防安全责任", "消防安全职责", "责任制"],
        "aliases": ["落实责任制", "履行消防职责", "消防安全职责"],
    },
    "消防控制室值班": {
        "kind": TYPE_DUTY,
        "verbs": ["值班", "安排", "落实"],
        "objects": ["消防控制室", "消控室"],
        "aliases": ["消控室值班", "消防控制室无人值班", "24小时值班"],
    },
    "公示消防安全信息": {
        "kind": TYPE_DUTY,
        "verbs": ["公示", "公告", "公开"],
        "objects": ["消防安全", "消防技术服务信息", "联系方式"],
        "aliases": ["公示消防责任人", "公开消防信息"],
    },
    # ---- 政府职责 ----
    "建设维护公共消防设施": {
        "kind": TYPE_GOV,
        "verbs": ["建设", "维护", "规划"],
        "objects": ["消防站", "消防供水", "消防水源", "公共消防设施"],
        "aliases": ["建消防站", "维护消防水源", "规划消防设施"],
    },
    "开展隐患排查整治": {
        "kind": TYPE_GOV,
        "verbs": ["排查", "整治", "治理"],
        "objects": ["火灾隐患", "重大火灾隐患", "消防安全"],
        "aliases": ["隐患排查", "消防整治", "火灾隐患整治"],
    },
}

OBJECTS = {
    "消火栓": ["消火栓", "消防栓"],
    "疏散通道": ["疏散通道", "逃生通道", "疏散走道"],
    "安全出口": ["安全出口"],
    "消防车通道": ["消防车通道", "消防通道"],
    "灭火器": ["灭火器", "灭火器材"],
    "防火间距": ["防火间距"],
    "消防设施": ["消防设施", "灭火系统", "喷淋", "自动喷水灭火系统"],
    "消防安全标志": ["消防安全标志", "疏散指示", "安全标志"],
    "易燃易爆危险品": ["易燃易爆危险品", "危险品", "汽油", "煤油", "液化气"],
    "建筑消防设施": ["建筑消防设施"],
    "电动自行车": ["电动自行车", "电动车", "电瓶车"],
    "充电设施": ["充电设施", "充电装置", "充电器"],
    "停车库": ["停车库", "停放场所", "充电场所"],
    "消防控制室": ["消防控制室", "消控室"],
    "防火防烟分区": ["防火分区", "防烟分区"],
    "装修装饰材料": ["易燃材料", "可燃材料", "装修装饰材料"],
    "公共消防设施": ["消防站", "消防供水", "消防水源", "公共消防设施"],
    "公共娱乐场所": ["公共娱乐场所", "歌舞娱乐场所"],
    "避难层": ["避难层", "避难间"],
    "微型消防站": ["微型消防站"],
    "志愿消防队": ["志愿消防队", "志愿消防队员"],
    "高层建筑": ["高层建筑", "高层民用建筑", "高层住宅", "超高层"],
}

PENALTIES = {
    "责令改正": ["责令改正", "责令限期改正"],
    "罚款": ["罚款", "处五千元", "元以下罚款", "元以上", "并处罚款"],
    "责令停产停业": ["责令停产停业", "停产停业"],
    "临时查封": ["临时查封", "查封"],
    "警告": ["警告"],
    "强制拆除": ["强制执行", "强制铲除", "拆除"],
    "拘留": ["拘留", "行政拘留", "五日以上十日以下拘留"],
}

# 处罚意图 / 罚则条款关键词
PUNISH_KEYWORDS = ["处罚", "罚款", "拘留", "后果", "违法", "责任", "怎么办", "怎么罚", "处理", "惩罚"]
PENALTY_CLAUSE_WORDS = ["处罚", "罚款", "拘留", "责令改正", "责令停产停业", "警告", "追究刑事"]

REL_PROHIBIT, REL_DUTY, REL_RESP = "禁止", "义务", "责任"
REL_PUNISH, REL_INVOLVE = "处罚", "涉及"
RELATIONS = [REL_PROHIBIT, REL_DUTY, REL_RESP, REL_PUNISH, REL_INVOLVE]

_STOP = set("的了在是我也你他它一个就都得吗呢吧啊呀")


def _cut(text):
    if _HAS_JIEBA:
        return [w for w in jieba.cut(text) if w.strip() and w not in _STOP]
    return [text[i:i + 2] for i in range(max(0, len(text) - 1))]


def _alias_terms(aliases):
    """别名集合 → 核心词集合（2字片段 + 完整别名，保证匹配鲁棒）"""
    terms = set()
    for a in aliases:
        a = a.strip()
        if len(a) >= 2:
            for i in range(len(a) - 1):
                terms.add(a[i:i + 2])
    return terms


# ------------------------------------------------------------
# 语义断句
# ------------------------------------------------------------
def split_sentences(text):
    parts = re.split(r"（[一二三四五六七八九十]+）", text)
    sents = []
    for p in parts:
        for s in re.split(r"[。；;\n]", p):
            s = s.strip().strip("（）：。；;")
            if s:
                sents.append(s)
    return sents


class GraphBuilder:
    def __init__(self):
        self.nodes = {}
        self.edges = []
        self.chunks = []
        self._build()

    def _add_node(self, name, ntype, aliases=()):
        nid = f"{ntype}:{name}"
        if nid not in self.nodes:
            self.nodes[nid] = {"id": nid, "name": name, "type": ntype,
                               "aliases": list(aliases), "degree": 0}
        return nid

    def _add_edge(self, s, t, rel, art, reason="", law=""):
        self.edges.append({"source": s, "target": t, "relation": rel,
                           "article": art, "reason": reason, "law": law or art.split("·", 1)[0]})
        self.nodes[s]["degree"] += 1
        self.nodes[t]["degree"] += 1

    def _find_subject(self, sent, default=None):
        """最长优先匹配主体名/别名；找不到时不强行默认「单位」。"""
        hits = []
        for s in SUBJECTS:
            if s and s in sent:
                hits.append(s)
            for alias in SUBJECT_ALIASES.get(s, []):
                if alias and alias in sent:
                    hits.append(s)
        if hits:
            return max(hits, key=len)
        return default

    def _find_behavior(self, sent):
        """返回 (name, aliases, kind) 或 (None, None, None)。"""
        for b, spec in BEHAVIORS.items():
            if any(a in sent for a in spec["aliases"]):
                return b, spec["aliases"], spec["kind"]
            verb = any(v in sent for v in spec["verbs"])
            obj = any(o in sent for o in spec["objects"])
            if verb and obj:
                return b, spec["aliases"], spec["kind"]
        return None, None, None

    def _find_object(self, sent):
        out = []
        for o, aliases in OBJECTS.items():
            if any(a in sent for a in aliases):
                out.append((o, aliases))
        return out

    def _find_penalty(self, sent):
        out = []
        for p, aliases in PENALTIES.items():
            if any(a in sent for a in aliases):
                out.append(p)
        return out

    def _judge_relation(self, sent, kind):
        if any(k in sent for k in ["不得", "禁止", "严禁"]):
            return REL_PROHIBIT
        if kind == TYPE_VIOLATION:
            return REL_PROHIBIT if any(k in sent for k in ["不得", "禁止", "严禁", "违法"]) else REL_RESP
        if any(k in sent for k in ["应当", "必须", "责任", "义务", "应当履行"]):
            return REL_DUTY
        if kind == TYPE_GOV:
            return REL_DUTY
        if kind == TYPE_DUTY:
            return REL_DUTY
        return REL_RESP

    def _is_penalty_clause(self, text):
        return any(k in text for k in PENALTY_CLAUSE_WORDS)

    def _build(self):
        # 预注册词典实体：口语链接不依赖语料是否写过全称
        for s in SUBJECTS:
            self._add_node(s, "主体", SUBJECT_ALIASES.get(s, []))
        for b, spec in BEHAVIORS.items():
            self._add_node(b, spec["kind"], spec["aliases"])
        for o, oa in OBJECTS.items():
            self._add_node(o, "消防对象", oa)
        for p, pa in PENALTIES.items():
            self._add_node(p, "处罚", pa)

        article_behaviors = {}

        for path, law_abbr in ARCHIVE_PATHS:
            with open(path, encoding="utf-8") as f:
                law = json.load(f)
            if "law_abbr" in law:
                law_abbr = law["law_abbr"]
            law_name = law.get("law_name", law_abbr)
            source_url = law.get("source_url", "")
            authority = law.get("authority", "")
            effective_date = law.get("effective_date", "")
            for ch in law["chapters"]:
                for art in ch["articles"]:
                    orig_num, txt = art["num"], art["text"]
                    num = f"{law_abbr}·{orig_num}"
                    sents = split_sentences(txt)
                    for si, st in enumerate(sents):
                        self.chunks.append({"id": f"{num}#{si}", "article": num,
                                            "title": art["title"], "text": st,
                                            "chapter": ch["chapter_title"], "law": law_abbr,
                                            "law_name": law_name, "source_url": source_url,
                                            "authority": authority, "effective_date": effective_date})
                        subject = self._find_subject(st)
                        behavior, aliases, kind = self._find_behavior(st)
                        if not behavior:
                            continue
                        # 无明确主体时：仅义务/禁止句才回落「单位」，避免枢纽污染
                        if not subject and any(k in st for k in ["应当", "必须", "不得", "禁止", "严禁"]):
                            subject = "单位"
                        beh_n = self._add_node(behavior, kind, aliases)
                        if subject:
                            sub_n = self._add_node(subject, "主体", SUBJECT_ALIASES.get(subject, []))
                            rel = self._judge_relation(st, kind)
                            self._add_edge(sub_n, beh_n, rel, num, f"句[{si}]: {st[:28]}")
                        article_behaviors.setdefault(num, set()).add(beh_n)
                        for o, oa in self._find_object(st):
                            obj_n = self._add_node(o, "消防对象", oa)
                            self._add_edge(beh_n, obj_n, REL_INVOLVE, num, f"涉及: {st[:24]}")
                        for p in self._find_penalty(st):
                            pn = self._add_node(p, "处罚", [])
                            self._add_edge(beh_n, pn, REL_PUNISH, num, f"罚则: {st[:24]}")

        # 罚则条款传播：有具体处罚才连边；不再用「依法处罚」万能兜底
        for path, law_abbr in ARCHIVE_PATHS:
            with open(path, encoding="utf-8") as f:
                law = json.load(f)
            if "law_abbr" in law:
                law_abbr = law["law_abbr"]
            for ch in law["chapters"]:
                for art in ch["articles"]:
                    orig_num, txt = art["num"], art["text"]
                    num = f"{law_abbr}·{orig_num}"
                    behs = article_behaviors.get(num)
                    if not behs:
                        continue
                    penalties = self._find_penalty(txt)
                    if not penalties:
                        continue
                    for beh_n in behs:
                        for p in penalties:
                            pn = self._add_node(p, "处罚", [])
                            self._add_edge(beh_n, pn, REL_PUNISH, num, f"罚则条款传播: {txt[:26]}")

        self._dedupe_edges()
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=2)
        with open(GRAPH_PATH, "w", encoding="utf-8") as f:
            json.dump({"nodes": list(self.nodes.values()), "edges": self.edges},
                      f, ensure_ascii=False, indent=2)

    def _dedupe_edges(self):
        seen, out = set(), []
        for e in self.edges:
            k = (e["source"], e["target"], e["relation"], e["article"])
            if k not in seen:
                seen.add(k); out.append(e)
        self.edges = out

    def stats(self):
        types = {}
        for n in self.nodes.values():
            types[n["type"]] = types.get(n["type"], 0) + 1
        return {"semantic_sents": len(self.chunks), "entities": len(self.nodes),
                "edges": len(self.edges), "articles": sorted({c["article"] for c in self.chunks}),
                "type_dist": types}


# ============================================================
# 二、图谱检索
# ============================================================
class GraphRetriever:
    def __init__(self, graph):
        self.nodes = {n["id"]: n for n in graph["nodes"]}
        self.edges = graph["edges"]
        self.adj = {}
        for e in self.edges:
            self.adj.setdefault(e["source"], []).append(e)
            self.adj.setdefault(e["target"], []).append(e)
        self.node_articles = {}
        for e in self.edges:
            for nid in (e["source"], e["target"]):
                self.node_articles.setdefault(nid, set()).add(e["article"])

    # ---- 实体链接：返回 [(nid, strength, ntype)] ----
    def link(self, q):
        hits = []
        aliases_by_id = {nid: self.nodes[nid]["aliases"] for nid in self.nodes}
        for nid, n in self.nodes.items():
            name, ntype = n["name"], n["type"]
            aliases = n.get("aliases", [])
            strength = None
            if name and name in q:
                strength = 1.0
            else:
                for a in aliases:
                    if a and a in q:
                        strength = 0.95
                        break
            if strength is None and _is_behavior_type(ntype):
                if self._fuzzy(q, aliases):
                    strength = 0.7
            if strength:
                hits.append((nid, strength, ntype))
        return hits

    def _fuzzy(self, q, aliases):
        terms = _alias_terms(aliases)
        m = [t for t in terms if t in q]
        return len(set(m)) >= 2

    def intent_weight(self, q):
        return 1.6 if any(k in q for k in PUNISH_KEYWORDS) else 1.0

    def search(self, q, top_k=5):
        seeds = self.link(q)
        pun = any(k in q for k in PUNISH_KEYWORDS)
        iw = 1.6 if pun else 1.0
        has_behavior_seed = any(_is_behavior_type(ntype) for _, _, ntype in seeds)

        art_score, art_seed, extra_path = {}, {}, {}

        # 第一优先级：seed节点直连条款
        for nid, strength, ntype in seeds:
            # 主体节点是枢纽，仅在也命中行为时采信，避免污染
            if ntype == "主体":
                if has_behavior_seed:
                    continue
                w = strength * 0.45
                for atcl in self.node_articles.get(nid, []):
                    art_score[atcl] = art_score.get(atcl, 0) + w
                    art_seed[atcl] = self.nodes[nid]["name"]
                continue
            w = strength * (iw if (_is_behavior_type(ntype) and pun) else 1.0)
            for atcl in self.node_articles.get(nid, []):
                art_score[atcl] = art_score.get(atcl, 0) + w
                art_seed[atcl] = self.nodes[nid]["name"]

        # 多跳：行为 seed 沿「处罚」边直达罚则条款
        for nid, strength, ntype in seeds:
            if _is_behavior_type(ntype):
                for e in self.adj.get(nid, []):
                    if e["relation"] == REL_PUNISH:
                        atcl = e["article"]
                        art_score[atcl] = art_score.get(atcl, 0) + strength * 0.8 * iw
                        extra_path[atcl] = (self.nodes[nid]["name"], self.nodes[e["target"]]["name"])

        if not art_score:
            return []
        mx = max(art_score.values()) or 1.0
        results = []
        for atcl, s in art_score.items():
            path = " → ".join(extra_path[atcl]) if atcl in extra_path else art_seed.get(atcl, "")
            results.append((atcl, s / mx, path))
        # 分数相同时按条款号稳定排序，保证评测与线上结果可复现。
        results.sort(key=lambda x: (-x[1], x[0]))
        return results[:top_k]


# ============================================================
# 构建入口
# ============================================================
def build_if_missing(force=False):
    if os.path.exists(GRAPH_PATH) and not force:
        return load_graph()
    print("[GraphRAG] 构建知识图谱...")
    g = GraphBuilder()
    graph = {"nodes": list(g.nodes.values()), "edges": g.edges}
    print(f"[GraphRAG] 完成：{len(graph['nodes'])}实体 / {len(graph['edges'])}边 / {len(g.chunks)}语义句")
    return graph


def load_graph():
    with open(GRAPH_PATH, encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    g = GraphBuilder()
    print(json.dumps(g.stats(), ensure_ascii=False, indent=2))
    r = GraphRetriever({"nodes": list(g.nodes.values()), "edges": g.edges})
    for q in ["楼道堆放杂物违反什么规定", "占用消防通道怎么处罚", "谎报火警有什么后果",
              "消火栓被埋了怎么办", "占用疏散通道怎么处罚"]:
        print(f"\nQ: {q}")
        for a, s, path in r.search(q):
            print(f"  {a}  score={s:.2f}  path={path}")
