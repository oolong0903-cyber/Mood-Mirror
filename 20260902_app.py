# -*- coding: utf-8 -*-
"""
中文情绪文本分析工具（Streamlit 单文件版）
================================================
功能：
  1) 情绪识别：喜悦 / 悲伤 / 愤怒 / 焦虑 / 平静 / 中性，
     并输出一条按强度占比分配宽度的三色构成条（消极 / 中性 / 积极），
     直接呈现整体情绪构成。不输出任何数值型效价/唤醒度分值（基准点未经标定）。
  2) LIWC 风格语言特征：情感词占比、认知加工词（我想/我觉得/我认为）、
     否定词（不/没/无…）、绝对化词（总是/从不/永远/一定）、第一人称「我」频次。
  3) 认知扭曲识别（CBT 启发，启发式匹配）：灾难化 / 非黑即白 / 过度概括 /
     读心术 / 应该陈述 / 情绪化推理 —— 每命中一类给一句温和、非评判的提示。
  4) 一段温和的文字解读（规则模板生成，非评判）。
  5) 页面底部附心理健康免责声明。

方法：
  * 纯规则词典匹配，不调用任何外部 API / 模型；
  * 内置精简中文情绪词典（仿 DUTIR 大连理工情感词汇本体结构），可整体替换；
  * 可选择加载同目录下外部词典文件（支持 DUTIR 风格），详见 load_lexicon_csv。

运行：  streamlit run 20260902_app.py（或双击同目录 启动.bat）
词典结构说明见文件内「词典区」注释与 DUTIR_CAT_MAP。
"""

import os
import re
from collections import Counter

import pandas as pd
import streamlit as st

# =====================================================================
# ══ 词典区 ①：情绪词词典（可直接整段替换成大连理工 DUTIR 的导出内容）══
# ---------------------------------------------------------------------
# 结构：每条 = (词语, 强度 1~5)。强度对标 DUTIR 的 1弱…5强（DUTIR 为 1,3,5,7,9，
# 导入时自动折算）。情绪词只在它真正表达情绪时计数，长词优先匹配。
# =====================================================================

EMOTION_WORDS = {
    "喜悦": [
        ("开心", 4), ("高兴", 4), ("快乐", 5), ("愉快", 4), ("喜悦", 5),
        ("欣喜", 5), ("欢喜", 4), ("兴奋", 4), ("雀跃", 3), ("满足", 3),
        ("幸福", 5), ("满意", 3), ("甜蜜", 3), ("舒心", 3), ("欢乐", 4),
        ("痛快", 4), ("轻松", 3), ("舒服", 3), ("舒坦", 3), ("美滋滋", 4),
        ("高兴坏了", 5), ("开心极了", 5), ("哈哈大笑", 5), ("眉开眼笑", 4),
        ("笑容", 3), ("笑眯眯", 3), ("欢呼", 4), ("庆幸", 3), ("自豪", 3),
        ("欣慰", 3), ("如愿", 4), ("喜出望外", 5), ("得意", 3), ("雀跃不已", 4),
        ("激动", 4), ("痛快淋漓", 4), ("开怀", 4), ("忍不住笑", 3),
        ("愉悦", 4), ("惊喜", 4), ("春风得意", 4),
        # —— 补充：日常口语与网络表达中的积极词（线上无 dutir.csv 时的主要来源）——
        ("爱", 3), ("喜欢", 3), ("喜爱", 3), ("热爱", 4), ("超喜欢", 4),
        ("好开心", 5), ("好高兴", 5), ("挺开心", 4), ("蛮开心", 4),
        ("开心多了", 4), ("心情好", 4), ("心情不错", 4), ("感觉不错", 3),
        ("好起来了", 4), ("缓过来了", 4), ("轻松多了", 4), ("踏实了", 3),
        ("有希望", 3), ("期待", 3), ("充满希望", 4), ("明天会更好", 4),
        ("被理解", 3), ("被支持", 3), ("有依靠", 3), ("安心多了", 4),
        ("值得", 3), ("有意义", 3), ("有收获", 3), ("成功了", 4),
        ("通过", 3), ("进步", 3), ("好起来", 4), ("笑出来", 3),
    ],
    "悲伤": [
        ("伤心", 4), ("难过", 4), ("悲伤", 5), ("悲哀", 5), ("悲痛", 5),
        ("哀伤", 5), ("伤感", 4), ("心碎", 5), ("痛苦", 5), ("苦闷", 4),
        ("郁闷", 4), ("沮丧", 4), ("低落", 3), ("消沉", 4), ("灰心", 3),
        ("绝望", 5), ("哭", 4), ("哭泣", 5), ("流泪", 4), ("眼泪", 4),
        ("心酸", 4), ("酸楚", 4), ("委屈", 3), ("无奈", 3), ("失落", 4),
        ("空虚", 3), ("孤单", 3), ("孤独", 4), ("寂寞", 3), ("想哭", 4),
        ("欲哭无泪", 5), ("沮丧透了", 5), ("无助", 4), ("悲从中来", 5),
        ("情绪低落", 4), ("郁郁寡欢", 5), ("消沉下去", 4), ("哀愁", 4),
        ("难过极了", 5), ("百感交集", 3),
        # —— 补充：扭曲表达、失败/无望类高频词 ——
        ("完蛋", 5), ("失败", 4), ("没用", 3), ("笑话", 3), ("白费", 4),
        ("浪费", 3), ("白忙", 4), ("一事无成", 5), ("无可救药", 5),
        ("没意义", 3), ("没意思", 3), ("提不起劲", 4), ("没劲", 3),
        ("撑不住", 4), ("受不了", 4), ("熬不下去", 5), ("活不下去", 5),
        ("想死", 5), ("生无可恋", 5), ("看不到希望", 5), ("没盼头", 5),
        ("糟糕", 4), ("糟心", 4), ("倒霉", 3), ("惨", 4), ("凄惨", 4),
        ("凄惨可怜", 4), ("可怜", 3), ("委屈", 3), ("憋屈", 3), ("心酸", 4),
        ("心累", 4), ("疲惫", 3), ("麻木", 4), ("无望", 5), ("绝望透了", 5),
        ("崩溃", 5), ("溃不成军", 5), ("一塌糊涂", 5), ("乱成一团", 4),
        ("痛不欲生", 5), ("生不如死", 5), ("万念俱灰", 5), ("心死", 5),
        ("孤零零", 3), ("一个人扛", 3), ("没人懂", 4), ("没人理", 4),
        ("被抛下", 4), ("被抛弃", 4), ("无依无靠", 4), ("孤独无助", 5),
        ("难过死了", 5), ("伤心死了", 5), ("糟糕透了", 5), ("惨透了", 5),
        # —— 极端消极表达：只收录「不会误伤」的组合式 ——
        # ⚠️ 单字「死」不收录：「美死了 / 笑死了 / 高兴死了 / 帅死了」在中文里是
        #    褒义程度副词（≈「很」），收进来会把这些句子全判成消极。
        #    「我要死了 / 我想死」这类由下方「想死 / 不想活 / 活不下去」覆盖。
        ("想死", 5), ("不想活", 5), ("活不下去", 5), ("结束生命", 5),
        ("了结自己", 5), ("自杀", 5), ("轻生", 5), ("自尽", 5),
        ("跳楼", 5), ("没命", 5), ("活该", 3), ("不得好死", 5),
    ],
    "愤怒": [
        ("生气", 4), ("愤怒", 5), ("气愤", 5), ("恼怒", 5), ("恼火", 4),
        ("发火", 3), ("火大", 4), ("震怒", 5), ("暴怒", 5), ("愤恨", 5),
        ("怨恨", 4), ("恨", 4), ("敌意", 4), ("抓狂", 4), ("气炸了", 5),
        ("气死", 5), ("怒火", 5), ("火冒三丈", 5), ("发飙", 4), ("炸了", 4),
        ("怒不可遏", 5), ("咬牙切齿", 4), ("满腔怒火", 5), ("恨得牙痒痒", 5),
        ("不满", 3), ("烦躁", 3), ("不耐烦", 3), ("憋屈", 3), ("恼人", 3),
        ("讨厌", 3), ("烦透了", 4), ("大发雷霆", 5), ("怒发冲冠", 5),
        ("暴跳如雷", 5), ("气冲冲", 4), ("恼羞成怒", 4), ("无名火", 3),
        # —— 补充：愤怒 / 敌意 / 抱怨类高频词 ——
        ("气", 2), ("气死", 5), ("气炸", 5), ("可气", 4), ("来气", 3),
        ("可恶", 4), ("可恨", 4), ("讨厌", 3), ("烦", 2), ("烦死", 4),
        ("烦人", 3), ("膈应", 3), ("不爽", 3), ("不痛快", 3), ("窝火", 3),
        ("憋气", 3), ("窝囊", 3), ("委屈愤怒", 4), ("忍不了", 4),
        ("忍无可忍", 5), ("爆发", 4), ("炸了", 4), ("火大", 4), ("翻脸", 4),
        ("受不了了", 4), ("凭什么", 3), ("不公平", 3), ("没道理", 3),
        ("讨人厌", 4), ("招人烦", 4), ("恨死", 5), ("怨", 3), ("埋怨", 3),
        ("恶心", 4), ("烦死人了", 5), ("气死人了", 5),
    ],
    "焦虑": [
        ("焦虑", 5), ("担心", 4), ("担忧", 4), ("害怕", 4), ("恐惧", 5),
        ("恐慌", 5), ("紧张", 4), ("不安", 4), ("心神不宁", 5), ("心慌", 4),
        ("慌乱", 4), ("忐忑", 4), ("忧虑", 4), ("忧心", 4), ("烦恼", 3),
        ("发愁", 3), ("愁", 3), ("愁眉苦脸", 4), ("焦躁", 5), ("急躁", 3),
        ("慌", 4), ("慌张", 4), ("惶恐", 5), ("不知所措", 4), ("坐立不安", 4),
        ("揪心", 4), ("战战兢兢", 4), ("惴惴不安", 5), ("提心吊胆", 4),
        ("如坐针毡", 5), ("心乱如麻", 5), ("忧心忡忡", 5), ("胆怯", 4),
        ("焦虑不安", 5), ("愁绪", 3), ("患得患失", 4), ("寝食难安", 5),
        ("放不下心", 4),
        # —— 补充：焦虑 / 担忧 / 压力类高频词 ——
        ("担心死了", 5), ("急", 3), ("着急", 4), ("很急", 4), ("急死", 4),
        ("压力", 3), ("压力大", 4), ("喘不过气", 5), ("窒息", 5),
        ("扛不住", 4), ("受不了", 4), ("没底", 3), ("心里没底", 4),
        ("七上八下", 4), ("心里没谱", 4), ("患得患失", 4), ("纠结", 3),
        ("矛盾", 3), ("犹豫", 3), ("难办", 3), ("麻烦", 2), ("费劲", 3),
        ("操心", 3), ("担心这担心那", 4), ("睡不着", 4), ("失眠", 4),
        ("心慌意乱", 5), ("忐忑不安", 5), ("惶恐", 5), ("害怕极了", 5),
        ("怕", 2), ("怕得要死", 5), ("吓", 3), ("吓人", 3), ("提心吊胆", 4),
        ("焦虑不安", 4), ("心乱", 3), ("坐立难安", 5), ("煎熬", 4),
    ],
    "平静": [
        ("平静", 4), ("淡定", 4), ("坦然", 3), ("平和", 4), ("安宁", 3),
        ("安详", 3), ("安稳", 2), ("冷静", 4), ("镇定", 3), ("从容", 3),
        ("舒缓", 2), ("安心", 4), ("踏实", 3), ("心安", 3), ("放松", 4),
        ("松弛", 2), ("悠然", 3), ("泰然", 3), ("释然", 3), ("平静下来", 4),
        ("不慌不忙", 3), ("心如止水", 4), ("怡然自得", 4), ("沉静", 3),
        ("松弛下来", 3), ("安之若素", 4),
        # —— 补充：平静 / 放松 / 恢复类词 ——
        ("还好", 2), ("还行", 2), ("没事", 2), ("挺好", 3), ("还不错", 3),
        ("安心", 3), ("放心", 3), ("静下心", 3), ("沉下来", 3),
        ("缓了缓", 3), ("喘口气", 3), ("松了口气", 4), ("松口气", 3),
        ("没那么糟", 3), ("还行吧", 3), ("正常", 2), ("平常心", 3),
        ("无所谓", 2), ("随便", 1), ("无所谓了", 3), ("看开了", 4),
        ("释怀", 4), ("放下", 3), ("想通", 4), ("过去了", 3),
    ],
}

# 六个可输出的情绪类别（作为合法类别的白名单，供词典加载时校验）。
# 历史上此处存放过各情绪在环状模型上的效价/唤醒度基准点，但那几个常数未经
# 量表标定、强度权重又会被加权平均约掉，故已不再用于计算任何对外指标。
# 保留本字典仅为校验外部词典的类别码是否合法。
EMOTION_BASE = {
    "喜悦": (0.62, 0.50),
    "平静": (0.28, 0.12),
    "悲伤": (-0.58, 0.30),
    "愤怒": (-0.60, 0.85),
    "焦虑": (-0.42, 0.70),
    "中性": (0.00, 0.15),
}

# 误收词剔除表：某些词在 DUTIR 里被标为情绪，但在日常句子里更多是情态副词 /
# 程度词 / 转折成分等中性用法（“肯定/一定/不可/成功”），会凭空拉高某类情绪。
# 无论内置还是外部 DUTIR 词典，构建查找表时都剔除它们（可按个人语料增删）。
EXCLUDE_EMOTION_WORDS = {"肯定", "一定", "不可", "成功", "其实", "下子"}


# 由词典结构生成：word -> (类别, 强度)
def _build_emotion_lookup():
    lookup = {}
    for cat, pairs in EMOTION_WORDS.items():
        for word, intensity in pairs:
            if word in EXCLUDE_EMOTION_WORDS:
                continue
            if word not in lookup:               # 跨类别同词时保留先出现的类别
                lookup[word] = (cat, max(1, min(5, int(intensity))))
    return lookup

EMOTION_LOOKUP = _build_emotion_lookup()

# =====================================================================
# ══ 词典区 ②：LIWC 风格特征词典（认知 / 否定 / 绝对化 / 第一人称）══
# =====================================================================

COG_WORDS = [   # 认知加工词（要求含 我想/我觉得/我认为 等自我参照）
    "我想", "我觉得", "我认为", "我感觉", "在我看来", "我想的是",
    "认为", "以为", "觉得", "感觉", "思考", "考虑", "琢磨", "寻思",
    "反省", "反思", "分析", "意识到", "明白", "理解", "发现",
    "想到", "想起", "记得", "猜测", "怀疑", "坚信", "相信",
    "打算", "计划", "决定", "选择", "推测", "总结",
]

# 否定词：多字词优先，避免“不会/没有”里的“不/没”被重复计数。
# 只保留高区分度的单字否定 不/没（不收录 无/未/非/别 —— “无论/未来/非常/别人”
# 等常用词里出现会大面积误报）。
NEG_WORDS = [
    "没有", "没什么", "毫无", "毫不", "绝非", "并非", "从未", "不再",
    "不是", "不要", "不用", "无法", "不会", "不能", "不可以", "没有过",
    "永不",
    "不", "没",
]

# 绝对化词（绝对化表达 / absolutist words）
ABS_WORDS = [
    "总是", "从不", "永远", "一定", "肯定", "绝对", "必须", "只能",
    "老是", "一直", "彻底", "完全", "全都", "全部", "所有", "每个",
    "每次", "没有人", "再也不", "再也没有", "永远都", "什么都不",
    "一点都", "根本", "简直",
]

FIRST_PERSON = {   # 第一人称用法分组（仅“我”相关，用于频次统计）
    "单数我": ["我"],                 # 独立单用“我”
    "我/我的": ["我的"],              # 所属形式
    "我们类": ["我们", "咱们", "咱"], # 复数 / 口语自称
    "自己类": ["自己", "本人"],       # 反身自称
}

# =====================================================================
# ══ 词典区 ③：认知扭曲启发式关键词（CBT，在原文上做子串匹配，不参与分词）══
# =====================================================================
# 只做“有无命中”提示，不追求召回率；按个人语料扩充/删减都很方便。
# 每个扭曲：patterns 为命中关键词/句式；tips 里的 {kw} 会被替换成第一个命中词。
# =====================================================================

CBT_PATTERNS = {
    "灾难化": {
        "patterns": [
            "完蛋了", "糟透了", "糟糕透顶", "彻底完了", "全完了", "一切都完了",
            "世界末日", "天塌", "没救了", "不堪设想", "灾难性的", "彻底失败",
            "肯定会失败", "一定会失败", "肯定完不成", "一定完不成", "什么都完了",
        ],
        "tip": "听起来有点把最坏的情形当成了必然（如「{kw}」）。"
               "把「万一……怎么办」写下来，通常会发现结果不止一种可能。",
    },
    "非黑即白": {
        "patterns": [
            "要么", "非黑即白", "全有或全无", "不是成功就是", "二选一",
            "只许成功", "不允许失败", "不能失败", "必须完美",
        ],
        "tip": "像在用「要么全有、要么全无」的标尺衡量（如「{kw}」）。"
               "生活大多落在中间地带，允许「做到一部分」也算数。",
    },
    "过度概括": {
        "patterns": [
            "总是", "从不", "从来都不", "从来没", "永远都", "每次都这样",
            "每次都是", "每次都", "所有人", "每个人", "从来没有人",
            "老是", "动不动就", "一直这样", "大家都不", "个个都",
            "什么都是", "没有任何人", "没有一个", "人人都", "没人愿意",
        ],
        "tip": "从一次或几次不顺利，推到了「总是 / 从不」（如「{kw}」）。"
               "这一次不等于每一次，这件事不等于所有事；试着换成「有时候」。",
    },
    "读心术": {
        "patterns": [
            "他肯定觉得", "她肯定觉得", "他一定觉得", "她一定觉得",
            "他们肯定觉得", "他们一定觉得", "别人肯定觉得", "别人一定觉得",
            "大家肯定觉得", "大家一定觉得", "他肯定认为", "她肯定认为",
            "他一定认为", "她一定认为", "他肯定在想", "她肯定在想",
            "他一定知道", "她一定知道", "他肯定很讨厌", "她肯定很讨厌",
            "他一定讨厌", "她一定讨厌", "他肯定看不起", "她肯定看不起",
            "他们肯定在", "他一定以为", "她一定以为", "他肯定很失望", "她肯定很失望",
        ],
        "tip": "似乎在替别人下结论（如「{kw}」）。他人的想法其实很难猜准，"
               "有疑问时直接去求证，会比一个人猜测更轻松。",
    },
    "应该陈述": {
        "patterns": ["应该", "应当", "理应", "本应该", "不该", "必须", "非得", "一定要"],
        "tip": "对自己或别人用了不少「应该 / 必须」式的指令（如「{kw}」）。"
               "试着把要求换成愿望——把「我应该」改成「我希望 / 我可以」，压力常常会小一些。",
    },
    "情绪化推理": {
        "patterns": [
            "我觉得自己很", "我觉得自己是", "我感觉自己很", "总觉得自己很",
            "觉得自己很", "感觉自己很", "觉得自己是", "感觉自己什么都",
            "觉得自己什么都不", "觉得大家都", "感觉大家都", "我有种感觉",
            "我有一种感觉", "我就是觉得", "总感觉", "感觉一切都不",
            "感觉什么都不", "我觉得我就是", "觉得自己就是",
        ],
        "tip": "把「我感觉……」直接当成了事实（如「{kw}」）。感觉是真实的体验，"
               "却不等于事情的真相；试着把「我感觉」和「实际情况」分开来看。",
    },
}

# 各扭曲主题色（柔和系）
DISTORTION_COLORS = {
    "灾难化": "#E0957B", "非黑即白": "#9F93C8", "过度概括": "#D6AE62",
    "读心术": "#7FA9B8", "应该陈述": "#8FA3B4", "情绪化推理": "#D694A6",
}


def detect_cognitive_distortions(text: str) -> list:
    """对原文做子串匹配，返回命中的认知扭曲列表（含命中的关键词样例）。"""
    found = []
    for name, spec in CBT_PATTERNS.items():
        matched = [p for p in spec["patterns"] if p in text]
        if matched:
            found.append({
                "name": name,
                "patterns": matched,          # 命中的关键词（按词典顺序）
                "example": matched[0],        # 第一个命中词，用于提示语
            })
    return found


def cbt_hint(d: dict) -> str:
    """用第一个命中词填入对应扭曲的温和提示语。"""
    return CBT_PATTERNS[d["name"]]["tip"].format(kw=d["example"])

# =====================================================================
# 基础工具：字符级最长匹配分词（纯词典、无需安装 jieba）
# =====================================================================

def _is_cjk(ch: str) -> bool:
    return "一" <= ch <= "鿿" or "㐀" <= ch <= "䶿"


def _split_ascii_run(text: str, start: int):
    """从 start 起截断连续的非中文片段（整段作为一词或空白丢弃）。"""
    j = start
    while j < len(text) and not _is_cjk(text[j]):
        j += 1
    seg = text[start:j].strip()
    return seg, j


MAX_WORD_LEN = 1
ALL_KEYWORDS = set()


def _rebuild_lexicon_index():
    """按当前 EMOTION_LOOKUP 重建分词用的最长词长与词典集合。

    模块顶部会先调用一次（内置词典）；若启动时用外部 DUTIR 词典覆盖了
    EMOTION_LOOKUP，也必须再次调用，否则新词不会参与分词。
    """
    global MAX_WORD_LEN, ALL_KEYWORDS
    ml = 1
    for w in EMOTION_LOOKUP:
        ml = max(ml, len(w))
    for grp in (COG_WORDS, NEG_WORDS, ABS_WORDS):
        for w in grp:
            ml = max(ml, len(w))
    for ws in FIRST_PERSON.values():
        for w in ws:
            ml = max(ml, len(w))
    MAX_WORD_LEN = ml
    ALL_KEYWORDS = (set(EMOTION_LOOKUP)
                    | set(COG_WORDS) | set(NEG_WORDS) | set(ABS_WORDS)
                    | set(w for ws in FIRST_PERSON.values() for w in ws))


_rebuild_lexicon_index()


def tokenize(text: str) -> list:
    """最长匹配分词。词典内词成词；中文单字逐个成词；英文/数字/标点整段成词。

    这样得到的“词数”只用于计算占比的近似分母，结果只做相对参照。
    """
    text = text.replace("　", " ").replace("\xa0", " ")
    n = len(text)
    i = 0
    toks = []
    while i < n:
        ch = text[i]
        if not _is_cjk(ch):
            seg, j = _split_ascii_run(text, i)
            if seg:
                toks.append(seg)
            i = j
            continue
        matched = None
        for L in range(min(MAX_WORD_LEN, n - i), 0, -1):
            if text[i:i + L] in ALL_KEYWORDS:
                matched = text[i:i + L]
                break
        if matched:
            toks.append(matched)
            i += len(matched)
        else:
            toks.append(ch)
            i += 1
    return toks


# =====================================================================
# LIWC 风格特征统计
# =====================================================================

def _feature_hits(text: str, patterns) -> Counter:
    """统计某类模式词的出现次数（长词优先，避免子串重复计数）。"""
    pats = sorted(set(patterns), key=len, reverse=True)
    hits = Counter()
    i, n = 0, len(text)
    while i < n:
        hit = None
        for p in pats:
            if p and text.startswith(p, i):
                hit = p
                break
        if hit:
            hits[hit] += 1
            i += len(hit)
        else:
            i += 1
    return hits


def _word_parts(token: str):
    """按“我”所在位置，把含“我”的分词结果归类到第一人称分组。"""
    for grp, ws in FIRST_PERSON.items():
        if token in ws:
            return grp
    if "我" in token:
        for grp, ws in FIRST_PERSON.items():
            if any(w in token for w in ws):
                return grp
    return None


# =====================================================================
# 核心分析入口：输入文本 → 结构化结果
# =====================================================================

def analyze(text: str) -> dict:
    """返回一条完整分析结果（dict），纯规则、可离线运行。"""
    text = (text or "").strip()
    toks = tokenize(text)
    total_tokens = len(toks)
    cjk_chars = sum(1 for ch in text if _is_cjk(ch))

    # ---- 1) 情绪：按分词结果统计各情绪词 ---- #
    # 否定域处理：若情绪词前面一小段（本子句内，最多回看 5 词）出现过否定词，
    # 说明它并非在表达该情绪，不计入。例如“没考好 / 不可能成功”里的“好、成功”。
    _NEG_MARKS = {"不", "没", "无", "别", "未", "非",
                  "不是", "没有", "不会", "不能", "不再", "从未", "不要",
                  "并非", "毫不", "没能", "没法", "无法"}
    _CONTRAST = {"但", "但是", "不过", "然而", "可", "却", "虽然", "尽管"}

    def _is_clause_boundary(tok):
        """标点或转折词视为子句边界：否定域不跨过它。"""
        if tok in _CONTRAST:
            return True
        return bool(tok) and not any(ch.isalnum() or _is_cjk(ch) for ch in tok)

    cat_counts = Counter()
    cat_intensity = Counter()
    matched_emotion_words = []          # (词, 类别, 强度)
    recent = []                          # 本子句内最近若干词，用于否定域判断
    for t in toks:
        if _is_clause_boundary(t):
            recent.clear()
        else:
            if t in EMOTION_LOOKUP:
                negated = any(m in _NEG_MARKS for m in recent[-5:])
                if not negated:
                    cat, inten = EMOTION_LOOKUP[t]
                    cat_counts[cat] += 1
                    cat_intensity[cat] += inten
                    matched_emotion_words.append((t, cat, inten))
            recent.append(t)
            if len(recent) > 8:
                recent.pop(0)

    emotion_token_n = sum(cat_counts.values())

    # 注：此处曾计算 valence / arousal（效价 / 唤醒度），已移除。
    # 原因：那六个基准点是手工常数、未经量表标定，且情绪强度在加权平均中会被
    # 约掉（单情绪下分子分母同时含强度，"有点难过"与"我要死了"输出相同）。
    # 该结果缺乏实证依据，不适合作为对外呈现的测量指标。
    # 情绪的「构成比例」改由 cat_intensity 经三色构成条呈现（见 composition_mix），
    # 刻意不输出任何数值型效价/唤醒度分值。

    # ---- 主导情绪 ---- #
    if emotion_token_n == 0:
        dominant = "中性"
        dominant_words = []
        rich = False
    else:
        dominant = max(cat_intensity, key=lambda c: (cat_intensity[c], cat_counts[c]))
        dominant_words = sorted(
            [(w, i) for (w, c, i) in matched_emotion_words if c == dominant],
            key=lambda x: -x[1],
        )[:6]
        # 情绪线索是否“稀疏”：情感词占词典词比例过低 → 提示结果弱
        rich = (emotion_token_n / max(1, total_tokens)) >= 0.02 or emotion_token_n >= 3

    # ---- 2) LIWC 风格特征 ---- #
    cog_hits = _feature_hits(text, COG_WORDS)
    neg_hits = _feature_hits(text, NEG_WORDS)
    abs_hits = _feature_hits(text, ABS_WORDS)

    first_person = Counter()
    for t in toks:
        g = _word_parts(t)
        if g:
            first_person[g] += 1
    wo_count = sum(first_person.values())
    wo_single = first_person["单数我"]

    def per_1000(n):
        return round(n / max(1, cjk_chars) * 1000, 2)

    # ---- 句数 / 平均句长（用句末标点粗估） ---- #
    sentences = [s for s in re.split(r"[。！？!?…；;]+\n*", text) if s.strip()]
    sent_n = len(sentences)

    # ---- 3) 认知扭曲（CBT 启发，附加在结果里，供渲染用）---- #
    distortions = detect_cognitive_distortions(text)

    return {
        "text": text,
        "total_chars": len(text),
        "cjk_chars": cjk_chars,
        "total_tokens": total_tokens,
        "sentence_n": sent_n,
        # 情绪
        "cat_counts": dict(cat_counts),
        "cat_intensity": dict(cat_intensity),
        "dominant": dominant,
        "dominant_words": dominant_words,
        "emotion_rich": rich,
        # 语言特征（计数 + 每千字率 + 占比）
        "emotion_token_n": emotion_token_n,
        "emotion_ratio_pct": round(emotion_token_n / max(1, total_tokens) * 100, 2),
        "cog": dict(cog_hits), "cog_n": sum(cog_hits.values()), "cog_ex": list(cog_hits),
        "neg": dict(neg_hits), "neg_n": sum(neg_hits.values()),
        "abs": dict(abs_hits), "abs_n": sum(abs_hits.values()),
        "wo_n": wo_count, "wo_single": wo_single, "wo_groups": dict(first_person),
        "rates": {
            "cog": per_1000(sum(cog_hits.values())),
            "neg": per_1000(sum(neg_hits.values())),
            "abs": per_1000(sum(abs_hits.values())),
            "wo": per_1000(wo_count),
        },
        # 认知扭曲
        "distortions": distortions,
    }


def valence_label(v: float) -> str:
    """已废弃：保留仅为向后兼容，界面与解读语均不再调用。

    原逻辑把 valence 分数映射为「偏负向/中性/偏正向」，但其基准点为手工常数、
    未经标定，且情绪强度会被加权平均约掉。现改用三色构成条呈现构成比例。
    """
    return ""


def arousal_label(a: float) -> str:
    """已废弃：保留仅为向后兼容，界面与解读语均不再调用（同上）。"""
    return ""


# 三色构成：把六类情绪按效价方向归入 消极 / 中性 / 积极 三档。
# 用「强度总和」而非词频计数——同一类情绪词写得越多越集中，条形越长，
# 这样条形能反映情绪的份量，而不只是命中次数。
MIX_GROUPS = [
    ("消极", ("悲伤", "愤怒", "焦虑"), "#9CA7BE"),
    ("中性", ("平静",),       "#B7BFA9"),
    ("积极", ("喜悦",),       "#7FB69A"),
]


def composition_mix(r: dict) -> list:
    """返回 [(档位名, 占比0~100, 颜色), ...]，占比按该档情绪词的强度总和计算。

    无情绪词命中时，中性档占满 100%（整段文字被视为情绪中性）。
    """
    inten = r.get("cat_intensity", {})
    if not inten:
        return [("中性", 100.0, "#B7BFA9")]
    total = sum(inten.values()) or 1
    out = []
    for name, cats, color in MIX_GROUPS:
        w = sum(inten.get(c, 0) for c in cats)
        if w > 0:
            out.append((name, round(w / total * 100, 1), color))
    return out or [("中性", 100.0, "#B7BFA9")]


def render_mixbar(mix: list) -> None:
    """渲染横向三色构成条 + 图例。"""
    segs = "".join(
        f'<span style="width:{pct:.1f}%;background:{color}" title="{name} {pct:.1f}%"></span>'
        for name, pct, color in mix
    )
    legend = "".join(
        f'<span><i style="background:{color}"></i>{name} <b>{pct:.1f}%</b></span>'
        for name, pct, color in mix
    )
    st.markdown(
        f'<div class="mixbar">{segs}</div><div class="mixlegend">{legend}</div>',
        unsafe_allow_html=True,
    )


# =====================================================================
# 外部词典加载（可整体替换为 DUTIR / 自建 CSV，按表头自动定位列）
# ---------------------------------------------------------------------
# 支持两种格式，启动时自动读取 情绪词典.csv 或 dutir.csv（找不到则用内置词典）：
#
#  ① 本工具简表：表头 词语, 类别, 强度, 极性
#     类别写 喜悦/悲伤/愤怒/焦虑/平静；强度 1~5；极性可留空。
#  ② DUTIR《情感词汇本体》原生格式（官方列：词语/词性种类/词义数/词义序号/
#     情感分类/强度/极性/辅助情感分类/…）——程序见表头含“情感分类”即按 DUTIR 解析。
#
# 情感分类两字母码是 DUTIR 的“情感小类”，说明文档（情感词汇本体库说明文档）
# 明确对应七大类 21 小类，见 DUTIR_CAT_MAP；强度 1/3/5/7/9 折算到 1~5；
# 极性 0中性/1褒/2贬/3兼有，仅用于过滤少数据。
# =====================================================================

# DUTIR 小类代码 → 本工具六类（按《情感词汇本体库说明文档》表2 校准）
DUTIR_CAT_MAP = {
    # —— 乐：快乐 PA；安心 PE ——
    "PA": "喜悦", "PE": "平静",
    # —— 好（褒义好感，统一归为正向“喜悦”）——
    "PD": "喜悦",   # 尊敬
    "PH": "喜悦",   # 赞扬
    "PG": "喜悦",   # 相信
    "PB": "喜悦",   # 喜爱
    "PK": "喜悦",   # 祝愿
    # —— 怒：愤怒 NA ——
    "NA": "愤怒",
    # —— 哀：悲伤 NB / 失望 NJ / 疚 NH / 思 PF ——
    "NB": "悲伤", "NJ": "悲伤", "NH": "悲伤", "PF": "悲伤",
    # —— 惧：慌 NI / 恐惧 NC / 羞 NG ——
    "NI": "焦虑", "NC": "焦虑", "NG": "焦虑",
    # —— 恶：烦闷 NE→焦虑；憎恶 ND / 贬责 NN → 愤怒；妒忌 NK / 怀疑 NL → 焦虑 ——
    "NE": "焦虑", "ND": "愤怒", "NN": "愤怒", "NK": "焦虑", "NL": "焦虑",
    # 注：惊(惊奇 PC) 无对应六类，默认忽略，不加载。
    # 中文类名直接写也可（自建简表用）
    "喜悦": "喜悦", "悲伤": "悲伤", "愤怒": "愤怒", "焦虑": "焦虑", "平静": "平静",
}


def _parse_inten(raw, is_dutir: bool) -> int:
    """强度列解析：DUTIR 的 1/3/5/7/9 按 (x+1)//2 折算到 1~5；否则视为已是 1~5。"""
    try:
        v = int(float(str(raw).strip()))
    except Exception:
        return 3
    if is_dutir:                 # 1→1, 3→2, 5→3, 7→4, 9→5
        v = (v + 1) // 2
    return max(1, min(5, v))


def _resolve_cols(df):
    """按表头名定位 词语/情感分类/强度/极性/词义序号 五列（官方列序可变也可定位）。"""
    headers = [str(c).strip() for c in df.columns]
    word_col = None
    for h in headers:
        if h in ("词语", "词", "word", "term", "词汇"):
            word_col = h
            break
    cat_col = str_col = pol_col = sense_col = None
    for h in headers:
        if not h:
            continue
        if "情感分类" in h or "类别" in h or "分类" in h:
            if cat_col is None:
                cat_col = h
        elif "词义序号" in h:
            if sense_col is None:
                sense_col = h
        elif "强度" in h:
            if str_col is None:
                str_col = h
        elif "极性" in h:
            if pol_col is None:
                pol_col = h
    word_col = word_col or headers[0]
    if cat_col is None and len(headers) > 1:
        cat_col = headers[1]
    return word_col, cat_col, str_col, pol_col, sense_col


def load_lexicon_csv(path: str):
    """从 CSV（含 DUTIR 原生格式）读取情绪词典，失败/无有效列返回 None。

    返回结构与内置 EMOTION_WORDS 相同：{类别: [(词语, 强度1~5), ...]}。
    一词多义时优先取“词义序号 = 1”的基本义（DUTIR 例：开心 有 PA快乐 与 NN贬 两义，
    若按文件顺序先到先得，会被排在前的负向义项抢走 → 改为优先基本义）。
    """
    frame = None
    for enc in ("utf-8-sig", "utf-8", "gbk"):
        try:
            frame = pd.read_csv(path, dtype=str, encoding=enc)
            break
        except Exception:
            continue
    if frame is None or frame.shape[1] < 2 or frame.shape[0] == 0:
        return None

    wc, cc, sc, pc, kc = _resolve_cols(frame)
    if not cc:
        return None
    is_dutir = "情感分类" in str(cc)          # 官方 DUTIR 以“情感分类”为类别列名
    df = frame.dropna(subset=[wc, cc])

    best = {}    # word -> (类别, 强度, 是否基本义/词义序号1)
    for _, row in df.iterrows():
        w = str(row[wc]).strip()
        cat_raw = str(row[cc]).strip()
        if not w or not cat_raw:
            continue
        code = cat_raw.upper().strip()
        if code == "PC" or cat_raw == "惊奇":       # 惊：六类无对应，忽略
            continue
        cat = DUTIR_CAT_MAP.get(code)
        if cat is None:                             # 兜底：类别文本内含中文类名
            for cname in ("喜悦", "悲伤", "愤怒", "焦虑", "平静"):
                if cname in cat_raw:
                    cat = cname
                    break
        if cat is None or cat not in EMOTION_BASE:
            continue
        inten_raw = str(row[sc]).strip() if sc else ""
        inten = _parse_inten(inten_raw, is_dutir) if inten_raw else 3
        sense = str(row[kc]).strip() if kc else ""
        is_primary = (sense == "1")
        old = best.get(w)
        if old is None or (is_primary and not old[2]):
            best[w] = (cat, inten, is_primary)

    words = {}
    for w, (cat, inten, _is1) in best.items():
        words.setdefault(cat, []).append((w, inten))
    return words if words else None


# 启动时探测外部词典；存在则覆盖内置情绪词典并重建分词索引
_SRCDIR = os.path.dirname(os.path.abspath(__file__))
for _f in ("情绪词典.csv", "dutir.csv"):
    _p = os.path.join(_SRCDIR, _f)
    if os.path.isfile(_p):
        _ext = load_lexicon_csv(_p)
        if _ext:
            EMOTION_WORDS.clear()
            EMOTION_WORDS.update(_ext)
            EMOTION_LOOKUP.clear()
            EMOTION_LOOKUP.update(_build_emotion_lookup())
            _rebuild_lexicon_index()
            break


# =====================================================================
# 温和解读（规则模板，语气始终为非评判、可操作）
# =====================================================================

def interpretation(r: dict) -> str:
    d = r["dominant"]
    parts = []

    if not r["emotion_rich"]:
        parts.append(
            "这段文字里的情绪线索比较稀疏（情感词很少），机器难以判断你当下的心情，"
            "只能暂记为「中性」。如果这只是例行的、说明性的文字，这是很正常的。"
        )
    else:
        wd = "、".join(w for w, _ in r["dominant_words"][:4]) if r["dominant_words"] else ""
        extra = f"（如「{wd}」）" if wd else ""
        # 构成描述直接由三色构成条的数据生成，避免复述无依据的数值指标
        mix_desc = "、".join(f"{n} {p:.0f}%" for n, p, _ in composition_mix(r))
        parts.append(
            f"整体来看，这段文字流露的情绪以「{d}」为主{extra}，"
            f"情绪构成上{('以' + mix_desc + '为主') if len(composition_mix(r)) > 1 else mix_desc}。"
            f"{_mood_hint(d)}"
        )

    cr = r["rates"]["cog"]
    if cr >= 8:
        parts.append(
            f"文中像「我觉得 / 我想 / 我认为」这类认知加工词出现较多（每千字约 {cr} 次）。"
            "这说明你在主动梳理和思考自己的感受，而不是停留在情绪表面——这是很宝贵的自我觉察。"
        )
    elif r["cog_n"] > 0:
        parts.append("文中也能看到一些认知加工的痕迹，表明表达并非只有情绪，还带着一层思考。")
    else:
        parts.append("文中几乎没有「我觉得/我认为」这类认知加工词，更多是直接的陈述或感受。")

    if r["neg_n"] >= 4 or r["rates"]["neg"] >= 12:
        parts.append(
            f"否定词出现得偏多（{r['neg_n']} 处）。多次使用「不 / 没 / 无」有时会让人更难看见已有的可能，"
            "试着把「我做不到」换成「我暂时还没做到」，语气会松动一些。"
        )
    if r["abs_n"] >= 2:
        parts.append(
            f"文里出现了 {r['abs_n']} 处「总是 / 永远 / 一定」这类绝对化表达。"
            "全有或全无的说法会放大压力，温和地把「总是」换成「有时」，也许能给自己留一点余地。"
        )
    if r["wo_single"] >= 8:
        parts.append(
            f"第一人称「我」出现频次不低（每千字约 {r['rates']['wo']} 次），"
            "叙述较多围绕自身的感受与处境，是一段比较“向内”的书写。"
        )
    elif r["wo_single"] == 0:
        parts.append("几乎没有用第一人称「我」，叙述显得比较抽离，也许是在陈述事实或他人的视角。")

    parts.append("提示：以上为词典规则的粗浅近似，仅适合练习与自我观察，不能替代专业评估。")
    return "\n\n".join(parts)


def _mood_hint(d: str) -> str:
    hints = {
        "喜悦": "这种轻快的心情值得被留意和保留。",
        "平静": "此刻内心比较平稳、放松，是难得的平衡状态。",
        "悲伤": "如果有这样的低落感，也请允许自己慢慢表达，不必急着“好起来”。",
        "愤怒": "怒气通常说明有让你在意、被越过的边界，可以试着温和地说出需要。",
        "焦虑": "不安往往指向你真正在意的事，把大问题拆小，一小步一小步来。",
        "中性": "",
    }
    return hints.get(d, "")


# =====================================================================
# 界面
# =====================================================================

PALETTE = {
    "bg": "#F5F7F5", "card": "#FFFFFF", "ink": "#3E4A43", "muted": "#7B887F",
    "accent": "#6E9E87", "accent2": "#A9CBB7", "line": "#E3EAE4",
    "emoji": {"喜悦": "#E3B066", "悲伤": "#9CA7BE", "愤怒": "#DA8F86",
              "焦虑": "#B08FC0", "平静": "#7FB69A", "中性": "#B7BFA9"},
    "dd": DISTORTION_COLORS,
}

CSS = f"""
<style>
    /* 页面背景、文字与主题变量一起锁定：无论浏览器/系统把主题切成暗色，
       文字都会回到深色，避免出现“浅色背景 + 白字”。 */
    .stApp {{
        background: {PALETTE['bg']} !important;
        color: {PALETTE['ink']};
        --background-color: {PALETTE['bg']};
        --secondary-background-color: #EEF3EE;
        --text-color: {PALETTE['ink']};
        --primary-color: {PALETTE['accent']};
    }}
    section[data-testid="stSidebar"] {{ background: #EEF3EE; }}
    h1, h2, h3 {{ color: {PALETTE['ink']} !important; }}
    /* 顶栏下方留足空白，避免标题被顶部功能栏遮挡 */
    .block-container {{ padding-top: 5.5rem; max-width: 880px; }}
    header[data-testid="stHeader"] {{
        background: transparent; border-bottom: 1px solid {PALETTE['line']};
    }}
    [data-testid="stAppDeployButton"] {{ display: none; }}
    /* 正文 / 输入控件文字显式着色（防暗色主题漏网的白字） */
    .stApp p, [data-testid="stMarkdownContainer"],
    [data-testid="stTextArea"] textarea,
    [data-testid="stTextInput"] input,
    .stApp label {{
        color: {PALETTE['ink']};
    }}
    .app-title {{ color:{PALETTE['ink']}; font-weight:700; letter-spacing:.5px; }}
    .subtitle {{ color:{PALETTE['muted']}; font-size:.95rem; margin-top:-.2rem; }}
    .softcard {{
        background: {PALETTE['card']}; border:1px solid {PALETTE['line']};
        border-radius:16px; padding:1.1rem 1.2rem; margin-bottom:.6rem;
        box-shadow:0 2px 10px rgba(110,158,135,.06);
    }}
    /* st.container(border=True) 统一成与 .softcard 一致的白底圆角卡片 */
    [data-testid="stVerticalBlockBorderWrapper"] {{
        background: #FFFFFF;
        border: 1px solid #E3EAE4 !important;
        border-radius: 16px;
        box-shadow: 0 2px 10px rgba(110,158,135,.06);
        padding: .55rem .8rem .2rem .8rem;
    }}
    .metric-value {{ font-size:1.9rem; font-weight:700; line-height:1.15; }}
    .metric-label {{ color:{PALETTE['muted']}; font-size:.8rem; margin-top:.15rem; }}
    /* 效价/唤醒度降级为定性标签：分值不再对外展示，只给三档色调 */
    .tone {{ font-size:1.3rem; font-weight:700; line-height:1.35; margin-top:.1rem; }}
    /* 三色构成条：消极 / 中性 / 积极，宽度按情绪强度占比分配 */
    .mixbar {{
        display:flex; height:26px; border-radius:999px; overflow:hidden;
        background:{PALETTE['line']}; margin:.45rem 0 .4rem;
    }}
    .mixbar > span {{ display:block; height:100%; }}
    .mixlegend {{
        display:flex; flex-wrap:wrap; gap:14px;
        color:{PALETTE['muted']}; font-size:.8rem; margin-top:.2rem;
    }}
    .mixlegend i {{
        display:inline-block; width:10px; height:10px; border-radius:2px;
        margin-right:5px; vertical-align:middle; font-style:normal;
    }}
    .mixlegend b {{ color:{PALETTE['ink']}; font-weight:500; }}
    .tone-cap {{
        color:{PALETTE['muted']}; font-size:.76rem; line-height:1.6;
        background:{PALETTE['bg']}; border:1px solid {PALETTE['line']};
        border-radius:10px; padding:.5rem .7rem; margin-top:.5rem;
    }}
    .chip {{
        display:inline-block; background:#EFF5F1; color:{PALETTE['ink']};
        border-radius:999px; padding:.18rem .65rem; font-size:.82rem;
        margin:.15rem .2rem .15rem 0;
    }}
    .emobadge {{
        display:inline-block; border-radius:12px; padding:.32rem .8rem;
        font-weight:700; font-size:1.05rem; color:#fff; letter-spacing:1px;
    }}
    .dd-item {{ border-left:4px solid transparent; }}
    .dd-badge {{
        display:inline-block; border-radius:8px; padding:.15rem .55rem;
        color:#fff; font-weight:600; font-size:.84rem; margin-right:.45rem;
        vertical-align:1px;
    }}
    .sec-title {{ color:{PALETTE['ink']}; font-weight:650; margin:.55rem 0 .3rem 0; }}
    .disclaimer {{
        background:#FFF6E7; border:1px solid #F0DDAE; border-radius:14px;
        padding:.95rem 1.15rem; margin-top:1.4rem; line-height:1.75;
        color:#6C5A38; font-size:.85rem;
    }}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

st.markdown('<div class="app-title">🌿 中文情绪文本分析</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">规则词典 · 情绪识别 + LIWC 风格语言特征 + 认知扭曲提示 · 温和解读（不依赖外部 API）</div>',
    unsafe_allow_html=True,
)
st.markdown("")

SAMPLE_MILD = (
    "今天拿到体检报告的时候，我其实很担心，害怕结果不好。坐在走廊里坐立不安，"
    "脑子里一直在想：会不会有事？后来医生说一切正常，我一下子就松了一口气，"
    "心里那块石头落了地，真的很开心，觉得生活又踏实了。"
)

SAMPLE_DISTORT = (
    "这次我又没考好，真的完蛋了。我真是个失败者，觉得自己什么都不行，"
    "每次都是这样，永远都不可能成功。他们肯定都在笑话我，她肯定觉得我特别没用，"
    "我应该做得更好才行，感觉天都塌了。"
)

if "input" not in st.session_state:
    st.session_state.input = ""


def _fill_input(text: str) -> None:
    """示例/清空按钮回调：在按钮事件阶段写 session_state。

    不能在 widget(key="input") 实例化之后再直接赋值，Streamlit 会抛
    StreamlitAPIException；通过 on_click 回调写入（发生在 widget 创建前）才合法。
    """
    st.session_state.input = text


col_in, col_btn = st.columns([6, 2])
with col_in:
    st.text_area(
        "✍️ 请粘贴一段中文文字", value=st.session_state.input, height=170,
        label_visibility="collapsed", key="input",
        placeholder="在这里粘贴想要分析的中文文字……（建议不少于 30 字，结果更稳定）",
    )
with col_btn:
    st.markdown('<div style="height:30px"></div>', unsafe_allow_html=True)
    st.button("📥 温和示例", on_click=_fill_input, args=(SAMPLE_MILD,), width="stretch")
    st.button("🧠 扭曲示例", on_click=_fill_input, args=(SAMPLE_DISTORT,), width="stretch")
    st.button("🧹 清空", on_click=_fill_input, args=("",), width="stretch")

def render_analysis(r: dict):
    """把一次分析结果渲染到页面上（含情绪、语言特征、认知扭曲、温和解读）。"""
    cat_counts = r["cat_counts"]

    # ---------- 1) 情绪识别 ----------
    st.markdown('<div class="sec-title">📊 情绪识别</div>', unsafe_allow_html=True)
    c1, c2 = st.columns([1, 2])
    dom_color = PALETTE["emoji"].get(r["dominant"], "#B7BFA9")
    with c1:
        st.markdown(
            f'<div class="softcard" style="margin-bottom:0"><span class="metric-label">主导情绪</span><br>'
            f'<span class="emobadge" style="background:{dom_color}">{r["dominant"]}</span></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            '<div class="softcard" style="margin-bottom:.4rem">'
            '<span class="metric-label">情绪构成 · 消极 / 中性 / 积极</span>',
            unsafe_allow_html=True,
        )
        render_mixbar(composition_mix(r))

    st.markdown(
        '<div class="tone-cap">色条按各类情绪词的<b>强度占比</b>分配宽度，'
        '混合情绪会各占一段——能直接看出消极与积极各占多少，而不是被折成一个中性的结论。'
        '无情绪词命中时整条视为中性。</div>',
        unsafe_allow_html=True,
    )

    # 篇幅信息 + 情绪分布：放进同一张白底卡片（左文右图）
    with st.container(border=True):
        txt_col, chart_col = st.columns([2, 3])
        with txt_col:
            st.markdown(
                f"**字数** {r['total_chars']}（汉字 {r['cjk_chars']}）　**约词数** "
                f"{r['total_tokens']}　**句子数** {r['sentence_n']}"
            )
            st.markdown(
                f"情感词命中 <b>{r['emotion_token_n']}</b> 处 · 占词典切分词的 "
                f"<b>{r['emotion_ratio_pct']}%</b>",
                unsafe_allow_html=True,
            )
        with chart_col:
            if cat_counts:
                df = pd.DataFrame({"次数": cat_counts})
                df = df.reindex(["喜悦", "平静", "悲伤", "愤怒", "焦虑"])
                df = df.fillna(0).astype(int)
                st.bar_chart(df, horizontal=True, color="#7FB69A", height=200)
            else:
                st.markdown("情绪词未命中：情绪分布为空。")

    # ---------- 2) LIWC 风格语言特征 ----------
    st.markdown('<div class="sec-title">🧩 LIWC 风格语言特征</div>', unsafe_allow_html=True)
    rows = [
        ("情感词", r["emotion_token_n"], f"{r['emotion_ratio_pct']}%"),
        ("认知加工词（我想/我觉得/我认为…）", r["cog_n"], f"{r['rates']['cog']}‰"),
        ("否定词（不/没/无…）", r["neg_n"], f"{r['rates']['neg']}‰"),
        ("绝对化词（总是/从不/永远/一定…）", r["abs_n"], f"{r['rates']['abs']}‰"),
        ("第一人称「我」", r["wo_single"], f"{r['rates']['wo']}‰"),
    ]
    ftab = pd.DataFrame(rows, columns=["特征", "出现次数", "每千字频次/占比"])

    # 统计表 + 命中词标签：同一张白底卡片
    with st.container(border=True):
        st.dataframe(ftab, hide_index=True, width="stretch")
        mcol1, mcol2 = st.columns(2)
        with mcol1:
            st.markdown(
                "**认知加工词**：" + (" ".join(f'<span class="chip">{w}</span>' for w in r["cog_ex"][:8]) or "（无）"),
                unsafe_allow_html=True,
            )
            st.markdown(
                "**第一人称**：" + "　".join(
                    f'{k.replace("类", "")}<span class="chip" style="margin-left:.2rem">{v}</span>'
                    for k, v in r["wo_groups"].items()
                ),
                unsafe_allow_html=True,
            )
        with mcol2:
            st.markdown(
                "**主导情绪中的典型词**：" + (
                    " ".join(f'<span class="chip">{w}×{i}</span>' for w, i in r["dominant_words"])
                    if r["dominant_words"] else "（无）"
                ),
                unsafe_allow_html=True,
            )

    # ---------- 3) 认知扭曲识别（CBT 启发） ----------
    st.markdown('<div class="sec-title">🧠 认知扭曲提示 · CBT 启发</div>', unsafe_allow_html=True)
    st.caption("按关键词/句式做启发式匹配，仅供自我觉察，不是诊断。命中越多越值得自己慢慢看，但任何一句都不必当真。")
    if not r["distortions"]:
        st.markdown(
            '<div class="softcard">未明显命中常见的六类认知扭曲'
            '（灾难化 / 非黑即白 / 过度概括 / 读心术 / 应该陈述 / 情绪化推理）。'
            '如果你本来就在区分「想法」与「事实」，这本身就是很好的自我觉察。'
            '当然，字面上没命中不代表完全没有，工具终究只是粗浅的提醒。</div>',
            unsafe_allow_html=True,
        )
    else:
        for d in r["distortions"]:
            dd_color = DISTORTION_COLORS[d["name"]]
            kw = "、".join(d["patterns"][:4])
            st.markdown(
                f'<div class="softcard dd-item" style="border-left-color:{dd_color}">'
                f'<span class="dd-badge" style="background:{dd_color}">{d["name"]}</span>'
                f'<span>{cbt_hint(d)}</span><br>'
                f'<span style="font-size:.78rem;color:{PALETTE["muted"]}">'
                f'命中的表述：{kw}　·　（温和提示，仅供觉察）</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # ---------- 4) 温和解读 ----------
    st.markdown('<div class="sec-title">💬 一段温和的解读</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="softcard">{interpretation(r)}</div>', unsafe_allow_html=True)


if st.button("✨ 开始分析", type="primary", width="stretch"):
    text = (st.session_state.input or "").strip()
    if len(text) < 5:
        st.warning("请先输入或粘贴一段文字（至少 5 个字）。")
    else:
        render_analysis(analyze(text))

# ---------- 5) 页底：心理健康免责声明（始终显示） ----------
st.markdown(
    '<div class="disclaimer">⚠️ <b>心理健康免责声明</b>：本工具仅供自我觉察与学习使用，'
    '所有识别结果都只是规则词典的粗浅近似，<b>不构成医疗建议或诊断</b>。'
    '若低落、焦虑等困扰持续存在并影响日常生活，请及时寻求专业心理咨询师或精神科医生的帮助。</div>',
    unsafe_allow_html=True,
)
