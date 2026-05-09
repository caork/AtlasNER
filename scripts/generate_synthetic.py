"""Generate Synthetic (S2) training data for structured extraction SFT.

Generates diverse training samples covering gaps in the Gold dataset:
- Negative examples (no extractable records)
- Low-field-count samples
- Cross-specialty samples
- Format variants (table-to-text, regulation style, notice style)
- Edge cases (long measures, multiple records, ambiguous risk/process)
"""

from __future__ import annotations

import json
import random
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "synthetic"

SYSTEM_PROMPT = (
    "你是电力领域结构化信息抽取专家。给定电力文档段落，提取风险管控记录。"
    "仅输出文档中确实存在的字段，不臆造信息。输出严格JSON格式。\n\n"
    "可提取的字段：\n"
    "- equipment: 设备名称\n"
    "- process: 工序/作业内容\n"
    "- risk: 风险描述（风险类型或可能导致的后果）\n"
    "- risk_level: 风险等级\n"
    "- prevention: 防范措施\n"
    "- control: 管控措施"
)

# ---- Vocabulary pools ----

EQUIPMENT_POOL = [
    "变压器", "断路器", "隔离开关", "GIS组合电器", "电容器", "电抗器",
    "避雷器", "互感器", "母线", "电缆", "架空输电线路", "杆塔",
    "继电保护装置", "自动化终端", "直流控制保护系统", "换流阀",
    "换流变压器", "接地装置", "电压互感器", "电流互感器",
    "SF6断路器", "真空断路器", "油浸式变压器", "干式变压器",
    "高压开关柜", "低压配电柜", "环网柜", "箱式变电站",
    "电缆终端", "电缆接头", "通信电源屏", "蓄电池组",
    "消弧线圈", "并联电容器", "串联电抗器", "站用变压器",
]

PROCESS_POOL = [
    "整装转运", "常规转运", "消防管道安装", "附件安装",
    "油处理", "真空注油", "器身检查", "绝缘试验",
    "回路绝缘检查", "二次接线", "保护校验", "调试验收",
    "倒闸操作", "停电检修", "带电作业", "工作许可",
    "基础开挖", "铁塔组立", "架线", "电缆敷设",
    "交叉跨越施工", "临时用电接线", "接地网施工",
    "高处作业", "起重吊装", "管母安装", "引线安装",
    "SF6气体回收", "SF6气体充装", "油色谱分析",
    "红外测温", "局部放电检测", "耐压试验",
    "直流系统检修", "阀厅检修", "换流站巡检",
]

RISK_POOL = [
    "机械伤害", "物体打击", "触电", "高处坠落", "火灾",
    "爆炸", "设备损坏", "人身伤害", "大面积停电",
    "误操作", "误送电", "误停电", "带负荷拉合刀闸",
    "气体中毒", "窒息", "灼伤", "电弧灼伤",
    "设备短路", "绝缘击穿", "油泄漏", "SF6泄漏",
    "套管断裂", "主变冲撞损坏", "二次回路短路",
    "坍塌", "起重伤害", "车辆伤害", "淹溺",
    "电缆接地", "母线失压", "保护误动", "保护拒动",
]

RISK_LEVEL_POOL = ["高", "中", "低", "重大风险", "较大风险", "一般风险", "低风险"]

STANDARD_POOL = [
    "GB 26860-2011", "GB 38755-2019", "GB/T 31464-2022",
    "DL/T 596", "DL/T 5168-2016", "DL/T 5210.1",
    "Q/GDW 12152-2021", "Q/GDW 10799.8-2023", "Q/GDW 1512-2014",
    "《国家电网有限公司电力安全工作规程》",
    "《电力设备预防性试验规程》",
    "《变电站运维管理规范》",
]

ORG_POOL = [
    "国家电网有限公司", "省电力公司", "市供电公司",
    "变电运维班", "输电运维班", "配电运维班",
    "调度控制中心", "安全监察部", "运维检修部",
    "施工项目部", "监理单位", "运行单位",
]

SPECIALTY_POOL = ["变电", "输电", "配电", "直流", "调度"]

# ---- Generator functions ----

def _rng() -> random.Random:
    return random.Random()

def make_id(idx: int) -> str:
    return f"syn_{idx:06d}"

def pick(pool: list, rng: random.Random, n: int = 1) -> list:
    return rng.sample(pool, min(n, len(pool)))

def join_risks(risks: list[str]) -> str:
    return "、".join(risks)


def gen_full_record(idx: int, rng: random.Random) -> dict:
    """Complete record with all 6 fields."""
    equip = pick(EQUIPMENT_POOL, rng)[0]
    proc = pick(PROCESS_POOL, rng)[0]
    risks = pick(RISK_POOL, rng, rng.randint(1, 3))
    level = pick(RISK_LEVEL_POOL, rng)[0]
    n_prev = rng.randint(2, 5)
    n_ctrl = rng.randint(2, 4)

    preventions = gen_measure_list(equip, proc, risks, n_prev, rng, "prevention")
    controls = gen_measure_list(equip, proc, risks, n_ctrl, rng, "control")

    fields = {
        "equipment": equip,
        "process": proc,
        "risk": join_risks(risks),
        "risk_level": level,
        "prevention": preventions,
        "control": controls,
    }

    passage = build_passage_full(fields, rng)
    think = build_think(fields)
    return build_sample(idx, passage, fields, think, "full_record", len(fields))


def gen_risk_prevention_only(idx: int, rng: random.Random) -> dict:
    """Only risk + prevention (2 fields, like Sheet3 rows)."""
    risks = pick(RISK_POOL, rng, rng.randint(1, 2))
    n_prev = rng.randint(1, 3)
    preventions = gen_simple_measures(n_prev, rng)

    fields = {
        "risk": join_risks(risks),
        "prevention": preventions,
    }

    passage = f"主要风险：{fields['risk']}\n\n风险防范措施：\n{preventions}"
    think = build_think(fields)
    return build_sample(idx, passage, fields, think, "risk_prevention_only", 2)


def gen_negative_example(idx: int, rng: random.Random) -> dict:
    """No extractable record — pure management text."""
    templates = [
        "各单位应加强安全生产管理，落实安全生产责任制，定期开展安全检查和隐患排查治理工作。",
        "工作负责人应在开工前组织全体作业人员进行安全交底，交代工作任务、安全措施及注意事项。",
        "施工单位应建立健全安全管理体系，配备专职安全管理人员，确保安全投入满足施工需要。",
        "各级调度机构应严格执行调度规程，加强电网运行方式管理，确保电网安全稳定运行。",
        f"根据{pick(STANDARD_POOL, rng)[0]}的相关规定，各单位应建立完善的技术档案管理制度。",
        f"{pick(ORG_POOL, rng)[0]}应定期组织开展应急演练，提高应急处置能力和水平。",
        "安全工器具应定期进行试验和检查，不合格的安全工器具应及时报废更换，严禁使用。",
        "现场工作结束后，工作负责人应清点工作人员和工器具，确认无遗留后方可向调度汇报。",
        "各单位应加强外包施工队伍管理，严格资质审查和人员培训考核，确保外包作业安全可控。",
        f"本通知自发布之日起施行，原有关规定与本通知不一致的，以本通知为准。请{pick(ORG_POOL, rng)[0]}遵照执行。",
        "安全培训应覆盖全体从业人员，新入职人员必须经过三级安全教育培训合格后方可上岗作业。",
        "变电站值班人员应严格执行交接班制度，认真做好设备巡视和运行记录，及时发现和处理异常情况。",
    ]
    passage = pick(templates, rng)[0]
    fields: dict[str, str] = {}
    think = "分析文档内容：\n该段落为一般性管理要求或制度规定，不包含具体的设备工序风险管控记录，无可提取的结构化信息。"
    return build_sample(idx, passage, fields, think, "negative", 0)


def gen_regulation_style(idx: int, rng: random.Random) -> dict:
    """Regulation/standard style text with structured content."""
    equip = pick(EQUIPMENT_POOL, rng)[0]
    proc = pick(PROCESS_POOL, rng)[0]
    risks = pick(RISK_POOL, rng, rng.randint(1, 2))
    std = pick(STANDARD_POOL, rng)[0]
    n_prev = rng.randint(2, 4)

    prev_items = []
    for i in range(1, n_prev + 1):
        prev_items.append(f"（{i}）{gen_single_measure(equip, rng)}")
    preventions = "\n".join(prev_items)

    passage = (
        f"第X条 {equip}{proc}安全要求\n\n"
        f"进行{equip}的{proc}作业时，应防范{join_risks(risks)}等风险，"
        f"并严格执行{std}的有关规定。具体安全措施如下：\n{preventions}"
    )

    fields = {
        "equipment": equip,
        "process": proc,
        "risk": join_risks(risks),
        "prevention": preventions,
    }
    think = build_think(fields)
    return build_sample(idx, passage, fields, think, "regulation_style", len(fields))


def gen_notice_style(idx: int, rng: random.Random) -> dict:
    """Notice/memo style document."""
    equip = pick(EQUIPMENT_POOL, rng)[0]
    proc = pick(PROCESS_POOL, rng)[0]
    risks = pick(RISK_POOL, rng, rng.randint(1, 2))
    level = pick(RISK_LEVEL_POOL, rng)[0]
    org = pick(ORG_POOL, rng)[0]
    n_prev = rng.randint(2, 3)

    prev_items = []
    for i in range(1, n_prev + 1):
        prev_items.append(f"{i}. {gen_single_measure(equip, rng)}")
    preventions = "\n".join(prev_items)

    passage = (
        f"关于加强{equip}{proc}风险管控的通知\n\n"
        f"各相关单位：\n"
        f"近期在{proc}作业中发生了{join_risks(risks)}事故，暴露出安全管控薄弱环节。"
        f"经{org}研究决定，对{equip}{proc}作业风险等级调整为{level}，"
        f"并要求落实以下防范措施：\n{preventions}"
    )

    fields = {
        "equipment": equip,
        "process": proc,
        "risk": join_risks(risks),
        "risk_level": level,
        "prevention": preventions,
    }
    think = build_think(fields)
    return build_sample(idx, passage, fields, think, "notice_style", len(fields))


def gen_table_to_text(idx: int, rng: random.Random) -> dict:
    """Simulates table-extracted text (compact, header-value pairs)."""
    equip = pick(EQUIPMENT_POOL, rng)[0]
    proc = pick(PROCESS_POOL, rng)[0]
    risks = pick(RISK_POOL, rng, rng.randint(1, 2))
    level = pick(RISK_LEVEL_POOL, rng)[0]
    n_prev = rng.randint(1, 3)

    prev_items = [f"{i}.{gen_single_measure(equip, rng)}" for i in range(1, n_prev + 1)]
    preventions = " ".join(prev_items)

    passage = (
        f"设备：{equip}　工序：{proc}　"
        f"风险：{join_risks(risks)}　等级：{level}　"
        f"防范措施：{preventions}"
    )

    fields = {
        "equipment": equip,
        "process": proc,
        "risk": join_risks(risks),
        "risk_level": level,
        "prevention": preventions,
    }
    think = build_think(fields)
    return build_sample(idx, passage, fields, think, "table_to_text", len(fields))


def gen_ambiguous_risk_process(idx: int, rng: random.Random) -> dict:
    """Cases where process and risk descriptions overlap."""
    ambiguous_pairs = [
        ("倒闸操作", "误操作", "操作过程中存在误操作风险"),
        ("带电作业", "触电", "带电作业过程中可能发生触电事故"),
        ("高处作业", "高处坠落", "高空作业时存在人员坠落风险"),
        ("起重吊装", "起重伤害", "起吊过程中存在起重伤害风险"),
        ("电缆敷设", "电缆接地", "电缆敷设不当可能导致电缆接地故障"),
        ("SF6气体回收", "SF6泄漏", "气体回收过程中存在SF6泄漏风险"),
    ]
    proc, risk, desc = pick(ambiguous_pairs, rng)[0]
    equip = pick(EQUIPMENT_POOL, rng)[0]
    n_prev = rng.randint(2, 3)
    prev_items = [f"{i}.{gen_single_measure(equip, rng)}" for i in range(1, n_prev + 1)]
    preventions = "\n".join(prev_items)

    passage = (
        f"{equip}{proc}\n\n{desc}。\n\n防范措施：\n{preventions}"
    )

    fields = {
        "equipment": equip,
        "process": proc,
        "risk": risk,
        "prevention": preventions,
    }
    think = (
        f"分析文档内容：\n"
        f"1. 设备：\"{equip}\" → equipment\n"
        f"2. 工序：\"{proc}\"是计划性作业，归为工序。→ process\n"
        f"3. 风险描述：\"{risk}\"是可能发生的风险事件。→ risk\n"
        f"   注意：\"{proc}\"是工序，\"{risk}\"是风险，二者相关但语义不同。\n"
        f"4. 风险等级：文档中未找到，跳过。\n"
        f"5. 防范措施：共{n_prev}条。→ 完整提取\n"
        f"6. 管控措施：文档中未找到，跳过。"
    )
    return build_sample(idx, passage, fields, think, "ambiguous_risk_process", len(fields))


def gen_multi_specialty(idx: int, rng: random.Random) -> dict:
    """Cross-specialty samples (直流, 输电, 配电 etc.)."""
    specialty = pick(SPECIALTY_POOL, rng)[0]
    specialty_equip = {
        "变电": ["变压器", "断路器", "隔离开关", "GIS组合电器", "母线"],
        "输电": ["架空输电线路", "杆塔", "电缆", "电缆终端", "电缆接头"],
        "配电": ["环网柜", "箱式变电站", "低压配电柜", "高压开关柜"],
        "直流": ["换流阀", "换流变压器", "直流控制保护系统", "通信电源屏"],
        "调度": ["继电保护装置", "自动化终端", "蓄电池组"],
    }
    equip = pick(specialty_equip.get(specialty, EQUIPMENT_POOL), rng)[0]
    proc = pick(PROCESS_POOL, rng)[0]
    risks = pick(RISK_POOL, rng, rng.randint(1, 2))
    level = pick(RISK_LEVEL_POOL, rng)[0]
    n_prev = rng.randint(2, 4)
    prev_items = [f"{i}.{gen_single_measure(equip, rng)}" for i in range(1, n_prev + 1)]
    preventions = "\n".join(prev_items)

    passage = (
        f"{specialty}专业 {equip}{proc}工序风险管控\n\n"
        f"主要风险：{join_risks(risks)}\n"
        f"风险等级：{level}\n\n"
        f"防范措施：\n{preventions}"
    )

    fields = {
        "equipment": equip,
        "process": proc,
        "risk": join_risks(risks),
        "risk_level": level,
        "prevention": preventions,
    }
    think = build_think(fields)
    return build_sample(idx, passage, fields, think, f"multi_specialty_{specialty}", len(fields))


# ---- Measure generation helpers ----

MEASURE_TEMPLATES = [
    "作业前应检查{equip}状态，确认具备作业条件。",
    "严格执行操作票制度，操作前核对设备名称和编号。",
    "作业人员必须正确佩戴安全帽、绝缘手套等个人防护用品。",
    "高处作业必须系好安全带，安全带应高挂低用。",
    "使用合格的安全工器具，不得使用过期或损坏的工器具。",
    "施工现场应设置安全警示标志和围栏，非工作人员禁止进入。",
    "带电部位附近作业时，应保持安全距离，必要时加装绝缘遮蔽。",
    "工作结束后应清点工器具和材料，确认无遗留。",
    "发现异常情况应立即停止作业，报告工作负责人处理。",
    "禁止在恶劣天气条件下进行室外高处作业和带电作业。",
    "{equip}检修前应办理工作票，履行工作许可手续。",
    "拆除和恢复{equip}接线时，应做好标记，防止接错。",
    "操作{equip}前应核对设备状态，确认无负荷后方可操作。",
    "使用起重机械时，钢丝绳不得超过额定负荷，严禁斜拉歪拽。",
    "电气设备停电检修时，应验电、接地，悬挂安全标示牌。",
    "二次回路作业时，应使用二次安全措施票，严禁擅自短接或断开。",
    "有限空间作业必须先通风、再检测、后作业，检测不合格严禁作业。",
    "{equip}运输过程中应控制振动值在允许范围内。",
    "临时用电应由持证电工操作，做到一机一闸一保护。",
    "作业现场应配备足够的消防器材，作业人员应熟悉灭火器使用方法。",
]


def gen_single_measure(equip: str, rng: random.Random) -> str:
    tpl = pick(MEASURE_TEMPLATES, rng)[0]
    return tpl.format(equip=equip)


def gen_measure_list(
    equip: str, proc: str, risks: list[str], n: int, rng: random.Random, kind: str
) -> str:
    items = []
    used = set()
    for i in range(1, n + 1):
        attempts = 0
        while attempts < 20:
            tpl = pick(MEASURE_TEMPLATES, rng)[0]
            if tpl not in used:
                used.add(tpl)
                items.append(f"{i}.{tpl.format(equip=equip)}")
                break
            attempts += 1
    return "\n".join(items)


def gen_simple_measures(n: int, rng: random.Random) -> str:
    items = []
    used = set()
    for i in range(1, n + 1):
        attempts = 0
        while attempts < 20:
            tpl = pick(MEASURE_TEMPLATES, rng)[0]
            if tpl not in used:
                used.add(tpl)
                items.append(f"{i}.{tpl.format(equip='设备')}")
                break
            attempts += 1
    return "\n".join(items)


# ---- Think chain builder ----

FIELD_ORDER = ["equipment", "process", "risk", "risk_level", "prevention", "control"]
FIELD_CN = {
    "equipment": "设备",
    "process": "工序",
    "risk": "风险描述",
    "risk_level": "风险等级",
    "prevention": "防范措施",
    "control": "管控措施",
}

def build_think(fields: dict[str, str]) -> str:
    lines = ["分析文档内容："]
    step = 1
    for key in FIELD_ORDER:
        cn = FIELD_CN[key]
        if key in fields:
            val = fields[key]
            if key in ("prevention", "control"):
                n = len([m for m in val.split("\n") if m.strip()])
                lines.append(f'{step}. {cn}：共{n}条。→ 完整提取')
            elif len(val) > 30:
                lines.append(f'{step}. {cn}："{val[:30]}..." → {key}')
            else:
                lines.append(f'{step}. {cn}："{val}" → {key}')
        else:
            lines.append(f"{step}. {cn}：文档中未找到，跳过。")
        step += 1
    return "\n".join(lines)


def build_passage_full(fields: dict[str, str], rng: random.Random) -> str:
    parts = []
    header_parts = []
    if "equipment" in fields:
        header_parts.append(fields["equipment"])
    if "process" in fields:
        header_parts.append(fields["process"])
    if header_parts:
        parts.append(" ".join(header_parts))
    if "risk" in fields:
        parts.append(f"主要风险：{fields['risk']}")
    if "risk_level" in fields:
        parts.append(f"风险等级：{fields['risk_level']}")
    if "prevention" in fields:
        parts.append(f"风险防范措施：\n{fields['prevention']}")
    if "control" in fields:
        parts.append(f"管控措施：\n{fields['control']}")
    return "\n\n".join(parts)


def build_sample(
    idx: int, passage: str, fields: dict[str, str], think: str,
    category: str, field_count: int,
) -> dict:
    output_json = json.dumps(fields, ensure_ascii=False, indent=2)
    assistant = f"<think>\n{think}\n</think>\n\n```json\n{output_json}\n```"
    return {
        "id": make_id(idx),
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"请从以下文档段落中提取结构化风险管控记录：\n\n<document>\n{passage}\n</document>",
            },
            {"role": "assistant", "content": assistant},
        ],
        "metadata": {
            "quality": "synthetic",
            "category": category,
            "field_count": field_count,
            "input_char_count": len(passage),
            "output_char_count": len(output_json),
            "split": "train",
        },
    }


# ---- Main generation ----

def main() -> None:
    rng = random.Random(42)
    samples: list[dict] = []
    idx = 0

    generators = [
        ("full_record", gen_full_record, 300),
        ("risk_prevention_only", gen_risk_prevention_only, 200),
        ("negative", gen_negative_example, 200),
        ("regulation_style", gen_regulation_style, 150),
        ("notice_style", gen_notice_style, 150),
        ("table_to_text", gen_table_to_text, 150),
        ("ambiguous_risk_process", gen_ambiguous_risk_process, 100),
        ("multi_specialty", gen_multi_specialty, 250),
    ]

    for name, gen_fn, count in generators:
        for _ in range(count):
            sample = gen_fn(idx, rng)
            samples.append(sample)
            idx += 1
        print(f"  Generated {count} {name} samples")

    rng.shuffle(samples)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "train.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"\nTotal: {len(samples)} synthetic samples")
    print(f"Written to {out_path}")

    # Stats
    from collections import Counter
    cat_counts = Counter(s["metadata"]["category"] for s in samples)
    fc_counts = Counter(s["metadata"]["field_count"] for s in samples)
    print("\nBy category:")
    for cat, cnt in sorted(cat_counts.items()):
        print(f"  {cat}: {cnt}")
    print("\nBy field count:")
    for fc, cnt in sorted(fc_counts.items()):
        print(f"  {fc} fields: {cnt}")


if __name__ == "__main__":
    main()
