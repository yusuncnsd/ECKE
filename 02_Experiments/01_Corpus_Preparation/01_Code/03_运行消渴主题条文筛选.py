from __future__ import annotations
import concurrent.futures
import hashlib
import json
import random
import re
import sys
import time
import urllib.error
import urllib.request
import zipfile
import threading
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from xml.etree import ElementTree as ET


# ======================== 请在此处填写 ========================
API密钥 = ("")

# 已通过试运行验收，当前固定为正式全量运行。
运行模式 = "正式运行"               # 可选：试运行 / 正式运行
试运行数量 = 10
并发线程数 = 5

# 模型与复现参数
模型名称 = "qwen-plus-2025-07-28"
温度 = 0.0
随机种子 = 20260813
最大重试次数 = 6
低置信阈值 = 0.80
# =============================================================


脚本目录 = Path(__file__).resolve().parent.parent
清洁古籍目录 = 脚本目录 / "04_Results/488_Cleaned_Books"
词表路径 = 脚本目录 / "03_Shared_Configuration_and_Lexicon/概念词表-消渴LJ-26.8.14去重.xlsx"
结果目录 = 脚本目录 / "04_Results/复现运行/Xiaoke_Passage_Corpus"
过程目录 = 结果目录 / "运行中"
证据单元缓存 = 过程目录 / "重建条文.jsonl"
证据单元清单 = 过程目录 / "重建条文清单.json"
候选缓存 = 过程目录 / "待复核候选.jsonl"
候选清单 = 过程目录 / "待复核候选清单.json"
首轮缓存 = 过程目录 / "LLM首轮结果.jsonl"
二轮缓存 = 过程目录 / "LLM二轮结果.jsonl"
异常缓存 = 过程目录 / "异常记录.jsonl"
正式JSONL = 结果目录 / "正式消渴主题条文库.jsonl"
正式TXT = 结果目录 / "正式消渴主题条文库.txt"
运行摘要 = 结果目录 / "构建摘要.json"

北京时间 = timezone(timedelta(hours=8))
API地址 = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
提示词版本 = "XIAOKE-CORPUS-FINAL-1.0"
规则版本 = "RECONSTRUCTION-FINAL-1.0"
写入锁 = threading.Lock()
API预检已通过 = False
专家知识版本 = "XIAOKE-EK-v1.0"
# 云端内容审核无法处理、且已经由专家作出明确结论的记录。
人工审核排除条文 = {
    "EU_0284026": "篇章及主要诊疗对象为强中/前阴诸疾，消渴仅作为可能并发结果，不能形成以消渴为中心的独立诊疗知识链。",
}
# 310条风险条文经专家复核后，有165条不采纳。使用稳定的evidence_unit_id固化，
# 避免删除Excel或重新编号后丢失人工结论；其余145条风险条文明确采纳。
专家终审不采纳证据单元 = set("""
EU_0002073,EU_0003191,EU_0003534,EU_0017647,EU_0019193,EU_0019880,EU_0020255,EU_0022514,EU_0023279,EU_0032041,EU_0032426,EU_0033431,EU_0038526,EU_0038752,EU_0038904,EU_0038905,EU_0038910,EU_0038912,EU_0038914,EU_0038915,EU_0038916,EU_0038924,EU_0046069,EU_0048667,EU_0048837,EU_0051207,EU_0051445,EU_0051458,EU_0067111,EU_0067112,EU_0067113,EU_0067116,EU_0077242,EU_0077710,EU_0077714,EU_0077717,EU_0077735,EU_0077759,EU_0077856,EU_0077870,EU_0077904,EU_0077953,EU_0077956,EU_0077957,EU_0077965,EU_0077994,EU_0078037,EU_0078043,EU_0078046,EU_0078053,EU_0078160,EU_0078162,EU_0078180,EU_0078195,EU_0078200,EU_0078214,EU_0078221,EU_0078233,EU_0078245,EU_0078293,EU_0078339,EU_0078344,EU_0078374,EU_0078377,EU_0078394,EU_0078410,EU_0078411,EU_0078414,EU_0078415,EU_0078458,EU_0091495,EU_0099063,EU_0109073,EU_0115835,EU_0134096,EU_0139504,EU_0140364,EU_0140988,EU_0140993,EU_0143054,EU_0148711,EU_0150750,EU_0155106,EU_0161144,EU_0164891,EU_0167826,EU_0171693,EU_0176190,EU_0178400,EU_0178884,EU_0181559,EU_0182024,EU_0182896,EU_0183302,EU_0183647,EU_0183724,EU_0184057,EU_0184927,EU_0186561,EU_0192621,EU_0192627,EU_0192793,EU_0193464,EU_0193760,EU_0194649,EU_0195615,EU_0196126,EU_0197940,EU_0198446,EU_0199358,EU_0199999,EU_0200830,EU_0201128,EU_0202012,EU_0216172,EU_0219431,EU_0219496,EU_0220759,EU_0222291,EU_0222346,EU_0233598,EU_0235742,EU_0239048,EU_0243692,EU_0244223,EU_0245639,EU_0247240,EU_0254865,EU_0258519,EU_0259167,EU_0259174,EU_0261276,EU_0261277,EU_0261278,EU_0261280,EU_0261283,EU_0266341,EU_0266802,EU_0267091,EU_0267097,EU_0274986,EU_0284372,EU_0291256,EU_0300375,EU_0300398,EU_0307067,EU_0307076,EU_0307094,EU_0309504,EU_0309726,EU_0320304,EU_0320646,EU_0320948,EU_0320954,EU_0320955,EU_0320961,EU_0322347,EU_0323735,EU_0346866,EU_0347017,EU_0347191,EU_0359357,EU_0370971,EU_0373566,EU_0380468
""".strip().split(","))

类别映射 = {
    "中药": "中药", "方剂": "方药", "疾病": "疾病", "症状": "症状",
    "病因病机": "病机", "病机": "病机", "病位": "病位",
    "治则治法": "治法", "治法": "治法",
}

强标题 = re.compile(
    r"^(?:书名|卷名|篇名|章节|目录|属性|内容)[：:]|^第?[一二三四五六七八九十百千万〇零两\d]+(?:卷|篇|章|门|论|证|候)"
)
方名标题 = re.compile(r"^.{1,24}(?:汤|丸|散|膏|丹|饮|煎|方|剂|酒|露|锭|饼|羹|粥)(?:方)?[：:]?$" )
中边界提示 = re.compile(r"^(?:论曰|经曰|曰|云|治曰|主治|治法|方如下|组成)[：:]?")
省略提示 = re.compile(r"^(?:同上|治前证|治前症|此证|前方|如前|上证|上症)")
句末 = set("。！？；")
纯目录 = re.compile(r"^(?:目录[：:]?)|^卷之?[一二三四五六七八九十百千万〇零两\d]+(?:目录)?$")
本草文件提示 = re.compile(r"本草|名医别录|药性|药解|药鉴|药征|药录|药品|药要|药笼")
本草正文提示 = re.compile(r"^(?:内容|属性|气味|性味|主治|释名)[：:]?")
消渴特异表现 = re.compile(
    r"(?:大渴|引饮|饮水(?:极|甚|过)?多|多饮|善饮|烦渴|口渴|咽干|舌燥|"
    r"善食|多食|消谷|易饥|善饥|食已即饥|"
    r"小便(?:极|甚|过)?多|小便数|溲数|尿多|尿如膏|小便如膏|小便如米泔|"
    r"肌肉消瘦|形体消瘦|羸瘦|羸弱|肌肤消削)"
)
歧义疾病词 = {"中消", "消烁", "热渴", "焦渴", "渴病", "食亦", "消疾", "火渴", "消肌", "消燥"}
歧义排除短语 = {"温中消食", "消烁寒毒", "大暑消烁", "烦渴病邪", "热渴尿涩"}
非药名开头 = ("即", "又", "一名", "此", "其", "按", "案", "云", "味", "主", "治", "生", "入", "得", "用", "无毒", "有毒")
非消渴疾病门 = re.compile(r"(?:暑|伤寒|温病|瘟疫|中风|淋|痢|霍乱|疟|咳|喘|痰|痧|黄疸|水肿|心痛|头痛|妇人|小儿|产后|痈疽|疮|历节|风)\S*门")
诊疗表达 = re.compile(r"(?:病|证|候|治|主治|疗|方|药|汤|丸|散|病机|因|由于|表现|症|渴|饮|小便|溲|消谷|善食)")
平行列表项 = re.compile(r"(?:(?<=。)|(?<=；)|(?<=\n))(?=[\u3400-\u9fff]{1,10}\s+(?:宜|忌|治|主|用))")

# ======================== 专家消渴主题知识 v1.0 ========================
# 原Excel仍负责广义7类领域词识别；此处只负责消渴专题的证据等级与组合约束。
专家明确疾病锚点 = {
    "三消", "糖尿病", "消瘅", "瘅渴", "消渴", "渴疾", "渴病", "消证",
    "上消", "中消", "下消", "内伤三消", "妊娠消渴", "胎前消渴", "小儿消渴",
}
专家高歧义词 = {"渴", "瘅", "消", "宣疾"}
专家上位词 = {"气血津液病证"}  # 仅用于提示词和本体说明，不参与候选召回。
专家下位词 = {"上消", "中消", "下消", "内伤三消", "妊娠消渴", "胎前消渴", "小儿消渴"}
专家高特异症状组 = {
    "饮水异常": {"多饮", "大渴引饮", "引饮", "饮水无度", "饮水极多", "饮水甚多", "饮一"},
    "小便异常": {"多尿", "小便多", "小便数", "溲数", "尿多", "饮一溲二"},
    "尿液特征": {"尿有甜味", "尿甜", "小便如膏", "尿如膏", "溲如膏", "小便如米泔", "小便如油"},
    "食欲异常": {"多食", "善食", "善饥", "易饥", "消谷", "食已即饥"},
}
专家弱特异症状 = {"乏力", "消瘦", "羸瘦", "羸弱", "肌肉消瘦", "形体消瘦", "肌肤消削"}
专家证型 = {"肺热津伤证", "肺热津伤", "胃热炽盛证", "胃热炽盛", "气阴亏虚证", "气阴亏虚", "肾阴亏虚证", "肾阴亏虚", "阴阳两虚证", "阴阳两虚"}
专家高特异方剂 = {"消渴方"}
专家相关方剂 = {"玉女煎", "七味白术散", "六味地黄丸", "金匮肾气丸"}


def 专家知识指纹() -> str:
    config = {
        "version": 专家知识版本,
        "anchors": sorted(专家明确疾病锚点), "ambiguous": sorted(专家高歧义词),
        "hypernym": sorted(专家上位词), "hyponyms": sorted(专家下位词),
        "symptom_groups": {k: sorted(v) for k, v in 专家高特异症状组.items()},
        "weak_symptoms": sorted(专家弱特异症状), "patterns": sorted(专家证型),
        "specific_formulas": sorted(专家高特异方剂), "related_formulas": sorted(专家相关方剂),
    }
    return hashlib.sha256(json.dumps(config, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def 文件哈希(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def 输入指纹() -> dict:
    files = sorted(清洁古籍目录.glob("*.txt"))
    if len(files) != 488:
        raise RuntimeError(f"清洁古籍应为488部，实际为{len(files)}部")
    listing = [{"name": p.name, "size": p.stat().st_size, "sha256": 文件哈希(p)} for p in files]
    combined = hashlib.sha256(json.dumps(listing, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    return {
        "rule_version": 规则版本, "expert_knowledge_version": 专家知识版本,
        "expert_knowledge_sha256": 专家知识指纹(), "book_count": 488,
        "books_fingerprint": combined, "files": listing, "lexicon_sha256": 文件哈希(词表路径),
    }


def 安全写文本(path: Path, text: str, encoding: str = "utf-8") -> None:
    """先写临时文件再原子替换，并容忍OneDrive的短暂文件占用。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    last_error = None
    for attempt in range(1, 9):
        try:
            temp.write_text(text, encoding=encoding)
            temp.replace(path)
            return
        except PermissionError as exc:
            last_error = exc
            time.sleep(min(0.25 * attempt, 2.0))
    raise RuntimeError(f"结果文件被同步程序占用，重试后仍无法写入：{path}") from last_error


def 保存清单(path: Path, base: dict, **extra) -> None:
    安全写文本(path, json.dumps({**base, **extra}, ensure_ascii=False, indent=2))


def 校验清单(path: Path, expected: dict, expected_kind: str) -> dict:
    if not path.exists():
        raise RuntimeError(f"发现{expected_kind}缓存但缺少配套清单：{path}。该缓存可能在写完清单前中止；请再次运行，程序将重建该可再生阶段。")
    actual = json.loads(path.read_text(encoding="utf-8-sig"))
    for key in ["rule_version", "expert_knowledge_version", "expert_knowledge_sha256", "book_count", "books_fingerprint", "lexicon_sha256"]:
        if actual.get(key) != expected.get(key):
            raise RuntimeError(f"{expected_kind}缓存清单与当前输入不一致：{key}")
    return actual


def 输出JSONL(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as f:
        total = len(rows)
        for index, row in enumerate(rows, 1):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            if total >= 50000 and (index % 50000 == 0 or index == total):
                print(f"写入进度[{path.name}]：{index}/{total}（{index/total:.1%}）")
    # OneDrive可能短暂占用目标文件；有限重试后再报告明确错误。
    last_error = None
    for attempt in range(1, 9):
        try:
            temp.replace(path)
            return
        except PermissionError as exc:
            last_error = exc
            time.sleep(min(0.25 * attempt, 2.0))
    raise RuntimeError(f"结果文件被同步程序占用，重试后仍无法写入：{path}") from last_error


def 追加JSONL(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # 线程锁、逐条flush与磁盘同步，保证异常退出前已经完成的结果可以续接。
    with 写入锁:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            try:
                import os
                os.fsync(f.fileno())
            except OSError:
                pass


def 读取JSONL(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    # 异常断电可能只损坏最后一行；忽略该行，后续按ID自动补跑。
                    if line_no == sum(1 for _ in path.open("r", encoding="utf-8-sig")):
                        print(f"警告：忽略异常退出形成的末行残片：{path.name}第{line_no}行")
                        continue
                    raise RuntimeError(f"JSONL损坏：{path}第{line_no}行：{exc}") from exc
            if line_no % 50000 == 0:
                print(f"读取进度[{path.name}]：已读取{line_no}条")
    print(f"读取完成[{path.name}]：{len(rows)}条")
    return rows


def excel列号(ref: str) -> int:
    letters = re.match(r"[A-Z]+", ref).group(0)
    value = 0
    for ch in letters:
        value = value * 26 + ord(ch) - 64
    return value - 1


def 读取词表xlsx(path: Path) -> dict[str, set[str]]:
    """只用Python标准库读取xlsx，避免本地PyCharm缺少openpyxl。"""
    if not path.exists():
        raise FileNotFoundError(f"词表不存在：{path}")
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(path) as z:
        shared = []
        if "xl/sharedStrings.xml" in z.namelist():
            root = ET.fromstring(z.read("xl/sharedStrings.xml"))
            for si in root.findall("m:si", ns):
                shared.append("".join(t.text or "" for t in si.iterfind(".//m:t", ns)))
        sheet = ET.fromstring(z.read("xl/worksheets/sheet1.xml"))
        values = []
        for row in sheet.findall(".//m:sheetData/m:row", ns):
            cells = [None] * 4
            for cell in row.findall("m:c", ns):
                idx = excel列号(cell.attrib["r"])
                if idx >= 4:
                    continue
                kind = cell.attrib.get("t")
                v = cell.find("m:v", ns)
                inline = cell.find("m:is/m:t", ns)
                if inline is not None:
                    value = inline.text
                elif v is None:
                    value = None
                elif kind == "s":
                    value = shared[int(v.text)]
                else:
                    value = v.text
                cells[idx] = value
            values.append(cells)

    lexicon: dict[str, set[str]] = defaultdict(set)
    for row in values:
        raw_type, concept, aliases, _ = row
        if not raw_type or not concept or str(raw_type).strip() not in 类别映射:
            continue
        entity_type = 类别映射[str(raw_type).strip()]
        terms = [str(concept).strip()]
        if aliases:
            terms.extend(x.strip() for x in re.split(r"[、,，;；/|]", str(aliases)) if x.strip())
        for term in terms:
            if len(term) >= 2 and not term.isdigit():
                lexicon[entity_type].add(term)
    if "疾病" not in lexicon or "消渴" not in lexicon["疾病"] or "三消" not in lexicon["疾病"]:
        raise RuntimeError("疾病强锚点读取异常：必须包含‘消渴’和‘三消’")
    return dict(lexicon)


def 应用人工审核(rows: list[dict]) -> tuple[list[dict], dict]:
    # Excel作为专家审核凭证单独保留；运行时使用稳定证据单元ID，避免重新编号造成错配。
    excluded_evidence_ids = 专家终审不采纳证据单元
    available_evidence_ids = {x["evidence_unit_id"] for x in rows}
    missing_excluded = excluded_evidence_ids - available_evidence_ids
    if missing_excluded:
        raise RuntimeError(f"当前待冻结库缺少{len(missing_excluded)}条已审核排除证据单元，不能可靠迁移人工结论")
    filtered = []
    for row in rows:
        if row["evidence_unit_id"] in excluded_evidence_ids:
            continue
        out = dict(row)
        filtered.append(out)
    for index, row in enumerate(filtered, 1):
        row["record_id"] = f"XK_{index:06d}"
    return filtered, {
        "人工审核已应用": True, "人工审核条文": 310,
        "人工采纳": 145, "人工不采纳": 165,
    }


def 切分过长段(text: str, max_chars: int = 900) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    parts, current = [], ""
    for piece in re.split(r"(?<=[。！？；])", text):
        if current and len(current) + len(piece) > max_chars:
            parts.append(current.strip())
            current = piece
        else:
            current += piece
    if current.strip():
        parts.append(current.strip())
    # 无标点超长行不猜分句，按最大长度切并加低置信标记。
    final = []
    for part in parts or [text]:
        final.extend(part[i:i + max_chars] for i in range(0, len(part), max_chars))
    return final


def 重建一书(path: Path, global_start: int) -> tuple[list[dict], int]:
    text = path.read_text(encoding="utf-8-sig", errors="strict")
    # 先保留每一行的精确字符范围，后续合并仍能回溯原TXT。
    lines = []
    cursor = 0
    for raw in text.splitlines(keepends=True):
        content = raw.rstrip("\r\n")
        lines.append((content, cursor, cursor + len(raw)))
        cursor += len(raw)
    if cursor < len(text):
        lines.append((text[cursor:], cursor, len(text)))

    units, buffer = [], []
    current_heading = ""
    current_topic_heading = ""
    current_herb = ""
    current_formula = ""
    is_materia_medica = bool(本草文件提示.search(path.name))

    def flush(boundary: str) -> None:
        nonlocal buffer
        if not buffer:
            return
        raw = "\n".join(x[0].strip() for x in buffer if x[0].strip()).strip()
        if not raw:
            buffer = []
            return
        source_start = buffer[0][1]
        source_end = buffer[-1][2]
        # 脉象宜忌、病名主治等平行列表优先拆为最小知识单元。
        list_parts = [x.strip() for x in 平行列表项.split(raw) if x.strip()]
        parts = []
        for list_part in list_parts:
            parts.extend(切分过长段(list_part))
        search_from = source_start
        for part in parts:
            unit_no = global_start + len(units) + 1
            raw_slice = text[search_from:source_end]
            relative = raw_slice.find(part)
            exact_start = search_from + relative if relative >= 0 else source_start
            exact_end = exact_start + len(part)
            search_from = exact_end
            header = current_herb if is_materia_medica else current_formula
            enriched = part
            header_type = ""
            if header and header not in part[: min(30, len(part))]:
                enriched = f"{'药名' if is_materia_medica else '方名'}：{header}\n{part}"
                header_type = "本草药名恢复" if is_materia_medica else "方名恢复"
            if current_topic_heading and current_topic_heading not in enriched[:80]:
                enriched = f"疾病门：{current_topic_heading}\n{enriched}"
                header_type = (header_type + "+疾病门恢复").strip("+")
            units.append({
                "evidence_unit_id": f"EU_{unit_no:07d}",
                "source_file": path.name,
                "book_name": re.sub(r"^\d+-", "", path.stem),
                "heading": current_heading,
                "unit_type": "本草" if is_materia_medica else ("方剂" if current_formula or 方名标题.match(part[:30]) else "条文"),
                "source_start": exact_start,
                "source_end": exact_end,
                "source_spans": [[exact_start, exact_end]],
                "source_text": part,
                "text": enriched,
                "boundary_basis": [boundary],
                "header_recovered": header_type,
                "quality_flags": (["超长条文切分"] if len(raw) > 900 else []) + (["缺少本草药名"] if is_materia_medica and not current_herb else []),
            })
        buffer = []

    for stripped_raw, line_start, line_end in lines:
        stripped = stripped_raw.strip()
        if not stripped:
            flush("空行")
            continue

        heading_match = re.match(r"^(?:篇名|章节|卷名)[：:]\s*(.+)$", stripped)
        if heading_match:
            flush("强标题")
            current_heading = stripped[:120]
            title_value = heading_match.group(1).strip()
            if is_materia_medica and title_value and not 纯目录.match(stripped):
                current_herb = title_value
            current_formula = ""
            continue

        # 目录只提供层级线索，不独立成为知识条文，也不能覆盖本草药名。
        if 纯目录.match(stripped):
            flush("目录边界")
            continue

        # “消渴门、痰饮门”等疾病门是后续正文的主题头，不作为独立条文。
        if re.fullmatch(r"[\u3400-\u9fff]{1,12}门", stripped):
            flush("疾病门标题")
            current_topic_heading = stripped
            current_heading = stripped
            current_formula = ""
            continue

        # 本草常见“内容：\n药名\n，生……”结构：药名行更新头实体，不独立输出。
        if is_materia_medica and not 本草正文提示.match(stripped) and re.fullmatch(r"[\u3400-\u9fff·]{1,16}", stripped):
            next_is_body = True
            if next_is_body and not stripped.startswith(非药名开头) and not stripped.endswith(("论", "经", "方", "曰")):
                flush("本草药名")
                current_herb = stripped
                continue

        if not is_materia_medica and 方名标题.match(stripped):
            flush("方名标题")
            current_formula = stripped.rstrip("：:")
            current_heading = stripped[:120]
            continue

        # “内容/属性/气味/主治”只是字段标签，不单独切断对应正文。
        if re.fullmatch(r"(?:内容|属性|气味|性味|主治|释名)[：:]?", stripped):
            continue

        buffer.append((stripped, line_start, line_end))
        # 以自然段为主；对方剂组成、本草条目不按短句机械切碎。
        joined_len = sum(len(x[0]) for x in buffer)
        if not is_materia_medica and not current_formula and stripped[-1] in 句末 and joined_len >= 180:
            flush("完整自然段")
        elif joined_len >= 900:
            flush("长度上限")
    flush("文件结束")
    # 合并明显被空行或版式错误截断的相邻单元，避免“羸瘦无力。小便”等残句进入抽取。
    merged, i = [], 0
    while i < len(units):
        current = dict(units[i])
        while i + 1 < len(units):
            tail = current["source_text"].rstrip()
            bracket_open = tail.count("（") + tail.count("(") > tail.count("）") + tail.count(")")
            dangling = bool(re.search(r"(?:小便|治|主治|方|用|各|或|及|与|入|加|上|下|一)$", tail))
            nxt = units[i + 1]
            same_structure = current["heading"] == nxt["heading"] and current["unit_type"] == nxt["unit_type"]
            if not same_structure or not (bracket_open or dangling) or len(current["text"]) + len(nxt["text"]) > 900:
                break
            current["source_end"] = nxt["source_end"]
            current["source_spans"].extend(nxt.get("source_spans", []))
            # source_text必须是原TXT的精确连续切片；知识文本另行规整，二者不可混用。
            current["source_text"] = text[current["source_start"]:current["source_end"]]
            header_lines = []
            for candidate_text in (current["text"], nxt["text"]):
                for line in candidate_text.splitlines():
                    if re.match(r"^(?:疾病门|药名|方名)[：:]", line.strip()):
                        header_lines.append(line.strip())
            body_clean = "\n".join(x.strip() for x in current["source_text"].splitlines() if x.strip())
            current["text"] = "\n".join([*dict.fromkeys(header_lines), body_clean]).strip()
            current["boundary_basis"].append("残缺相邻单元合并")
            current["quality_flags"] = sorted(set(current.get("quality_flags", []) + ["残缺边界已恢复"]))
            i += 1
        merged.append(current); i += 1
    units = merged
    for local_index, unit in enumerate(units, 1):
        unit["evidence_unit_id"] = f"EU_{global_start + local_index:07d}"
    return units, global_start + len(units)


def 添加有限上下文(units: list[dict]) -> None:
    for i, unit in enumerate(units):
        same_prev = i > 0 and units[i - 1]["source_file"] == unit["source_file"]
        same_next = i + 1 < len(units) and units[i + 1]["source_file"] == unit["source_file"]
        need_prev = bool(省略提示.match(unit["text"]))
        need_next = unit["unit_type"] == "方剂" or len(unit["text"]) < 35 or unit["text"].endswith(("治：", "主治：", "方："))
        unit["context_before"] = units[i - 1]["text"][-300:] if need_prev and same_prev else ""
        unit["context_after"] = units[i + 1]["text"][:500] if need_next and same_next else ""
        unit["context_recovered"] = bool(unit["context_before"] or unit["context_after"] or unit.get("header_recovered"))


def 验收重建条文(units: list[dict], input_manifest: dict) -> None:
    if not units:
        raise RuntimeError("重建条文为空")
    files = {x["source_file"] for x in units}
    expected_files = {x["name"] for x in input_manifest["files"]}
    if files != expected_files:
        missing = sorted(expected_files - files)
        extra = sorted(files - expected_files)
        raise RuntimeError(f"重建条文未覆盖全部488部古籍；缺失{missing[:10]}；异常{extra[:10]}")
    ids = [x["evidence_unit_id"] for x in units]
    if len(ids) != len(set(ids)):
        raise RuntimeError("重建条文ID不唯一")
    # 按书分组，每部古籍只读取一次；旧实现逐条读取整本书会产生数十万次磁盘访问。
    by_book: dict[str, list[dict]] = defaultdict(list)
    for row in units:
        by_book[row["source_file"]].append(row)
    print(f"开始来源回溯验收：{len(units)}条，{len(by_book)}部古籍")
    for index, (file_name, rows) in enumerate(sorted(by_book.items()), 1):
        content = (清洁古籍目录 / file_name).read_text(encoding="utf-8-sig")
        for row in rows:
            start, end = row["source_start"], row["source_end"]
            source_text = row["source_text"]
            # 优先检查记录区间，因换行规整造成不完全相等时再在全书中检索。
            if source_text not in content[max(0, start - 2): min(len(content), end + 2)] and source_text not in content:
                raise RuntimeError(f"来源无法回溯：{row['evidence_unit_id']}（{file_name}）")
        if index % 10 == 0 or index == len(by_book):
            print(f"来源验收进度：{index}/{len(by_book)}部（{index/len(by_book):.1%}）")
    print("来源回溯验收完成")


def 重建全部条文(input_manifest: dict) -> list[dict]:
    files = sorted(清洁古籍目录.glob("*.txt"))
    if len(files) != 488:
        raise RuntimeError(f"清洁古籍应为488部，实际为{len(files)}部：{清洁古籍目录}")
    units, offset = [], 0
    for index, path in enumerate(files, 1):
        book_units, offset = 重建一书(path, offset)
        units.extend(book_units)
        if index % 25 == 0 or index == len(files):
            print(f"条文重建进度：{index}/{len(files)}部；累计{len(units)}条")
    添加有限上下文(units)
    print("条文重建完成，开始全量来源验收……")
    验收重建条文(units, input_manifest)
    print(f"开始写入重建条文：{len(units)}条（文件较大，请稍候）")
    输出JSONL(证据单元缓存, units)
    print(f"重建条文写入完成：{证据单元缓存}")
    保存清单(证据单元清单, input_manifest, kind="reconstructed_units", records=len(units), source_book_count=len(set(x["source_file"] for x in units)))
    return units


def 命中词(text: str, terms: set[str]) -> list[str]:
    return sorted((term for term in terms if term in text), key=lambda x: (-len(x), x))


def 专家证据(text: str) -> dict:
    symptom_groups = {
        group: sorted((term for term in terms if term in text), key=lambda x: (-len(x), x))
        for group, terms in 专家高特异症状组.items()
    }
    symptom_groups = {k: v for k, v in symptom_groups.items() if v}
    return {
        "明确疾病": 命中词(text, 专家明确疾病锚点),
        "高歧义词": 命中词(text, 专家高歧义词),
        "下位词": 命中词(text, 专家下位词),
        "典型症状组": symptom_groups,
        "弱特异症状": 命中词(text, 专家弱特异症状),
        "证型": 命中词(text, 专家证型),
        "高特异方剂": 命中词(text, 专家高特异方剂),
        "相关方剂": 命中词(text, 专家相关方剂),
    }


def 有效疾病命中(text: str, disease_terms: set[str]) -> tuple[list[str], list[str]]:
    valid, ambiguous = [], []
    all_disease_terms = set(disease_terms) | 专家明确疾病锚点
    # 单字“渴、瘅、消”和待核“宣疾”绝不能直接成为疾病强锚点。
    for term in all_disease_terms - 专家高歧义词:
        if term not in text:
            continue
        # “方见消渴门/并见消渴门”只是跨章节索引，不等于当前正文讨论消渴。
        without_cross_ref = re.sub(r"(?:方|并|三方)?见[^。；\n]{0,12}(?:门|篇)", "", text)
        if term not in without_cross_ref:
            ambiguous.append(term)
            continue
        if any(phrase in text for phrase in 歧义排除短语 if term in phrase):
            ambiguous.append(term)
            continue
        if term in 歧义疾病词:
            # 歧义词需要标题/治病语境或消渴特异表现共同支持。
            contextual = bool(
                re.search(rf"(?:篇名|门|证|候|病|治|主治)[：:]?.{{0,8}}{re.escape(term)}|{re.escape(term)}(?:病|证|候|门)", text)
                or 消渴特异表现.search(text)
            )
            (valid if contextual else ambiguous).append(term)
        else:
            valid.append(term)
    return sorted(valid, key=lambda x: (-len(x), x)), sorted(ambiguous)


def 标题病名支持(row: dict, disease_terms: set[str], expert: dict) -> list[str]:
    """标题病名不是强锚点；只有正文同时具有诊疗表达和专题组合证据才参与召回。"""
    heading_text = "\n".join(x for x in (row.get("heading", ""), row.get("text", "").split("\n", 1)[0]) if x)
    title_hits, _ = 有效疾病命中(heading_text, disease_terms)
    body = row.get("source_text", "")
    support = bool(
        len(expert["典型症状组"]) >= 1
        or expert["证型"] or expert["高特异方剂"] or expert["相关方剂"]
        or (expert["弱特异症状"] and 消渴特异表现.search(body))
    )
    return title_hits if support and 诊疗表达.search(body) else []


def 生成候选(units: list[dict], lexicon: dict[str, set[str]], input_manifest: dict) -> list[dict]:
    candidates = []
    total_units = len(units)
    for index, unit in enumerate(units, 1):
        core = unit["text"]
        body = unit.get("source_text", core)
        evidence = {kind: 命中词(core, terms) for kind, terms in lexicon.items()}
        # 正文病名才是强锚点，恢复的标题不得冒充正文病名。
        disease, ambiguous_disease = 有效疾病命中(body, lexicon.get("疾病", set()))
        expert = 专家证据(body)
        title_disease = 标题病名支持(unit, lexicon.get("疾病", set()), expert)
        symptom_group_count = len(expert["典型症状组"])
        has_specific_formula = bool(expert["高特异方剂"])
        has_high_specific_expression = bool(re.search(r"(?:饮一溲二|尿有甜味|尿甜|小便如膏|尿如膏|溲如膏|小便如油)", body))
        # 从严门控：无病名时必须命中两个不同症状维度，或命中高度特异复合表达。
        # 普通症状+通用证型/相关方剂不再单独召回，避免“口干+六味地黄丸”等误收。
        implicit_supported = (
            symptom_group_count >= 2
            or has_high_specific_expression
            or (has_specific_formula and bool(re.search(r"(?:治|主治|主|疗)", core)))
        )
        selected = bool(disease) or bool(title_disease) or implicit_supported
        if not selected:
            continue
        # 纯目录、纯标题、纯交叉引用不直接调用LLM；应由重建环节合并，未合并者排除。
        substantive = re.sub(r"^(?:疾病门|药名|方名)[：:].*\n", "", core).strip()
        if 纯目录.match(substantive) or (len(substantive) < 25 and re.search(r"(?:见.+门|方见|篇名|目录)", substantive)):
            continue
        row = dict(unit)
        row["anchor_terms"] = disease
        row["title_anchor_terms"] = title_disease
        row["ambiguous_anchor_terms"] = ambiguous_disease
        row["expert_evidence"] = expert
        row["knowledge_evidence"] = {k: v for k, v in evidence.items() if k != "疾病" and v}
        if disease:
            row["candidate_basis"] = "正文疾病强锚点"
        elif title_disease:
            row["candidate_basis"] = "标题病名支持"
        elif symptom_group_count >= 2:
            row["candidate_basis"] = "典型症状组合"
        elif has_high_specific_expression:
            row["candidate_basis"] = "高度特异复合表现"
        elif has_specific_formula:
            row["candidate_basis"] = "消渴专方"
        else:
            row["candidate_basis"] = "其他支持"
        candidates.append(row)
        if index % 10000 == 0 or index == total_units:
            print(f"候选扫描进度：{index}/{total_units}（{index/total_units:.1%}）；累计候选{len(candidates)}条")
    print(f"开始写入候选缓存：{len(candidates)}条")
    输出JSONL(候选缓存, candidates)
    print(f"候选缓存写入完成：{候选缓存}")
    保存清单(
        候选清单, input_manifest, kind="candidates", records=len(candidates),
        source_book_count=len(set(x["source_file"] for x in candidates)),
        reconstructed_records=len(units),
        reconstructed_manifest_sha256=文件哈希(证据单元清单),
    )
    return candidates


def 提取JSON(content: str) -> dict:
    content = content.strip()
    content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.I)
    try:
        obj = json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, flags=re.S)
        if not match:
            raise ValueError("模型输出中没有JSON对象")
        obj = json.loads(match.group(0))
    if not isinstance(obj, dict):
        raise ValueError("模型输出不是JSON对象")
    return obj


def 首轮提示(row: dict) -> str:
    return f"""你是中医古籍消渴专题语料审核专家。请判断【核心条文】是否围绕消渴、三消及其明确诊疗知识展开。

专家主题知识：消渴上位属气血津液病证，下含上消、中消、下消、内伤三消、妊娠/胎前消渴和小儿消渴。典型表现包括多饮、多尿、多食善饥、消瘦，以及尿甜、尿如膏油；常见证型包括肺热津伤、胃热炽盛、气阴亏虚、肾阴亏虚、阴阳两虚。玉女煎、七味白术散、六味地黄丸、金匮肾气丸也治疗其他疾病，不能仅凭方名收录。“渴、瘅、消”是高歧义单字，不能单独证明消渴。

收录：条文能围绕消渴构建至少一项明确知识链，如疾病—症状、症状—病机、病机—病位、治法—病机、方药/中药—治疗—消渴或方药—组成。
排除：普通口渴、阴虚、肾虚、清热或常见药物；其他疾病引起的渴；只有目录/篇名/交叉引用；证据仅来自无关上下文。
消渴仅作为其他疾病辨证列表中的偶发表现，且不能独立形成消渴知识链时排除；处于其他篇章但正文明确转入或兼论消渴诊疗者可以收录。
章节主题优先：如果条文明示属于暑证、伤寒、温病、淋证、中风等其他疾病门，且正文没有明确提及消渴/三消或转为、兼有消渴，即使出现大渴引饮、烦渴、白虎汤等也必须排除。
本草条文必须包含准确药名；书名以及“叶、根、实、子、花、皮”等脱离上位药物的单字不得作为药名。方剂条文必须识别证据附近真正对应的方名；一段多方时不得沿用段首错误方名。若主体不明确，判为uncertain。
相关方剂和通用证型只能支持已经成立的消渴判断，绝不能据此反推该条属于消渴。
若核心条文超过500字，只抽取能够独立表达消渴知识链的最小连续原文片段；不得把无关长段作为抽取正文。
上下文仅用于解决“同上”、标题、方名与主治割裂，不得把上下文本身当作核心条文证据。
必须逐字复制核心条文中的最小支持证据；不得改写、补充或凭常识推断。

条文编号：{row['evidence_unit_id']}
来源：{row['source_file']}
篇章：{row['heading']}
规则召回依据：{row['candidate_basis']}
命中病名：{row['anchor_terms']}
标题支持病名：{row.get('title_anchor_terms', [])}
歧义病名提示：{row.get('ambiguous_anchor_terms', [])}
命中领域知识：{json.dumps(row['knowledge_evidence'], ensure_ascii=False)}
专家组合证据：{json.dumps(row.get('expert_evidence', {}), ensure_ascii=False)}
前置有限上下文：{row['context_before'] or '无'}
【核心条文】：{row['text']}
后置有限上下文：{row['context_after'] or '无'}

只返回一个JSON对象，所有证据必须逐字复制核心条文：
{{"evidence_unit_id":"...","decision":"include|exclude|uncertain","judgement_type":"明确病名|隐式相关|排除|不确定","subject":"本条真正的药名/方名/疾病主体，必须逐字存在于核心条文；无则空","minimum_evidence":"最小支持证据或空字符串","extraction_span":"围绕最小证据、语义完整的连续原文片段；排除时为空，最长500字","symptom_dimensions":["饮水异常/小便异常/食欲异常/尿液特征/消瘦"],"other_disease_context":"明确的其他疾病或空字符串","knowledge_chain":"可由本条构建的消渴知识链；排除时为空","reason":"简短理由","confidence":0到1}}"""


def 二轮提示(row: dict, first: dict) -> str:
    return f"""请对一条消渴古籍候选进行第二次独立复核。第一轮可能错误，不要迎合第一轮结论。
消渴典型证据包括上中下消、多饮、多尿、多食善饥、消瘦、尿甜/如膏及肺热津伤、胃热炽盛、气阴亏虚、肾阴亏虚、阴阳两虚等组合。“渴、瘅、消”是高歧义单字；玉女煎、七味白术散、六味地黄丸、金匮肾气丸不是消渴专属方。
请只裁决该条是否能构建明确的消渴知识链。相关方剂或通用证型不能反推消渴。明确属于其他疾病且未转入或兼论消渴者排除。本草/方剂主体必须是证据附近真实主体，不能用书名、单字部位或段首无关方名。长段只保留最长500字的独立连续核心片段。

条文编号：{row['evidence_unit_id']}
来源及篇章：{row['source_file']}；{row['heading']}
前文：{row['context_before'] or '无'}
核心条文：{row['text']}
后文：{row['context_after'] or '无'}
第一轮结果：{json.dumps(first, ensure_ascii=False)}

只返回JSON：
{{"evidence_unit_id":"...","decision":"include|exclude|uncertain","judgement_type":"明确病名|隐式相关|排除|不确定","subject":"核心条文中逐字存在的真实主体或空字符串","minimum_evidence":"最小原文证据或空字符串","extraction_span":"语义完整的连续原文片段；排除时为空，最长500字","symptom_dimensions":[],"other_disease_context":"其他疾病或空字符串","knowledge_chain":"消渴知识链；排除时为空","reason":"简短理由","confidence":0到1}}"""


def 调用模型(prompt: str, ann_id: str) -> tuple[dict, dict]:
    if not API密钥.strip():
        raise RuntimeError("请在脚本前部填写API密钥后再运行")
    payload = {
        "model": 模型名称,
        "messages": [
            {"role": "system", "content": "你进行古籍主题审核，并始终输出合法JSON对象。"},
            {"role": "user", "content": prompt},
        ],
        "temperature": 温度,
        "seed": 随机种子,
        "response_format": {"type": "json_object"},
    }
    last_error = None
    for attempt in range(1, 最大重试次数 + 1):
        started = time.time()
        request = urllib.request.Request(
            API地址,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Authorization": f"Bearer {API密钥.strip()}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                raw = json.loads(response.read().decode("utf-8"))
            content = raw["choices"][0]["message"]["content"]
            obj = 提取JSON(content)
            usage = raw.get("usage", {})
            return obj, {"id": ann_id, "attempt": attempt, "seconds": round(time.time() - started, 2), "usage": usage}
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            last_error = f"HTTP {exc.code}: {body[:500]}"
            if exc.code in {400, 401, 403} and "data_inspection_failed" not in body:
                break
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        time.sleep(min(2 ** attempt, 30))
    raise RuntimeError(f"{ann_id}调用失败：{last_error}")


def API连通性预检() -> None:
    """每次程序启动只验证一次密钥，避免失效密钥对全部待处理记录逐条报错。"""
    global API预检已通过
    if API预检已通过:
        return
    try:
        obj, _ = 调用模型('请只返回JSON对象：{"status":"ok"}', "API预检")
    except Exception as exc:
        raise RuntimeError(
            "API密钥预检失败，尚未开始处理待审核条文。请在脚本前部填写新的有效密钥后重新运行。"
            f"\n原始错误：{exc}"
        ) from exc
    if not isinstance(obj, dict):
        raise RuntimeError("API预检返回格式异常，请重新运行")
    API预检已通过 = True
    print("API密钥与模型连通性预检：通过")


def 校验LLM(obj: dict, row: dict) -> dict:
    if obj.get("evidence_unit_id") != row["evidence_unit_id"]:
        raise ValueError("返回条文编号不一致")
    decision = str(obj.get("decision", "")).lower()
    if decision not in {"include", "exclude", "uncertain"}:
        raise ValueError(f"非法decision：{decision}")
    evidence = str(obj.get("minimum_evidence") or "").strip()
    extraction_span = str(obj.get("extraction_span") or "").strip()
    subject = str(obj.get("subject") or "").strip()
    knowledge_chain = str(obj.get("knowledge_chain") or "").strip()
    other_disease = str(obj.get("other_disease_context") or "").strip()
    symptom_dimensions = obj.get("symptom_dimensions") or []
    if not isinstance(symptom_dimensions, list):
        symptom_dimensions = []
    ignored = re.compile(r"[\s，。；：！？、,.!?;:'\"“”‘’（）()【】\[\]《》<>—…·-]")
    def 定位原文(fragment: str, label: str) -> tuple[str, str]:
        if not fragment:
            return "", "exact"
        if fragment in row["text"]:
            return fragment, "exact"
        target = ignored.sub("", fragment)
        chars, positions = [], []
        for pos, char in enumerate(row["text"]):
            if not ignored.fullmatch(char):
                chars.append(char); positions.append(pos)
        normalized = "".join(chars)
        if not target or target not in normalized:
            # 模型若改写了原文，不能把改写文本作为证据；降级为疑难样本，
            # 而不是作为API调用异常反复请求同一条。
            return "", "unmatched"
        start = normalized.index(target)
        return row["text"][positions[start]:positions[start + len(target) - 1] + 1], "normalized"
    evidence, evidence_match_mode = 定位原文(evidence, "最小证据")
    extraction_span, extraction_match_mode = 定位原文(extraction_span, "抽取片段")
    if len(extraction_span) > 500:
        extraction_span = ""
        extraction_match_mode = "overlength"
    try:
        confidence = float(obj.get("confidence", 0))
    except (TypeError, ValueError):
        confidence = 0.0
    if not 0 <= confidence <= 1:
        confidence = 0.0
    if evidence_match_mode == "unmatched" or extraction_match_mode in {"unmatched", "overlength"}:
        decision = "uncertain"
        confidence = min(confidence, 0.5)
    if decision == "include" and not evidence:
        decision = "uncertain"
        confidence = min(confidence, 0.5)
    if decision == "include" and (not extraction_span or not knowledge_chain):
        decision = "uncertain"
        confidence = min(confidence, 0.5)
    if decision == "include" and evidence not in extraction_span:
        decision = "uncertain"
        confidence = min(confidence, 0.5)
    if decision == "exclude" and knowledge_chain:
        decision = "uncertain"
        confidence = min(confidence, 0.5)
    # 防止把书名或脱离上位主体的部位单字当成本草药名。
    if decision == "include" and row.get("unit_type") == "本草" and (
        subject in {"叶", "根", "实", "子", "花", "皮"}
        or subject == row.get("book_name")
        or not subject
    ):
        decision = "uncertain"
        confidence = min(confidence, 0.5)
    if decision == "include" and row.get("unit_type") in {"本草", "方剂"} and subject not in row["text"]:
        decision = "uncertain"
        confidence = min(confidence, 0.5)
    # 明确他病优先于隐式症状。只有正文病名锚点或明确兼论消渴才允许保留。
    explicit_other = bool(other_disease or re.search(r"(?:黄疸|疸症|水病|欲作水|伤寒|霍乱|痢|痘|疝|淋|瘴|鼻衄)", row.get("heading", "") + row["text"][:180]))
    if decision == "include" and explicit_other and not row.get("anchor_terms") and not re.search(r"(?:兼|转为|又为|名曰)(?:消渴|三消|上消|中消|下消)", row["text"]):
        decision = "exclude"
        confidence = max(confidence, 0.9)
        evidence = extraction_span = knowledge_chain = ""
    if decision == "include" and row.get("unit_type") == "本草" and (
        "缺少本草药名" in row.get("quality_flags", []) or not re.search(r"^药名：", row["text"])
    ):
        decision = "uncertain"
        confidence = min(confidence, 0.5)
    if decision == "include" and (纯目录.match(row["text"].strip()) or len(row["text"].strip()) < 12):
        decision = "uncertain"
        confidence = min(confidence, 0.5)
    if decision == "include" and 非消渴疾病门.search(row["text"].split("\n", 1)[0]) and not row.get("anchor_terms"):
        decision = "exclude"
        confidence = max(confidence, 0.9)
        evidence = ""
    reason = str(obj.get("reason") or "")[:500]
    conclusion_conflict = bool(
        (decision == "exclude" and re.search(r"(?:符合.*(?:收录|纳入)|应(?:当)?收录|支持.*消渴知识链)", reason))
        or (decision == "include" and re.search(r"(?:不符合|应排除|无法构建|不能构建)", reason))
    )
    return {
        "evidence_unit_id": row["evidence_unit_id"],
        "decision": decision,
        "judgement_type": str(obj.get("judgement_type") or "不确定"),
        "minimum_evidence": evidence,
        "extraction_span": extraction_span,
        "subject": subject,
        "symptom_dimensions": [str(x) for x in symptom_dimensions],
        "other_disease_context": other_disease,
        "knowledge_chain": knowledge_chain,
        "evidence_match_mode": evidence_match_mode,
        "extraction_match_mode": extraction_match_mode,
        "reason": reason,
        "conclusion_conflict": conclusion_conflict,
        "confidence": confidence,
    }


def 审核阶段(name: str, rows: list[dict], cache: Path, prompt_builder) -> list[dict]:
    existing = {
        x["evidence_unit_id"]: x for x in 读取JSONL(cache)
        if x.get("decision") in {"include", "exclude", "uncertain"}
        and x.get("judgement_type") != "调用异常"
    }
    # 专家终审优先于模型，仅在首轮写入一次可追溯的排除结果。
    if name == "LLM首轮":
        for row in rows:
            evidence_id = row["evidence_unit_id"]
            if evidence_id in 人工审核排除条文 and evidence_id not in existing:
                reviewed = {
                    "evidence_unit_id": evidence_id, "decision": "exclude",
                    "judgement_type": "人工审核排除", "minimum_evidence": "",
                    "extraction_span": "", "subject": "", "symptom_dimensions": [],
                    "other_disease_context": "强中/前阴诸疾", "knowledge_chain": "",
                    "evidence_match_mode": "manual", "extraction_match_mode": "manual",
                    "reason": 人工审核排除条文[evidence_id],
                    "conclusion_conflict": False, "confidence": 1.0,
                    "review_source": "专家人工审核",
                }
                追加JSONL(cache, reviewed)
                existing[evidence_id] = reviewed
                print(f"人工审核结论已写入：{evidence_id} → 不纳入")
    pending = [row for row in rows if row["evidence_unit_id"] not in existing]
    print(f"{name}：总数{len(rows)}；已完成{len(rows)-len(pending)}；待处理{len(pending)}；并发{并发线程数}")
    if pending:
        API连通性预检()

    def worker(row: dict) -> dict:
        try:
            obj, log = 调用模型(prompt_builder(row), row["evidence_unit_id"])
            result = 校验LLM(obj, row)
            result["call_log"] = log
            return result
        except Exception as exc:
            if "data_inspection_failed" in str(exc) or "inappropriate content" in str(exc):
                # 古籍中的生殖、创伤等合法医学描述可能触发云端内容审核。
                # 此类记录保存为可审计的未决项，不再无限重试，也不自动纳入正式库。
                error = {"stage": name, "evidence_unit_id": row["evidence_unit_id"], "error": str(exc), "time": datetime.now(北京时间).isoformat()}
                追加JSONL(异常缓存, error)
                return {
                    "evidence_unit_id": row["evidence_unit_id"], "decision": "uncertain",
                    "judgement_type": "内容审核拦截", "minimum_evidence": "",
                    "extraction_span": "", "subject": "", "symptom_dimensions": [],
                    "other_disease_context": "", "knowledge_chain": "",
                    "evidence_match_mode": "blocked", "extraction_match_mode": "blocked",
                    "reason": "云端模型内容审核拦截，保留为人工审核项，不自动纳入正式库。",
                    "conclusion_conflict": False, "confidence": 0.0,
                }
            # 单条失败不终止全量运行；异常只写日志，不写完成缓存，重启后自动重试。
            error = {"stage": name, "evidence_unit_id": row["evidence_unit_id"], "error": str(exc), "time": datetime.now(北京时间).isoformat()}
            追加JSONL(异常缓存, error)
            return {"evidence_unit_id": row["evidence_unit_id"], "decision": "uncertain", "judgement_type": "调用异常", "minimum_evidence": "", "reason": str(exc)[:500], "confidence": 0.0, "_do_not_cache": True}

    done = len(rows) - len(pending)
    failed_ids = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=并发线程数) as pool:
        future_map = {pool.submit(worker, row): row for row in pending}
        for future in concurrent.futures.as_completed(future_map):
            result = future.result()
            existing[result["evidence_unit_id"]] = result
            if result.pop("_do_not_cache", False):
                failed_ids.append(result["evidence_unit_id"])
            else:
                追加JSONL(cache, result)
            done += 1
            print(f"{name}进度：{done}/{len(rows)}（{done / max(1,len(rows)):.1%}）")
    # 不允许带着调用异常进入下一阶段；成功记录已经落盘，重新运行只补跑失败项。
    if failed_ids:
        preview = "、".join(failed_ids[:10])
        raise RuntimeError(
            f"{name}仍有{len(failed_ids)}条调用异常（如：{preview}）。"
            "成功结果均已保存，请直接重新运行本脚本，程序将只补跑失败记录。"
        )
    # LLM缓存只逐条追加，不在阶段结束时替换文件，避免OneDrive文件锁；读取时按ID自然去重。
    return [existing[row["evidence_unit_id"]] for row in rows]


def 分层试运行(candidates: list[dict], limit: int) -> list[dict]:
    if limit >= len(candidates):
        return candidates
    rng = random.Random(随机种子)
    groups = defaultdict(list)
    for row in candidates:
        if row.get("candidate_basis") == "标题病名支持":
            key = "标题病名支持"
        elif row["anchor_terms"]:
            key = "正文明确病名"
        elif row.get("unit_type") == "本草":
            key = "本草隐式"
        elif row.get("unit_type") == "方剂":
            key = "方剂隐式"
        elif row.get("ambiguous_anchor_terms"):
            key = "歧义病名"
        elif row.get("quality_flags"):
            key = "结构风险"
        else:
            key = row.get("candidate_basis", "其他隐式")
        groups[key].append(row)
    sample = []
    keys = sorted(groups)
    while len(sample) < limit and any(groups.values()):
        for key in keys:
            if groups[key] and len(sample) < limit:
                idx = rng.randrange(len(groups[key]))
                sample.append(groups[key].pop(idx))
    return sorted(sample, key=lambda x: x["evidence_unit_id"])


def 规范去重文本(text: str) -> str:
    return re.sub(r"[\s，。；：！？、,.!?;:'\"“”‘’（）()【】\[\]《》<>—…·-]+", "", text)


def 构建知识文本(row: dict, chosen: dict) -> tuple[list[str], str, str]:
    """固定为最小主语补全：只补缺失的真实药名/方名，不加入疾病门或结构标签。"""
    span = chosen.get("extraction_span", "").strip()
    subject = chosen.get("subject", "").strip()
    # 删除模型截取片段中遗留的结构标签；内容本身保持原文。
    span = re.sub(r"^(?:(?:疾病门|方名|药名)[：:][^\n]*\n)+", "", span).strip()
    span = re.sub(r"^(?:属性|内容)[：:]\s*", "", span).strip()
    semantic_headers = []
    heading_value = re.sub(r"^(?:篇名|章节|卷名)[：:]\s*", "", row.get("heading", "")).strip()
    # 篇章本身是方名时，它比模型自由返回的症状焦点更可靠，直接作为最小主语。
    if re.search(r"(?:汤|丸|散|膏|丹|饮|煎|方|剂|酒|露)$", heading_value):
        subject = heading_value
    subject_is_structure = bool(
        subject and (
            row.get("unit_type") in {"方剂", "本草"}
            or subject == heading_value
            or bool(re.search(r"(?:汤|丸|散|膏|丹|饮|煎|方|剂|酒|露)$", subject))
        )
    )
    if subject_is_structure and subject not in span[:80]:
        semantic_headers = [subject]
        knowledge_text = f"{subject}。{span}".strip()
    else:
        knowledge_text = span
    # 保留“主治行＋组成行”等有效分行，只修复句中错误断行。
    # 前一行以句末标点结束时保留一个换行；语义未完成时直接拼接。
    raw_lines = [x.strip() for x in re.split(r"\r?\n", knowledge_text) if x.strip()]
    normalized_text = raw_lines[0] if raw_lines else ""
    for next_line in raw_lines[1:]:
        separator = "\n" if normalized_text.rstrip().endswith(tuple("。！？；：")) else ""
        normalized_text += separator + next_line
    # “@@@”是部分电子古籍遗留的版式占位符，不承载医学语义；仅从对外知识文本移除，
    # 原始source_text和字符位置保持不变，仍可严格回溯。
    knowledge_text = re.sub(r"@{2,}", "", normalized_text).strip()
    return semantic_headers, span, knowledge_text


def 最终决策(row: dict, first: dict, second: dict | None) -> str:
    if second is None:
        return first["decision"]
    # 二轮仅针对疑难；两轮一致才自动收录，冲突或仍不确定不进入正式库。
    if first["decision"] == second["decision"]:
        return first["decision"]
    return "uncertain"


def 去重并成库(candidates: list[dict], first_results: list[dict], second_results: list[dict]) -> tuple[list[dict], dict]:
    first_map = {x["evidence_unit_id"]: x for x in first_results}
    second_map = {x["evidence_unit_id"]: x for x in second_results}
    included = []
    unresolved = 0
    for row in candidates:
        first = first_map[row["evidence_unit_id"]]
        second = second_map.get(row["evidence_unit_id"])
        decision = 最终决策(row, first, second)
        if decision == "uncertain":
            unresolved += 1
        if decision != "include":
            continue
        chosen = second if second and second["decision"] == "include" else first
        # 方剂/本草主体由结构头约束；模型负责核验，不允许用疾病或症状替代。
        if row.get("unit_type") in {"方剂", "本草"}:
            label = "方名" if row["unit_type"] == "方剂" else "药名"
            match = re.search(rf"^{label}[：:]\s*([^\n]+)", row["text"])
            structural_subject = match.group(1).strip() if match else ""
            if structural_subject:
                chosen = dict(chosen)
                chosen["subject"] = structural_subject
        out = dict(row)
        out["llm_judgement_type"] = chosen["judgement_type"]
        out["llm_minimum_evidence"] = chosen["minimum_evidence"]
        out["subject"] = chosen.get("subject", "")
        out["semantic_context"], out["extraction_text"], out["knowledge_text"] = 构建知识文本(row, chosen)
        out["knowledge_chain"] = chosen.get("knowledge_chain", "")
        out["symptom_dimensions"] = chosen.get("symptom_dimensions", [])
        out["other_disease_context"] = chosen.get("other_disease_context", "")
        out["llm_confidence"] = chosen["confidence"]
        out["llm_second_reviewed"] = second is not None
        included.append(out)

    # 书内和跨书统一按规范化全文去重，仅保留来源编号靠前且边界较完整的一条。
    grouped = defaultdict(list)
    for row in included:
        grouped[规范去重文本(row["knowledge_text"])].append(row)
    final = []
    within, cross = 0, 0
    for _, group in grouped.items():
        group.sort(key=lambda x: (len(x["extraction_text"]), x["source_file"], x["source_start"]))
        keep = group[0]
        books = {x["source_file"] for x in group}
        if len(group) > 1:
            if len(books) == 1:
                within += len(group) - 1
            else:
                cross += len(group) - 1
        keep["duplicate_removed"] = len(group) - 1
        final.append(keep)
    final.sort(key=lambda x: (x["source_file"], x["source_start"], x["evidence_unit_id"]))
    for index, row in enumerate(final, 1):
        row["record_id"] = f"XK_{index:06d}"
        # 删除运行期冗余字段，正式库保留可追溯和后续抽取需要的内容。
        row.pop("context_before", None)
        row.pop("context_after", None)
    return final, {"书内重复删除": within, "跨书重复删除": cross, "未决未纳入": unresolved}


def 验收正式库(rows: list[dict]) -> None:
    required = {"record_id", "evidence_unit_id", "source_file", "source_start", "source_end", "source_text", "text", "extraction_text", "semantic_context", "knowledge_text", "subject", "knowledge_chain", "llm_minimum_evidence"}
    ids, source_keys, normalized = set(), set(), set()
    source_cache = {}
    total = len(rows)
    for index, row in enumerate(rows, 1):
        missing = required - row.keys()
        if missing:
            raise RuntimeError(f"{row.get('record_id')}缺字段：{missing}")
        if row["record_id"] in ids:
            raise RuntimeError(f"重复record_id：{row['record_id']}")
        ids.add(row["record_id"])
        key = (row["source_file"], row["source_start"], row["source_end"])
        if key in source_keys:
            raise RuntimeError(f"重复来源位置：{key}")
        source_keys.add(key)
        norm = 规范去重文本(row["knowledge_text"])
        if norm in normalized:
            raise RuntimeError(f"去重后仍有重复正文：{row['record_id']}")
        normalized.add(norm)
        source = 清洁古籍目录 / row["source_file"]
        if not source.exists():
            raise RuntimeError(f"来源文件不存在：{source}")
        if row["source_file"] not in source_cache:
            source_cache[row["source_file"]] = source.read_text(encoding="utf-8-sig")
        source_content = source_cache[row["source_file"]]
        if row["source_text"] not in source_content:
            raise RuntimeError(f"来源正文无法回溯：{row['record_id']}")
        if not row["llm_minimum_evidence"] or row["llm_minimum_evidence"] not in row["text"]:
            raise RuntimeError(f"LLM证据无效：{row['record_id']}")
        if not row["extraction_text"] or row["extraction_text"] not in row["text"] or len(row["extraction_text"]) > 500:
            raise RuntimeError(f"抽取正文无效：{row['record_id']}")
        # 原文中的错误换行可能已在展示正文中修复，证据包含关系按字符内容等价判断。
        flatten = lambda s: re.sub(r"@{2,}", "", re.sub(r"\s+", "", s)).strip()
        evidence = row["llm_minimum_evidence"].strip()
        # 最小证据允许携带“疾病门/方名/药名”等结构上下文；验收时剥离元数据行，
        # 再核对剩余实质证据。若证据只有结构标题，则抽取正文必须自身含消渴病名锚点。
        substantive_lines = []
        has_structural_evidence = False
        for evidence_line in re.split(r"\r?\n", evidence):
            evidence_line = evidence_line.strip()
            if not evidence_line:
                continue
            if (
                re.fullmatch(r"(?:疾病门|方名|药名|篇名|章节|卷名)[：:][^\n]+", evidence_line)
                or re.fullmatch(r"(?:疾病门[：:])?消渴门", evidence_line)
            ):
                has_structural_evidence = True
                continue
            substantive_lines.append(re.sub(r"^(?:属性|内容)[：:]\s*", "", evidence_line))
        substantive_evidence = "".join(substantive_lines)
        evidence_valid_for_extraction = (
            (substantive_evidence and flatten(substantive_evidence) in flatten(row["extraction_text"]))
            or (not substantive_evidence and has_structural_evidence)
        )
        if not evidence_valid_for_extraction:
            raise RuntimeError(f"最小证据未包含在抽取正文：{row['record_id']}")
        # 只修复错误断行；忽略换行后仍须完整包含核心原文。
        # 验收只比较字符内容；段内换行及其两侧版式空白视为等价。
        if flatten(row["extraction_text"]) not in flatten(row["knowledge_text"]):
            raise RuntimeError(f"知识文本没有完整包含抽取正文：{row['record_id']}")
        if index % 1000 == 0 or index == total:
            print(f"正式库验收进度：{index}/{total}（{index/max(1,total):.1%}）")


def 写正式结果(rows: list[dict], summary: dict) -> None:
    # 对外JSONL采用固定数据契约；详细审核字段只留在运行中缓存，不进入正式语料库。
    public_rows = [{
        "record_id": row["record_id"],
        "source": row["book_name"],
        "source_file": row["source_file"],
        "chapter": row["heading"],
        "text": row["knowledge_text"],
        "source_start": row["source_start"],
        "source_end": row["source_end"],
    } for row in rows]
    输出JSONL(正式JSONL, public_rows)
    blocks = []
    for row in rows:
        blocks.extend([
            f"条文编号：{row['record_id']}",
            f"来源：{row['book_name']}（{row['source_file']}）",
            f"篇章：{row['heading']}",
            f"完整条文：{row['knowledge_text']}", "",
        ])
    安全写文本(正式TXT, "\n".join(blocks), encoding="utf-8-sig")
    安全写文本(运行摘要, json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass
    if 运行模式 not in {"试运行", "正式运行"}:
        raise RuntimeError("运行模式只能是‘试运行’或‘正式运行’")
    if not API密钥.strip():
        raise RuntimeError("请先在脚本前部填写API密钥")
    结果目录.mkdir(parents=True, exist_ok=True)
    过程目录.mkdir(parents=True, exist_ok=True)

    print("================ 消渴主题条文筛选（正式版） ================")
    print(f"模式：{运行模式}；模型：{模型名称}；并发：{并发线程数}")
    print("阶段1/6：读取并冻结消渴概念词表")
    manifest = 输入指纹()
    lexicon = 读取词表xlsx(词表路径)
    print("；".join(f"{k}{len(v)}个表面词" for k, v in lexicon.items()))
    print(f"疾病强锚点：{len(lexicon['疾病'])}个：{'、'.join(sorted(lexicon['疾病']))}")

    print("阶段2/6：古籍感知条文重建与有限上下文恢复")
    if 证据单元缓存.exists() != 证据单元清单.exists():
        print("检测到重建阶段在缓存与清单之间中止，自动清除不完整的可再生缓存并重建。")
        证据单元缓存.unlink(missing_ok=True); 证据单元清单.unlink(missing_ok=True)
    if 证据单元缓存.exists():
        unit_manifest = 校验清单(证据单元清单, manifest, "重建条文")
        units = 读取JSONL(证据单元缓存)
        if len(units) != unit_manifest.get("records") or len({x["source_file"] for x in units}) != 488:
            raise RuntimeError("重建条文缓存数量或古籍覆盖与清单不一致")
        验收重建条文(units, manifest)
    else:
        units = 重建全部条文(manifest)
    print(f"证据单元：{len(units)}条")

    print("阶段3/6：领域词表高召回候选生成")
    if 候选缓存.exists() != 候选清单.exists():
        print("检测到候选阶段在缓存与清单之间中止，自动清除不完整的可再生缓存并重建。")
        候选缓存.unlink(missing_ok=True); 候选清单.unlink(missing_ok=True)
    if 候选缓存.exists():
        candidate_manifest = 校验清单(候选清单, manifest, "候选")
        if candidate_manifest.get("reconstructed_records") != len(units) or candidate_manifest.get("reconstructed_manifest_sha256") != 文件哈希(证据单元清单):
            raise RuntimeError("候选缓存与当前重建条文不是同一运行批次")
        candidates = 读取JSONL(候选缓存)
        if len(candidates) != candidate_manifest.get("records"):
            raise RuntimeError("候选缓存数量与清单不一致")
    else:
        candidates = 生成候选(units, lexicon, manifest)
    working = 分层试运行([dict(x) for x in candidates], 试运行数量) if 运行模式 == "试运行" else candidates
    print(f"全量候选：{len(candidates)}条；本次审核：{len(working)}条")

    # 试运行与正式运行使用独立缓存，避免30条结果污染正式全量版本。
    suffix = "_试运行" if 运行模式 == "试运行" else ""
    first_cache = 首轮缓存.with_name(首轮缓存.stem + suffix + 首轮缓存.suffix)
    second_cache = 二轮缓存.with_name(二轮缓存.stem + suffix + 二轮缓存.suffix)

    print("阶段4/6：通义千问全量候选主题复核")
    first = 审核阶段("LLM首轮", working, first_cache, 首轮提示)
    first_map = {x["evidence_unit_id"]: x for x in first}
    difficult = [
        row for row in working
        if first_map[row["evidence_unit_id"]].get("judgement_type") != "内容审核拦截"
        and (
        first_map[row["evidence_unit_id"]]["decision"] == "uncertain"
        or first_map[row["evidence_unit_id"]]["confidence"] < 低置信阈值
        or first_map[row["evidence_unit_id"]].get("conclusion_conflict", False)
        or len(row.get("text", "")) > 500
        or bool(row.get("ambiguous_anchor_terms"))
        or (first_map[row["evidence_unit_id"]]["decision"] == "include" and len(first_map[row["evidence_unit_id"]]["minimum_evidence"]) < 4)
        )
    ]
    print(f"阶段5/6：疑难二次复核，共{len(difficult)}条")
    second = 审核阶段(
        "LLM二轮", difficult, second_cache,
        lambda row: 二轮提示(row, first_map[row["evidence_unit_id"]]),
    ) if difficult else []

    print("阶段6/6：最终判定、书内/跨书去重与验收")
    final, dedup_stats = 去重并成库(working, first, second)
    final, manual_stats = 应用人工审核(final)
    验收正式库(final)
    decision_counts = Counter(x["decision"] for x in first)
    summary = {
        "generated_at": datetime.now(北京时间).isoformat(),
        "version": "Xiaoke Corpus FINAL 1.0",
        "run_mode": 运行模式,
        "prompt_version": 提示词版本,
        "expert_knowledge_version": 专家知识版本,
        "model": 模型名称,
        "temperature": 温度,
        "seed": 随机种子,
        "clean_books": 488,
        "evidence_units": len(units),
        "all_candidates": len(candidates),
        "reviewed_this_run": len(working),
        "first_round_decisions": dict(decision_counts),
        "second_round_reviewed": len(difficult),
        "final_records": len(final),
        **dedup_stats,
        **manual_stats,
        "note": "试运行结果不是正式语料库；正式200条金标准未参与候选规则和阈值设定。",
    }
    if 运行模式 == "试运行":
        global 正式JSONL, 正式TXT, 运行摘要
        正式JSONL = 结果目录 / "试运行消渴主题条文库.jsonl"
        正式TXT = 结果目录 / "试运行消渴主题条文库.txt"
        运行摘要 = 结果目录 / "试运行构建摘要.json"
    写正式结果(final, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"结果：{正式JSONL}")


if __name__ == "__main__":
    main()
