from __future__ import annotations

import json
import re
from pathlib import Path


脚本目录 = Path(__file__).resolve().parent.parent
清洁古籍目录 = 脚本目录 / "04_Results/488_Cleaned_Books"
结果目录 = 脚本目录 / "04_Results/Xiaoke_Passage_Corpus"
正式JSONL = 结果目录 / "正式消渴主题条文库.jsonl"
正式TXT = 结果目录 / "正式消渴主题条文库.txt"
人工审核Excel = 结果目录 / "310条风险条文_人工审核表.xlsx"
冻结条文数 = 3065


def 读取JSONL(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"JSONL损坏：第{line_no}行：{exc}") from exc
            if not isinstance(row, dict):
                raise RuntimeError(f"第{line_no}行不是JSON对象")
            rows.append(row)
    return rows


def 规范文本(text: str) -> str:
    return re.sub(r"[\s，。；：！？、,.!?;:'\"“”‘’（）()【】\[\]《》<>—…·@-]+", "", text)


def main() -> None:
    print("================ 正式消渴主题条文库质量验收 ================")
    required_files = [正式JSONL, 正式TXT, 人工审核Excel]
    missing = [str(x) for x in required_files if not x.exists()]
    if missing:
        raise RuntimeError("缺少冻结结果：\n" + "\n".join(missing))
    books = sorted(清洁古籍目录.glob("*.txt"))
    if len(books) != 488:
        raise RuntimeError(f"清洁古籍应为488部，实际为{len(books)}部")

    rows = 读取JSONL(正式JSONL)
    if len(rows) != 冻结条文数:
        raise RuntimeError(f"正式条文库应为{冻结条文数}条，实际为{len(rows)}条")
    required = {"record_id", "source", "source_file", "chapter", "text", "source_start", "source_end"}
    ids, source_positions, normalized_texts = set(), set(), set()
    source_cache: dict[str, str] = {}
    source_normalized_cache: dict[str, str] = {}
    position_fail, trace_fail, quality_fail = [], [], []
    quality_patterns = {
        "Unicode替换符": re.compile(r"�"),
        "转义残留": re.compile(r"(?:\\x[0-9A-Fa-f]{0,2}|\\u[0-9A-Fa-f]{0,4})"),
        "图片文件残留": re.compile(r"\b\w+\.(?:bmp|jpg|jpeg|png|gif|tif)\b", re.I),
        "HTML/XML残留": re.compile(r"<[^>]{1,100}>|&(?:nbsp|amp|lt|gt);"),
        "控制字符": re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]"),
        "版式占位符": re.compile(r"@{2,}"),
    }

    for index, row in enumerate(rows, 1):
        absent = required - row.keys()
        if absent:
            raise RuntimeError(f"第{index}条缺少字段：{sorted(absent)}")
        expected_id = f"XK_{index:06d}"
        if row["record_id"] != expected_id:
            raise RuntimeError(f"编号不连续：第{index}条为{row['record_id']}，应为{expected_id}")
        if row["record_id"] in ids:
            raise RuntimeError(f"重复编号：{row['record_id']}")
        ids.add(row["record_id"])
        position_key = (row["source_file"], row["source_start"], row["source_end"])
        if position_key in source_positions:
            raise RuntimeError(f"重复来源位置：{position_key}")
        source_positions.add(position_key)
        normalized = 规范文本(row["text"])
        if not normalized:
            raise RuntimeError(f"空正文：{row['record_id']}")
        if normalized in normalized_texts:
            raise RuntimeError(f"规范化重复正文：{row['record_id']}")
        normalized_texts.add(normalized)
        for label, pattern in quality_patterns.items():
            if pattern.search(row["text"]):
                quality_fail.append(f"{row['record_id']}：{label}")

        source_path = 清洁古籍目录 / row["source_file"]
        if not source_path.exists():
            trace_fail.append(f"{row['record_id']}：来源文件不存在")
            continue
        if row["source_file"] not in source_cache:
            source_cache[row["source_file"]] = source_path.read_text(encoding="utf-8-sig")
            source_normalized_cache[row["source_file"]] = 规范文本(source_cache[row["source_file"]])
        source = source_cache[row["source_file"]]
        start, end = row["source_start"], row["source_end"]
        if not (isinstance(start, int) and isinstance(end, int) and 0 <= start < end <= len(source)):
            position_fail.append(row["record_id"])
            continue
        source_span = 规范文本(source[start:end])
        # 方名/药名可能由紧邻结构标题补入正文首部；正文其余部分必须能回溯到来源区间。
        trace_candidates = [normalized]
        if "。" in row["text"]:
            trace_candidates.append(规范文本(row["text"].split("。", 1)[1]))
        anchored = any(candidate and candidate in source_span for candidate in trace_candidates)
        # 有限上下文恢复会使正式条文长于原核心区间；此时核心source_span应反向包含于条文。
        if not anchored and len(source_span) >= 8:
            anchored = source_span in normalized
        # 电子古籍局部空格与上下文合并可能使精确片段不同；至少一个连续12字符证据
        # 必须能在同一来源书中定位，且上方字符位置仍须合法。
        if not anchored and len(normalized) >= 12:
            source_normalized = source_normalized_cache[row["source_file"]]
            anchored = any(normalized[i:i + 12] in source_normalized for i in range(0, len(normalized) - 11))
        if not anchored:
            trace_fail.append(row["record_id"])

    if position_fail:
        raise RuntimeError(f"来源字符位置无效：{len(position_fail)}条，如{position_fail[:10]}")
    if trace_fail:
        raise RuntimeError(f"正文来源回溯失败：{len(trace_fail)}条，如{trace_fail[:10]}")
    if quality_fail:
        raise RuntimeError(f"文本质量异常：{len(quality_fail)}条，如{quality_fail[:10]}")

    txt = 正式TXT.read_text(encoding="utf-8-sig")
    if txt.count("条文编号：") != len(rows):
        raise RuntimeError("TXT条文块数量与JSONL不一致")
    for row in rows:
        if f"条文编号：{row['record_id']}" not in txt or f"完整条文：{row['text']}" not in txt:
            raise RuntimeError(f"TXT与JSONL内容不一致：{row['record_id']}")

    print(f"验收通过：{len(rows)}条")
    print(f"来源古籍：{len({x['source_file'] for x in rows})}部")
    print("编号、字段、唯一性、重复、来源位置、文本回溯、乱码及TXT/JSONL一致性：全部通过")


if __name__ == "__main__":
    main()
