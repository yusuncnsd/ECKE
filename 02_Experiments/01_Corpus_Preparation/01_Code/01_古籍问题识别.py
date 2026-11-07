from __future__ import annotations

import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


项目目录 = Path(__file__).resolve().parent.parent
原始古籍目录 = 项目目录 / "02_Input_Data/488_Original_Books"
人工审核清单 = 项目目录 / "04_Results/古籍疑难项_人工审核清单.txt"

# 只识别可复核的数字化载体残留；不使用金标准，也不判断中医语义。
规则 = {
    "图片文件占位符": re.compile(
        r"(?i)(?<![A-Za-z0-9_])[\\/]?[A-Za-z0-9_-]{1,80}\.(?:bmp|jpe?g|png|gif|tiff?|wmf|emf)(?![A-Za-z0-9_])"
    ),
    "字面转义换行": re.compile(r"\\(?:r|n|t)"),
    "字面转义x": re.compile(r"\\x(?:[0-9A-Fa-f]{2})?"),
    "网页实体": re.compile(r"&(?:nbsp|amp|lt|gt|quot|apos|#\d+|#x[0-9A-Fa-f]+);"),
    "网址": re.compile(r"(?i)(?:https?://|www\.)[^\s<>]+"),
    "电子邮箱": re.compile(r"(?i)\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b"),
    "可疑字母数字串": re.compile(
        r"(?i)(?<![A-Za-z0-9])[A-Za-z]{1,20}\d{1,10}(?:[._-][A-Za-z0-9]+)*(?![A-Za-z0-9])"
    ),
    "替换字符": re.compile("\ufffd"),
    "疑似乱码词": re.compile(r"锟斤拷|烫烫烫|屯屯屯|鈥|銆|鐨|鏄|绔|�"),
    "零宽字符": re.compile("[\u200b-\u200f\u202a-\u202e\u2060\ufeff]"),
    "XML_HTML标签": re.compile(r"</?[A-Za-z][^>\r\n]{0,100}>"),
    "中文结构标签": re.compile(r"<\s*(?:篇名|目录|属性|内容|书名|卷名|章节)\s*>"),
}


def 读取文本(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig", errors="replace")


def 上下文(text: str, start: int, end: int, radius: int = 35) -> str:
    return text[max(0, start - radius): min(len(text), end + radius)].replace("\r", "↵").replace("\n", "↵")


def 识别问题(显示示例: bool = True) -> dict:
    files = sorted(原始古籍目录.glob("*.txt"))
    if len(files) != 489:
        raise RuntimeError(f"原始古籍应为489部，实际为{len(files)}部：{原始古籍目录}")

    counts: Counter[str] = Counter()
    affected: dict[str, set[str]] = defaultdict(set)
    examples: dict[str, list[dict]] = defaultdict(list)
    category_counts: Counter[str] = Counter()

    for index, path in enumerate(files, 1):
        text = 读取文本(path)
        for ch in text:
            category = unicodedata.category(ch)
            if category in {"Cc", "Cs", "Co"} and ch not in "\r\n\t":
                category_counts[category] += 1

        for name, pattern in 规则.items():
            matches = list(pattern.finditer(text))
            counts[name] += len(matches)
            if matches:
                affected[name].add(path.name)
            for match in matches:
                if len(examples[name]) < 12:
                    examples[name].append({
                        "文件": path.name,
                        "命中": match.group(0),
                        "上下文": 上下文(text, match.start(), match.end()),
                    })
        if index % 50 == 0 or index == len(files):
            print(f"问题识别进度：{index}/{len(files)}（{index / len(files):.1%}）")

    report = {
        "古籍数量": len(files),
        "问题计数": dict(counts),
        "涉及文件数": {name: len(paths) for name, paths in affected.items()},
        "Unicode异常类别": dict(category_counts),
        "问题示例": dict(examples),
        "处理原则": {
            "自动清除": ["图片文件占位符", "字面转义残留", "网页实体/网址/邮箱", "控制及零宽字符"],
            "保留文字仅去标签": ["篇名", "目录", "属性", "内容", "书名", "卷名", "章节"],
            "仅报告不猜改": ["疑似乱码汉字", "无法由原文确定的OCR错字", "疑似缺文"],
        },
    }
    if 显示示例:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def 生成疑难项审核清单() -> None:
    """输出需由人工定规则的原文位置；不修改任何古籍。"""
    # 可疑字母数字串包含图片名的词干，因此先排除完整图片占位符。
    suspect_rules = {
        "A_可疑字母数字串（不含图片占位符）": 规则["可疑字母数字串"],
        "B_替换字符或典型乱码词": re.compile(r"\ufffd|锟斤拷|烫烫烫|屯屯屯|鈥|銆|鐨|鏄|绔"),
        "C_Unicode私用区字符": re.compile(r"[\ue000-\uf8ff]"),
        "D_孤立或连续反斜杠": re.compile(r"\\+"),
        "E_异常长无标点行": re.compile(r"^.{500,}$"),
        "F_连续异常标点": re.compile(r"[，。；：！？、]{4,}|[.]{4,}|[-_=*#]{4,}"),
    }
    grouped: dict[str, list[dict]] = defaultdict(list)
    totals: Counter[str] = Counter()
    category_limits = {
        "A_可疑字母数字串（不含图片占位符）": 300,
        "B_替换字符或典型乱码词": 100,
        "C_Unicode私用区字符": 100,
        "D_孤立或连续反斜杠": 40,
        "E_异常长无标点行": 40,
        "F_连续异常标点": 100,
    }

    for path in sorted(原始古籍目录.glob("*.txt")):
        text = 读取文本(path)
        lines = text.splitlines()
        offset = 0
        for line_no, line in enumerate(lines, 1):
            for name, pattern in suspect_rules.items():
                for match in pattern.finditer(line):
                    hit = match.group(0)
                    if name.startswith("A_"):
                        # 完整图片占位符属于确定删除项，不再要求人工判断。
                        span_start, span_end = match.span()
                        window = line[max(0, span_start - 2): min(len(line), span_end + 8)]
                        if 规则["图片文件占位符"].search(window):
                            continue
                    totals[name] += 1
                    # 每类保留足以制定规则的代表样本；稀有乱码与私用字全部保留。
                    same_hit = sum(1 for item in grouped[name] if item["命中"] == hit)
                    if len(grouped[name]) < category_limits[name] and same_hit < 5:
                        shown_hit = hit
                        if name.startswith("E_"):
                            shown_hit = f"异常长行（{len(line)}字符）"
                        elif len(shown_hit) > 120:
                            shown_hit = shown_hit[:60] + "……" + shown_hit[-40:]
                        if len(line) <= 160:
                            shown_context = line
                        elif name.startswith("E_"):
                            shown_context = line[:75] + " …… " + line[-75:]
                        else:
                            shown_context = line[max(0, match.start() - 60): min(len(line), match.end() + 60)]
                        grouped[name].append({
                            "文件": path.name,
                            "行号": line_no,
                            "命中": shown_hit,
                            "上下文": shown_context,
                        })
            offset += len(line) + 1

    certain_counts: Counter[str] = Counter()
    certain_files: dict[str, set[str]] = defaultdict(set)
    for path in sorted(原始古籍目录.glob("*.txt")):
        text = 读取文本(path)
        for name in ["图片文件占位符", "字面转义换行", "字面转义x", "网页实体", "网址", "电子邮箱", "零宽字符"]:
            matches = list(规则[name].finditer(text))
            certain_counts[name] += len(matches)
            if matches:
                certain_files[name].add(path.name)

    out = []
    out.append("489部中医古籍疑难项人工审核清单")
    out.append("=" * 50)
    out.append("说明：本清单只用于制定清洗规则，不修改原始古籍。行号以原始TXT为准。")
    out.append("")
    out.append("一、可确定为数字化载体残留（建议统一自动处理）")
    for name, count in certain_counts.items():
        out.append(f"{name}：{count}处，涉及{len(certain_files[name])}部古籍")
    out.append("")
    out.append("二、需要人工确定规则的疑难项")
    out.append("建议在每一类标题后批注：删除／替换为何值／保留／仅标记。")
    out.append("")
    for name, items in grouped.items():
        out.append(f"【{name}】全量命中{totals[name]}处；以下列出不超过{category_limits[name]}个代表出处")
        if not items:
            out.append("未发现。")
        for index, item in enumerate(items, 1):
            out.append(f"{index}. 出处：{item['文件']}，第{item['行号']}行")
            out.append(f"   命中：{item['命中']}")
            out.append(f"   上下文：{item['上下文']}")
            out.append("   人工决定：____________________")
        out.append("")
    out.append("三、无法通过字符模式可靠识别的项目")
    out.append("OCR同形错字、古字/异体字、原书脱文、句读错误无法仅凭正则安全修复；建议在下游证据单元中标质量标记，不由清洗程序猜改正文。")
    人工审核清单.write_text("\n".join(out) + "\n", encoding="utf-8-sig")
    print(f"已生成：{人工审核清单}")
    print(f"疑难类别：{len(grouped)}；清单条目：{sum(len(x) for x in grouped.values())}")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass
    print("================ 489部古籍数字化问题识别 ================")
    report = 识别问题(显示示例=False)
    print("问题识别汇总：")
    for name, count in report["问题计数"].items():
        if count:
            print(f"  {name}：{count}处，涉及{report['涉及文件数'][name]}部")
    生成疑难项审核清单()
