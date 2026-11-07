from __future__ import annotations

import hashlib
import html
import re
import shutil
import unicodedata
from collections import Counter
from pathlib import Path


项目目录 = Path(__file__).resolve().parent.parent
原始目录 = 项目目录 / "02_Input_Data/488_Original_Books"
输出目录 = 项目目录 / "04_Results/复现运行/488_Cleaned_Books"
临时目录 = 项目目录 / "04_Results/复现运行/488部清洁古籍_生成中"

# 用户人工审核后冻结的规则（2026-08-13）：
# 1. 异常长行与连续标点保留；2. 数字化图片名、字母数字乱码、私用字删除；
# 3. 反斜杠和字面转义只恢复版面边界；4. 英文/全角标点规范为中文标点；
# 5. 不猜测、不补写OCR缺字，不改写古籍语义。

图片占位符 = re.compile(
    r"(?i)(?<![A-Za-z0-9_])[\\/]?[A-Za-z0-9_-]{1,100}\.(?:bmp|jpe?g|png|gif|tiff?|wmf|emf)(?![A-Za-z0-9_])"
)
字母数字乱码 = re.compile(
    r"(?i)(?<![A-Za-z0-9])[A-Za-z]{1,24}\d{1,12}(?:[._-][A-Za-z0-9]+)*(?![A-Za-z0-9])"
)
网址 = re.compile(r"(?i)(?:https?://|www\.)[^\s<>]+")
邮箱 = re.compile(r"(?i)\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")


def 读取(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig", errors="replace")


def 规范标点(text: str, stats: Counter[str]) -> str:
    """处理中英文/全角标点；保护数字小数点，不切分或重写长句。"""
    # 先把Unicode全角ASCII归一，中文标点随后统一恢复。
    normalized = unicodedata.normalize("NFKC", text)
    if normalized != text:
        stats["全角字符识别规范化"] += 1
    text = normalized

    # 仅替换正文中的ASCII标点；数字之间的点/逗号保留，防止破坏小数和数值格式。
    replacements = {
        ";": "；", ":": "：", "?": "？", "!": "！",
        "(": "（", ")": "）", "[": "【", "]": "】",
    }
    for old, new in replacements.items():
        count = text.count(old)
        if count:
            text = text.replace(old, new)
            stats["英文标点转中文"] += count

    text, n = re.subn(r"(?<!\d),(?!\d)|(?<=\D),(?=\d)|(?<=\d),(?=\D)", "，", text)
    stats["英文标点转中文"] += n
    text, n = re.subn(r"(?<!\d)\.(?!\d)|(?<=\D)\.(?=\d)|(?<=\d)\.(?=\D)", "。", text)
    stats["英文标点转中文"] += n

    # 常见混合标点统一；不压缩用户明确要求保留的连续标点。
    text = text.replace("｡", "。").replace("､", "、").replace("﹐", "，").replace("﹑", "、")
    text = text.replace("﹒", "。").replace("﹔", "；").replace("﹕", "：")
    text = text.replace("﹖", "？").replace("﹗", "！")
    return text


def 清洗(text: str) -> tuple[str, Counter[str]]:
    stats: Counter[str] = Counter()
    text = text.lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n")

    # 必须先删除完整图片名，避免后续只删词干而残留扩展名。
    text, n = 图片占位符.subn("", text)
    stats["图片文件占位符"] += n

    # \r、\n、\x及反斜杠是旧电子书版面分隔符，转成边界而非拼接前后正文。
    text, n1 = re.subn(r"\\(?:r|n)", "\n", text)
    text, n2 = re.subn(r"\\t", " ", text)
    text, n3 = re.subn(r"\\x(?:[0-9A-Fa-f]{2})?", "\n", text)
    text, n4 = re.subn(r"\\+", "\n", text)
    stats["转义与反斜杠分隔符"] += n1 + n2 + n3 + n4

    decoded = html.unescape(text)
    if decoded != text:
        stats["网页实体"] += 1
    text = decoded
    text, n = 网址.subn("", text); stats["网址"] += n
    text, n = 邮箱.subn("", text); stats["邮箱"] += n

    # XML/HTML载体标签删除；中文结构标签保留语义名称并转为清晰行边界。
    text, n = re.subn(r"<\s*(篇名|目录|属性|内容|书名|卷名|章节)\s*>", r"\n\1：", text)
    stats["中文结构标签"] += n
    text, n = re.subn(r"</?[A-Za-z][^>\r\n]{0,120}>", "", text)
    stats["XML_HTML标签"] += n

    # 按人工代表标注推广：所有“字母+数字”嵌入乱码删除。
    text, n = 字母数字乱码.subn("", text)
    stats["字母数字乱码"] += n

    # 人工标注确认删除替换字符、典型乱码词与Unicode私用字。
    text, n1 = re.subn(r"\ufffd|锟斤拷|烫烫烫|屯屯屯|鈥|銆|鐨|鏄|绔", "", text)
    text, n2 = re.subn(r"[\ue000-\uf8ff]", "", text)
    stats["乱码及私用字"] += n1 + n2

    # 删除不可见控制、代理与零宽排版符，不删除换行/制表符。
    chars = []
    removed = 0
    for ch in text:
        cat = unicodedata.category(ch)
        if ch in "\n\t":
            chars.append(ch)
        elif cat in {"Cc", "Cs", "Co"} or ch in "\u200b\u200c\u200d\u200e\u200f\u202a\u202b\u202c\u202d\u202e\u2060\ufeff":
            removed += 1
        else:
            chars.append(ch)
    text = "".join(chars)
    stats["控制零宽字符"] += removed

    text = 规范标点(text, stats)

    # NFKC可能把全角反斜杠或全角字母数字还原为新的可识别残留，末端二次清除。
    text, n1 = 图片占位符.subn("", text)
    text, n2 = re.subn(r"\\(?:r|n)", "\n", text)
    text, n3 = re.subn(r"\\t", " ", text)
    text, n4 = re.subn(r"\\x(?:[0-9A-Fa-f]{2})?", "\n", text)
    text, n5 = re.subn(r"\\+", "\n", text)
    text, n6 = 字母数字乱码.subn("", text)
    stats["末端二次残留清除"] += n1 + n2 + n3 + n4 + n5 + n6

    # 仅规整空白，不拆分长句、不压缩连续标点。
    text = re.sub(r"[ \u3000\t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"
    if not text.strip():
        raise ValueError("清洗后正文为空")
    return text, stats


def 验收(files: list[Path]) -> None:
    residual = {
        "图片占位符": 图片占位符,
        "字面转义": re.compile(r"\\(?:r|n|t|x(?:[0-9A-Fa-f]{2})?)"),
        "字母数字乱码": 字母数字乱码,
        "替换/私用字符": re.compile(r"\ufffd|[\ue000-\uf8ff]"),
        "控制零宽字符": re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\u200b-\u200f\u202a-\u202e\u2060\ufeff]"),
    }
    errors = []
    hashes: dict[str, list[str]] = {}
    for path in files:
        text = 读取(path)
        for name, pattern in residual.items():
            match = pattern.search(text)
            if match:
                errors.append(f"{path.name}仍含{name}：{match.group(0)!r}")
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        hashes.setdefault(digest, []).append(path.name)
    duplicate_groups = [names for names in hashes.values() if len(names) > 1]
    if errors:
        raise RuntimeError("验收失败：\n" + "\n".join(errors[:30]))
    if duplicate_groups:
        raise RuntimeError(f"清洗后出现全文重复古籍：{duplicate_groups[:5]}")


def main() -> None:
    print("================ 488部中医古籍深度清洗 ================")
    files = sorted(原始目录.glob("*.txt"))
    if len(files) != 488:
        raise RuntimeError(f"输入应为488部TXT，实际为{len(files)}部：{原始目录}")
    if 临时目录.exists():
        shutil.rmtree(临时目录)
    临时目录.mkdir()
    total: Counter[str] = Counter()
    try:
        for index, source in enumerate(files, 1):
            cleaned, stats = 清洗(读取(source))
            (临时目录 / source.name).write_text(cleaned, encoding="utf-8")
            total.update(stats)
            if index % 20 == 0 or index == len(files):
                print(f"清洗进度：{index}/{len(files)}（{index / len(files):.1%}）")
        generated = sorted(临时目录.glob("*.txt"))
        if len(generated) != 488:
            raise RuntimeError(f"输出应为488部，实际为{len(generated)}部")
        print("正在执行全量残留、空文件和重复文件验收……")
        验收(generated)
        if 输出目录.exists():
            shutil.rmtree(输出目录)
        临时目录.rename(输出目录)
    except Exception:
        if 临时目录.exists():
            shutil.rmtree(临时目录)
        raise

    print("\n清洗与验收完成：")
    print(f"输入古籍：{len(files)}部；输出古籍：{len(list(输出目录.glob('*.txt')))}部")
    for name, count in total.items():
        print(f"{name}：{count}处")
    print(f"输出目录：{输出目录}")


if __name__ == "__main__":
    main()
