"""
把城市旅游攻略 PDF 转成 guides 知识库条目。

不写入 attractions：PDF 排版杂、票价易过时，适合作为「攻略切片」
（交通/行程/餐饮/玩法），检索时用 city + doc_type=guide。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from pypdf import PdfReader

def _chunk_text(text: str, chunk_size: int = 700, overlap: int = 80) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks

SECTION_ORDER = [
    "概述",
    "亮点",
    "有问必答",
    "行前准备",
    "交通",
    "景点",
    "活动",
    "住宿",
    "餐饮",
    "购物",
    "实用信息",
    "推荐行程",
    "更多路线",
    "背景",
    "跟着他们去旅行",
]

_HEADER_NOISE = re.compile(
    r"^(.{1,8}[·•]攻略|目录\s*[·.]\s*CATALOG|关于\s*[·.]\s*ABOUT|"
    r"印象\s*[·.]\s*IMPRESSION|亮点\s*[·.]\s*HIGH\s*LIGHTS)$"
)
_PAGE_NUM = re.compile(r"^\d{1,3}$")
_PHOTO_CREDIT = re.compile(r"(by\s+\S+|摄影[:：]|图[:：])", re.I)

_SECTION_LINE = re.compile(
    r"^(?P<name>"
    + "|".join(re.escape(s) for s in SECTION_ORDER)
    + r")(?:\s*[·.．]\s*\w+)?(?:\s+\d{1,2})?$"
)


def extract_pdf_text(pdf_path: str) -> Tuple[str, Dict]:
    reader = PdfReader(pdf_path)
    pages = []
    empty = 0
    for i, page in enumerate(reader.pages):
        text = (page.extract_text() or "").strip()
        if not text:
            empty += 1
        pages.append(text)
    meta = reader.metadata or {}
    created = str(getattr(meta, "creation_date", None) or meta.get("/CreationDate") or "")
    year = ""
    m = re.search(r"20\d{2}", created)
    if m:
        year = m.group(0)
    stats = {
        "pages": len(pages),
        "empty_pages": empty,
        "chars": sum(len(p) for p in pages),
        "source_year": year,
        "title": str(meta.get("/Title") or Path(pdf_path).stem),
    }
    joined = "\n\n".join(pages)
    return joined, stats


def _clean_lines(raw: str) -> List[str]:
    lines = []
    for line in raw.splitlines():
        text = line.strip()
        if not text:
            if lines and lines[-1] != "":
                lines.append("")
            continue
        if _PAGE_NUM.match(text) or _HEADER_NOISE.match(text):
            continue
        if _PHOTO_CREDIT.search(text) and len(text) < 40:
            continue
        lines.append(text)
    return lines


def split_sections(raw: str) -> List[Dict[str, str]]:
    lines = _clean_lines(raw)
    sections: List[Dict[str, str]] = []
    current = "概述"
    buf: List[str] = []

    def flush():
        body = "\n".join(buf).strip()
        if body:
            sections.append({"section": current, "text": body})

    for line in lines:
        compact = re.sub(r"\s+", "", line)
        matched = None
        m = _SECTION_LINE.match(line) or _SECTION_LINE.match(compact)
        if m:
            matched = m.group("name")
        else:
            for name in SECTION_ORDER:
                if compact.startswith(name) and len(compact) <= len(name) + 8:
                    matched = name
                    break
        if matched and matched != current:
            flush()
            current = matched
            buf = []
            continue
        buf.append(line)
    flush()
    return sections


def _split_attraction_blocks(section_text: str) -> List[Tuple[str, str]]:
    """景点章节再按「短标题 + 门票/地址」切成小块。"""
    lines = section_text.splitlines()
    blocks: List[Tuple[str, str]] = []
    title = "景点综述"
    buf: List[str] = []

    def flush():
        body = "\n".join(buf).strip()
        if body:
            blocks.append((title, body))

    for line in lines:
        stripped = line.strip()
        looks_like_title = (
            2 <= len(stripped) <= 16
            and "：" not in stripped
            and ":" not in stripped
            and not stripped.endswith("。")
            and re.fullmatch(r"[\u4e00-\u9fffA-Za-z0-9·、]+", stripped)
        )
        if looks_like_title and buf and len("".join(buf)) > 40:
            flush()
            title = stripped
            buf = []
            continue
        buf.append(line)
    flush()
    return blocks or [("景点综述", section_text)]


def sections_to_documents(
    city: str,
    pdf_name: str,
    stats: Dict,
    sections: List[Dict[str, str]],
    chunk_size: int = 700,
    chunk_overlap: int = 80,
) -> List[Dict]:
    docs: List[Dict] = []
    year = stats.get("source_year") or ""

    for sec in sections:
        section = sec["section"]
        pieces: List[Tuple[str, str]]
        if section == "景点":
            pieces = _split_attraction_blocks(sec["text"])
        else:
            pieces = [(section, sec["text"])]

        for title, body in pieces:
            chunks = _chunk_text(body, chunk_size=chunk_size, overlap=chunk_overlap) or [body]
            for idx, chunk in enumerate(chunks):
                text = chunk.strip()
                if len(text) < 40:
                    continue
                content = (
                    f"【{city}旅行攻略 · {section}】{title}\n"
                    f"{text}"
                )
                if year:
                    content += f"\n（资料年份约 {year}，价格与班次请以最新信息为准）"
                docs.append(
                    {
                        "content": content,
                        "metadata": {
                            "title": title,
                            "city": city,
                            "category": "攻略",
                            "doc_type": "guide",
                            "section": section,
                            "source": pdf_name,
                            "source_year": year,
                            "chunk_index": idx,
                            "rating": 0,
                            "price": 0,
                            "tags": f"攻略,{section}",
                        },
                    }
                )
    return docs


def parse_guide_pdf(pdf_path: str, city: str) -> Tuple[List[Dict], Dict]:
    raw, stats = extract_pdf_text(pdf_path)
    sections = split_sections(raw)
    docs = sections_to_documents(city, Path(pdf_path).name, stats, sections)
    stats["sections"] = {s["section"]: len(s["text"]) for s in sections}
    stats["documents"] = len(docs)
    return docs, stats
