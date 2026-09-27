"""
把城市攻略 PDF 转成 app/knowledge_base/guides/{city}.json

默认只预览，不写入向量库。确认切片合理后再 --save。
多城市：文件名带城市即可，例如 杭州旅游攻略.pdf

用法（项目根目录）:
  .venv\\Scripts\\python.exe scripts\\ingest_guide_pdf.py --pdf 上海旅游攻略.pdf --city 上海
  .venv\\Scripts\\python.exe scripts\\ingest_guide_pdf.py --pdf 上海旅游攻略.pdf --city 上海 --save
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def _load_parser():
    path = os.path.join(ROOT, "app", "rag", "pdf_guide_parser.py")
    spec = importlib.util.spec_from_file_location("pdf_guide_parser", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def preview(docs, stats, city: str) -> None:
    print("=" * 60)
    print(f"城市: {city}")
    print(f"页数: {stats['pages']}  空页: {stats['empty_pages']}  字符: {stats['chars']}")
    print(f"资料年份: {stats.get('source_year') or '未知'}")
    print(f"切出文档块: {stats['documents']}")
    print("章节字符数:")
    for name, n in (stats.get("sections") or {}).items():
        print(f"  - {name}: {n}")
    print("=" * 60)
    shown = {}
    for doc in docs:
        sec = doc["metadata"]["section"]
        if sec in shown:
            continue
        shown[sec] = True
        print(f"\n[{sec}] {doc['metadata']['title']}")
        print(doc["content"][:280].replace("\n", " / "))
        print("...")


def main() -> int:
    parser = argparse.ArgumentParser(description="攻略 PDF → guides 知识库")
    parser.add_argument("--pdf", required=True, help="PDF 路径")
    parser.add_argument("--city", required=True, help="城市名，如 上海")
    parser.add_argument("--save", action="store_true", help="写入 app/knowledge_base/guides/{city}.json")
    args = parser.parse_args()

    pdf_path = args.pdf
    if not os.path.isabs(pdf_path):
        pdf_path = os.path.join(ROOT, pdf_path)
    if not os.path.exists(pdf_path):
        print(f"找不到 PDF: {pdf_path}")
        return 1

    guide = _load_parser()
    docs, stats = guide.parse_guide_pdf(pdf_path, args.city)
    preview(docs, stats, args.city)

    if not args.save:
        print("\n未写入文件。确认切片可用后加 --save")
        return 0

    out_dir = os.path.join(ROOT, "app", "knowledge_base", "guides")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"{args.city}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)
    print(f"\n已保存 {len(docs)} 条 → {out}")
    print("注意：默认不写入 Chroma。景点主库仍用 attractions JSON；攻略是补充层。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
