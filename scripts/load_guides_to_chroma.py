"""
把 app/knowledge_base/guides/*.json 写入独立的 Chroma collection（默认 travel_guides）。
不触碰景点库 travel_knowledge。

用法（项目根目录）:
  .venv\\Scripts\\python.exe scripts\\load_guides_to_chroma.py
  .venv\\Scripts\\python.exe scripts\\load_guides_to_chroma.py --city 上海
"""

from __future__ import annotations

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))


def main() -> int:
    parser = argparse.ArgumentParser(description="攻略 JSON → 独立 Chroma collection")
    parser.add_argument("--city", default="", help="只导入该城市；默认导入 guides 目录全部")
    parser.add_argument("--rebuild", action="store_true", help="先删除攻略 collection 再写入")
    args = parser.parse_args()

    from app.rag.data_loader import KnowledgeBaseLoader
    from app.rag.vector_store import get_guide_vector_store

    os.environ["VECTOR_BACKEND"] = os.getenv("VECTOR_BACKEND", "chroma")
    store = get_guide_vector_store()
    print(f"攻略 collection: {store.default_collection}")
    print(f"persist: {getattr(store, 'persist_directory', '')}")

    attraction_name = os.getenv("CHROMA_COLLECTION", "travel_knowledge")
    if store.default_collection == attraction_name:
        print("拒绝写入：攻略 collection 名不能和景点库相同")
        return 1

    loader = KnowledgeBaseLoader()
    cities = [args.city] if args.city else [
        name[:-5]
        for name in os.listdir(os.path.join(PROJECT_ROOT, "app", "knowledge_base", "guides"))
        if name.endswith(".json")
    ]
    if not cities:
        print("guides 目录没有 JSON")
        return 1

    docs = []
    for city in cities:
        city_docs = loader.load_guide_data(city)
        print(f"  {city}: {len(city_docs)} 条")
        docs.extend(city_docs)

    if not docs:
        print("没有可写入的攻略")
        return 1

    if args.rebuild and store.collection_exists():
        print(f"删除攻略集合 {store.default_collection} ...")
        store.delete_collection()

    print(f"写入 {len(docs)} 条到 {store.default_collection} ...")
    store.add_documents(docs)
    stats = store.get_collection_stats()
    print(f"攻略库统计: {stats}")

    probe = store.search("上海三天怎么玩 城隍庙 外滩", filters={"city": "上海"}, k=3)
    print(f"试检索 {len(probe)} 条:")
    for i, item in enumerate(probe, 1):
        md = item.get("metadata") or {}
        print(f"  {i}. [{md.get('section')}] {md.get('title')} doc_type={md.get('doc_type')}")

    from app.rag.vector_store import get_vector_store

    attraction_store = get_vector_store()
    print(
        f"景点库仍是: {attraction_store.default_collection} "
        f"count={attraction_store.get_collection_stats().get('vector_count')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
