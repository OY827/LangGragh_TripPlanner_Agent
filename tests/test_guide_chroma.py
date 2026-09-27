"""
攻略 collection 手工测试（与景点库 travel_knowledge 隔离）

用法（项目根目录）:
  .\\.venv\\Scripts\\python.exe tests/test_guide_chroma.py
  .\\.venv\\Scripts\\python.exe tests/test_guide_chroma.py --city 上海
"""

from __future__ import annotations

import argparse
import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
except ImportError:
    pass

os.environ.setdefault("VECTOR_BACKEND", "chroma")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")


def _print_hits(results, limit=5):
    if not results:
        print("    (无结果)")
        return
    for i, item in enumerate(results[:limit], 1):
        meta = item.get("metadata") or {}
        score = item.get("score")
        score_txt = f"{float(score):.4f}" if score is not None else "n/a"
        print(
            f"    {i}. [{meta.get('section') or meta.get('category')}] "
            f"{meta.get('title')}  city={meta.get('city')}  "
            f"doc_type={meta.get('doc_type')}  source={meta.get('source')}  "
            f"score={score_txt}"
        )
        snippet = (item.get("content") or "").replace("\n", " ")[:90]
        if snippet:
            print(f"       {snippet}...")


def main() -> int:
    parser = argparse.ArgumentParser(description="测试独立攻略 Chroma")
    parser.add_argument("--city", default="上海")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()
    city = args.city
    top_k = args.top_k
    queries = [
        f"{city}三天怎么玩",
        f"{city} 外滩 城隍庙",
        f"{city} 美食 小笼包",
    ]

    from app.rag.retriever import KnowledgeRetriever
    from app.rag.vector_store import get_guide_vector_store, get_vector_store

    guide_store = get_guide_vector_store()
    attraction_store = get_vector_store()
    print("=" * 64)
    print("collection 隔离")
    print("=" * 64)
    g_stats = guide_store.get_collection_stats()
    a_stats = attraction_store.get_collection_stats()
    print(f"  攻略: {g_stats.get('collection_name')}  count={g_stats.get('vector_count')}")
    print(f"  景点: {a_stats.get('collection_name')}  count={a_stats.get('vector_count')}")
    if g_stats.get("collection_name") == a_stats.get("collection_name"):
        print("  [fail] 攻略和景点用了同一个 collection")
        return 2
    if not g_stats.get("vector_count"):
        print("  [fail] 攻略库为空，先跑 scripts/load_guides_to_chroma.py")
        return 2

    retriever = KnowledgeRetriever()
    ok = True

    for query in queries:
        print("\n" + "=" * 64)
        print(f"retrieve_guides  query={query!r}  city={city}")
        print("=" * 64)
        t0 = time.time()
        hits = retriever.retrieve_guides(query, city=city, top_k=top_k)
        print(f"  耗时 {time.time() - t0:.2f}s  条数={len(hits)}")
        _print_hits(hits, top_k)
        bad_type = [h for h in hits if (h.get("metadata") or {}).get("doc_type") != "guide"]
        bad_city = [h for h in hits if (h.get("metadata") or {}).get("city") not in (None, city)]
        if not hits:
            print("  [fail] 无攻略命中")
            ok = False
        if bad_type:
            print("  [fail] 混入非 guide 文档")
            ok = False
        if bad_city:
            print("  [fail] 城市过滤泄漏")
            ok = False
        if hits and not bad_type and not bad_city:
            print("  [ok] 只命中攻略层")

    print("\n" + "=" * 64)
    print(f"对照：景点库 search  query={city!r} 外滩  city={city}")
    print("=" * 64)
    t0 = time.time()
    poi_hits = attraction_store.search(f"{city} 外滩", filters={"city": city}, k=top_k)
    print(f"  耗时 {time.time() - t0:.2f}s  条数={len(poi_hits)}")
    _print_hits(poi_hits, top_k)
    leaked_guide = [h for h in poi_hits if (h.get("metadata") or {}).get("doc_type") == "guide"]
    if leaked_guide:
        print("  [fail] 景点检索打到了攻略")
        ok = False
    elif poi_hits:
        print("  [ok] 景点库未混入攻略")
    else:
        print("  [warn] 景点库该城市无命中（不影响攻略层）")

    print("\n" + "=" * 64)
    print(f"旅行攻略工具  city={city}  theme=三天行程")
    print("=" * 64)
    from app.rag.rag_tools import GuideRecommendTool

    tool = GuideRecommendTool(retriever=retriever)
    t0 = time.time()
    text = tool.invoke({"city": city, "theme": "三天行程"})
    print(f"  耗时 {time.time() - t0:.2f}s")
    print(text[:1200])
    if "暂无" in text or "出错" in text:
        print("  [fail] 工具未返回攻略")
        ok = False
    else:
        print("  [ok] 工具返回攻略文本")

    print("\n" + "=" * 64)
    print("通过" if ok else "含失败项")
    print("=" * 64)
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
