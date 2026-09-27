"""
本地 Chroma RAG 手工测试脚本

默认只测向量库检索（需要 DashScope Embedding）。
可选：--generate 走 LLM 生成；--hybrid 走 HybridRAG（查询改写）。

用法（在项目根目录）:
  .\\.venv\\Scripts\\python.exe tests/test_chroma_rag.py
  .\\.venv\\Scripts\\python.exe tests/test_chroma_rag.py --city 杭州 --query 西湖
  .\\.venv\\Scripts\\python.exe tests/test_chroma_rag.py --generate
  .\\.venv\\Scripts\\python.exe tests/test_chroma_rag.py --hybrid
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
        title = meta.get("title", "未知")
        city = meta.get("city", "")
        category = meta.get("category", "")
        score = item.get("score")
        rerank = item.get("rerank_score")
        extra = f" rerank={float(rerank):.3f}" if rerank is not None else ""
        score_txt = f"{float(score):.4f}" if score is not None else "n/a"
        loc = meta.get("location") or {}
        print(f"    {i}. {title}  city={city}  cat={category}  score={score_txt}{extra}")
        if loc:
            print(f"       loc={loc.get('longitude')},{loc.get('latitude')}")
        snippet = (item.get("content") or "").replace("\n", " ")[:80]
        if snippet:
            print(f"       {snippet}...")


def test_store(city: str, query: str, top_k: int):
    from app.rag.vector_store import get_vector_store

    print("\n" + "=" * 64)
    print("1) 本地 Chroma 集合")
    print("=" * 64)
    store = get_vector_store()
    stats = store.get_collection_stats()
    for key in ("backend", "collection_name", "vector_count", "dimension", "persist_directory"):
        print(f"  {key}: {stats.get(key)}")

    if stats.get("error") or not stats.get("vector_count"):
        print("  [fail] 集合为空。请先运行: python rebuild_kb.py")
        return None, False

    ok = True
    print("\n" + "=" * 64)
    print(f"2) 向量检索  query={query!r}  filter city={city}")
    print("=" * 64)
    t0 = time.time()
    vector_hits = store.search(query, filters={"city": city}, k=top_k)
    print(f"  耗时 {time.time() - t0:.2f}s  条数={len(vector_hits)}")
    _print_hits(vector_hits, top_k)
    leaked = [h for h in vector_hits if (h.get("metadata") or {}).get("city") not in (None, city)]
    if leaked:
        print(f"  [fail] 城市过滤泄漏: {[h['metadata'].get('title') for h in leaked]}")
        ok = False
    elif not vector_hits:
        print("  [fail] 无检索结果")
        ok = False
    else:
        print("  [ok] 城市过滤通过")

    print("\n" + "=" * 64)
    print("3) BM25 关键词检索")
    print("=" * 64)
    t0 = time.time()
    bm25_hits = store.bm25_search(query, filters={"city": city}, k=top_k)
    print(f"  耗时 {time.time() - t0:.2f}s  条数={len(bm25_hits)}")
    _print_hits(bm25_hits, top_k)

    print("\n" + "=" * 64)
    print("4) Hybrid = 向量 * 0.5 + BM25 * 0.5")
    print("=" * 64)
    t0 = time.time()
    hybrid_hits = store.hybrid_search(query, filters={"city": city}, k=top_k, vector_weight=0.5)
    print(f"  耗时 {time.time() - t0:.2f}s  条数={len(hybrid_hits)}")
    _print_hits(hybrid_hits, top_k)

    return store, ok


def test_retriever(store, city: str, query: str, top_k: int):
    from app.rag.retriever import KnowledgeRetriever

    retriever = KnowledgeRetriever(store)
    print("\n" + "=" * 64)
    print("5) KnowledgeRetriever similarity / mmr / retrieve_attractions")
    print("=" * 64)

    print("\n  -- similarity --")
    hits = retriever.retrieve(query, filters={"city": city}, top_k=top_k, search_type="similarity")
    _print_hits(hits, top_k)

    print("\n  -- mmr --")
    hits = retriever.retrieve(f"{city} 景点推荐", filters={"city": city}, top_k=top_k, search_type="mmr")
    _print_hits(hits, top_k)

    print("\n  -- retrieve_attractions(历史文化) --")
    hits = retriever.retrieve_attractions(
        city=city,
        preferences=["历史文化"],
        travel_days=2,
        search_type="mmr",
    )
    _print_hits(hits, top_k)
    return retriever


def test_generate(retriever, city: str, query: str):
    from app.rag.generator import AnswerGenerator

    print("\n" + "=" * 64)
    print("6) Retrieve → AnswerGenerator（会调用 LLM）")
    print("=" * 64)
    docs = retriever.retrieve(query, filters={"city": city}, top_k=3)
    if not docs:
        print("  [skip] 没有检索结果，无法生成")
        return
    generator = AnswerGenerator(provider="aliyun")
    t0 = time.time()
    answer = generator.generate(query=query, context=docs)
    print(f"  耗时 {time.time() - t0:.2f}s")
    print("  --- 回答 ---")
    print(answer)


def test_hybrid_rag(city: str, query: str, enable_rerank: bool):
    from app.rag.hybrid_rag import HybridRAG, HybridRAGConfig

    print("\n" + "=" * 64)
    print("7) HybridRAG retrieve_with_validation（查询改写 + 多路检索）")
    print("=" * 64)
    config = HybridRAGConfig(
        enable_query_rewrite=True,
        enable_retrieval_validation=False,
        enable_answer_validation=False,
        enable_rerank=enable_rerank,
        rerank_model="BAAI/bge-reranker-base",
        retrieval_top_k=12,
        rerank_top_k=5,
        top_k=5,
        max_retrieval_rounds=1,
        use_mmr=False,
    )
    rag = HybridRAG(config=config)
    t0 = time.time()
    docs, meta = rag.retrieve_with_validation(query=query, city=city, preferences=["历史文化"])
    print(f"  耗时 {time.time() - t0:.2f}s")
    print(f"  variants={meta.get('query_variants')}")
    print(f"  rounds={meta.get('retrieval_rounds')} total_docs={meta.get('total_docs')}")
    print(f"  timing={meta.get('timing')}")
    _print_hits(docs, 5)


def parse_args():
    parser = argparse.ArgumentParser(description="测试本地 Chroma RAG")
    parser.add_argument("--city", default="北京")
    parser.add_argument("--query", default="历史文化景点 寺庙")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--generate", action="store_true", help="检索后调用 LLM 生成回答")
    parser.add_argument("--hybrid", action="store_true", help="跑 HybridRAG（含查询改写）")
    parser.add_argument("--rerank", action="store_true", help="HybridRAG 启用 BGE rerank（需下载模型）")
    return parser.parse_args()


def main():
    args = parse_args()
    print("本地 Chroma RAG 测试")
    print(f"  VECTOR_BACKEND={os.getenv('VECTOR_BACKEND')}")
    print(f"  CHROMA_PATH={os.getenv('CHROMA_PATH', 'app/data/chroma')}")
    print(f"  city={args.city}  query={args.query}")

    store, ok = test_store(args.city, args.query, args.top_k)
    if store is None:
        sys.exit(1)

    retriever = test_retriever(store, args.city, args.query, args.top_k)

    if args.generate:
        test_generate(retriever, args.city, args.query)

    if args.hybrid:
        test_hybrid_rag(args.city, args.query, enable_rerank=args.rerank)

    print("\n" + "=" * 64)
    print("完成" if ok else "完成（含失败项，见上方 [fail]）")
    print("=" * 64)
    sys.exit(0 if ok else 2)


if __name__ == "__main__":
    main()
