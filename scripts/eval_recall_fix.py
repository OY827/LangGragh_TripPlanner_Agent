"""
RAG 召回率修复：行业标准评估 + 检索优化

问题诊断：
1. 原 Recall@10 = top10相关数/GT总数，GT有91个文档时上限仅11%
2. Ground Truth标注过宽，69%的景点都被标为相关
3. 缺少Hit Rate等行业标准指标

修复方案：
1. 新指标：Hit Rate@K (至少1个相关)、Recall@20/50、分类命中率
2. 严格GT：仅用category匹配，不用关键词弱匹配
3. 检索优化：BGE-zh + 分类过滤 + 更大检索窗口
"""

import sys
import os
import time
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from app.rag.vector_store import QdrantVectorStore
from app.rag.retriever import KnowledgeRetriever
from app.rag.hybrid_rag import HybridRAG, HybridRAGConfig
from app.rag.data_loader import KnowledgeBaseLoader


CITY = "上海"
NEW_COLLECTION = "attraction_kb_bge_test"


# ============ 严格的 Ground Truth（仅分类匹配） ============
# 只用 category 字段定义相关性，避免关键词弱匹配导致GT过宽
QUERY_INTENTS = [
    {
        "query": "上海的历史文化景点",
        "city": "上海",
        "intent_categories": ["历史文化", "博物馆", "艺术文化"],
        "intent_keywords": [],
        "intent_price_filter": None,
        "description": "历史文化类",
    },
    {
        "query": "上海适合亲子游玩的地方",
        "city": "上海",
        "intent_categories": ["主题乐园"],
        "intent_keywords": [],
        "intent_price_filter": None,
        "description": "亲子游玩类",
    },
    {
        "query": "上海免费景点推荐",
        "city": "上海",
        "intent_categories": ["历史文化", "自然风光", "城市漫步", "艺术文化", "博物馆"],
        "intent_keywords": [],
        "intent_price_filter": 0,
        "description": "免费景点",
    },
]


def build_strict_ground_truth(attractions, intent):
    """
    严格标注：仅用 category 匹配 + 价格筛选
    避免关键词弱匹配导致GT过宽
    """
    relevant_docs = []
    relevance_scores = {}
    grade_dist = {0: 0, 1: 0, 2: 0, 3: 0}

    intent_categories = intent.get("intent_categories", [])
    price_filter = intent.get("intent_price_filter")

    for attr in attractions:
        category = attr.get("category", "")
        price = attr.get("price")

        # 分类匹配
        if category not in intent_categories:
            grade_dist[0] += 1
            continue

        # 价格筛选
        if price_filter == 0 and price != 0:
            grade_dist[0] += 1
            continue

        # 强相关：分类匹配 + 价格匹配
        grade = 3
        grade_dist[grade] += 1
        relevant_docs.append(attr["title"])
        relevance_scores[attr["title"]] = grade

    return {
        "relevant_docs": relevant_docs,
        "relevance_scores": relevance_scores,
        "total_attractions": len(attractions),
        "relevant_count": len(relevant_docs),
        "grade_distribution": grade_dist,
    }


def compute_metrics(retrieved_titles, relevant_set, relevance_scores, k_values):
    """
    计算行业标准指标：
    - Precision@K: 前K个中相关文档的比例
    - Recall@K: 前K个中相关文档 / GT总数
    - Hit Rate@K: 至少1个相关文档的比例（二值）
    - NDCG@K: 归一化折损累积增益
    - MRR: 平均倒数排名
    """
    results = {}

    for k in k_values:
        top_k = retrieved_titles[:k]

        # Precision@K
        relevant_in_top = sum(1 for t in top_k if t in relevant_set)
        precision = relevant_in_top / k if k > 0 else 0

        # Recall@K
        recall = relevant_in_top / len(relevant_set) if len(relevant_set) > 0 else 0

        # Hit Rate@K (二值)
        hit = 1 if relevant_in_top > 0 else 0

        # NDCG@K
        dcg = 0.0
        for i, title in enumerate(top_k):
            rel = relevance_scores.get(title, 0)
            dcg += (2 ** rel - 1) / math.log2(i + 2)

        ideal_scores = sorted(relevance_scores.values(), reverse=True)[:k]
        idcg = 0.0
        for i, rel in enumerate(ideal_scores):
            idcg += (2 ** rel - 1) / math.log2(i + 2)
        ndcg = dcg / idcg if idcg > 0 else 0.0

        results[k] = {
            "precision": precision,
            "recall": recall,
            "hit_rate": hit,
            "ndcg": ndcg,
        }

    # MRR (全局，不依赖K)
    mrr = 0.0
    for rank, title in enumerate(retrieved_titles, 1):
        if relevance_scores.get(title, 0) >= 1:
            mrr = 1.0 / rank
            break

    results["mrr"] = mrr
    return results


def load_all_attractions(retriever, city):
    """从向量库加载所有景点"""
    all_docs = retriever.retrieve(
        query=f"{city} 景点 全部",
        filters={"city": city},
        top_k=500
    )
    attractions = []
    for doc in all_docs:
        metadata = doc.get("metadata", {})
        if metadata.get("city") == city:
            attractions.append({
                "title": metadata.get("title", ""),
                "category": metadata.get("category", ""),
                "tags": metadata.get("tags", ""),
                "description": metadata.get("description", ""),
                "price": metadata.get("price"),
            })
    return attractions


def evaluate_query(rag_instance, intent, gt, k_values):
    """评估单个查询"""
    query = intent["query"]
    city = intent["city"]
    relevant_set = set(gt["relevant_docs"])
    relevance_scores = gt["relevance_scores"]

    t0 = time.time()
    docs, meta = rag_instance.retrieve_with_validation(
        query=query, city=city, preferences=[]
    )
    elapsed = time.time() - t0

    retrieved_titles = [
        d.get("metadata", {}).get("title", "") for d in docs
    ]

    metrics = compute_metrics(retrieved_titles, relevant_set, relevance_scores, k_values)
    metrics["elapsed"] = elapsed
    metrics["total_retrieved"] = len(retrieved_titles)
    metrics["relevant_in_top10"] = sum(1 for t in retrieved_titles[:10] if t in relevant_set)
    metrics["relevant_in_top20"] = sum(1 for t in retrieved_titles[:20] if t in relevant_set)
    metrics["relevant_in_top50"] = sum(1 for t in retrieved_titles[:50] if t in relevant_set)
    metrics["retrieved_titles"] = retrieved_titles[:20]

    return metrics


def main():
    print("=" * 72)
    print("🔬 RAG 召回率修复：行业标准评估")
    print("=" * 72)

    k_values = [3, 5, 10, 20, 50]

    # ============ 加载向量库 ============
    print("\n📦 加载 BGE-zh 向量库...")
    vector_store = QdrantVectorStore(
        embedding_model_name="BAAI/bge-base-zh-v1.5",
        embedding_backend="huggingface"
    )
    vector_store.default_collection = NEW_COLLECTION
    retriever_bge = KnowledgeRetriever(vector_store=vector_store)

    print("📦 加载 MiniLM 向量库...")
    retriever_minilm = KnowledgeRetriever()

    # ============ 构建 Ground Truth ============
    print("\n🏷️  构建严格 Ground Truth（仅分类匹配）...")

    # BGE GT
    bge_attractions = load_all_attractions(retriever_bge, CITY)
    gt_bge = {}
    for intent in QUERY_INTENTS:
        gt = build_strict_ground_truth(bge_attractions, intent)
        gt_bge[intent["query"]] = gt
        gd = gt["grade_distribution"]
        print(f"\n  [{intent['description']}] '{intent['query']}'")
        print(f"    总景点: {gt['total_attractions']}, 相关: {gt['relevant_count']}")
        print(f"    相关分类: {intent['intent_categories']}")
        print(f"    相关景点: {gt['relevant_docs']}")

    # MiniLM GT（景点数相同，应该一样）
    minilm_attractions = load_all_attractions(retriever_minilm, CITY)
    gt_minilm = {}
    for intent in QUERY_INTENTS:
        gt = build_strict_ground_truth(minilm_attractions, intent)
        gt_minilm[intent["query"]] = gt

    # ============ 初始化 RAG 实例 ============
    print("\n🔧 初始化 RAG 实例...")

    # MiniLM - 无Rerank
    config_minilm = HybridRAGConfig(
        enable_rerank=False,
        top_k=50,
        max_retrieval_rounds=1,
        enable_query_rewrite=False,
    )
    rag_minilm = HybridRAG(retriever=retriever_minilm, config=config_minilm)

    # BGE-zh - 无Rerank
    config_bge = HybridRAGConfig(
        enable_rerank=False,
        top_k=50,
        max_retrieval_rounds=1,
        enable_query_rewrite=False,
    )
    rag_bge = HybridRAG(retriever=retriever_bge, config=config_bge)

    # BGE-zh + Rerank
    config_bge_rerank = HybridRAGConfig(
        enable_rerank=True,
        rerank_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        retrieval_top_k=50,
        rerank_top_k=50,
        top_k=50,
        max_retrieval_rounds=1,
        enable_query_rewrite=False,
        enable_retrieval_validation=False,
        use_mmr=False,
    )
    rag_bge_rerank = HybridRAG(retriever=retriever_bge, config=config_bge_rerank)

    # ============ 评估 ============
    print(f"\n{'=' * 72}")
    print("📊 评估结果")
    print(f"{'=' * 72}")

    all_results = {}

    for intent in QUERY_INTENTS:
        query = intent["query"]
        gt = gt_bge[query]
        gt_min = gt_minilm[query]

        print(f"\n{'─' * 72}")
        print(f"🧪 [{intent['description']}] '{query}'")
        print(f"   相关景点数: {gt['relevant_count']}")
        print(f"{'─' * 72}")

        # MiniLM
        res_minilm = evaluate_query(rag_minilm, intent, gt_min, k_values)

        # BGE-zh
        res_bge = evaluate_query(rag_bge, intent, gt, k_values)

        # BGE-zh + Rerank
        res_bge_rerank = evaluate_query(rag_bge_rerank, intent, gt, k_values)

        print(f"\n  Top-K 指标对比:")
        header = f"  {'配置':<25}"
        for k in k_values:
            header += f"  {'P@'+str(k):>8} {'R@'+str(k):>8} {'H@'+str(k):>8}"
        header += f"  {'MRR':>8} {'Top10命中':>10}"
        print(header)
        print(f"  {'─' * (25 + 11 * len(k_values) + 30)}")

        for config_name, res in [
            ("MiniLM", res_minilm),
            ("BGE-zh", res_bge),
            ("BGE-zh+Rerank", res_bge_rerank),
        ]:
            row = f"  {config_name:<23}"
            for k in k_values:
                m = res[k]
                row += f"  {m['precision']:>8.3f} {m['recall']:>8.3f} {m['hit_rate']:>8.1f}"
            row += f"  {res['mrr']:>8.3f} {res['relevant_in_top10']:>10}"
            print(row)

        # Top-10 详情
        print(f"\n  📋 Top-10 检索结果 (BGE-zh):")
        for j, t in enumerate(res_bge["retrieved_titles"][:10], 1):
            mark = "✓" if t in set(gt["relevant_docs"]) else "✗"
            print(f"    {j}. [{mark}] {t}")

        all_results[query] = {
            "minilm": res_minilm,
            "bge": res_bge,
            "bge_rerank": res_bge_rerank,
            "gt": gt,
        }

    # ============ 汇总统计 ============
    print(f"\n{'=' * 72}")
    print("📈 汇总统计（3个查询的平均值）")
    print(f"{'=' * 72}")

    for k in k_values:
        print(f"\n  Top-{k}:")
        print(f"  {'配置':<25} {'Precision':>12} {'Recall':>12} {'Hit Rate':>12} {'NDCG':>12}")
        print(f"  {'─' * 73}")

        for config_name, config_key in [
            ("MiniLM", "minilm"),
            ("BGE-zh", "bge"),
            ("BGE-zh+Rerank", "bge_rerank"),
        ]:
            queries = [i["query"] for i in QUERY_INTENTS]
            p_vals = [all_results[q][config_key][k]["precision"] for q in queries]
            r_vals = [all_results[q][config_key][k]["recall"] for q in queries]
            h_vals = [all_results[q][config_key][k]["hit_rate"] for q in queries]
            n_vals = [all_results[q][config_key][k]["ndcg"] for q in queries]

            avg_p = sum(p_vals) / len(p_vals)
            avg_r = sum(r_vals) / len(r_vals)
            avg_h = sum(h_vals) / len(h_vals)
            avg_n = sum(n_vals) / len(n_vals)

            print(f"  {config_name:<23} {avg_p:>12.4f} {avg_r:>12.4f} {avg_h:>12.1f} {avg_n:>12.4f}")

    # ============ 关键指标汇总 ============
    print(f"\n{'=' * 72}")
    print("🎯 关键指标：Hit Rate@10 和 Recall@20")
    print(f"{'=' * 72}")

    queries = [i["query"] for i in QUERY_INTENTS]

    for config_name, config_key in [
        ("MiniLM", "minilm"),
        ("BGE-zh", "bge"),
        ("BGE-zh+Rerank", "bge_rerank"),
    ]:
        hr10_vals = [all_results[q][config_key][10]["hit_rate"] for q in queries]
        r20_vals = [all_results[q][config_key][20]["recall"] for q in queries]
        r50_vals = [all_results[q][config_key][50]["recall"] for q in queries]
        p10_vals = [all_results[q][config_key][10]["precision"] for q in queries]

        avg_hr10 = sum(hr10_vals) / len(hr10_vals) * 100
        avg_r20 = sum(r20_vals) / len(r20_vals) * 100
        avg_r50 = sum(r50_vals) / len(r50_vals) * 100
        avg_p10 = sum(p10_vals) / len(p10_vals) * 100

        print(f"\n  {config_name}:")
        print(f"    Hit Rate@10: {avg_hr10:.1f}%  (查询的Top-10中至少有1个相关文档的比例)")
        print(f"    Recall@20:   {avg_r20:.1f}%  (Top-20中相关文档占GT总数的比例)")
        print(f"    Recall@50:   {avg_r50:.1f}%  (Top-50中相关文档占GT总数的比例)")
        print(f"    Precision@10:{avg_p10:.1f}%  (Top-10中相关文档的比例)")

    # ============ 命中详情 ============
    print(f"\n{'=' * 72}")
    print("📋 命中详情：每个查询Top-10命中的相关文档数")
    print(f"{'=' * 72}")

    for config_name, config_key in [
        ("MiniLM", "minilm"),
        ("BGE-zh", "bge"),
        ("BGE-zh+Rerank", "bge_rerank"),
    ]:
        print(f"\n  {config_name}:")
        for intent in QUERY_INTENTS:
            query = intent["query"]
            gt = all_results[query]["gt"]
            res = all_results[query][config_key]
            hit10 = res["relevant_in_top10"]
            total_gt = gt["relevant_count"]
            pct = hit10 / total_gt * 100 if total_gt > 0 else 0
            print(f"    '{query}': Top-10命中 {hit10}/{total_gt} ({pct:.0f}%)")

    print(f"\n{'=' * 72}")
    print("✅ 评估完成")
    print("=" * 72)


if __name__ == "__main__":
    main()
