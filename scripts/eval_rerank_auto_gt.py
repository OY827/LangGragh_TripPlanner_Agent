"""
Rerank 效果量化评估脚本（基于知识库自动标注 Ground Truth）

自动标注逻辑：
1. 从向量库加载指定城市的所有景点
2. 根据查询意图定义匹配规则（分类/标签/关键词/价格）
3. 对每个景点计算相关性等级（0-3）：
   - 3分：分类匹配 + 标签/关键词匹配 + 价格匹配（强相关）
   - 2分：分类匹配 或 标签/关键词匹配（相关）
   - 1分：仅标题/描述关键词弱匹配（弱相关）
   - 0分：不相关
4. 用自动标注的 Ground Truth 评估 Precision/Recall/NDCG/MRR
"""

import sys
import os
import time
import math
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from app.rag.hybrid_rag import HybridRAG, HybridRAGConfig
from app.rag.retriever import KnowledgeRetriever


# ============ 查询意图配置（用于自动标注） ============
# 每个查询定义：
#   intent_categories: 匹配的分类（metadata.category）
#   intent_keywords: 匹配的关键词（匹配 tags / description / title）
#   intent_price_filter: 价格筛选（None=不限，0=免费，(0,50)=50元以下）
#   exclude_categories: 需要排除的分类
QUERY_INTENTS = [
    {
        "query": "上海的历史文化景点",
        "city": "上海",
        "intent_categories": ["历史文化", "博物馆", "艺术文化"],
        "intent_keywords": [
            "历史", "古迹", "遗址", "古镇", "故居", "博物馆", "纪念",
            "园林", "文物", "寺庙", "教堂", "故居", "旧址", "文化馆"
        ],
        "intent_price_filter": None,
        "exclude_categories": ["主题乐园"],
        "description": "历史文化类查询",
    },
    {
        "query": "上海适合亲子游玩的地方",
        "city": "上海",
        "intent_categories": ["主题乐园", "自然风光"],
        "intent_keywords": [
            "亲子", "遛娃", "儿童", "家庭", "乐园", "海洋", "动物",
            "自然", "森林", "公园", "沙滩"
        ],
        "intent_price_filter": None,
        "exclude_categories": [],
        "description": "亲子游查询",
    },
    {
        "query": "上海免费景点推荐",
        "city": "上海",
        "intent_categories": ["历史文化", "自然风光", "城市漫步", "艺术文化", "博物馆"],
        "intent_keywords": [
            "免费", "开放", "公园", "绿地", "广场", "街区", "路",
            "文化", "历史", "艺术"
        ],
        "intent_price_filter": 0,  # 只看免费
        "exclude_categories": [],
        "description": "免费景点查询",
    },
]


def load_all_attractions(city: str) -> list:
    """从向量库加载指定城市的所有景点"""
    retriever = KnowledgeRetriever()
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
                "rating": metadata.get("rating"),
                "metadata": metadata,
                "score": doc.get("score", 0),
            })

    return attractions


def auto_annotate_relevance(attraction: dict, intent: dict) -> int:
    """
    自动计算景点与查询意图的相关性等级（0-3）

    评分规则：
    +3: 分类匹配 + (标签/关键词匹配 或 价格匹配)
    +2: 分类匹配 或 标签/关键词匹配
    +1: 仅标题/描述关键词弱匹配
    排除: 命中 exclude_categories 直接0分
    """
    category = attraction.get("category", "")
    tags = attraction.get("tags", "")
    title = attraction.get("title", "")
    desc = attraction.get("description", "")
    price = attraction.get("price")

    intent_categories = intent.get("intent_categories", [])
    intent_keywords = intent.get("intent_keywords", [])
    price_filter = intent.get("intent_price_filter")
    exclude_cats = intent.get("exclude_categories", [])

    # 排除规则
    if category in exclude_cats:
        return 0

    # 分类匹配
    category_match = category in intent_categories

    # 关键词匹配（在 tags + description + title 中搜索）
    text_to_check = f"{tags} {desc} {title}"
    keyword_hits = sum(1 for kw in intent_keywords if kw in text_to_check)
    keyword_match = keyword_hits > 0

    # 价格匹配
    price_match = True
    if price_filter is not None:
        if price == 0 and price_filter == 0:
            price_match = True
        elif isinstance(price_filter, tuple) and len(price_filter) == 2:
            if price is not None and price_filter[0] <= price <= price_filter[1]:
                price_match = True
            else:
                price_match = False
        elif price == 0 and price_filter == 0:
            price_match = True
        else:
            price_match = (price == 0) == (price_filter == 0) if price_filter == 0 else True

    if not price_match:
        return 0

    # 计算等级
    if category_match and keyword_match:
        return 3
    elif category_match:
        return 2
    elif keyword_match:
        return 1
    else:
        return 0


def build_ground_truth(attractions: list, intent: dict) -> dict:
    """
    构建自动标注的 Ground Truth

    Returns:
        {
            "relevant_docs": 标题列表（相关性 >= 1）,
            "relevance_scores": {标题: 相关性等级},
            "total_attractions": 景点总数,
            "relevant_count": 相关景点数,
            "grade_distribution": {0: x, 1: x, 2: x, 3: x}
        }
    """
    relevant_docs = []
    relevance_scores = {}
    grade_dist = {0: 0, 1: 0, 2: 0, 3: 0}

    for attr in attractions:
        grade = auto_annotate_relevance(attr, intent)
        grade_dist[grade] += 1

        if grade >= 1:
            title = attr["title"]
            relevant_docs.append(title)
            relevance_scores[title] = grade

    return {
        "relevant_docs": relevant_docs,
        "relevance_scores": relevance_scores,
        "total_attractions": len(attractions),
        "relevant_count": len(relevant_docs),
        "grade_distribution": grade_dist,
    }


def compute_precision_at_k(retrieved_titles, relevant_set, k):
    """Precision@K = 前K个结果中相关文档数 / K"""
    top_k = retrieved_titles[:k]
    relevant_in_top = sum(1 for t in top_k if t in relevant_set)
    return relevant_in_top / k if k > 0 else 0


def compute_recall_at_k(retrieved_titles, relevant_set, k):
    """Recall@K = 前K个结果中相关文档数 / 总相关文档数"""
    top_k = retrieved_titles[:k]
    relevant_in_top = sum(1 for t in top_k if t in relevant_set)
    return relevant_in_top / len(relevant_set) if len(relevant_set) > 0 else 0


def compute_ndcg_at_k(retrieved_titles, relevance_scores, k):
    """
    NDCG@K = DCG@K / IDCG@K

    使用自动标注的多级相关性（0-3）
    """
    top_k = retrieved_titles[:k]

    dcg = 0.0
    for i, title in enumerate(top_k):
        rel = relevance_scores.get(title, 0)
        dcg += (2 ** rel - 1) / math.log2(i + 2)

    # IDCG: 理想排序（按相关性等级降序排列）
    ideal_scores = sorted(relevance_scores.values(), reverse=True)[:k]
    idcg = 0.0
    for i, rel in enumerate(ideal_scores):
        idcg += (2 ** rel - 1) / math.log2(i + 2)

    return dcg / idcg if idcg > 0 else 0.0


def compute_mrr(retrieved_titles, relevance_scores):
    """MRR = 1 / 第一个相关文档（等级>=1）的排名"""
    for rank, title in enumerate(retrieved_titles, 1):
        if relevance_scores.get(title, 0) >= 1:
            return 1.0 / rank
    return 0.0


def evaluate_single_query(rag_instance, intent, gt, top_k=10):
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
        d.get("metadata", {}).get("title", "") for d in docs[:top_k]
    ]

    precision = compute_precision_at_k(retrieved_titles, relevant_set, top_k)
    recall = compute_recall_at_k(retrieved_titles, relevant_set, top_k)
    ndcg = compute_ndcg_at_k(retrieved_titles, relevance_scores, top_k)
    mrr = compute_mrr(retrieved_titles, relevance_scores)

    return {
        "query": query,
        "city": city,
        "top_k": top_k,
        "precision": precision,
        "recall": recall,
        "ndcg": ndcg,
        "mrr": mrr,
        "retrieved_titles": retrieved_titles,
        "relevant_count": len(relevant_set),
        "elapsed": elapsed,
        "rerank_enabled": meta.get("rerank_enabled", False),
    }


def print_metric_comparison(label, val_wo, val_w):
    diff = val_w - val_wo
    pct = (diff / val_wo * 100) if val_wo > 0 else float("inf")
    arrow = "↑" if diff > 0 else ("↓" if diff < 0 else "→")
    if abs(pct) == float("inf"):
        pct_str = "+∞%" if diff > 0 else "-∞%"
    else:
        pct_str = f"{pct:+.1f}%"
    print(f"  {label}: {val_wo:.4f} → {val_w:.4f}  {arrow} {pct_str} ({diff:+.4f})")


def main():
    print("=" * 72)
    print("🔬 Rerank 效果量化评估（自动标注 Ground Truth）")
    print("=" * 72)

    top_k_values = [3, 5, 10]

    # ============ 初始化 ============
    print("\n📦 加载知识库数据...")
    retriever = KnowledgeRetriever()

    all_attractions = {}
    for intent in QUERY_INTENTS:
        city = intent["city"]
        if city not in all_attractions:
            all_attractions[city] = load_all_attractions(city)
            print(f"  {city}: 加载 {len(all_attractions[city])} 个景点")

    # ============ 自动标注 Ground Truth ============
    print("\n🏷️  自动标注 Ground Truth...")
    ground_truths = {}
    for intent in QUERY_INTENTS:
        city = intent["city"]
        attractions = all_attractions[city]
        gt = build_ground_truth(attractions, intent)
        ground_truths[intent["query"]] = gt

        gd = gt["grade_distribution"]
        print(f"\n  查询: '{intent['query']}'")
        print(f"    总景点数: {gt['total_attractions']}")
        print(f"    相关景点: {gt['relevant_count']}")
        print(f"    等级分布: 强相关(3)={gd[3]}, 相关(2)={gd[2]}, 弱相关(1)={gd[1]}, 不相关(0)={gd[0]}")
        print(f"    相关景点列表: {gt['relevant_docs'][:15]}{'...' if len(gt['relevant_docs']) > 15 else ''}")

    # ============ 初始化 RAG 实例 ============
    print("\n🔧 初始化 HybridRAG 实例...")

    config_no_rerank = HybridRAGConfig(
        enable_rerank=False,
        top_k=10,
        max_retrieval_rounds=1,
        enable_query_rewrite=False,
    )
    rag_no_rerank = HybridRAG(config=config_no_rerank)

    config_with_rerank = HybridRAGConfig(
        enable_rerank=True,
        rerank_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        retrieval_top_k=30,
        rerank_top_k=10,
        top_k=10,
        max_retrieval_rounds=1,
        enable_query_rewrite=False,
        enable_retrieval_validation=False,
        use_mmr=False,
    )
    rag_with_rerank = HybridRAG(config=config_with_rerank)

    # ============ 评估 ============
    all_results = []

    for intent in QUERY_INTENTS:
        gt = ground_truths[intent["query"]]
        relevant_set = set(gt["relevant_docs"])

        print(f"\n{'─' * 72}")
        print(f"🧪 查询: '{intent['query']}'")
        print(f"   Ground Truth: {len(relevant_set)} 个相关景点")
        print(f"{'─' * 72}")

        for top_k in top_k_values:
            res_wo = evaluate_single_query(rag_no_rerank, intent, gt, top_k)
            res_w = evaluate_single_query(rag_with_rerank, intent, gt, top_k)

            print(f"\n  📊 Top-{top_k} 指标对比:")
            print_metric_comparison("Precision", res_wo["precision"], res_w["precision"])
            print_metric_comparison("Recall", res_wo["recall"], res_w["recall"])
            print_metric_comparison("NDCG", res_wo["ndcg"], res_w["ndcg"])
            print_metric_comparison("MRR", res_wo["mrr"], res_w["mrr"])
            print_metric_comparison("响应时间(s)", res_wo["elapsed"], res_w["elapsed"])

            # Top-K 文档详情
            if top_k == 5:
                print(f"\n  📋 Top-{top_k} 检索结果:")
                print(f"    无Rerank:")
                for j, t in enumerate(res_wo["retrieved_titles"][:top_k], 1):
                    grade = gt["relevance_scores"].get(t, 0)
                    mark = f"G{grade}" if grade > 0 else "✗"
                    print(f"      {j}. [{mark}] {t}")
                print(f"    有Rerank:")
                for j, t in enumerate(res_w["retrieved_titles"][:top_k], 1):
                    grade = gt["relevance_scores"].get(t, 0)
                    mark = f"G{grade}" if grade > 0 else "✗"
                    print(f"      {j}. [{mark}] {t}")

            all_results.append({
                "query": intent["query"],
                "top_k": top_k,
                "wo_rerank": {
                    "precision": res_wo["precision"],
                    "recall": res_wo["recall"],
                    "ndcg": res_wo["ndcg"],
                    "mrr": res_wo["mrr"],
                    "elapsed": res_wo["elapsed"],
                },
                "with_rerank": {
                    "precision": res_w["precision"],
                    "recall": res_w["recall"],
                    "ndcg": res_w["ndcg"],
                    "mrr": res_w["mrr"],
                    "elapsed": res_w["elapsed"],
                },
            })

    # ============ 汇总统计 ============
    print(f"\n{'=' * 72}")
    print("📈 汇总统计（自动标注 Ground Truth）")
    print(f"{'=' * 72}")

    for top_k in top_k_values:
        subset = [r for r in all_results if r["top_k"] == top_k]
        if not subset:
            continue

        def avg(key):
            vals_wo = [r["wo_rerank"][key] for r in subset]
            vals_w = [r["with_rerank"][key] for r in subset]
            return sum(vals_wo) / len(vals_wo), sum(vals_w) / len(vals_w)

        p_wo, p_w = avg("precision")
        r_wo, r_w = avg("recall")
        n_wo, n_w = avg("ndcg")
        m_wo, m_w = avg("mrr")
        t_wo, t_w = avg("elapsed")

        print(f"\n  Top-{top_k}:")
        print_metric_comparison("Precision@K", p_wo, p_w)
        print_metric_comparison("Recall@K", r_wo, r_w)
        print_metric_comparison("NDCG@K", n_wo, n_w)
        print_metric_comparison("MRR", m_wo, m_w)
        print_metric_comparison("响应时间(s)", t_wo, t_w)

    # ============ 提升率总表 ============
    print(f"\n{'=' * 72}")
    print("📊 指标提升率汇总")
    print(f"{'=' * 72}")
    print(f"\n  {'指标':<16} {'Top-3':>12} {'Top-5':>12} {'Top-10':>12}")
    print(f"  {'─' * 56}")

    for metric, key in [
        ("Precision@K", "precision"),
        ("Recall@K", "recall"),
        ("NDCG@K", "ndcg"),
        ("MRR", "mrr"),
    ]:
        row = f"  {metric:<16}"
        for top_k in top_k_values:
            subset = [r for r in all_results if r["top_k"] == top_k]
            vals_wo = [r["wo_rerank"][key] for r in subset]
            vals_w = [r["with_rerank"][key] for r in subset]
            avg_wo = sum(vals_wo) / len(vals_wo)
            avg_w = sum(vals_w) / len(vals_w)
            pct = ((avg_w - avg_wo) / avg_wo * 100) if avg_wo > 0 else float("inf")
            row += f" {pct:>+10.1f}%"
        print(row)

    print(f"\n  {'响应时间':<16} ", end="")
    for top_k in top_k_values:
        subset = [r for r in all_results if r["top_k"] == top_k]
        vals_wo = [r["wo_rerank"]["elapsed"] for r in subset]
        vals_w = [r["with_rerank"]["elapsed"] for r in subset]
        avg_wo = sum(vals_wo) / len(vals_wo)
        avg_w = sum(vals_w) / len(vals_w)
        overhead = (avg_w - avg_wo) / avg_wo * 100 if avg_wo > 0 else float("inf")
        print(f"{overhead:>+10.1f}%  ", end="")
    print()

    print(f"\n{'=' * 72}")
    print("✅ 评估完成")
    print("=" * 72)


if __name__ == "__main__":
    main()
