"""
启用查询增强 + top_k=50 + GT=45 的评估脚本

对比配置：
1. BGE-zh + 分类Boost（无查询增强） - 基线
2. BGE-zh + 分类Boost + 查询增强 - 实验组
3. BGE-zh + 分类Boost + 查询增强 + Rerank - 完整方案

Ground Truth：使用多数共识标注（分类匹配 + 关键词匹配 ≈ 45个历史文化景点）
"""

import sys
import os
import time
import math
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from app.rag.vector_store import QdrantVectorStore
from app.rag.retriever import KnowledgeRetriever
from app.rag.hybrid_rag import HybridRAG, HybridRAGConfig


CITY = "上海"
NEW_COLLECTION = "attraction_kb_bge_test"

# 历史文化相关关键词（用于GT标注的关键词匹配，仅在title+tags中搜索）
# 精选关键词，避免过宽匹配（如"路"、"大楼"会匹配现代商业设施）
HISTORY_KEYWORDS = [
    # 通用历史类
    "古迹", "遗址", "古镇", "故居", "博物馆", "纪念馆", "文物",
    "寺庙", "教堂", "旧址", "抗战", "会址",
    # 上海特色历史文化地标（被错误分类的）
    "城隍庙", "外滩", "外白渡桥", "张园", "思南公馆", "思南路",
    "田子坊", "新天地", "武康大楼", "武康路", "南京路", "淮海路",
    "龙华寺", "静安寺", "古猗园", "大观园",
]


# ============ 查询意图配置 ============
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


def load_attractions_from_json(city):
    """从 JSON 文件加载所有景点（含完整 metadata 和 content）"""
    path = f"d:/Projects/LangChain/app/knowledge_base/attractions/{city}.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_ground_truth(attractions, intent):
    """
    构建 Ground Truth：分类匹配 + 关键词匹配（多数共识方式）

    对于历史文化类查询：
    - 分类匹配（历史文化/博物馆/艺术文化）→ 3分
    - 关键词匹配（title+tags中含历史相关关键词）→ 2分
    - 两者都匹配 → 3分

    这样可以覆盖被错误分类的历史文化景点（如"外滩"被分为"自然风光"）
    """
    relevant_docs = []
    relevance_scores = {}
    grade_dist = {0: 0, 1: 0, 2: 0, 3: 0}

    intent_categories = intent.get("intent_categories", [])
    price_filter = intent.get("intent_price_filter")
    is_history_query = intent["description"] == "历史文化类"

    for attr in attractions:
        category = attr.get("category", "")
        title = attr.get("title", "")
        tags = attr.get("tags", "")
        price = attr.get("price")

        # 1. 分类匹配
        category_match = category in intent_categories

        # 2. 关键词匹配（仅历史文化类查询，在title+tags中搜索）
        keyword_match = False
        if is_history_query:
            text = f"{title} {tags}"
            keyword_match = any(kw in text for kw in HISTORY_KEYWORDS)

        is_relevant = category_match or keyword_match

        # 价格过滤
        if is_relevant and price_filter == 0 and price != 0:
            is_relevant = False

        if not is_relevant:
            grade_dist[0] += 1
            continue

        # 评分：分类匹配=3分，仅关键词匹配=2分
        grade = 3 if category_match else 2
        grade_dist[grade] += 1
        relevant_docs.append(title)
        relevance_scores[title] = grade

    return {
        "relevant_docs": relevant_docs,
        "relevance_scores": relevance_scores,
        "total_attractions": len(attractions),
        "relevant_count": len(relevant_docs),
        "grade_distribution": grade_dist,
    }


def compute_metrics(retrieved_titles, relevant_set, relevance_scores, k_values):
    """计算指标：Precision/Recall/HitRate/NDCG/MRR"""
    results = {}

    for k in k_values:
        top_k = retrieved_titles[:k]
        relevant_in_top = sum(1 for t in top_k if t in relevant_set)
        precision = relevant_in_top / k if k > 0 else 0
        recall = relevant_in_top / len(relevant_set) if len(relevant_set) > 0 else 0
        hit = 1 if relevant_in_top > 0 else 0

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
            "relevant_in_top": relevant_in_top,
        }

    mrr = 0.0
    for rank, title in enumerate(retrieved_titles, 1):
        if relevance_scores.get(title, 0) >= 1:
            mrr = 1.0 / rank
            break
    results["mrr"] = mrr

    return results


def load_all_attractions_from_json(city):
    """从 JSON 文件加载所有景点（用于GT构建，含完整metadata）"""
    raw = load_attractions_from_json(city)
    attractions = []
    for item in raw:
        metadata = item.get("metadata", {})
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

    retrieved_titles = [d.get("metadata", {}).get("title", "") for d in docs]
    metrics = compute_metrics(retrieved_titles, relevant_set, relevance_scores, k_values)
    metrics["elapsed"] = elapsed
    metrics["total_retrieved"] = len(retrieved_titles)
    metrics["retrieved_titles"] = retrieved_titles[:20]
    metrics["query_variants"] = meta.get("query_variants", [])

    return metrics


def main():
    print("=" * 72)
    print("🔬 查询增强 + top_k=50 评估（GT=45）")
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

    # ============ 构建 Ground Truth ============
    print("\n🏷️  构建 Ground Truth（分类匹配 + 关键词匹配）...")
    attractions = load_all_attractions_from_json(CITY)
    print(f"  {CITY} 景点总数: {len(attractions)}")

    gt_bge = {}
    for intent in QUERY_INTENTS:
        gt = build_ground_truth(attractions, intent)
        gt_bge[intent["query"]] = gt
        print(f"\n  [{intent['description']}] '{intent['query']}'")
        print(f"    相关景点: {gt['relevant_count']}")
        print(f"    等级分布: {gt['grade_distribution']}")
        if intent["description"] == "历史文化类":
            print(f"    相关景点列表: {gt['relevant_docs']}")

    # ============ 初始化 RAG 实例（三种配置） ============
    print("\n🔧 初始化 RAG 实例...")

    # 配置1: 基线（无查询增强）
    config_baseline = HybridRAGConfig(
        enable_rerank=False,
        top_k=50,
        max_retrieval_rounds=1,
        enable_query_rewrite=False,
    )
    rag_baseline = HybridRAG(retriever=retriever_bge, config=config_baseline)

    # 配置2: + 查询增强
    config_query_rewrite = HybridRAGConfig(
        enable_rerank=False,
        top_k=50,
        max_retrieval_rounds=1,
        enable_query_rewrite=True,
        max_query_variants=3,
    )
    rag_query_rewrite = HybridRAG(retriever=retriever_bge, config=config_query_rewrite)

    # 配置3: + 查询增强 + Rerank（使用中文Rerank模型 BAAI/bge-reranker-base）
    config_full = HybridRAGConfig(
        enable_rerank=True,
        rerank_model="BAAI/bge-reranker-base",
        retrieval_top_k=60,
        rerank_top_k=50,
        top_k=50,
        max_retrieval_rounds=1,
        enable_query_rewrite=True,
        max_query_variants=3,
        enable_retrieval_validation=False,
        use_mmr=False,
    )
    rag_full = HybridRAG(retriever=retriever_bge, config=config_full)

    # ============ 评估 ============
    print(f"\n{'=' * 72}")
    print("📊 评估结果")
    print(f"{'=' * 72}")

    all_results = {}

    for intent in QUERY_INTENTS:
        query = intent["query"]
        gt = gt_bge[query]

        print(f"\n{'─' * 72}")
        print(f"🧪 [{intent['description']}] '{query}'")
        print(f"   GT 相关景点数: {gt['relevant_count']}")
        print(f"{'─' * 72}")

        # 基线
        res_baseline = evaluate_query(rag_baseline, intent, gt, k_values)

        # + 查询增强
        res_qr = evaluate_query(rag_query_rewrite, intent, gt, k_values)

        # + 查询增强 + Rerank
        res_full = evaluate_query(rag_full, intent, gt, k_values)

        # 打印查询变体
        if res_qr["query_variants"]:
            print(f"\n  📝 查询增强变体:")
            for i, v in enumerate(res_qr["query_variants"][:3], 1):
                print(f"    {i}. {v}")

        # 打印指标对比
        print(f"\n  Top-K 指标对比:")
        header = f"  {'配置':<25}"
        for k in k_values:
            header += f"  {'P@'+str(k):>8} {'R@'+str(k):>8} {'H@'+str(k):>8}"
        header += f"  {'MRR':>8}"
        print(header)
        print(f"  {'─' * (25 + 11 * len(k_values) + 10)}")

        for config_name, res in [
            ("基线(无QR)", res_baseline),
            ("+查询增强", res_qr),
            ("+QR+Rerank", res_full),
        ]:
            row = f"  {config_name:<23}"
            for k in k_values:
                m = res[k]
                row += f"  {m['precision']:>8.3f} {m['recall']:>8.3f} {m['hit_rate']:>8.1f}"
            row += f"  {res['mrr']:>8.3f}"
            print(row)

        # Top-10 详情
        print(f"\n  📋 Top-10 检索结果:")
        for config_name, res in [
            ("基线", res_baseline),
            ("+QR", res_qr),
            ("+QR+Rerank", res_full),
        ]:
            print(f"    [{config_name}]:")
            for j, t in enumerate(res["retrieved_titles"][:10], 1):
                mark = "✓" if t in set(gt["relevant_docs"]) else "✗"
                print(f"      {j}. [{mark}] {t}")

        all_results[query] = {
            "baseline": res_baseline,
            "query_rewrite": res_qr,
            "full": res_full,
            "gt": gt,
        }

    # ============ 汇总统计 ============
    print(f"\n{'=' * 72}")
    print("📈 汇总统计（3个查询的平均值）")
    print(f"{'=' * 72}")

    queries = [i["query"] for i in QUERY_INTENTS]

    for k in k_values:
        print(f"\n  Top-{k}:")
        print(f"  {'配置':<25} {'Precision':>12} {'Recall':>12} {'Hit Rate':>12} {'NDCG':>12}")
        print(f"  {'─' * 73}")

        for config_name, config_key in [
            ("基线(无QR)", "baseline"),
            ("+查询增强", "query_rewrite"),
            ("+QR+Rerank", "full"),
        ]:
            p_vals = [all_results[q][config_key][k]["precision"] for q in queries]
            r_vals = [all_results[q][config_key][k]["recall"] for q in queries]
            h_vals = [all_results[q][config_key][k]["hit_rate"] for q in queries]
            n_vals = [all_results[q][config_key][k]["ndcg"] for q in queries]

            avg_p = sum(p_vals) / len(p_vals)
            avg_r = sum(r_vals) / len(r_vals)
            avg_h = sum(h_vals) / len(h_vals)
            avg_n = sum(n_vals) / len(n_vals)

            print(f"  {config_name:<23} {avg_p:>12.4f} {avg_r:>12.4f} {avg_h:>12.1f} {avg_n:>12.4f}")

    # ============ 提升率对比 ============
    print(f"\n{'=' * 72}")
    print("📊 提升率对比（相对基线）")
    print(f"{'=' * 72}")

    for k in k_values:
        print(f"\n  Top-{k}:")
        print(f"  {'配置':<25} {'Precision':>12} {'Recall':>12} {'Hit Rate':>12} {'NDCG':>12}")
        print(f"  {'─' * 73}")

        baseline_p = sum(all_results[q]["baseline"][k]["precision"] for q in queries) / len(queries)
        baseline_r = sum(all_results[q]["baseline"][k]["recall"] for q in queries) / len(queries)
        baseline_h = sum(all_results[q]["baseline"][k]["hit_rate"] for q in queries) / len(queries)
        baseline_n = sum(all_results[q]["baseline"][k]["ndcg"] for q in queries) / len(queries)

        for config_name, config_key in [
            ("+查询增强", "query_rewrite"),
            ("+QR+Rerank", "full"),
        ]:
            p = sum(all_results[q][config_key][k]["precision"] for q in queries) / len(queries)
            r = sum(all_results[q][config_key][k]["recall"] for q in queries) / len(queries)
            h = sum(all_results[q][config_key][k]["hit_rate"] for q in queries) / len(queries)
            n = sum(all_results[q][config_key][k]["ndcg"] for q in queries) / len(queries)

            def pct(new, base):
                if base == 0:
                    return "+∞%" if new > 0 else "0%"
                return f"{(new - base) / base * 100:+.1f}%"

            print(f"  {config_name:<23} {pct(p, baseline_p):>12} {pct(r, baseline_r):>12} {pct(h, baseline_h):>12} {pct(n, baseline_n):>12}")

    # ============ 命中详情 ============
    print(f"\n{'=' * 72}")
    print("📋 命中详情：每个查询Top-10/20/50命中的相关文档数")
    print(f"{'=' * 72}")

    for config_name, config_key in [
        ("基线(无QR)", "baseline"),
        ("+查询增强", "query_rewrite"),
        ("+QR+Rerank", "full"),
    ]:
        print(f"\n  {config_name}:")
        for intent in QUERY_INTENTS:
            query = intent["query"]
            gt = all_results[query]["gt"]
            res = all_results[query][config_key]
            hit10 = res[10]["relevant_in_top"]
            hit20 = res[20]["relevant_in_top"]
            hit50 = res[50]["relevant_in_top"]
            total_gt = gt["relevant_count"]
            print(f"    '{query}' (GT={total_gt}): Top-10命中 {hit10}, Top-20命中 {hit20}, Top-50命中 {hit50}")

    print(f"\n{'=' * 72}")
    print("✅ 评估完成")
    print("=" * 72)


if __name__ == "__main__":
    main()
