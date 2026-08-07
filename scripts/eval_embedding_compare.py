"""
测试不同 Embedding 模型对 RAG 召回率的影响

对比模型：
1. all-MiniLM-L6-v2 (英文, 384维) - 当前使用
2. BAAI/bge-base-zh-v1.5 (中文优化, 768维) - 候选

流程：
1. 用新 embedding 模型创建新集合（不影响原集合）
2. 加载上海景点数据
3. 运行自动标注评估
4. 输出对比结果
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


# ============ 测试配置 ============
CITY = "上海"
NEW_COLLECTION = "attraction_kb_bge_test"
NEW_EMBEDDING_MODEL = "BAAI/bge-base-zh-v1.5"

# ============ 查询意图（用于自动标注） ============
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
    },
    {
        "query": "上海免费景点推荐",
        "city": "上海",
        "intent_categories": ["历史文化", "自然风光", "城市漫步", "艺术文化", "博物馆"],
        "intent_keywords": [
            "免费", "开放", "公园", "绿地", "广场", "街区", "路",
            "文化", "历史", "艺术"
        ],
        "intent_price_filter": 0,
        "exclude_categories": [],
    },
]


def auto_annotate_relevance(attraction, intent):
    """自动标注相关性等级（0-3）"""
    category = attraction.get("category", "")
    tags = attraction.get("tags", "")
    title = attraction.get("title", "")
    desc = attraction.get("description", "")
    price = attraction.get("price")

    intent_categories = intent.get("intent_categories", [])
    intent_keywords = intent.get("intent_keywords", [])
    price_filter = intent.get("intent_price_filter")
    exclude_cats = intent.get("exclude_categories", [])

    if category in exclude_cats:
        return 0

    category_match = category in intent_categories
    text_to_check = f"{tags} {desc} {title}"
    keyword_hits = sum(1 for kw in intent_keywords if kw in text_to_check)
    keyword_match = keyword_hits > 0

    if price_filter == 0:
        if price != 0 and price != "免费":
            return 0

    if category_match and keyword_match:
        return 3
    elif category_match:
        return 2
    elif keyword_match:
        return 1
    else:
        return 0


def build_ground_truth(attractions, intent):
    """构建自动标注 Ground Truth"""
    relevant_docs = []
    relevance_scores = {}
    grade_dist = {0: 0, 1: 0, 2: 0, 3: 0}

    for attr in attractions:
        grade = auto_annotate_relevance(attr, intent)
        grade_dist[grade] += 1
        if grade >= 1:
            relevant_docs.append(attr["title"])
            relevance_scores[attr["title"]] = grade

    return {
        "relevant_docs": relevant_docs,
        "relevance_scores": relevance_scores,
        "total_attractions": len(attractions),
        "relevant_count": len(relevant_docs),
        "grade_distribution": grade_dist,
    }


def compute_precision_at_k(retrieved_titles, relevant_set, k):
    top_k = retrieved_titles[:k]
    relevant_in_top = sum(1 for t in top_k if t in relevant_set)
    return relevant_in_top / k if k > 0 else 0


def compute_recall_at_k(retrieved_titles, relevant_set, k):
    top_k = retrieved_titles[:k]
    relevant_in_top = sum(1 for t in top_k if t in relevant_set)
    return relevant_in_top / len(relevant_set) if len(relevant_set) > 0 else 0


def compute_ndcg_at_k(retrieved_titles, relevance_scores, k):
    top_k = retrieved_titles[:k]
    dcg = 0.0
    for i, title in enumerate(top_k):
        rel = relevance_scores.get(title, 0)
        dcg += (2 ** rel - 1) / math.log2(i + 2)

    ideal_scores = sorted(relevance_scores.values(), reverse=True)[:k]
    idcg = 0.0
    for i, rel in enumerate(ideal_scores):
        idcg += (2 ** rel - 1) / math.log2(i + 2)

    return dcg / idcg if idcg > 0 else 0.0


def compute_mrr(retrieved_titles, relevance_scores):
    for rank, title in enumerate(retrieved_titles, 1):
        if relevance_scores.get(title, 0) >= 1:
            return 1.0 / rank
    return 0.0


def load_all_attractions_from_store(retriever, city):
    """从向量库加载所有景点（用于自动标注）"""
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


def evaluate_single_query(rag_instance, intent, gt, top_k=10):
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

    return {
        "precision": compute_precision_at_k(retrieved_titles, relevant_set, top_k),
        "recall": compute_recall_at_k(retrieved_titles, relevant_set, top_k),
        "ndcg": compute_ndcg_at_k(retrieved_titles, relevance_scores, top_k),
        "mrr": compute_mrr(retrieved_titles, relevance_scores),
        "retrieved_titles": retrieved_titles,
        "elapsed": elapsed,
    }


def build_new_vector_store(embedding_model, collection_name):
    """用指定 embedding 模型创建新的向量存储"""
    print(f"\n📦 初始化新向量存储...")
    print(f"   Embedding 模型: {embedding_model}")
    print(f"   集合名: {collection_name}")

    vector_store = QdrantVectorStore(
        embedding_model_name=embedding_model,
        embedding_backend="huggingface"
    )

    # 删除旧集合（如果存在）
    try:
        if vector_store.client.collection_exists(collection_name=collection_name):
            print(f"   删除已存在的集合: {collection_name}")
            vector_store.delete_collection(collection_name)
    except Exception:
        pass

    # 加载景点数据
    print(f"   加载 {CITY} 景点数据...")
    loader = KnowledgeBaseLoader()
    all_docs = loader.load_attraction_data(CITY)
    print(f"   加载 {len(all_docs)} 条文档")

    # 添加到新集合
    print(f"   向量化并写入集合 {collection_name}...")
    t0 = time.time()
    vector_store.add_documents(all_docs, collection_name=collection_name)
    print(f"   ✅ 写入完成，耗时 {time.time() - t0:.1f}s")

    # 统计
    stats = vector_store.get_collection_stats(collection_name)
    print(f"   集合统计: {stats}")

    # 设置默认集合为新集合，确保 retriever 使用新集合
    vector_store.default_collection = collection_name

    return vector_store


def main():
    print("=" * 72)
    print(f"🔬 Embedding 模型对比测试（{CITY}）")
    print(f"   原模型: all-MiniLM-L6-v2 (384维, 英文)")
    print(f"   新模型: {NEW_EMBEDDING_MODEL} (768维, 中文优化)")
    print("=" * 72)

    top_k_values = [3, 5, 10]

    # ============ Step 1: 构建新向量库 ============
    new_vector_store = build_new_vector_store(NEW_EMBEDDING_MODEL, NEW_COLLECTION)
    new_retriever = KnowledgeRetriever(vector_store=new_vector_store)

    # ============ Step 2: 加载景点并自动标注 ============
    print(f"\n🏷️  加载景点并自动标注 Ground Truth...")
    all_attractions = load_all_attractions_from_store(new_retriever, CITY)
    print(f"   {CITY} 景点总数: {len(all_attractions)}")

    ground_truths = {}
    for intent in QUERY_INTENTS:
        gt = build_ground_truth(all_attractions, intent)
        ground_truths[intent["query"]] = gt
        gd = gt["grade_distribution"]
        print(f"\n  查询: '{intent['query']}'")
        print(f"    相关景点: {gt['relevant_count']}/{gt['total_attractions']}")
        print(f"    等级: 强相关(3)={gd[3]}, 相关(2)={gd[2]}, 弱相关(1)={gd[1]}, 不相关(0)={gd[0]}")

    # ============ Step 3: 初始化 RAG 实例 ============
    print(f"\n🔧 初始化 HybridRAG 实例...")

    # 新模型 - 无Rerank
    config_no_rerank = HybridRAGConfig(
        enable_rerank=False,
        top_k=10,
        max_retrieval_rounds=1,
        enable_query_rewrite=False,
    )
    rag_bge_no_rerank = HybridRAG(
        retriever=new_retriever,
        config=config_no_rerank
    )

    # 新模型 - 有Rerank
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
    rag_bge_with_rerank = HybridRAG(
        retriever=new_retriever,
        config=config_with_rerank
    )

    # 原模型（all-MiniLM-L6-v2）- 使用默认 retriever
    print(f"\n   初始化原模型 (all-MiniLM-L6-v2) 实例...")
    original_retriever = KnowledgeRetriever()

    rag_minilm_no_rerank = HybridRAG(
        retriever=original_retriever,
        config=config_no_rerank
    )
    rag_minilm_with_rerank = HybridRAG(
        retriever=original_retriever,
        config=config_with_rerank
    )

    # ============ Step 4: 评估 ============
    print(f"\n{'=' * 72}")
    print("📊 开始评估")
    print(f"{'=' * 72}")

    # 用原模型加载景点（用于标注对比）
    original_attractions = load_all_attractions_from_store(original_retriever, CITY)
    if len(original_attractions) != len(all_attractions):
        print(f"⚠️ 警告: 两个集合景点数不同 (原:{len(original_attractions)}, 新:{len(all_attractions)})")
        # 用原模型的标注
        ground_truths_orig = {}
        for intent in QUERY_INTENTS:
            gt = build_ground_truth(original_attractions, intent)
            ground_truths_orig[intent["query"]] = gt
    else:
        ground_truths_orig = ground_truths

    all_results = []

    for intent in QUERY_INTENTS:
        query = intent["query"]
        gt_new = ground_truths[query]
        gt_orig = ground_truths_orig[query]

        print(f"\n{'─' * 72}")
        print(f"🧪 查询: '{query}'")
        print(f"{'─' * 72}")

        for top_k in top_k_values:
            # MiniLM (原模型)
            res_minilm_wo = evaluate_single_query(rag_minilm_no_rerank, intent, gt_orig, top_k)
            res_minilm_w = evaluate_single_query(rag_minilm_with_rerank, intent, gt_orig, top_k)

            # BGE (新模型)
            res_bge_wo = evaluate_single_query(rag_bge_no_rerank, intent, gt_new, top_k)
            res_bge_w = evaluate_single_query(rag_bge_with_rerank, intent, gt_new, top_k)

            print(f"\n  📊 Top-{top_k} 对比:")
            print(f"  {'配置':<30} {'Precision':>10} {'Recall':>10} {'NDCG':>10} {'MRR':>10} {'耗时(s)':>10}")
            print(f"  {'─' * 80}")
            print(f"  {'MiniLM (无Rerank)':<28} {res_minilm_wo['precision']:>10.4f} {res_minilm_wo['recall']:>10.4f} {res_minilm_wo['ndcg']:>10.4f} {res_minilm_wo['mrr']:>10.4f} {res_minilm_wo['elapsed']:>10.2f}")
            print(f"  {'MiniLM + Rerank':<28} {res_minilm_w['precision']:>10.4f} {res_minilm_w['recall']:>10.4f} {res_minilm_w['ndcg']:>10.4f} {res_minilm_w['mrr']:>10.4f} {res_minilm_w['elapsed']:>10.2f}")
            print(f"  {'BGE-zh (无Rerank)':<28} {res_bge_wo['precision']:>10.4f} {res_bge_wo['recall']:>10.4f} {res_bge_wo['ndcg']:>10.4f} {res_bge_wo['mrr']:>10.4f} {res_bge_wo['elapsed']:>10.2f}")
            print(f"  {'BGE-zh + Rerank':<28} {res_bge_w['precision']:>10.4f} {res_bge_w['recall']:>10.4f} {res_bge_w['ndcg']:>10.4f} {res_bge_w['mrr']:>10.4f} {res_bge_w['elapsed']:>10.2f}")

            # Top-5 详情
            if top_k == 5:
                print(f"\n  📋 Top-{top_k} 检索结果 (BGE-zh 无Rerank):")
                for j, t in enumerate(res_bge_wo["retrieved_titles"][:top_k], 1):
                    grade = gt_new["relevance_scores"].get(t, 0)
                    mark = f"G{grade}" if grade > 0 else "✗"
                    print(f"    {j}. [{mark}] {t}")

                print(f"\n  📋 Top-{top_k} 检索结果 (BGE-zh + Rerank):")
                for j, t in enumerate(res_bge_w["retrieved_titles"][:top_k], 1):
                    grade = gt_new["relevance_scores"].get(t, 0)
                    mark = f"G{grade}" if grade > 0 else "✗"
                    print(f"    {j}. [{mark}] {t}")

            all_results.append({
                "query": query,
                "top_k": top_k,
                "minilm_no_rerank": {k: res_minilm_wo[k] for k in ["precision", "recall", "ndcg", "mrr", "elapsed"]},
                "minilm_with_rerank": {k: res_minilm_w[k] for k in ["precision", "recall", "ndcg", "mrr", "elapsed"]},
                "bge_no_rerank": {k: res_bge_wo[k] for k in ["precision", "recall", "ndcg", "mrr", "elapsed"]},
                "bge_with_rerank": {k: res_bge_w[k] for k in ["precision", "recall", "ndcg", "mrr", "elapsed"]},
            })

    # ============ 汇总 ============
    print(f"\n{'=' * 72}")
    print("📈 平均指标汇总")
    print(f"{'=' * 72}")

    for top_k in top_k_values:
        subset = [r for r in all_results if r["top_k"] == top_k]

        def avg(config_key, metric_key):
            vals = [r[config_key][metric_key] for r in subset]
            return sum(vals) / len(vals) if vals else 0

        print(f"\n  Top-{top_k}:")
        print(f"  {'配置':<30} {'Precision':>10} {'Recall':>10} {'NDCG':>10} {'MRR':>10} {'耗时(s)':>10}")
        print(f"  {'─' * 80}")

        for config_name, config_key in [
            ("MiniLM (无Rerank)", "minilm_no_rerank"),
            ("MiniLM + Rerank", "minilm_with_rerank"),
            ("BGE-zh (无Rerank)", "bge_no_rerank"),
            ("BGE-zh + Rerank", "bge_with_rerank"),
        ]:
            p = avg(config_key, "precision")
            r = avg(config_key, "recall")
            n = avg(config_key, "ndcg")
            m = avg(config_key, "mrr")
            t = avg(config_key, "elapsed")
            print(f"  {config_name:<28} {p:>10.4f} {r:>10.4f} {n:>10.4f} {m:>10.4f} {t:>10.2f}")

    # ============ 提升率对比 ============
    print(f"\n{'=' * 72}")
    print("📊 提升率对比（相对 MiniLM 无Rerank 基线）")
    print(f"{'=' * 72}")
    print(f"\n  {'配置':<25} {'Precision':>12} {'Recall':>12} {'NDCG':>12} {'MRR':>12}")
    print(f"  {'─' * 73}")

    for top_k in top_k_values:
        subset = [r for r in all_results if r["top_k"] == top_k]
        print(f"\n  Top-{top_k}:")

        def avg(config_key, metric_key):
            vals = [r[config_key][metric_key] for r in subset]
            return sum(vals) / len(vals) if vals else 0

        baseline_p = avg("minilm_no_rerank", "precision")
        baseline_r = avg("minilm_no_rerank", "recall")
        baseline_n = avg("minilm_no_rerank", "ndcg")
        baseline_m = avg("minilm_no_rerank", "mrr")

        for config_name, config_key in [
            ("MiniLM + Rerank", "minilm_with_rerank"),
            ("BGE-zh (无Rerank)", "bge_no_rerank"),
            ("BGE-zh + Rerank", "bge_with_rerank"),
        ]:
            p = avg(config_key, "precision")
            r = avg(config_key, "recall")
            n = avg(config_key, "ndcg")
            m = avg(config_key, "mrr")

            def pct(new, base):
                if base == 0:
                    return "+∞%" if new > 0 else "0%"
                return f"{(new - base) / base * 100:+.1f}%"

            print(f"  {config_name:<23} {pct(p, baseline_p):>12} {pct(r, baseline_r):>12} {pct(n, baseline_n):>12} {pct(m, baseline_m):>12}")

    print(f"\n{'=' * 72}")
    print("✅ 评估完成")
    print("=" * 72)


if __name__ == "__main__":
    main()
