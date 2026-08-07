"""
Rerank 效果量化评估脚本（上海测试用例）

评估指标：
1. Precision@K    - 检索结果中相关文档的比例
2. Recall@K       - 知识库相关文档被检索到的比例
3. NDCG@K         - 归一化折损累积增益（衡量排序质量）
4. MRR            - 平均倒数排名（第一个相关文档的位置）
5. 响应时间       - 端到端耗时
6. 分类命中率     - Top-K结果中匹配查询意图分类的比例
"""

import sys
import os
import time
import math
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from app.rag.hybrid_rag import HybridRAG, HybridRAGConfig
from app.rag.retriever import KnowledgeRetriever


def _load_eval_dataset():
    """优先加载 KB 对齐后的 GT；不存在则回退到内置占位（应先跑 recalibrate）。"""
    gt_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "evaluation",
        "gt_shanghai_kb_aligned.json",
    )
    if not os.path.exists(gt_path):
        raise FileNotFoundError(
            f"未找到 KB 对齐 GT: {gt_path}\n"
            f"请先运行: python scripts/recalibrate_shanghai_gt.py"
        )
    with open(gt_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    # 启动时校验：GT 标题必须全部存在于上海知识库
    kb_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "app",
        "knowledge_base",
        "attractions",
        "上海.json",
    )
    with open(kb_path, "r", encoding="utf-8") as f:
        kb = json.load(f)
    kb_titles = {
        (item.get("metadata") or item).get("title")
        for item in kb
    }
    for item in dataset:
        missing = [t for t in item["relevant_docs"] if t not in kb_titles]
        if missing:
            raise ValueError(f"GT 标题不在知识库中 ({item['query']}): {missing[:10]}")
    return dataset


# ============ 评估数据集（KB title 对齐版） ============
EVAL_DATASET = _load_eval_dataset()


def build_no_rerank_rag():
    """无 Rerank 配置"""
    config = HybridRAGConfig(
        enable_rerank=False,
        top_k=10,
        max_retrieval_rounds=1,
        enable_query_rewrite=False,
    )
    return HybridRAG(config=config)


def build_with_rerank_rag():
    """有 Rerank 配置（与线上工具一致：BGE + 意图加权）"""
    config = HybridRAGConfig(
        enable_rerank=True,
        rerank_model="BAAI/bge-reranker-base",
        retrieval_top_k=30,
        rerank_top_k=10,
        top_k=10,
        max_retrieval_rounds=1,
        enable_query_rewrite=False,
        enable_retrieval_validation=False,
        use_mmr=False,
        enable_intent_boost=True,
        intent_boost_weight=0.12,
    )
    return HybridRAG(config=config)


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


def compute_ndcg_at_k(retrieved_titles, relevant_set, k):
    """NDCG@K = DCG@K / IDCG@K"""
    top_k = retrieved_titles[:k]

    # DCG: 每个位置i的增益 = 2^rel(i) - 1，除以 log2(i+1)
    # rel(i) = 1 如果文档相关，0 否则
    dcg = 0.0
    for i, title in enumerate(top_k):
        rel = 1.0 if title in relevant_set else 0.0
        # 位置从1开始
        dcg += (2 ** rel - 1) / math.log2(i + 2)

    # IDCG: 理想排序（所有相关文档排在最前）
    num_relevant = min(len(relevant_set), k)
    idcg = 0.0
    for i in range(num_relevant):
        idcg += (2 ** 1 - 1) / math.log2(i + 2)

    return dcg / idcg if idcg > 0 else 0.0


def compute_mrr(retrieved_titles, relevant_set):
    """MRR = 1 / 第一个相关文档的排名"""
    for rank, title in enumerate(retrieved_titles, 1):
        if title in relevant_set:
            return 1.0 / rank
    return 0.0


def compute_category_hit_rate(retrieved_docs, intent_categories, k):
    """分类命中率：Top-K结果中匹配查询意图分类的比例"""
    top_k = retrieved_docs[:k]
    if not top_k:
        return 0.0, []

    matched = 0
    matched_titles = []
    for doc in top_k:
        metadata = doc.get("metadata", {})
        category = metadata.get("category", "")
        tags = metadata.get("tags", "")
        title = metadata.get("title", "")

        text = f"{category} {tags} {title}"
        if any(cat in text for cat in intent_categories):
            matched += 1
            matched_titles.append(title)

    return matched / len(top_k), matched_titles


def evaluate_single_query(rag_instance, item, top_k=10):
    """评估单个查询"""
    query = item["query"]
    city = item["city"]
    relevant_set = set(item["relevant_docs"])
    intent_categories = item["intent_categories"]

    t0 = time.time()
    docs, meta = rag_instance.retrieve_with_validation(
        query=query,
        city=city,
        preferences=list(item.get("intent_categories") or []),
    )
    elapsed = time.time() - t0

    retrieved_titles = [
        d.get("metadata", {}).get("title", "") for d in docs[:top_k]
    ]

    precision = compute_precision_at_k(retrieved_titles, relevant_set, top_k)
    recall = compute_recall_at_k(retrieved_titles, relevant_set, top_k)
    ndcg = compute_ndcg_at_k(retrieved_titles, relevant_set, top_k)
    mrr = compute_mrr(retrieved_titles, relevant_set)
    cat_hit_rate, cat_matched = compute_category_hit_rate(docs, intent_categories, top_k)

    return {
        "query": query,
        "city": city,
        "top_k": top_k,
        "precision": precision,
        "recall": recall,
        "ndcg": ndcg,
        "mrr": mrr,
        "category_hit_rate": cat_hit_rate,
        "category_matched": cat_matched,
        "retrieved_titles": retrieved_titles,
        "relevant_count": len(relevant_set),
        "elapsed": elapsed,
        "rerank_enabled": meta.get("rerank_enabled", False),
        "rerank_info": meta.get("rerank_info", {}),
    }


def print_metric_comparison(name_wo, name_w, label, val_wo, val_w):
    diff = val_w - val_wo
    pct = (diff / val_wo * 100) if val_wo > 0 else float("inf")
    arrow = "↑" if diff > 0 else ("↓" if diff < 0 else "→")
    print(f"  {label}: {val_wo:.4f} → {val_w:.4f}  {arrow} {pct:+.1f}% ({diff:+.4f})")


def main():
    print("=" * 72)
    print("🔬 Rerank 效果量化评估（上海测试用例 · KB title 对齐 GT）")
    print("=" * 72)
    print(f"GT 来源: evaluation/gt_shanghai_kb_aligned.json")
    print(f"Rerank 模型: BAAI/bge-reranker-base")

    top_k_values = [3, 5, 10]

    # ============ 初始化 ============
    print("\n📦 初始化 HybridRAG 实例...")
    rag_no_rerank = build_no_rerank_rag()
    rag_with_rerank = build_with_rerank_rag()

    # ============ 逐查询评估 ============
    all_results = []

    for item in EVAL_DATASET:
        print(f"\n{'─' * 72}")
        print(f"🧪 查询: '{item['query']}' (城市: {item['city']})")
        print(f"   意图分类: {item['intent_categories']}")
        print(f"   Ground Truth 相关文档数: {len(item['relevant_docs'])}")
        print(f"{'─' * 72}")

        for top_k in top_k_values:
            res_wo = evaluate_single_query(rag_no_rerank, item, top_k)
            res_w = evaluate_single_query(rag_with_rerank, item, top_k)

            print(f"\n  📊 Top-{top_k} 指标对比:")
            print_metric_comparison(
                "无Rerank", "有Rerank", "Precision",
                res_wo["precision"], res_w["precision"]
            )
            print_metric_comparison(
                "无Rerank", "有Rerank", "Recall",
                res_wo["recall"], res_w["recall"]
            )
            print_metric_comparison(
                "无Rerank", "有Rerank", "NDCG",
                res_wo["ndcg"], res_w["ndcg"]
            )
            print_metric_comparison(
                "无Rerank", "有Rerank", "MRR",
                res_wo["mrr"], res_w["mrr"]
            )
            print_metric_comparison(
                "无Rerank", "有Rerank", "分类命中率",
                res_wo["category_hit_rate"], res_w["category_hit_rate"]
            )
            print_metric_comparison(
                "无Rerank", "有Rerank", "响应时间(s)",
                res_wo["elapsed"], res_w["elapsed"]
            )

            # Top-5 文档对比
            if top_k == 5:
                print(f"\n  📋 Top-{top_k} 检索结果详情:")
                print(f"    无Rerank:")
                for j, t in enumerate(res_wo["retrieved_titles"][:top_k], 1):
                    mark = "✓" if t in set(item["relevant_docs"]) else "✗"
                    print(f"      {j}. {mark} {t}")
                print(f"    有Rerank:")
                for j, t in enumerate(res_w["retrieved_titles"][:top_k], 1):
                    mark = "✓" if t in set(item["relevant_docs"]) else "✗"
                    print(f"      {j}. {mark} {t}")

            all_results.append({
                "query": item["query"],
                "top_k": top_k,
                "wo_rerank": {
                    "precision": res_wo["precision"],
                    "recall": res_wo["recall"],
                    "ndcg": res_wo["ndcg"],
                    "mrr": res_wo["mrr"],
                    "category_hit_rate": res_wo["category_hit_rate"],
                    "elapsed": res_wo["elapsed"],
                },
                "with_rerank": {
                    "precision": res_w["precision"],
                    "recall": res_w["recall"],
                    "ndcg": res_w["ndcg"],
                    "mrr": res_w["mrr"],
                    "category_hit_rate": res_w["category_hit_rate"],
                    "elapsed": res_w["elapsed"],
                },
            })

    # ============ 汇总统计 ============
    print(f"\n{'=' * 72}")
    print("📈 汇总统计（所有查询的平均值）")
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
        c_wo, c_w = avg("category_hit_rate")
        t_wo, t_w = avg("elapsed")

        print(f"\n  Top-{top_k}:")
        print_metric_comparison("无Rerank", "有Rerank", "Precision@K", p_wo, p_w)
        print_metric_comparison("无Rerank", "有Rerank", "Recall@K", r_wo, r_w)
        print_metric_comparison("无Rerank", "有Rerank", "NDCG@K", n_wo, n_w)
        print_metric_comparison("无Rerank", "有Rerank", "MRR", m_wo, m_w)
        print_metric_comparison("无Rerank", "有Rerank", "分类命中率", c_wo, c_w)
        print_metric_comparison("无Rerank", "有Rerank", "响应时间(s)", t_wo, t_w)

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
        ("分类命中率", "category_hit_rate"),
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
