"""
Rerank 检索测试脚本

以上海为测试用例，对比 Rerank 前后的检索结果差异
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from app.rag.hybrid_rag import HybridRAG, HybridRAGConfig


def test_rerank():
    """测试 Rerank 检索效果"""

    print("=" * 70)
    print("🔧 Rerank 检索测试（上海）")
    print("=" * 70)

    # ========== 配置1: 不使用Rerank（对照组） ==========
    print("\n📋 配置1: 不使用Rerank（对照组）")
    config_no_rerank = HybridRAGConfig(
        enable_rerank=False,
        top_k=10,
        max_retrieval_rounds=1,
        enable_query_rewrite=False
    )

    rag_no_rerank = HybridRAG(config=config_no_rerank)

    # ========== 配置2: 使用Rerank（实验组） ==========
    print("\n📋 配置2: 使用Rerank（实验组）")
    config_with_rerank = HybridRAGConfig(
        enable_rerank=True,
        rerank_model="cross-encoder/ms-marco-MiniLM-L-6-v2",  # 轻量模型(~80MB)
        retrieval_top_k=30,     # 初步检索30篇
        rerank_top_k=10,        # Rerank后保留10篇
        top_k=10,
        max_retrieval_rounds=1,
        enable_query_rewrite=False,
        enable_retrieval_validation=False,  # Rerank模式下禁用检索验证
        use_mmr=False           # Rerank模式不用MMR
    )

    rag_with_rerank = HybridRAG(config=config_with_rerank)

    # ========== 测试查询 ==========
    test_cases = [
        {
            "query": "上海的历史文化景点",
            "city": "上海",
            "description": "历史文化类查询"
        },
        {
            "query": "上海适合亲子游玩的地方",
            "city": "上海",
            "description": "亲子游查询"
        },
        {
            "query": "上海免费景点推荐",
            "city": "上海",
            "description": "免费景点查询"
        },
    ]

    for i, case in enumerate(test_cases, 1):
        print(f"\n{'─' * 70}")
        print(f"🧪 测试{i}: {case['description']}")
        print(f"   查询: '{case['query']}'")
        print(f"   城市: {case['city']}")
        print(f"{'─' * 70}")

        # 无Rerank
        print(f"\n  📌 无Rerank结果:")
        t1 = time.time()
        docs_no_rerank, meta_no = rag_no_rerank.retrieve_with_validation(
            query=case["query"],
            city=case["city"],
            preferences=[]
        )
        time_no = time.time() - t1

        for j, doc in enumerate(docs_no_rerank[:5], 1):
            title = doc.get("metadata", {}).get("title", "?")
            score = doc.get("score", 0)
            category = doc.get("metadata", {}).get("category", "?")
            price = doc.get("metadata", {}).get("price", "?")
            print(f"     {j}. [{category}] {title} (门票:{price}元, 向量分:{score:.4f})")
        print(f"   ⏱️ 耗时: {time_no:.2f}s, 文档数: {len(docs_no_rerank)}")

        # 有Rerank
        print(f"\n  📌 有Rerank结果:")
        t2 = time.time()
        docs_rerank, meta_rerank = rag_with_rerank.retrieve_with_validation(
            query=case["query"],
            city=case["city"],
            preferences=[]
        )
        time_rerank = time.time() - t2

        for j, doc in enumerate(docs_rerank[:5], 1):
            title = doc.get("metadata", {}).get("title", "?")
            rerank_score = doc.get("rerank_score", 0)
            original_score = doc.get("score", 0)
            category = doc.get("metadata", {}).get("category", "?")
            price = doc.get("metadata", {}).get("price", "?")
            print(f"     {j}. [{category}] {title} (门票:{price}元, 重排分:{rerank_score:.4f}, 原始分:{original_score:.4f})")
        print(f"   ⏱️ 耗时: {time_rerank:.2f}s, 文档数: {len(docs_rerank)}")
        print(f"   📊 Rerank模式: {meta_rerank.get('rerank_info', {}).get('mode', 'N/A')}")

        # 对比分析
        print(f"\n  📊 对比分析:")

        # 排名变化
        no_rerank_titles = [d.get("metadata", {}).get("title", "") for d in docs_no_rerank[:5]]
        rerank_titles = [d.get("metadata", {}).get("title", "") for d in docs_rerank[:5]]

        changed = 0
        for j in range(min(len(no_rerank_titles), len(rerank_titles))):
            if no_rerank_titles[j] != rerank_titles[j]:
                changed += 1
        print(f"     Top-5排名变化: {changed} 个位置调整")

        # 新进入Top-5的文档
        new_in_top5 = set(rerank_titles) - set(no_rerank_titles)
        if new_in_top5:
            print(f"     新进入Top-5: {new_in_top5}")

        # 跌出Top-5的文档
        dropped = set(no_rerank_titles) - set(rerank_titles)
        if dropped:
            print(f"     跌出Top-5: {dropped}")

        # 耗时对比
        print(f"     耗时对比: {time_no:.2f}s → {time_rerank:.2f}s (+{time_rerank-time_no:.2f}s)")

    print(f"\n{'=' * 70}")
    print("✅ Rerank 测试完成")
    print("=" * 70)


if __name__ == "__main__":
    test_rerank()
