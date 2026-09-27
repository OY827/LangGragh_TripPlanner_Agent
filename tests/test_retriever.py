"""Retriever 模块测试脚本"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.rag import KnowledgeBaseLoader, get_vector_store, KnowledgeRetriever


def test_retriever():
    """测试检索器功能"""

    print("=" * 60)
    print("🔍 Retriever 模块测试")
    print("=" * 60)

    # 初始化组件
    print("\n📦 初始化组件...")
    loader = KnowledgeBaseLoader()
    vector_store = get_vector_store()
    retriever = KnowledgeRetriever(vector_store)
    print("✅ 组件初始化完成")

    # 测试1: 加载知识库数据
    print("\n" + "-" * 40)
    print("📚 测试1: 加载知识库数据")
    print("-" * 40)
    docs = loader.load_attraction_data("北京")
    print(f"   加载 {len(docs)} 条景点数据")

    # 测试2: 添加数据到向量库
    print("\n" + "-" * 40)
    print("💾 测试2: 添加数据到向量库")
    print("-" * 40)
    try:
        vector_store.add_documents(docs)
        print("   ✅ 数据添加成功")
    except Exception as e:
        print(f"   ⚠️ 添加失败（可能已存在）: {e}")

    # 测试3: 基础相似度检索
    print("\n" + "-" * 40)
    print("🔎 测试3: 基础相似度检索")
    print("-" * 40)
    results = retriever.retrieve(
        query="北京历史文化景点",
        filters={"city": "北京"},
        top_k=3,
        search_type="similarity"
    )
    print(f"   检索到 {len(results)} 条结果:")
    for i, r in enumerate(results, 1):
        title = r["metadata"].get("title", "未知")
        score = r.get("score", 0)
        print(f"   {i}. {title} (相似度: {score:.4f})")

    # 测试4: MMR 多样性检索
    print("\n" + "-" * 40)
    print("🎯 测试4: MMR 多样性检索")
    print("-" * 40)
    results = retriever.retrieve(
        query="北京景点推荐",
        filters={"city": "北京"},
        top_k=3,
        search_type="mmr"
    )
    print(f"   检索到 {len(results)} 条结果:")
    for i, r in enumerate(results, 1):
        title = r["metadata"].get("title", "未知")
        category = r["metadata"].get("category", "未知")
        print(f"   {i}. {title} [{category}]")

    # 测试5: 景点专用检索
    print("\n" + "-" * 40)
    print("🏛️ 测试5: 景点专用检索")
    print("-" * 40)
    results = retriever.retrieve_attractions(
        city="北京",
        preferences=["历史文化"],
        travel_days=2
    )
    print(f"   推荐 {len(results)} 个景点:")
    for i, r in enumerate(results, 1):
        title = r["metadata"].get("title", "未知")
        rating = r["metadata"].get("rating", "N/A")
        price = r["metadata"].get("ticket_price", "N/A")
        duration = r["metadata"].get("visit_duration", 60)
        print(f"   {i}. {title}")
        print(f"      评分: {rating} | 票价: {price}元 | 游览时长: {duration}分钟")

    # 测试6: 批量检索
    print("\n" + "-" * 40)
    print("📦 测试6: 批量检索")
    print("-" * 40)
    queries = ["故宫历史", "长城攻略", "北京美食"]
    batch_results = retriever.batch_retrieve(
        queries=queries,
        filters={"city": "北京"},
        top_k=2
    )
    for query, results in batch_results.items():
        print(f"   查询「{query}」: {len(results)} 条结果")
        for r in results[:2]:
            print(f"      - {r['metadata'].get('title', '未知')}")

    print("\n" + "=" * 60)
    print("✅ 所有测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    try:
        test_retriever()
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
