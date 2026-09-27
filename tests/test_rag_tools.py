"""RAG 工具测试脚本"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.rag import (
    RAGQueryTool,
    AttractionRecommendTool,
    AttractionDetailTool,
    GuideRecommendTool,
    RAGToolkit,
    KnowledgeBaseLoader,
    get_vector_store,
    KnowledgeRetriever,
)


def test_rag_tools():
    """测试RAG工具"""

    print("=" * 60)
    print("🔧 RAG 工具测试")
    print("=" * 60)

    # 初始化知识库数据
    print("\n📦 初始化知识库...")
    loader = KnowledgeBaseLoader()
    vector_store = get_vector_store()
    docs = loader.load_attraction_data("北京")
    try:
        vector_store.add_documents(docs)
        print(f"   ✅ 添加 {len(docs)} 条数据")
    except Exception as e:
        print(f"   ⚠️ 数据可能已存在: {e}")

    # 创建工具包
    print("\n🔨 创建RAG工具包...")
    retriever = KnowledgeRetriever(vector_store)
    toolkit = RAGToolkit(retriever)
    print(f"   ✅ 创建 {len(toolkit.tools)} 个工具:")
    for tool in toolkit.tools:
        print(f"      - {tool.name}")

    # 测试1: 通用查询工具
    print("\n" + "-" * 40)
    print("🔍 测试1: RAG通用查询工具")
    print("-" * 40)
    rag_query = toolkit.get_tool_by_name("rag_通用查询")
    result = rag_query.invoke({
        "query": "北京历史景点推荐",
        "top_k": 3,
        "city": "北京"
    })
    print(result)

    # 测试2: 景点推荐工具
    print("\n" + "-" * 40)
    print("🏛️ 测试2: 景点推荐工具")
    print("-" * 40)
    recommend_tool = toolkit.get_tool_by_name("景点推荐")
    result = recommend_tool.invoke({
        "city": "北京",
        "preferences": "历史文化,自然风光",
        "travel_days": 2
    })
    print(result)

    # 测试3: 景点详情工具
    print("\n" + "-" * 40)
    print("📍 测试3: 景点详情工具")
    print("-" * 40)
    detail_tool = toolkit.get_tool_by_name("景点详情")
    result = detail_tool.invoke({
        "attraction_name": "故宫",
        "city": "北京"
    })
    print(result)

    # 测试4: 旅行攻略工具
    print("\n" + "-" * 40)
    print("🗺️ 测试4: 旅行攻略工具")
    print("-" * 40)
    guide_tool = toolkit.get_tool_by_name("旅行攻略")
    result = guide_tool.invoke({
        "city": "北京",
        "theme": "深度游"
    })
    print(result)

    print("\n" + "=" * 60)
    print("✅ RAG工具测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    try:
        test_rag_tools()
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
