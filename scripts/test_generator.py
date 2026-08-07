"""生成器模块测试脚本"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.rag import AnswerGenerator, KnowledgeBaseLoader, QdrantVectorStore, KnowledgeRetriever


def test_generator():
    """测试生成器功能"""

    print("=" * 60)
    print("📝 生成器模块测试")
    print("=" * 60)

    # 初始化组件
    print("\n📦 初始化组件...")
    generator = AnswerGenerator(provider="aliyun")
    loader = KnowledgeBaseLoader()
    vector_store = QdrantVectorStore(url="http://localhost:6333")
    retriever = KnowledgeRetriever(vector_store)
    print("✅ 组件初始化完成")

    # 准备测试数据
    print("\n📚 准备测试数据...")
    docs = loader.load_attraction_data("北京")
    try:
        vector_store.add_documents(docs)
        print(f"   ✅ 添加 {len(docs)} 条数据")
    except Exception as e:
        print(f"   ⚠️ 数据可能已存在: {e}")

    # 测试1: 问答生成
    print("\n" + "-" * 40)
    print("💬 测试1: 问答生成")
    print("-" * 40)
    context = [
        {
            "content": "故宫是明清两代的皇家宫殿，位于北京中心。始建于1406年，历时14年建成，是世界上现存规模最大、保存最为完整的木质结构古建筑群。",
            "metadata": {"title": "故宫"}
        },
        {
            "content": "故宫占地面积约72万平方米，建筑面积约15万平方米，有宫殿9999间半。著名景点包括太和殿、中和殿、保和殿等。",
            "metadata": {"title": "故宫介绍"}
        }
    ]
    answer = generator.generate("故宫是什么？", context)
    print(f"问题: 故宫是什么？")
    print(f"回答:\n{answer}")

    # 测试2: 行程规划生成
    print("\n" + "-" * 40)
    print("🗺️ 测试2: 行程规划生成")
    print("-" * 40)
    attractions = [
        {
            "metadata": {
                "title": "故宫",
                "category": "历史文化",
                "rating": 4.9,
                "ticket_price": 60,
                "visit_duration": 240
            },
            "content": "故宫是世界上现存规模最大的木质结构古建筑群。"
        },
        {
            "metadata": {
                "title": "天坛",
                "category": "历史文化",
                "rating": 4.8,
                "ticket_price": 35,
                "visit_duration": 120
            },
            "content": "天坛是明清两代皇帝祭天的场所。"
        },
        {
            "metadata": {
                "title": "颐和园",
                "category": "自然风光",
                "rating": 4.8,
                "ticket_price": 30,
                "visit_duration": 180
            },
            "content": "颐和园是中国现存规模最大的皇家园林。"
        }
    ]
    itinerary = generator.generate_itinerary("北京", 2, attractions, ["历史文化", "自然风光"])
    print(f"行程:\n{itinerary}")

    # 测试3: 景点介绍生成
    print("\n" + "-" * 40)
    print("📍 测试3: 景点介绍生成")
    print("-" * 40)
    intro = generator.generate_attraction_intro("故宫", "北京", context)
    print(f"景点介绍:\n{intro}")

    # 测试4: 完整RAG流程（检索+生成）
    print("\n" + "-" * 40)
    print("🔄 测试4: 完整RAG流程")
    print("-" * 40)
    query = "北京有哪些历史文化景点值得推荐？"
    results = retriever.retrieve(query, filters={"city": "北京"}, top_k=3)
    print(f"检索到 {len(results)} 条相关知识")
    rag_answer = generator.generate(query, results)
    print(f"RAG回答:\n{rag_answer}")

    print("\n" + "=" * 60)
    print("✅ 生成器测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    print("\n⚠️ 注意: 本测试需要配置阿里云通义千问API密钥")
    print("请确保已设置环境变量 ALIBABA_CLOUD_API_KEY\n")

    try:
        test_generator()
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
