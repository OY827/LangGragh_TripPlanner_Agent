import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from app.rag.data_loader import KnowledgeBaseLoader
from app.rag.vector_store import get_vector_store


def init_knowledge_base():
    """初始化知识库到本地向量数据库（默认 Chroma）"""

    print("⏳ 正在加载知识库数据...")
    loader = KnowledgeBaseLoader()
    documents = loader.load_all_knowledge()

    if not documents:
        print("❌ 未找到知识库数据，请先添加数据")
        return

    print(f"✅ 成功加载 {len(documents)} 条文档")

    print("⏳ 正在初始化向量数据库...")
    try:
        vector_store = get_vector_store()
        print("✅ 向量库初始化成功")
    except Exception as e:
        print(f"❌ 向量库初始化失败: {e}")
        return

    if vector_store.collection_exists():
        print(f"删除已有集合 '{vector_store.default_collection}'...")
        vector_store.delete_collection()

    print("⏳ 正在向向量库添加文档...")
    vector_store.add_documents(documents)

    stats = vector_store.get_collection_stats()
    print("✅ 知识库初始化完成！")
    print(f"   - 集合名称: {stats.get('collection_name')}")
    print(f"   - 向量数量: {stats.get('vector_count')}")
    print(f"   - 向量维度: {stats.get('dimension')}")
    if stats.get("persist_directory"):
        print(f"   - 本地路径: {stats['persist_directory']}")

    print("\n🔍 测试检索功能...")
    results = vector_store.search("北京历史景点", filters={"city": "北京"}, k=3)
    print(f"搜索结果 ({len(results)} 条):")
    for i, result in enumerate(results, 1):
        print(f"  {i}. {result['metadata'].get('title', '未知')} (相似度: {result['score']:.4f})")


if __name__ == "__main__":
    init_knowledge_base()
