import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.rag.data_loader import KnowledgeBaseLoader
from app.rag.vector_store import QdrantVectorStore


def init_knowledge_base():
    """初始化知识库到Qdrant向量数据库"""
    
    print("⏳ 正在加载知识库数据...")
    loader = KnowledgeBaseLoader()
    documents = loader.load_all_knowledge()
    
    if not documents:
        print("❌ 未找到知识库数据，请先添加数据")
        return
    
    print(f"✅ 成功加载 {len(documents)} 条文档")
    
    print("⏳ 正在连接Qdrant向量数据库...")
    try:
        vector_store = QdrantVectorStore(url="http://localhost:6333")
        print("✅ Qdrant连接成功")
    except Exception as e:
        print(f"❌ Qdrant连接失败: {e}")
        print("请确保Qdrant服务已启动，或使用Docker启动：")
        print("docker run -p 6333:6333 -v $(pwd)/qdrant_data:/qdrant/storage qdrant/qdrant")
        return
    
    print("⏳ 正在向向量库添加文档...")
    vector_store.add_documents(documents)
    
    # 获取统计信息
    stats = vector_store.get_collection_stats()
    print(f"✅ 知识库初始化完成！")
    print(f"   - 集合名称: {stats['collection_name']}")
    print(f"   - 向量数量: {stats['vector_count']}")
    print(f"   - 向量维度: {stats['dimension']}")
    
    # 测试检索功能
    print("\n🔍 测试检索功能...")
    results = vector_store.search("北京历史景点", k=3)
    print(f"搜索结果 ({len(results)} 条):")
    for i, result in enumerate(results, 1):
        print(f"  {i}. {result['metadata'].get('title', '未知')} (相似度: {result['score']:.4f})")


if __name__ == "__main__":
    init_knowledge_base()