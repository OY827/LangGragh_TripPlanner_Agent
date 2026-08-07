"""
将知识库数据加载到云端Qdrant向量数据库
"""

import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

from app.rag.vector_store import QdrantVectorStore
from app.rag.data_collector import AttractionDataCollector


def main():
    print("="*60)
    print("将知识库数据加载到云端Qdrant向量数据库")
    print("="*60)
    
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    qdrant_collection = os.getenv("QDRANT_COLLECTION", "travel_knowledge")
    
    print(f"QDRANT_URL: {qdrant_url}")
    print(f"QDRANT_COLLECTION: {qdrant_collection}")
    
    vector_store = QdrantVectorStore(use_memory=False)
    print("✅ 初始化云端Qdrant成功")
    
    collector = AttractionDataCollector()
    
    print("\n1. 加载知识库数据:")
    all_data = collector.load_from_knowledge_base()
    print(f"   加载了 {len(all_data)} 条数据")
    
    if not all_data:
        print("❌ 没有找到数据")
        return
    
    print("\n2. 开始向Qdrant添加数据:")
    print("   这可能需要几分钟时间...")
    
    try:
        vector_store.add_documents(all_data)
        print(f"✅ 成功添加 {len(all_data)} 条数据到Qdrant")
    except Exception as e:
        print(f"❌ 添加数据失败: {e}")
        return
    
    print("\n3. 验证数据:")
    try:
        collection_info = vector_store.client.get_collection(collection_name=vector_store.default_collection)
        print(f"   集合名称: {vector_store.default_collection}")
        print(f"   向量数量: {collection_info.points_count}")
        print(f"   向量维度: {collection_info.config.params.vectors.size}")
    except Exception as e:
        print(f"❌ 获取统计信息失败: {e}")
        print("   尝试使用替代方法...")
        try:
            info = vector_store.get_collection_stats()
            print(f"   统计信息: {info}")
        except Exception as e2:
            print(f"   替代方法也失败: {e2}")
    
    print("\n4. 测试检索功能:")
    try:
        results = vector_store.search("北京历史文化景点", filters={"city": "北京"}, k=3)
        print(f"   检索到 {len(results)} 条结果")
        for i, result in enumerate(results):
            content = result.get("content", "")
            metadata = result.get("metadata", {})
            print(f"   {i+1}. {metadata.get('title', '')} - 评分: {metadata.get('rating', '')} - 价格: {metadata.get('price', '')}元")
    except Exception as e:
        print(f"❌ 检索测试失败: {e}")
    
    print("\n" + "="*60)
    print("加载完成!")
    print("="*60)


if __name__ == "__main__":
    main()