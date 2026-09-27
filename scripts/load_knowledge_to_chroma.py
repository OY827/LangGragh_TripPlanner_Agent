"""
将知识库数据加载到本地 Chroma 向量数据库
"""

import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

from app.rag.vector_store import get_vector_store
from app.rag.data_collector import AttractionDataCollector


def main():
    print("=" * 60)
    print("将知识库数据加载到本地 Chroma")
    print("=" * 60)

    os.environ["VECTOR_BACKEND"] = os.getenv("VECTOR_BACKEND", "chroma")
    vector_store = get_vector_store()
    print(f"✅ 初始化向量库成功: {vector_store.default_collection}")
    stats_before = vector_store.get_collection_stats()
    if not stats_before.get("error"):
        print(f"   已有向量: {stats_before.get('vector_count')}")

    collector = AttractionDataCollector()

    print("\n1. 加载知识库数据:")
    all_data = collector.load_from_knowledge_base()
    print(f"   加载了 {len(all_data)} 条数据")

    if not all_data:
        print("❌ 没有找到数据")
        return

    if vector_store.collection_exists():
        print(f"\n2. 删除旧集合 '{vector_store.default_collection}' 后重建")
        vector_store.delete_collection()
    else:
        print("\n2. 集合不存在，直接写入")

    print("   正在写入（需要调用 Embedding API，可能需要几分钟）...")
    try:
        vector_store.add_documents(all_data)
        print(f"✅ 成功添加 {len(all_data)} 条数据")
    except Exception as e:
        print(f"❌ 添加数据失败: {e}")
        return

    print("\n3. 验证数据:")
    info = vector_store.get_collection_stats()
    print(f"   统计信息: {info}")

    print("\n4. 测试检索功能:")
    try:
        results = vector_store.search("北京历史文化景点", filters={"city": "北京"}, k=3)
        print(f"   检索到 {len(results)} 条结果")
        for i, result in enumerate(results):
            metadata = result.get("metadata", {})
            print(
                f"   {i + 1}. {metadata.get('title', '')} - "
                f"评分: {metadata.get('rating', '')} - "
                f"价格: {metadata.get('price', '')}元"
            )
    except Exception as e:
        print(f"❌ 检索测试失败: {e}")

    print("\n" + "=" * 60)
    print("加载完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
