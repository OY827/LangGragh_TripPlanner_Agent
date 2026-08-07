"""删除Qdrant集合并重新构建知识库"""

import os
# os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7897'

from app.rag.vector_store import QdrantVectorStore
from app.rag.data_collector import AttractionDataCollector
from app.rag.splitter import DocumentSplitter

print("=== 删除旧集合 ===")
vector_store = QdrantVectorStore(use_memory=False)
collection_name = vector_store.default_collection

if vector_store.client.collection_exists(collection_name=collection_name):
    print(f"删除集合 '{collection_name}'...")
    vector_store.client.delete_collection(collection_name=collection_name)
    print("✅ 集合已删除")
else:
    print("集合不存在")

print("\n=== 重新构建知识库 ===")
collector = AttractionDataCollector()
splitter = DocumentSplitter()

all_data = collector.load_from_knowledge_base()
print(f"已加载 {len(all_data)} 条数据")

print(f"\n📊 共加载 {len(all_data)} 条数据")

print("\n=== 准备文档 ===")
documents = []
for item in all_data:
    documents.append({
        'content': item['content'],
        'metadata': item['metadata']
    })
print(f'共 {len(documents)} 个景点文档（不切分，保持完整性）')

print("\n=== 存储到向量数据库 ===")
vector_store.add_documents(documents)
print('✅ 数据已存储到 Qdrant')

print("\n=== 验证数据 ===")
results = vector_store.search('寺庙', filters={'city': '北京'}, k=5)
print(f"找到 {len(results)} 条结果:")
for r in results:
    title = r['metadata'].get('title', '未知')
    price = r['metadata'].get('price', 'N/A')
    duration = r['metadata'].get('duration', 'N/A')
    print(f"  - {title}: 票价={price}, 时长={duration}分钟")

print("\n✅ 知识库重建完成！")