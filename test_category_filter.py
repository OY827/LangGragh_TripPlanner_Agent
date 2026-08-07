"""测试分类过滤检索"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.rag.retriever import KnowledgeRetriever

def test_category_filter():
    """测试分类过滤检索"""
    print("=== 测试分类过滤检索 ===")
    
    retriever = KnowledgeRetriever()
    
    test_cases = [
        ("上海", "历史文化"),
        ("北京", "历史文化"),
        ("上海", "自然风光"),
    ]
    
    for city, category in test_cases:
        print(f"\n📍 查询：{city} {category}")
        results = retriever.retrieve(
            query=category,
            filters={"city": city, "category": category},
            top_k=10,
            search_type="hybrid"
        )
        
        print(f"找到 {len(results)} 条结果:")
        for r in results:
            title = r['metadata'].get('title', '未知')
            cat = r['metadata'].get('category', '未知')
            score = r.get('score', 0)
            print(f"   - {title} (分类: {cat}, 相似度: {score:.4f})")

if __name__ == "__main__":
    test_category_filter()