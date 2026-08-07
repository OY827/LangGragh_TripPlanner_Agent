"""
连接云Qdrant的测试脚本
用于测试云Qdrant连接和真实数据评估
"""

import json
import os
import sys
from datetime import datetime
from typing import List, Dict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from app.rag.retriever import KnowledgeRetriever


class CloudRAGEvaluator:
    """连接云Qdrant的RAG评估器"""

    _CATEGORY_KEYWORDS = {
        "历史文化": {
            "positive": ["历史", "古迹", "遗址", "文物", "古城", "古镇", "古寺", "故居", "博物馆", "博物院", "陵墓", "宫殿", "王府", "塔", "桥梁", "革命", "纪念馆", "故宫", "天坛", "颐和园", "圆明园", "长城", "寺庙", "庙宇", "古桥", "石窟", "石刻"],
            "negative": ["展览馆", "展览", "会展", "会议", "美术馆", "艺术", "剧院", "剧场", "影城", "科技馆", "科学", "数码", "娱乐", "游戏", "演唱会", "晚会"]
        },
        "自然风光": {
            "positive": ["自然", "山", "湖", "河", "海", "瀑布", "森林", "湿地", "溪流", "茶园", "植物", "风景", "景区", "峡谷", "温泉", "岛", "滩", "江", "溪"],
            "negative": ["公园", "乐园", "广场", "城市", "体育", "运动", "度假", "购物", "商场"]
        },
        "美食": {
            "positive": ["美食", "小吃", "餐厅", "夜市", "餐饮", "特色菜", "美食街"],
            "negative": ["购物", "商场", "商业街", "步行街", "公园"]
        },
        "购物": {
            "positive": ["购物", "商场", "商业街", "购物中心", "步行街"],
            "negative": ["美食", "小吃", "餐厅", "公园", "自然", "风景"]
        },
        "主题乐园": {
            "positive": ["乐园", "游乐场", "欢乐谷", "迪士尼", "环球影城", "动物园", "海洋公园", "度假区"],
            "negative": ["公园", "自然", "风景", "购物", "商场"]
        },
    }

    def __init__(self):
        print("正在连接云Qdrant...")
        self.retriever = KnowledgeRetriever()
        print("✅ 云Qdrant连接成功")

    def _match_category(self, category_name: str, doc_category: str, doc_tags: str, doc_title: str) -> bool:
        """判断文档是否匹配指定类别（严格匹配）"""
        category_config = self._CATEGORY_KEYWORDS.get(category_name, {})
        positive_keywords = category_config.get("positive", [])
        negative_keywords = category_config.get("negative", [])
        
        text_to_check = f"{doc_category} {doc_tags} {doc_title}"
        
        has_positive = any(keyword in text_to_check for keyword in positive_keywords)
        has_negative = any(keyword in text_to_check for keyword in negative_keywords)
        
        return has_positive and not has_negative

    def _get_relevant_docs_from_knowledge_base(self, query: str, filters: Dict) -> List[str]:
        """从知识库中动态获取所有相关文档"""
        matched_categories = []
        for category_name, config in self._CATEGORY_KEYWORDS.items():
            if any(keyword in query for keyword in config.get("positive", [])):
                matched_categories.append(category_name)
        
        if not matched_categories:
            return []
        
        all_docs = self.retriever.retrieve(query, filters=filters, top_k=200)
        relevant_docs = []
        for doc in all_docs:
            metadata = doc.get("metadata", {})
            doc_category = metadata.get("category", "")
            doc_tags = metadata.get("tags", "")
            doc_title = metadata.get("title", "")
            
            if any(self._match_category(cat, doc_category, doc_tags, doc_title) for cat in matched_categories):
                relevant_docs.append(doc_title)
        
        return list(set(relevant_docs))

    def evaluate(self, eval_dataset: List[Dict], top_k: int = 10) -> Dict:
        """评估召回率和精确率"""
        results = []
        
        for item in eval_dataset:
            query = item["query"]
            filters = item.get("filters", {})
            city = filters.get("city", "")
            
            print(f"\n正在评估: {query}")
            
            relevant_docs = self._get_relevant_docs_from_knowledge_base(query, filters)
            print(f"  知识库中相关文档数: {len(relevant_docs)}")
            
            if relevant_docs:
                print(f"  相关文档列表: {', '.join(relevant_docs[:10])}{'...' if len(relevant_docs) > 10 else ''}")
            
            if not relevant_docs:
                results.append({
                    "query": query,
                    "city": city,
                    "relevant_docs_count": 0,
                    "recall": 0,
                    "precision": 0,
                    "matched_docs": [],
                    "retrieved_titles": [],
                })
                continue
            
            retrieved = self.retriever.retrieve(query, filters=filters, top_k=top_k)
            retrieved_titles = [r.get("metadata", {}).get("title", "") for r in retrieved]
            
            print(f"  检索结果: {', '.join(retrieved_titles)}")
            
            matched_count = 0
            matched_docs = []
            for doc in relevant_docs:
                if any(doc in title for title in retrieved_titles):
                    matched_count += 1
                    matched_docs.append(doc)
            
            misidentified = [t for t in retrieved_titles if t not in matched_docs]
            
            recall = matched_count / len(relevant_docs)
            precision = matched_count / len(retrieved_titles) if retrieved_titles else 0
            
            print(f"  命中相关文档: {', '.join(matched_docs)}")
            print(f"  误检文档: {', '.join(misidentified) if misidentified else '无'}")
            print(f"  召回率: {recall:.2%} (命中 {matched_count}/{len(relevant_docs)})")
            print(f"  精确率: {precision:.2%} (相关 {matched_count}/{len(retrieved_titles)})")
            
            results.append({
                "query": query,
                "city": city,
                "relevant_docs_count": len(relevant_docs),
                "matched_count": matched_count,
                "matched_docs": matched_docs,
                "retrieved_titles": retrieved_titles,
                "misidentified": misidentified,
                "recall": recall,
                "precision": precision,
            })
        
        avg_recall = sum(r["recall"] for r in results) / len(results) if results else 0
        avg_precision = sum(r["precision"] for r in results) / len(results) if results else 0
        
        return {
            "avg_recall": avg_recall,
            "avg_precision": avg_precision,
            "num_queries": len(results),
            "results": results
        }


def build_eval_dataset() -> List[Dict]:
    """构建评估数据集"""
    return [
        {"query": "上海历史文化景点", "filters": {"city": "上海"}},
        {"query": "北京历史古迹", "filters": {"city": "北京"}},
        {"query": "杭州自然风光", "filters": {"city": "杭州"}},
        {"query": "成都美食", "filters": {"city": "成都"}},
        {"query": "广州购物", "filters": {"city": "广州"}},
    ]


def print_report(results: Dict):
    """打印评估报告"""
    print("\n" + "=" * 70)
    print("云Qdrant RAG评估报告")
    print("=" * 70)
    print(f"评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"查询数量: {results['num_queries']}")
    print(f"平均召回率: {results['avg_recall']:.2%}")
    print(f"平均精确率: {results['avg_precision']:.2%}")


if __name__ == "__main__":
    print("=" * 70)
    print("云Qdrant RAG评估测试")
    print("=" * 70)
    print(f"Qdrant URL: {os.getenv('QDRANT_URL')}")
    print(f"Qdrant Collection: {os.getenv('QDRANT_COLLECTION')}")
    print(f"Embedding Backend: {os.getenv('EMBEDDING_BACKEND')}")
    print("=" * 70)
    
    try:
        evaluator = CloudRAGEvaluator()
        eval_dataset = build_eval_dataset()
        results = evaluator.evaluate(eval_dataset, top_k=10)
        print_report(results)
    except Exception as e:
        print(f"\n❌ 评估失败: {e}")
        import traceback
        traceback.print_exc()