"""
RAG离线评估脚本 - RAG Offline Evaluator
评估RAG系统的召回率、精确率等指标

改进方案：基于分类标签的动态评估
不再预设固定答案列表，而是根据景点的category/tags标签动态判断相关性
"""

import json
import os
import sys
from datetime import datetime
from typing import List, Dict, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.rag.retriever import KnowledgeRetriever


class RAGEvaluator:
    """RAG评估器"""

    _CATEGORY_KEYWORDS = {
        "历史文化": {
            "positive": ["历史", "古迹", "遗址", "文物", "古城", "古镇", "古寺", "故居", "博物馆", "陵墓", "宫殿", "王府", "塔", "桥梁", "革命", "纪念馆", "故居"],
            "negative": ["展览馆", "展览", "会展", "会议", "中心", "美术馆", "艺术", "剧院", "剧场", "影城", "科技馆", "科学"]
        },
        "自然风光": {
            "positive": ["自然", "山", "湖", "河", "海", "瀑布", "森林", "湿地", "溪流", "茶园", "植物", "风景", "景区", "峡谷", "温泉"],
            "negative": ["公园", "乐园", "广场", "城市", "体育", "运动"]
        },
        "美食": {
            "positive": ["美食", "小吃", "餐厅", "夜市"],
            "negative": ["购物", "商场", "商业街", "步行街"]
        },
        "购物": {
            "positive": ["购物", "商场", "商业街", "购物中心"],
            "negative": ["美食", "小吃", "餐厅", "公园"]
        },
        "主题乐园": {
            "positive": ["乐园", "游乐场", "欢乐谷", "迪士尼", "环球影城", "动物园"],
            "negative": ["公园", "自然", "风景"]
        },
    }

    def __init__(self, retriever=None):
        self.retriever = retriever or KnowledgeRetriever()
        self.results = []

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

    def evaluate_recall(self, eval_dataset: List[Dict], top_k: int = 10) -> Dict:
        """评估召回率 Recall@K"""
        results = []
        
        for item in eval_dataset:
            query = item["query"]
            filters = item.get("filters", {})
            
            relevant_docs = self._get_relevant_docs_from_knowledge_base(query, filters)
            
            if not relevant_docs:
                results.append({
                    "query": query,
                    "relevant_docs_count": 0,
                    "retrieved_titles": [],
                    "matched_count": 0,
                    "matched_docs": [],
                    "recall": 0,
                    "mrr": 0,
                    "total_relevant": 0
                })
                continue
            
            retrieved = self.retriever.retrieve(query, filters=filters, top_k=top_k)
            retrieved_titles = [r.get("metadata", {}).get("title", "") for r in retrieved]
            
            matched_count = 0
            matched_docs = []
            for doc in relevant_docs:
                if any(doc in title for title in retrieved_titles):
                    matched_count += 1
                    matched_docs.append(doc)
            
            recall = matched_count / len(relevant_docs) if relevant_docs else 0
            
            mrr = 0
            for rank, title in enumerate(retrieved_titles, 1):
                if any(doc in title for doc in relevant_docs):
                    mrr = 1 / rank
                    break
            
            results.append({
                "query": query,
                "relevant_docs_count": len(relevant_docs),
                "retrieved_titles": retrieved_titles[:top_k],
                "matched_count": matched_count,
                "matched_docs": matched_docs,
                "recall": recall,
                "mrr": mrr,
                "total_relevant": len(relevant_docs),
                "max_possible_recall": min(top_k, len(relevant_docs)) / len(relevant_docs)
            })
        
        avg_recall = sum(r["recall"] for r in results) / len(results) if results else 0
        avg_mrr = sum(r["mrr"] for r in results) / len(results) if results else 0
        
        return {
            "metric": "recall",
            "avg_recall": avg_recall,
            "avg_mrr": avg_mrr,
            "top_k": top_k,
            "num_queries": len(results),
            "detailed": results
        }

    def evaluate_precision(self, eval_dataset: List[Dict], top_k: int = 10) -> Dict:
        """评估精确率 Precision@K"""
        results = []
        
        for item in eval_dataset:
            query = item["query"]
            filters = item.get("filters", {})
            
            relevant_docs = self._get_relevant_docs_from_knowledge_base(query, filters)
            retrieved = self.retriever.retrieve(query, filters=filters, top_k=top_k)
            retrieved_titles = [r.get("metadata", {}).get("title", "") for r in retrieved]
            
            relevant_retrieved = 0
            for title in retrieved_titles:
                if any(doc in title for doc in relevant_docs):
                    relevant_retrieved += 1
            
            precision = relevant_retrieved / len(retrieved_titles) if retrieved_titles else 0
            
            results.append({
                "query": query,
                "precision": precision,
                "relevant_retrieved": relevant_retrieved,
                "total_retrieved": len(retrieved_titles),
                "relevant_docs_count": len(relevant_docs)
            })
        
        avg_precision = sum(r["precision"] for r in results) / len(results) if results else 0
        
        return {
            "metric": "precision",
            "avg_precision": avg_precision,
            "top_k": top_k,
            "num_queries": len(results),
            "detailed": results
        }

    def evaluate_f1(self, eval_dataset: List[Dict], top_k: int = 10) -> Dict:
        """评估F1分数"""
        recall_result = self.evaluate_recall(eval_dataset, top_k)
        precision_result = self.evaluate_precision(eval_dataset, top_k)
        
        recall = recall_result["avg_recall"]
        precision = precision_result["avg_precision"]
        
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            "metric": "f1",
            "avg_f1": f1,
            "avg_precision": precision,
            "avg_recall": recall,
            "top_k": top_k,
            "num_queries": len(eval_dataset)
        }

    def run_full_evaluation(self, eval_dataset: List[Dict], top_k_list: List[int] = [5, 10, 20, 50]) -> Dict:
        """运行完整评估"""
        results = {}
        
        for top_k in top_k_list:
            recall_result = self.evaluate_recall(eval_dataset, top_k)
            precision_result = self.evaluate_precision(eval_dataset, top_k)
            f1_result = self.evaluate_f1(eval_dataset, top_k)
            
            results[f"top_{top_k}"] = {
                "recall": recall_result,
                "precision": precision_result,
                "f1": f1_result
            }
        
        return {
            "evaluation_time": datetime.now().isoformat(),
            "num_queries": len(eval_dataset),
            "results": results
        }

    def save_results(self, results: Dict, filename: str):
        """保存评估结果"""
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    def load_eval_dataset(self, filename: str) -> List[Dict]:
        """加载评估数据集"""
        with open(filename, "r", encoding="utf-8") as f:
            return json.load(f)


def build_default_eval_dataset() -> List[Dict]:
    """构建默认评估数据集（只包含查询和过滤条件，不再预设固定答案）"""
    return [
        {
            "query": "上海历史文化景点",
            "filters": {"city": "上海"}
        },
        {
            "query": "北京历史古迹",
            "filters": {"city": "北京"}
        },
        {
            "query": "杭州自然风光",
            "filters": {"city": "杭州"}
        },
        {
            "query": "成都美食街",
            "filters": {"city": "成都"}
        },
        {
            "query": "广州购物景点",
            "filters": {"city": "广州"}
        },
        {
            "query": "西安兵马俑",
            "filters": {"city": "西安"}
        },
        {
            "query": "苏州园林",
            "filters": {"city": "苏州"}
        },
        {
            "query": "重庆洪崖洞",
            "filters": {"city": "重庆"}
        },
        {
            "query": "南京历史景点",
            "filters": {"city": "南京"}
        },
        {
            "query": "深圳主题乐园",
            "filters": {"city": "深圳"}
        }
    ]


def print_evaluation_report(results: Dict):
    """打印评估报告"""
    print("=" * 70)
    print("RAG 评估报告")
    print(f"评估时间: {results['evaluation_time']}")
    print(f"查询数量: {results['num_queries']}")
    print("=" * 70)
    
    for top_k, metrics in results["results"].items():
        print(f"\n--- {top_k} ---")
        print(f"  Recall@K:   {metrics['recall']['avg_recall']:.2%}")
        print(f"  MRR:        {metrics['recall']['avg_mrr']:.4f}")
        print(f"  Precision@K: {metrics['precision']['avg_precision']:.2%}")
        print(f"  F1 Score:   {metrics['f1']['avg_f1']:.4f}")
    
    print("\n--- 详细结果 ---")
    for top_k, metrics in results["results"].items():
        print(f"\n{top_k} 详细:")
        for item in metrics['recall']['detailed']:
            print(f"  查询: {item['query']}")
            print(f"    知识库中相关文档数: {item['relevant_docs_count']}")
            print(f"    最大可能召回率: {item.get('max_possible_recall', 0):.2%}")
            print(f"    召回率: {item['recall']:.2%} (命中 {item['matched_count']}/{item['total_relevant']})")
            print(f"    MRR: {item['mrr']:.4f}")
            if item['matched_docs']:
                print(f"    命中景点: {', '.join(item['matched_docs'][:10])}{'...' if len(item['matched_docs']) > 10 else ''}")


if __name__ == "__main__":
    print("初始化RAG评估器...")
    evaluator = RAGEvaluator()
    
    print("加载评估数据集...")
    eval_dataset = build_default_eval_dataset()
    
    print(f"开始评估，共 {len(eval_dataset)} 个查询...")
    results = evaluator.run_full_evaluation(eval_dataset, top_k_list=[5, 10, 20, 50])
    
    print_evaluation_report(results)
    
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"rag_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    evaluator.save_results(results, output_file)
    print(f"\n评估结果已保存到: {output_file}")