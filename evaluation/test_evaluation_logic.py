"""
测试改进后的RAG评估逻辑
使用模拟数据演示基于分类标签的动态评估方法
"""

from typing import List, Dict


class MockRAGEvaluator:
    """模拟RAG评估器，演示评估逻辑"""

    _CATEGORY_KEYWORDS = {
        "历史文化": {
            "positive": ["历史", "古迹", "遗址", "文物", "古城", "古镇", "古寺", "故居", "博物馆", "陵墓", "宫殿", "王府", "塔", "桥梁", "革命", "纪念馆"],
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
        
        all_docs = self._mock_retrieve(query, filters, top_k=200)
        relevant_docs = []
        for doc in all_docs:
            doc_category = doc.get("category", "")
            doc_tags = doc.get("tags", "")
            doc_title = doc.get("title", "")
            
            if any(self._match_category(cat, doc_category, doc_tags, doc_title) for cat in matched_categories):
                relevant_docs.append(doc_title)
        
        return list(set(relevant_docs))

    def _mock_retrieve(self, query: str, filters: Dict, top_k: int) -> List[Dict]:
        """模拟知识库"""
        city = filters.get("city", "")
        
        mock_knowledge_base = {
            "上海": [
                {"title": "田子坊", "category": "历史文化", "tags": "历史 文化 创意园区"},
                {"title": "静安寺", "category": "历史文化", "tags": "历史 寺庙 佛教"},
                {"title": "豫园", "category": "历史文化", "tags": "历史 园林 古迹"},
                {"title": "外滩", "category": "历史文化", "tags": "历史 建筑 江景"},
                {"title": "城隍庙", "category": "历史文化", "tags": "历史 庙宇 小吃"},
                {"title": "上海博物馆", "category": "历史文化", "tags": "历史 文物 博物馆"},
                {"title": "中共一大会址", "category": "历史文化", "tags": "历史 革命 纪念馆"},
                {"title": "东方明珠", "category": "现代建筑", "tags": "地标 观光"},
                {"title": "上海迪士尼", "category": "主题乐园", "tags": "乐园 游乐"},
                {"title": "南京路步行街", "category": "购物", "tags": "购物 商业街"},
                {"title": "人民广场", "category": "城市公园", "tags": "公园 广场"},
                {"title": "陆家嘴", "category": "现代建筑", "tags": "地标 金融"},
                {"title": "世纪公园", "category": "城市公园", "tags": "公园 绿地"},
                {"title": "野生动物园", "category": "主题乐园", "tags": "乐园 动物"},
                {"title": "杜莎夫人蜡像馆", "category": "主题乐园", "tags": "乐园 蜡像"},
                {"title": "上海科技馆", "category": "现代建筑", "tags": "科技 展览 中心"},
                {"title": "中华艺术宫", "category": "现代建筑", "tags": "艺术 展览 中心"},
                {"title": "上海展览馆", "category": "现代建筑", "tags": "展览 会展 中心"},
                {"title": "上海大剧院", "category": "现代建筑", "tags": "剧院 艺术 演出"},
                {"title": "上海影城", "category": "现代建筑", "tags": "影城 电影 娱乐"},
            ],
            "北京": [
                {"title": "故宫", "category": "历史文化", "tags": "历史 宫殿 古迹"},
                {"title": "天坛", "category": "历史文化", "tags": "历史 祭祀 古迹"},
                {"title": "颐和园", "category": "历史文化", "tags": "历史 园林 古迹"},
                {"title": "圆明园", "category": "历史文化", "tags": "历史 遗址 园林"},
                {"title": "八达岭长城", "category": "历史文化", "tags": "历史 长城 古迹"},
                {"title": "明十三陵", "category": "历史文化", "tags": "历史 陵墓 古迹"},
                {"title": "恭王府", "category": "历史文化", "tags": "历史 王府 古迹"},
                {"title": "鸟巢", "category": "现代建筑", "tags": "地标 体育"},
                {"title": "王府井", "category": "购物", "tags": "购物 商业街"},
                {"title": "奥林匹克公园", "category": "城市公园", "tags": "公园 体育"},
                {"title": "水立方", "category": "现代建筑", "tags": "地标 体育"},
                {"title": "三里屯", "category": "购物", "tags": "购物 时尚"},
                {"title": "朝阳公园", "category": "城市公园", "tags": "公园 休闲"},
                {"title": "欢乐谷", "category": "主题乐园", "tags": "乐园 游乐"},
                {"title": "环球影城", "category": "主题乐园", "tags": "乐园 电影"},
                {"title": "中国科学技术馆", "category": "现代建筑", "tags": "科技 展览 中心"},
                {"title": "国家美术馆", "category": "现代建筑", "tags": "艺术 展览 中心"},
                {"title": "北京展览馆", "category": "现代建筑", "tags": "展览 会展 中心"},
                {"title": "国家大剧院", "category": "现代建筑", "tags": "剧院 艺术 演出"},
                {"title": "北京影城", "category": "现代建筑", "tags": "影城 电影 娱乐"},
            ],
            "杭州": [
                {"title": "西湖", "category": "自然风光", "tags": "自然 湖景 风景"},
                {"title": "千岛湖", "category": "自然风光", "tags": "自然 湖景 岛屿"},
                {"title": "西溪湿地", "category": "自然风光", "tags": "自然 湿地 生态"},
                {"title": "龙井村", "category": "自然风光", "tags": "自然 茶园 乡村"},
                {"title": "九溪", "category": "自然风光", "tags": "自然 溪流 风景"},
                {"title": "灵隐寺", "category": "历史文化", "tags": "历史 寺庙 古迹"},
                {"title": "雷峰塔", "category": "历史文化", "tags": "历史 塔 古迹"},
                {"title": "断桥", "category": "历史文化", "tags": "历史 桥梁 传说"},
                {"title": "六和塔", "category": "历史文化", "tags": "历史 塔 古迹"},
                {"title": "河坊街", "category": "购物", "tags": "购物 商业街"},
                {"title": "宋城", "category": "主题乐园", "tags": "乐园 文化"},
                {"title": "银泰城", "category": "购物", "tags": "购物 商场"},
                {"title": "植物园", "category": "自然风光", "tags": "自然 植物"},
                {"title": "动物园", "category": "主题乐园", "tags": "乐园 动物"},
                {"title": "滨江公园", "category": "城市公园", "tags": "公园 江景"},
                {"title": "杭州科技馆", "category": "现代建筑", "tags": "科技 展览 中心"},
                {"title": "杭州艺术中心", "category": "现代建筑", "tags": "艺术 展览 中心"},
                {"title": "杭州展览馆", "category": "现代建筑", "tags": "展览 会展 中心"},
                {"title": "杭州大剧院", "category": "现代建筑", "tags": "剧院 艺术 演出"},
                {"title": "杭州影城", "category": "现代建筑", "tags": "影城 电影 娱乐"},
            ],
        }
        
        return mock_knowledge_base.get(city, [])[:top_k]

    def _mock_search(self, query: str, filters: Dict, top_k: int) -> List[str]:
        """模拟检索结果（带噪声）"""
        city = filters.get("city", "")
        
        mock_search_results = {
            "上海": {
                "上海历史文化景点": ["田子坊", "上海科技馆", "静安寺", "上海展览馆", "豫园", "中华艺术宫", "外滩", "城隍庙", "上海博物馆", "中共一大会址"],
            },
            "北京": {
                "北京历史古迹": ["故宫", "中国科学技术馆", "天坛", "颐和园", "北京展览馆", "圆明园", "八达岭长城", "国家美术馆", "明十三陵", "恭王府"],
            },
            "杭州": {
                "杭州自然风光": ["西湖", "杭州科技馆", "千岛湖", "西溪湿地", "杭州艺术中心", "九溪", "龙井村", "植物园", "滨江公园", "杭州展览馆"],
            },
        }
        
        return mock_search_results.get(city, {}).get(query, [])[:top_k]

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
                    "recall": 0,
                    "mrr": 0,
                })
                continue
            
            retrieved_titles = self._mock_search(query, filters, top_k)
            
            matched_count = 0
            matched_docs = []
            for doc in relevant_docs:
                if any(doc in title for title in retrieved_titles):
                    matched_count += 1
                    matched_docs.append(doc)
            
            recall = matched_count / len(relevant_docs)
            
            mrr = 0
            for rank, title in enumerate(retrieved_titles, 1):
                if any(doc in title for title in retrieved_titles):
                    mrr = 1 / rank
                    break
            
            max_possible_recall = min(top_k, len(relevant_docs)) / len(relevant_docs)
            
            results.append({
                "query": query,
                "relevant_docs_count": len(relevant_docs),
                "retrieved_titles": retrieved_titles,
                "matched_count": matched_count,
                "matched_docs": matched_docs,
                "recall": recall,
                "mrr": mrr,
                "max_possible_recall": max_possible_recall,
            })
        
        avg_recall = sum(r["recall"] for r in results) / len(results) if results else 0
        avg_mrr = sum(r["mrr"] for r in results) / len(results) if results else 0
        
        return {
            "metric": "recall",
            "avg_recall": avg_recall,
            "avg_mrr": avg_mrr,
            "top_k": top_k,
            "detailed": results
        }

    def evaluate_precision(self, eval_dataset: List[Dict], top_k: int = 10) -> Dict:
        """评估精确率 Precision@K"""
        results = []
        
        for item in eval_dataset:
            query = item["query"]
            filters = item.get("filters", {})
            
            relevant_docs = self._get_relevant_docs_from_knowledge_base(query, filters)
            retrieved_titles = self._mock_search(query, filters, top_k)
            
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
                "relevant_docs_count": len(relevant_docs),
            })
        
        avg_precision = sum(r["precision"] for r in results) / len(results) if results else 0
        
        return {
            "metric": "precision",
            "avg_precision": avg_precision,
            "top_k": top_k,
            "detailed": results
        }


def print_evaluation_report(evaluator, eval_dataset, top_k_list=[5, 10, 20]):
    """打印评估报告"""
    print("=" * 70)
    print("改进后的RAG评估逻辑演示")
    print("=" * 70)
    
    for top_k in top_k_list:
        recall_result = evaluator.evaluate_recall(eval_dataset, top_k)
        precision_result = evaluator.evaluate_precision(eval_dataset, top_k)
        
        print(f"\n--- top_{top_k} ---")
        print(f"  平均召回率: {recall_result['avg_recall']:.2%}")
        print(f"  平均精确率: {precision_result['avg_precision']:.2%}")
    
    print("\n" + "=" * 70)
    print("详细分析")
    print("=" * 70)
    
    for top_k in [10]:
        recall_result = evaluator.evaluate_recall(eval_dataset, top_k)
        for item in recall_result["detailed"]:
            print(f"\n查询: {item['query']}")
            print(f"  知识库中相关文档数: {item['relevant_docs_count']}")
            print(f"  最大可能召回率: {item['max_possible_recall']:.2%}")
            print(f"  检索结果: {', '.join(item['retrieved_titles'])}")
            print(f"  命中相关文档: {', '.join(item['matched_docs'])}")
            print(f"  未命中相关文档: {', '.join(sorted(set([d for d in evaluator._mock_retrieve(item['query'], item.get('filters', {}), 200) if evaluator._match_category('历史文化' if '历史' in item['query'] else '自然风光', d.get('category', ''), d.get('tags', ''), d.get('title', ''))]) - set(item['matched_docs'])))}")
            print(f"  误检文档（不相关但被检索到）: {', '.join([t for t in item['retrieved_titles'] if t not in item['matched_docs']])}")
            print(f"  召回率: {item['recall']:.2%} (命中 {item['matched_count']}/{item['relevant_docs_count']})")
            print(f"  精确率: {sum(1 for t in item['retrieved_titles'] if t in item['matched_docs']) / len(item['retrieved_titles']):.2%} (相关 {sum(1 for t in item['retrieved_titles'] if t in item['matched_docs'])}/{len(item['retrieved_titles'])})")
    
    print("\n" + "=" * 70)
    print("改进点说明")
    print("=" * 70)
    print("""
问题1：召回率计算的局限性
  - 改进前：知识库有50个相关文档，top_k=5，召回率最大只能是10%
  - 改进后：增加"最大可能召回率"指标，同时评估不同top_k值
  
问题2：相关性定义过于宽泛
  - 改进前：只要包含历史关键词就被标记为相关，导致"展览馆"、"科技馆"被误判
  - 改进后：使用正/负关键词双重匹配，排除非历史文化类景点

分类匹配规则：
  历史文化类：必须包含正关键词（历史、古迹、博物馆等）且不包含负关键词（展览、科技、艺术等）
  自然风光类：必须包含正关键词（自然、湖、山等）且不包含负关键词（公园、城市、体育等）
""")


if __name__ == "__main__":
    evaluator = MockRAGEvaluator()
    
    eval_dataset = [
        {"query": "上海历史文化景点", "filters": {"city": "上海"}},
        {"query": "北京历史古迹", "filters": {"city": "北京"}},
        {"query": "杭州自然风光", "filters": {"city": "杭州"}},
    ]
    
    print_evaluation_report(evaluator, eval_dataset)