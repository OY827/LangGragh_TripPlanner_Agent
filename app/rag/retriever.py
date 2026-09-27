from typing import Any, List, Dict, Optional, Union
from enum import Enum
from .vector_store import get_vector_store, get_guide_vector_store


class SearchType(Enum):
    """检索类型枚举"""
    SIMILARITY = "similarity"  # 纯相似度检索
    MMR = "mmr"  # 最大边际相关性（多样性检索）
    SIMILARITY_SCORE_THRESHOLD = "similarity_score_threshold"  # 阈值过滤
    HYBRID = "hybrid"  # 混合检索（向量检索 + BM25）


class KnowledgeRetriever:
    """知识检索器 - 支持多种检索策略"""

    def __init__(self, vector_store: Optional[Any] = None, guide_store: Optional[Any] = None):
        self.vector_store = vector_store or get_vector_store()
        self.guide_store = guide_store
        self.default_top_k = 5

    def _get_guide_store(self):
        if self.guide_store is None:
            self.guide_store = get_guide_vector_store()
        return self.guide_store

    def retrieve_guides(
        self,
        query: str,
        city: Optional[str] = None,
        top_k: int = 5,
        search_type: Union[str, SearchType] = "similarity",
    ) -> List[Dict]:
        """只查攻略 collection，不碰景点库。"""
        filters = {"city": city} if city else None
        store = self._get_guide_store()
        if isinstance(search_type, str):
            search_type = SearchType(search_type)
        if search_type == SearchType.HYBRID:
            return store.hybrid_search(query, filters, top_k)
        return store.search(query, filters, top_k)

    def retrieve(
        self,
        query: str,
        filters: Optional[Dict] = None,
        top_k: int = 5,
        search_type: Union[str, SearchType] = "similarity"
    ) -> List[Dict]:
        """
        检索相关文档

        Args:
            query: 检索查询
            filters: 过滤条件（如 {"city": "北京", "type": "attraction"}）
            top_k: 返回结果数量
            search_type: 检索类型
                - similarity: 纯相似度检索
                - mmr: 最大边际相关性（结果更有多样性）
                - similarity_score_threshold: 相似度阈值过滤

        Returns:
            检索结果列表
        """
        if isinstance(search_type, str):
            search_type = SearchType(search_type)

        if search_type == SearchType.MMR:
            return self._mmr_search(query, filters, top_k)
        elif search_type == SearchType.SIMILARITY_SCORE_THRESHOLD:
            return self._threshold_search(query, filters, top_k, threshold=0.7)
        elif search_type == SearchType.HYBRID:
            return self.vector_store.hybrid_search(query, filters, top_k)
        else:
            return self.vector_store.search(query, filters, top_k)

    def _mmr_search(
        self,
        query: str,
        filters: Optional[Dict],
        top_k: int,
        fetch_k: int = None
    ) -> List[Dict]:
        """
        最大边际相关性检索（MMR）

        MMR在保证相关性的同时，增加结果的多样性。
        适用于需要推荐多个相似但不同类型结果的场景。

        Args:
            query: 检索查询
            filters: 过滤条件
            top_k: 最终返回结果数量
            fetch_k: 初始检索数量（默认 top_k * 2）
        """
        if fetch_k is None:
            fetch_k = top_k * 3

        # 1. 初始检索，获取更多候选
        initial_results = self.vector_store.search(query, filters, fetch_k)

        if not initial_results:
            return []

        # 2. MMR 选择
        selected = []
        selected_texts = []

        for result in initial_results:
            # 如果已选结果少于目标数量，直接添加
            if len(selected) >= top_k:
                break

            # 检查与已选结果的最大相似度
            max_sim = max(
                self._text_similarity(result["content"], text)
                for text in selected_texts
            ) if selected_texts else 0

            # 如果与所有已选结果都不太相似（<0.85），则添加
            # MMR公式: score = α * similarity - (1-α) * max_sim
            # 这里简化为：相似度 >= 0.85 的才排除
            if max_sim < 0.85:
                selected.append(result)
                selected_texts.append(result["content"])

        # 如果选得不够，补充剩余结果
        if len(selected) < top_k:
            for result in initial_results:
                if result not in selected:
                    selected.append(result)
                    if len(selected) >= top_k:
                        break

        return selected

    def _threshold_search(
        self,
        query: str,
        filters: Optional[Dict],
        top_k: int,
        threshold: float = 0.7
    ) -> List[Dict]:
        """
        相似度阈值检索

        只返回相似度高于阈值的結果，适用于需要高质量结果的场景。

        Args:
            query: 检索查询
            filters: 过滤条件
            top_k: 返回结果数量上限
            threshold: 相似度阈值（0-1）
        """
        # 检索更多候选以便过滤
        candidates = self.vector_store.search(query, filters, top_k * 3)

        # 过滤低于阈值的结果
        filtered = [r for r in candidates if r.get("score", 0) >= threshold]

        return filtered[:top_k]

    def retrieve_attractions(
        self,
        city: str,
        preferences: Optional[List[str]] = None,
        travel_days: int = 1,
        search_type: str = "mmr"
    ) -> List[Dict]:
        """
        检索景点（旅行规划专用）

        Args:
            city: 城市名称
            preferences: 用户偏好列表，如 ["历史文化", "自然风光"]
            travel_days: 旅行天数（决定返回景点数量）
            search_type: 检索策略

        Returns:
            景点列表，每天推荐约3个景点
        """
        # 构建查询文本
        query = self._build_attraction_query(city, preferences)

        # 过滤条件
        filters = {
            "city": city
        }

        # 每天约3个景点
        top_k = travel_days * 3

        # 执行检索
        results = self.retrieve(query, filters, top_k, search_type)

        # 按评分排序
        results.sort(key=lambda x: x.get("metadata", {}).get("rating", 0), reverse=True)

        return results

    def retrieve_by_semantic(
        self,
        query: str,
        filters: Optional[Dict] = None,
        top_k: int = 5
    ) -> List[Dict]:
        """
        语义检索

        基于查询的语义理解进行检索，适合复杂问句。
        """
        return self.vector_store.search(query, filters, top_k)

    def retrieve_by_keyword(
        self,
        keywords: List[str],
        filters: Optional[Dict] = None,
        top_k: int = 5
    ) -> List[Dict]:
        """
        关键词检索

        基于关键词匹配进行检索，适合精确查找。
        """
        # 将关键词组合为查询
        query = " ".join(keywords)
        return self.vector_store.search(query, filters, top_k)

    def hybrid_retrieve(
        self,
        query: str,
        keywords: Optional[List[str]] = None,
        filters: Optional[Dict] = None,
        top_k: int = 5,
        semantic_weight: float = 0.7
    ) -> List[Dict]:
        """
        混合检索

        结合语义检索和关键词检索，结果进行加权融合。

        Args:
            query: 语义查询
            keywords: 关键词列表
            filters: 过滤条件
            top_k: 返回数量
            semantic_weight: 语义检索权重（0-1），关键词权重 = 1 - semantic_weight
        """
        # 1. 语义检索
        semantic_results = self.vector_store.search(query, filters, top_k * 2)

        # 2. 关键词检索
        keyword_results = []
        if keywords:
            keyword_results = self.retrieve_by_keyword(keywords, filters, top_k * 2)

        # 3. 结果融合
        combined = self._merge_results(
            semantic_results,
            keyword_results,
            semantic_weight=semantic_weight,
            top_k=top_k
        )

        return combined

    def _merge_results(
        self,
        semantic_results: List[Dict],
        keyword_results: List[Dict],
        semantic_weight: float,
        top_k: int
    ) -> List[Dict]:
        """
        合并语义检索和关键词检索的结果

        Args:
            semantic_results: 语义检索结果
            keyword_results: 关键词检索结果
            semantic_weight: 语义权重
            top_k: 返回数量

        Returns:
            合并后的结果
        """
        # 为每个结果计算综合分数
        scores = {}
        seen = {}

        # 处理语义结果
        max_semantic = max((r.get("score", 0) for r in semantic_results), default=1)
        for r in semantic_results:
            key = r.get("metadata", {}).get("title", r.get("content", "")[:50])
            norm_score = r.get("score", 0) / max_semantic
            scores[key] = scores.get(key, 0) + norm_score * semantic_weight
            seen[key] = r

        # 处理关键词结果
        max_keyword = max((r.get("score", 0) for r in keyword_results), default=1)
        for r in keyword_results:
            key = r.get("metadata", {}).get("title", r.get("content", "")[:50])
            norm_score = r.get("score", 0) / max_keyword
            scores[key] = scores.get(key, 0) + norm_score * (1 - semantic_weight)
            seen[key] = r

        # 排序并返回
        sorted_keys = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

        return [seen[k] for k in sorted_keys[:top_k] if k in seen]

    def _build_attraction_query(
        self,
        city: str,
        preferences: Optional[List[str]] = None
    ) -> str:
        """
        构建景点检索查询

        Args:
            city: 城市名称
            preferences: 用户偏好

        Returns:
            组合后的查询文本
        """
        query_parts = [city, "景点"]

        if preferences:
            query_parts.extend(preferences)

        query_parts.extend(["推荐", "攻略", "介绍"])

        return " ".join(query_parts)

    def _text_similarity(self, text1: str, text2: str) -> float:
        """
        计算两个文本的相似度（基于词重叠）

        Args:
            text1: 文本1
            text2: 文本2

        Returns:
            相似度分数（0-1）
        """
        if not text1 or not text2:
            return 0.0

        # 简单的词重叠计算
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        intersection = words1 & words2
        union = words1 | words2

        if not union:
            return 0.0

        return len(intersection) / len(union)

    def get_as_langchain_retriever(self, **kwargs):
        """
        获取 LangChain Retriever 接口

        便于集成到 LangChain 的 Chains 和 Agents 中。

        Args:
            **kwargs: 传递给 LangChain Retriever 的参数

        Returns:
            LangChain Retriever 对象
        """
        return self.vector_store.as_retriever(**kwargs)

    def batch_retrieve(
        self,
        queries: List[str],
        filters: Optional[Dict] = None,
        top_k: int = 5
    ) -> Dict[str, List[Dict]]:
        """
        批量检索

        一次执行多个查询，适用于需要从不同角度检索的场景。

        Args:
            queries: 查询列表
            filters: 过滤条件
            top_k: 每个查询返回的结果数

        Returns:
            {query: results} 格式的字典
        """
        results = {}
        for query in queries:
            results[query] = self.retrieve(query, filters, top_k)
        return results
