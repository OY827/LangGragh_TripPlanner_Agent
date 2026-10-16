"""
Hybrid RAG 模块

结合 2步 RAG 和 Agentic RAG 的特点，引入中间步骤实现更高灵活性：
1. 查询增强 (Query Enhancement)：LLM 重写查询、生成变体、扩展上下文
2. 检索验证 (Retrieval Validation)：评估检索结果相关性，不足时优化重试
3. 答案验证 (Answer Validation)：检查答案与源对齐、完整性、准确性
"""

import time
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass, field

from app.llm import get_llm
from app.rag.retriever import KnowledgeRetriever
from app.services.structured_output import AnswerCheck, QueryVariants, as_dict, invoke_structured


def _price_amount(value: Any) -> Optional[float]:
    """把 0、30、'30元'、'免费' 收成数字。无法识别时返回 None。"""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    if "免费" in text or text in {"免票", "免费"}:
        return 0.0
    number = []
    seen_dot = False
    for ch in text:
        if ch.isdigit():
            number.append(ch)
        elif ch == "." and not seen_dot:
            number.append(ch)
            seen_dot = True
        elif number:
            break
    if not number or number == ["."]:
        return None
    try:
        return float("".join(number))
    except ValueError:
        return None


@dataclass
class RetrievalResult:
    """检索结果"""
    docs: List[Dict[str, Any]]
    query_used: str
    scores: List[float]
    validation_passed: bool = True
    validation_feedback: str = ""


@dataclass
class HybridRAGConfig:
    """Hybrid RAG 配置"""
    max_query_variants: int = 3
    min_relevant_docs: int = 5
    relevance_threshold: float = 0.5
    max_retrieval_rounds: int = 2
    enable_query_rewrite: bool = True
    enable_retrieval_validation: bool = True
    enable_answer_validation: bool = True
    use_mmr: bool = True
    top_k: int = 10
    # Rerank 配置
    enable_rerank: bool = True
    rerank_model: str = "BAAI/bge-reranker-base"
    rerank_top_k: int = 10          # Rerank后返回的文档数
    retrieval_top_k: int = 30       # 初步检索的文档数（Rerank前）
    # 意图感知：改写重排查询 + 关键词加权，缓解亲子等意图被通用地标冲淡
    enable_intent_boost: bool = True
    intent_boost_weight: float = 0.12


class HybridRAG:
    """
    Hybrid RAG 引擎

    处理流程：
    Query → [增强] → [检索] → [验证] → [生成] → [验证] → Answer
              ↑         ↓        ↓
              └── 不足则重试 ←── 不通过 ←──┘
    """

    def __init__(
        self,
        retriever: KnowledgeRetriever = None,
        llm_provider: str = "aliyun",
        llm_model: str = None,
        config: HybridRAGConfig = None
    ):
        self.retriever = retriever or KnowledgeRetriever()
        self.llm = get_llm(provider=llm_provider, model=llm_model, temperature=0.3)
        self.config = config or HybridRAGConfig()

        self._timing: Dict[str, float] = {}

        # 初始化 Reranker
        self.reranker = None
        if self.config.enable_rerank:
            from app.rag.reranker import Reranker
            self.reranker = Reranker(model_name=self.config.rerank_model)

    def retrieve_with_validation(
        self,
        query: str,
        city: str = "",
        preferences: List[str] = None,
        **kwargs
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        带验证的检索主流程

        Returns:
            (文档列表, 元信息包括查询变体、耗时、验证结果)
        """
        t0 = time.time()
        meta: Dict[str, Any] = {
            "query_variants": [],
            "retrieval_rounds": 0,
            "total_docs": 0,
            "validation_passed": False,
            "timing": {},
        }

        # Step 1: 查询增强
        t1 = time.time()
        query_variants = self._enhance_query(query, city, preferences)
        meta["query_variants"] = query_variants
        meta["timing"]["query_enhancement"] = round(time.time() - t1, 2)

        # Step 2: 多轮检索 + 验证
        all_docs = []
        for round_idx in range(self.config.max_retrieval_rounds):
            t2 = time.time()
            round_docs = self._multi_query_retrieve(query_variants, city, preferences)
            meta["retrieval_rounds"] = round_idx + 1
            meta["timing"][f"retrieval_round_{round_idx + 1}"] = round(time.time() - t2, 2)

            # Step 3: 检索验证
            if self.config.enable_retrieval_validation:
                is_valid, feedback = self._validate_retrieval(round_docs, query)
                if is_valid:
                    all_docs = round_docs
                    meta["validation_passed"] = True
                    break
                elif round_idx < self.config.max_retrieval_rounds - 1:
                    query_variants = self._generate_alternative_queries(query, feedback)
                    meta["query_variants"].extend(query_variants)
            else:
                all_docs = round_docs
                meta["validation_passed"] = True
                break

        # 去重
        seen = set()
        unique_docs = []
        for doc in all_docs:
            title = doc.get("metadata", {}).get("title", doc.get("content", "")[:50])
            if title and title not in seen:
                seen.add(title)
                unique_docs.append(doc)

        # Step 4: Rerank 精排（候选>1 即精排，避免短列表跳过排序）
        if self.config.enable_rerank and self.reranker and len(unique_docs) > 1:
            t3 = time.time()
            rerank_query = self._build_rerank_query(query, city, preferences)
            meta["rerank_query"] = rerank_query
            reranked_docs = self.reranker.rerank(
                query=rerank_query,
                docs=unique_docs,
                top_k=max(self.config.rerank_top_k, len(unique_docs)),
            )
            if self.config.enable_intent_boost:
                reranked_docs = self._apply_intent_boost(
                    reranked_docs, query=query, preferences=preferences
                )
                # family 意图已在 _apply_intent_boost 内按 intent_hit 优先排序
                if self._detect_rerank_intent(query, preferences) != "family":
                    reranked_docs = sorted(
                        reranked_docs,
                        key=lambda x: x.get("rerank_score", 0),
                        reverse=True,
                    )
            unique_docs = reranked_docs[: self.config.rerank_top_k]
            meta["timing"]["rerank"] = round(time.time() - t3, 2)
            meta["rerank_enabled"] = True
            meta["rerank_info"] = self.reranker.get_info()
            meta["intent_boost"] = self.config.enable_intent_boost
        else:
            meta["rerank_enabled"] = False

        meta["total_docs"] = len(unique_docs)
        meta["timing"]["total"] = round(time.time() - t0, 2)

        return unique_docs, meta

    def _detect_rerank_intent(self, query: str, preferences: List[str] = None) -> str:
        """重排专用意图（勿与检索侧返回 dict 的 _detect_intent 混淆）"""
        text = f"{query} {' '.join(preferences or [])}"
        if any(k in text for k in ["亲子", "儿童", "家庭", "遛娃", "孩子", "乐园"]):
            return "family"
        if any(k in text for k in ["免费", "0元", "免票"]):
            return "free"
        if any(k in text for k in ["历史", "文化", "古迹", "古镇", "博物馆", "故居", "寺庙"]):
            return "history"
        return "general"

    def _build_rerank_query(
        self,
        query: str,
        city: str = "",
        preferences: List[str] = None,
    ) -> str:
        """为 CrossEncoder 构造意图更尖锐的重排查询，减少通用地标抢分"""
        intent = self._detect_rerank_intent(query, preferences)
        city_prefix = city or ""
        if intent == "family":
            return (
                f"{city_prefix}亲子乐园 儿童游玩 主题乐园 动物园 海洋馆 "
                f"适合带孩子 {query}"
            ).strip()
        if intent == "free":
            return f"{city_prefix}免费开放景点 免门票 {query}".strip()
        if intent == "history":
            return f"{city_prefix}历史文化 古迹 古镇 博物馆 故居 {query}".strip()
        return query

    def _apply_intent_boost(
        self,
        docs: List[Dict[str, Any]],
        query: str,
        preferences: List[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        在 BGE 分数上叠加意图关键词加权。
        亲子意图额外保证：有乐园/动物园等信号的文档整体排在通用地标之前。
        """
        intent = self._detect_rerank_intent(query, preferences)
        weight = float(getattr(self.config, "intent_boost_weight", 0.12) or 0.12)

        family_kw = [
            "乐园", "迪士尼", "乐高", "海洋", "水族", "野生动物", "动物园", "玛雅",
            "精灵", "昆虫", "科技馆", "自然博物馆", "植物园", "森林", "东方绿舟",
            "海昌", "天文馆", "影视乐园", "水世界", "冰雪", "航海博物馆",
        ]
        family_cat = {"主题乐园"}
        family_penalty = [
            "金融中心", "观光厅", "人民广场", "陆家嘴", "步行街",
            "东方明珠", "外滩", "世纪公园", "展馆深度",
        ]

        history_kw = ["古镇", "寺", "庙", "故居", "会址", "纪念馆", "遗址", "博物馆", "外滩", "豫园", "城隍"]
        history_cat = {"历史文化", "博物馆"}

        free_kw = ["免费"]
        free_penalty = ["乐园", "迪士尼", "乐高", "野生动物", "游船", "体验"]

        for doc in docs:
            meta = doc.get("metadata") or {}
            title = str(meta.get("title") or "")
            category = str(meta.get("category") or "")
            tags = meta.get("tags") or ""
            if isinstance(tags, list):
                tags = " ".join(str(t) for t in tags)
            price = str(meta.get("price") or "")
            blob = f"{title} {category} {tags}"
            base = float(doc.get("rerank_score") or doc.get("rerank_score_raw") or 0)
            boost = 0.0
            intent_hit = 0

            if intent == "family":
                hits = sum(1 for k in family_kw if k in blob)
                if hits:
                    boost += weight * min(hits, 3)
                    intent_hit = 1
                if category in family_cat:
                    boost += weight
                    intent_hit = 1
                if any(p in title for p in family_penalty) and hits == 0:
                    # 通用地标硬降权，避免分数饱和时仍压过乐园
                    base = base * 0.5
                    boost -= weight
            elif intent == "history":
                hits = sum(1 for k in history_kw if k in blob)
                if hits:
                    boost += weight * min(hits, 3)
                    intent_hit = 1
                if category in history_cat:
                    boost += weight
                    intent_hit = 1
            elif intent == "free":
                if price in ("0", "0.0", "免费", "0元") or any(k in blob for k in free_kw):
                    boost += weight
                    intent_hit = 1
                if any(p in title for p in free_penalty):
                    boost -= weight
                    intent_hit = 0

            doc["intent_hit"] = intent_hit
            doc["intent_boost"] = round(boost, 4)
            doc["rerank_score_raw"] = float(doc.get("rerank_score_raw") or doc.get("rerank_score") or base)
            doc["rerank_score"] = base + boost

        if intent == "family":
            # 有亲子信号的整体优先，再比分数
            docs = sorted(
                docs,
                key=lambda x: (int(x.get("intent_hit") or 0), float(x.get("rerank_score") or 0)),
                reverse=True,
            )
        return docs

    def _enhance_query(
        self,
        query: str,
        city: str = "",
        preferences: List[str] = None
    ) -> List[str]:
        """
        查询增强：生成多个查询变体（同义改写，保持泛化性）

        使用 LLM 将原始查询改写为语义等价的变体，扩大检索覆盖范围。
        关键原则：同义改写而非具体化，保留原始意图和约束。
        """
        if not self.config.enable_query_rewrite or not self.llm:
            return [query]

        try:
            prompt = f"""你是一个旅游搜索优化专家。请将用户查询改写为{self.config.max_query_variants}个语义等价的搜索变体，用于扩大向量检索的覆盖范围。

城市: {city}
用户查询: {query}
用户偏好: {', '.join(preferences) if preferences else '无'}

改写要求:
1. 变体必须是"同义改写"，保持与原始查询相同的意图和泛化程度
2. 不要使用具体景点名称（如"上海博物馆"、"广富林"等），应使用类别词（如"博物馆"、"古镇"、"故居"）
3. 必须保留原始查询中的关键约束（如"免费"→变体也必须是免费相关的；"亲子"→变体也必须适合亲子）
4. 每个变体从不同角度表达相同意图
5. 变体应简短（3-8个字 + 城市名），适合向量语义检索

示例:
- "上海的历史文化景点" → ["上海古迹遗址", "上海博物馆纪念馆", "上海古镇故居"]
- "上海免费景点推荐" → ["上海免费游览地", "上海免费开放景区", "上海0元景点"]
- "上海适合亲子游玩的地方" → ["上海亲子乐园", "上海家庭游玩景点", "上海儿童游乐场"]

返回 JSON 格式: {{"queries": ["变体1", "变体2", "变体3"]}}"""

            parsed = invoke_structured(self.llm, QueryVariants, prompt)
            variants = []
            if isinstance(parsed, QueryVariants):
                variants = parsed.queries
            else:
                data = as_dict(parsed) or {}
                raw_queries = data.get("queries") or []
                if isinstance(raw_queries, list):
                    variants = [str(item) for item in raw_queries if str(item).strip()]
            if variants:
                result = [query] + [item for item in variants if item != query]
                return result[:self.config.max_query_variants]

            variants = self._rule_based_expand(query, city, preferences)
            return variants

        except Exception:
            return self._rule_based_expand(query, city, preferences)

    def _rule_based_expand(
        self,
        query: str,
        city: str = "",
        preferences: List[str] = None
    ) -> List[str]:
        """规则化查询扩展（fallback）"""
        category_keywords = {
            "历史文化": ["古迹", "博物馆", "古镇", "故居", "遗址", "纪念", "园林", "文物"],
            "自然风光": ["公园", "山水", "园林", "湖", "森林", "瀑布", "自然", "生态"],
            "主题乐园": ["乐园", "主题公园", "游乐园", "海洋馆", "景区"],
            "艺术文化": ["美术馆", "剧院", "文化", "展览", "艺术", "画廊"],
            "购物休闲": ["步行街", "商场", "美食", "小吃", "商业街", "夜市"],
            "宗教": ["寺庙", "教堂", "清真寺", "道观", "宗教"],
        }

        text = f"{query} {' '.join(preferences) if preferences else ''}"
        matched = []

        for category, keywords in category_keywords.items():
            for kw in keywords:
                if kw in text:
                    matched.append(kw)
                    break
            if any(kw in text for kw in category_keywords[category]):
                matched.extend(category_keywords[category])

        if not matched:
            for category, keywords in category_keywords.items():
                if category in text:
                    matched.extend(keywords)
                    break

        if not matched:
            matched = ["景点", "名胜", "热门", "推荐"]

        variants = []
        for kw in matched[:self.config.max_query_variants]:
            if city:
                variants.append(f"{city} {kw}")
            else:
                variants.append(kw)

        if not variants:
            variants = [query]

        return variants

    def _multi_query_retrieve(
        self,
        query_variants: List[str],
        city: str = "",
        preferences: List[str] = None
    ) -> List[Dict[str, Any]]:
        """多查询变体并行检索 + 分类Boost排序"""
        all_docs = []

        retrieval_k = self.config.retrieval_top_k if self.config.enable_rerank else self.config.top_k

        # 检测查询意图
        intent_info = self._detect_intent(query_variants[0] if query_variants else "", preferences or [])

        for variant in query_variants:
            # 1. 意图分类检索（带分类过滤）
            intent_categories = intent_info.get("categories", [])
            for cat in intent_categories:
                try:
                    docs = self.retriever.retrieve(
                        query=variant,
                        filters={"city": city, "category": cat} if city else {"category": cat},
                        top_k=retrieval_k,
                        search_type="hybrid"
                    )
                    # 标记分类匹配
                    for d in docs:
                        d["_intent_match"] = True
                    all_docs.extend(docs)
                except Exception:
                    pass

            # 2. 价格过滤检索（免费景点等）
            if intent_info.get("price_filter") == 0:
                try:
                    docs = self.retriever.retrieve(
                        query=variant,
                        filters={"city": city, "price": 0} if city else {"price": 0},
                        top_k=retrieval_k,
                        search_type="hybrid"
                    )
                    for d in docs:
                        d["_intent_match"] = True
                    all_docs.extend(docs)
                except Exception:
                    pass

            # 3. 全局检索（无分类过滤）
            try:
                docs = self.retriever.retrieve(
                    query=variant,
                    filters={"city": city} if city else None,
                    top_k=retrieval_k,
                    search_type="hybrid"
                )
                all_docs.extend(docs)
            except Exception:
                pass

        # 去重
        unique_docs = self._deduplicate(all_docs)

        # 分类Boost排序
        unique_docs = self._boost_by_intent(unique_docs, intent_info)

        # Rerank模式下不做MMR（让Reranker来排序）
        if not self.config.enable_rerank and self.config.use_mmr and len(unique_docs) > self.config.top_k:
            unique_docs = self._mmr_select(unique_docs, self.config.top_k)

        return unique_docs

    def _detect_intent(self, query: str, preferences: List[str]) -> Dict[str, Any]:
        """检测查询意图（分类 + 价格）"""
        text = f"{query} {' '.join(preferences)}"

        # 分类映射：查询关键词 → 实际元数据分类
        category_map = {
            "历史文化": ["历史", "古迹", "文化", "古镇", "故居", "遗址", "纪念", "文物", "园林", "故居", "旧址"],
            "博物馆": ["博物馆", "展馆", "纪念馆", "美术馆"],
            "艺术文化": ["艺术", "艺术文化", "剧院", "画廊", "展览"],
            "自然风光": ["自然", "山水", "风景", "公园", "园林", "湖", "森林", "瀑布", "生态", "沙滩", "湿地"],
            "城市漫步": ["步行街", "街区", "citywalk", "漫步", "路", "街", "广场"],
            "主题乐园": ["乐园", "主题公园", "游乐园", "海洋馆", "海洋", "动物", "亲子", "遛娃", "儿童"],
        }

        matched_categories = set()
        for cat, keywords in category_map.items():
            if cat in text or any(kw in text for kw in keywords):
                matched_categories.add(cat)

        # 价格过滤
        price_filter = None
        if any(kw in text for kw in ["免费", "不要钱", "免费景点", "开放", "免票"]):
            price_filter = 0

        return {
            "categories": list(matched_categories),
            "price_filter": price_filter,
        }

    def _boost_by_intent(self, docs: List[Dict], intent_info: Dict[str, Any]) -> List[Dict]:
        """根据查询意图对检索结果进行Boost排序"""
        if not docs or not intent_info:
            return docs

        intent_categories = intent_info.get("categories", [])
        price_filter = intent_info.get("price_filter")

        for doc in docs:
            metadata = doc.get("metadata", {})
            doc_category = metadata.get("category", "")
            doc_price = metadata.get("price")
            base_score = doc.get("score", 0)

            boost = 1.0

            # 分类Boost：匹配意图分类 +50%
            if doc_category in intent_categories:
                boost += 0.5

            # 已标记的意图匹配再 +30%
            if doc.get("_intent_match"):
                boost += 0.3

            # 价格过滤：免费查询时，免费景点 +100%，付费景点 -50%
            # 库里的价格经常是 "30元" 这种字符串，不能直接和数字比较
            if price_filter == 0:
                price_amount = _price_amount(doc_price)
                if price_amount == 0:
                    boost += 1.0
                elif price_amount is not None and price_amount > 0:
                    boost *= 0.5

            # 收藏数/评分Boost（使用热门度排序）
            hot = metadata.get("hot", "")
            if hot and str(hot).replace(".", "").isdigit():
                hot_score = float(hot) / 10.0
                boost += hot_score * 0.2

            doc["_boosted_score"] = base_score * boost

        # 按boosted score排序
        docs.sort(key=lambda x: x.get("_boosted_score", x.get("score", 0)), reverse=True)

        return docs

    def _validate_retrieval(
        self,
        docs: List[Dict[str, Any]],
        original_query: str
    ) -> Tuple[bool, str]:
        """
        检索验证：检查结果的相关性和充分性

        Returns:
            (是否通过, 反馈信息用于下一轮查询优化)
        """
        if not docs:
            return False, "未检索到任何文档"

        # 检查数量
        if len(docs) < self.config.min_relevant_docs:
            return False, f"结果数量不足（{len(docs)} < {self.config.min_relevant_docs}）"

        # 检查相关性分数
        low_score_count = sum(
            1 for d in docs
            if d.get("score", 0) < self.config.relevance_threshold
        )
        if low_score_count > len(docs) * 0.5:
            return False, f"超过50%的文档相关性低于阈值({self.config.relevance_threshold})"

        # 检查内容多样性
        titles = set()
        for d in docs:
            title = d.get("metadata", {}).get("title", "")
            if title:
                titles.add(title)

        if len(titles) < self.config.min_relevant_docs * 0.6:
            return False, f"文档内容重复率过高"

        return True, "检索结果充足且相关"

    def _generate_alternative_queries(
        self,
        original_query: str,
        feedback: str
    ) -> List[str]:
        """根据验证反馈生成替代查询"""
        if not self.llm:
            return self._rule_based_expand(original_query, "", [])

        try:
            prompt = f"""检索失败，需要重新生成查询。

原始查询: {original_query}
失败原因: {feedback}

请生成3个全新的、更具体的查询变体来改善检索结果。
返回 JSON: {{"queries": ["查询1", "查询2", "查询3"]}}"""

            parsed = invoke_structured(self.llm, QueryVariants, prompt)
            if isinstance(parsed, QueryVariants):
                return parsed.queries[:self.config.max_query_variants]
            data = as_dict(parsed) or {}
            queries = data.get("queries") or []
            if isinstance(queries, list) and queries:
                return [str(item) for item in queries if str(item).strip()][:self.config.max_query_variants]
            return self._rule_based_expand(original_query, "", [])
        except Exception:
            return self._rule_based_expand(original_query, "", [])

    def _deduplicate(self, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """基于标题去重"""
        seen = set()
        unique = []
        for doc in docs:
            title = doc.get("metadata", {}).get("title", "")
            if not title:
                title = doc.get("content", "")[:50]
            if title and title not in seen:
                seen.add(title)
                unique.append(doc)
        return unique

    def _mmr_select(self, docs: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        """MMR 多样性选择"""
        if len(docs) <= top_k:
            return docs

        selected = []
        selected_texts = []

        for doc in sorted(docs, key=lambda x: x.get("score", 0), reverse=True):
            if len(selected) >= top_k:
                break

            text = doc.get("content", "")
            max_sim = max(
                self._text_similarity(text, st)
                for st in selected_texts
            ) if selected_texts else 0

            if max_sim < 0.85:
                selected.append(doc)
                selected_texts.append(text)

        if len(selected) < top_k:
            for doc in docs:
                if doc not in selected:
                    selected.append(doc)
                    if len(selected) >= top_k:
                        break

        return selected

    def _text_similarity(self, text1: str, text2: str) -> float:
        """文本相似度（词重叠）"""
        if not text1 or not text2:
            return 0.0
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        intersection = words1 & words2
        union = words1 | words2
        return len(intersection) / len(union) if union else 0.0

    def validate_answer(
        self,
        answer: str,
        source_docs: List[Dict[str, Any]],
        original_query: str
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        答案验证：检查准确性、完整性、源对齐

        Returns:
            (是否通过, 验证详情)
        """
        result = {
            "grounded": False,
            "completeness": 0.0,
            "alignment_score": 0.0,
            "issues": [],
        }

        if not self.config.enable_answer_validation or not self.llm:
            result["grounded"] = True
            return True, result

        try:
            # 构建完整的源文档标题列表（所有文档，不只前5个）
            all_titles = []
            source_texts = []
            for doc in source_docs:
                meta = doc.get("metadata", {})
                title = meta.get("title", "")
                rating = meta.get("rating", "N/A")
                desc = meta.get("description", "")
                if title:
                    all_titles.append(title)
                content = doc.get("content", "")
                if title:
                    source_texts.append(f"[{title}] (评分{rating}): {content[:300]}")

            titles_text = ", ".join(all_titles) if all_titles else "无"
            sources_text = "\n".join(source_texts[:8])

            prompt = f"""请验证以下答案是否基于提供的源文档。

【源文档列表】（所有可用景点）:
{titles_text}

【源文档详细内容】:
{sources_text}

【原始问题】: {original_query}
【生成答案】: {answer[:800]}

请仔细检查:
1. 列出答案中提到的所有景点名称
2. 逐一核对每个景点是否在【源文档列表】中存在
3. 检查答案内容是否与源文档中的描述一致
4. grounded (bool): 答案是否基于源文档（只有编造的信息才算幻觉，源文档有的景点不算幻觉）
5. completeness (0-1): 答案是否完整回答了问题
6. alignment_score (0-1): 答案与源文档的对齐程度
7. issues: 发现的问题列表

注意：只有当答案提到源文档中完全不存在的景点/信息时，才算幻觉。如果景点在源文档列表中，即使详细描述有差异，也不算幻觉。

请直接返回纯 JSON:
{{"grounded": true, "completeness": 0.8, "alignment_score": 0.7, "issues": []}}"""

            parsed = invoke_structured(self.llm, AnswerCheck, prompt)
            if isinstance(parsed, AnswerCheck):
                result["grounded"] = parsed.grounded
                result["completeness"] = parsed.completeness
                result["alignment_score"] = parsed.alignment_score
                result["issues"] = parsed.issues
            else:
                data = as_dict(parsed)
                if data:
                    result["grounded"] = bool(data.get("grounded", True))
                    result["completeness"] = float(data.get("completeness", 0.5) or 0.5)
                    result["alignment_score"] = float(data.get("alignment_score", 0.5) or 0.5)
                    issues = data.get("issues") or []
                    result["issues"] = issues if isinstance(issues, list) else [str(issues)]
                else:
                    result["grounded"] = True
                    result["completeness"] = 0.7
                    result["alignment_score"] = 0.7

            passed = result["grounded"] and result["completeness"] >= 0.5
            return passed, result

        except Exception:
            result["grounded"] = True
            result["completeness"] = 0.7
            result["alignment_score"] = 0.7
            return True, result
