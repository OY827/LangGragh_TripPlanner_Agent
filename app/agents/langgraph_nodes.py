"""
LangGraph 节点函数集合

每个节点是一个接受 State、返回 State 更新的函数。
集成 PromptManager 实现渐进披露式提示词（基于 md 模板）。
分层组织：
- 景点检索节点
- 天气节点
- 酒店节点
- 主图节点
"""

import json
import re
import time
from typing import Dict, Any, List, Optional

from app.llm import get_llm
from app.rag.retriever import KnowledgeRetriever
from app.rag.hybrid_rag import HybridRAG, HybridRAGConfig
from app.services.amap_service import AmapService
from app.memory.memory_manager import TravelMemoryManager
from app.monitoring import get_metrics, get_logger
from app.prompts import get_prompt_manager
from app.services.structured_output import TripPlanOutput, as_dict, invoke_structured


class GraphNodes:
    """
    LangGraph 节点函数集合
    
    所有节点函数都是无状态的（stateless），通过 State 传递数据。
    集成 PromptManager 加载渐进披露式 md 模板。
    """

    # 当天时间轴：09:00 出发，21:00 前结束。
    # 游览写入知识库里的完整分钟数；高德路程再加 10 分钟缓冲。
    # 查不到路线时按 30 分钟估算。500 米以内按步行计。
    # 3 小时及以上的景点排在上午，其余按最近邻。
    # 午饭 11:30–14:00、晚饭 17:00–19:30，各留 60 分钟，插在两段游览之间。
    DAY_START = 9
    DAY_END = 21
    DAY_START_MIN = 9 * 60
    DAY_END_MIN = 21 * 60
    TRAVEL_BUFFER_MIN = 10
    UNKNOWN_TRAVEL_MIN = 30
    WALK_MAX_METERS = 500
    LONG_VISIT_MIN = 180
    MEAL_MINUTES = 60
    LUNCH_START = 11 * 60 + 30
    LUNCH_END = 14 * 60
    DINNER_START = 17 * 60
    DINNER_END = 19 * 60 + 30
    OPEN_WAIT_MAX = 180
    GENERIC_STAYS = frozenset({
        "经济型", "舒适型", "豪华型", "高档型", "中档", "低档",
        "经济", "舒适", "豪华", "高档", "酒店", "住宿", "宾馆",
    })
    
    def __init__(
        self,
        llm_provider: str = "aliyun",
        llm_model: str = None,
        enable_monitoring: bool = True
    ):
        self.llm = get_llm(provider=llm_provider, model=llm_model, temperature=0.7)
        self.retriever = KnowledgeRetriever()
        self.hybrid_rag = HybridRAG(
            retriever=self.retriever,
            llm_provider=llm_provider,
            llm_model=llm_model,
            config=HybridRAGConfig(
                max_query_variants=3,
                min_relevant_docs=5,
                max_retrieval_rounds=2,
            )
        )
        self.amap_service = AmapService()
        self.memory_manager = TravelMemoryManager()
        self.enable_monitoring = enable_monitoring
        self.metrics = get_metrics() if enable_monitoring else None
        self.logger = get_logger() if enable_monitoring else None
        
        self.prompt_manager = get_prompt_manager()
        self._agent_prompts = {}
        self._hotel_agent = None
    
    def _get_agent_prompt(self, agent_name: str) -> str:
        """获取渐进披露式提示词（带缓存）"""
        if agent_name not in self._agent_prompts:
            self._agent_prompts[agent_name] = self.prompt_manager.get_agent_prompt(agent_name)
        return self._agent_prompts[agent_name]
    
    def _build_planner_messages(self, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """基于 md 模板 + 动态数据构建 LLM 消息列表（渐进披露式）"""
        city = state.get("city", "")
        start_date = state.get("start_date", "")
        end_date = state.get("end_date", "")
        travel_days = state.get("travel_days", 1)
        transportation = state.get("transportation", "")
        accommodation = state.get("accommodation", "")
        preferences = state.get("preferences", [])
        free_text = state.get("free_text_input", "")
        
        memory_info = state.get("memory_info", "")
        attraction_response = state.get("attraction_response", "")
        weather_response = state.get("weather_response", "")
        hotel_response = state.get("hotel_response", "")
        route_info = state.get("route_info", "")
        
        system_prompt = self._get_agent_prompt("planner")
        
        context_lines = [
            f"## 用户请求参数",
            f"- 城市: {city}",
            f"- 日期: {start_date} 至 {end_date}",
            f"- 天数: {travel_days}天",
            f"- 交通方式: {transportation}",
            f"- 住宿: {accommodation}",
            f"- 偏好: {', '.join(preferences) if preferences else '无'}",
        ]
        
        if memory_info:
            context_lines.append(f"\n## 旅行记忆（memory_skill）")
            context_lines.append(memory_info)
        
        if route_info:
            context_lines.append(f"\n## 路线信息")
            context_lines.append(route_info)
        
        attraction_candidates = state.get("merged_results") or attraction_response
        context_lines.append(f"\n## 景点候选")
        context_lines.append(attraction_candidates if attraction_candidates else "暂无景点信息")

        guide_info = self._get_guide_reference(city, travel_days, preferences)
        if guide_info:
            context_lines.append(f"\n## 参考攻略")
            context_lines.append(guide_info)
        
        context_lines.append(f"\n## 天气（只用于室内外安排，不新增景点）")
        context_lines.append(weather_response if weather_response else "暂无天气信息")
        
        context_lines.append(f"\n## 酒店候选（已按价格从低到高）")
        context_lines.append(hotel_response if hotel_response else "暂无酒店信息")
        
        if free_text:
            context_lines.append(f"\n## 补充要求")
            context_lines.append(free_text)
        
        context_lines.append(f"\n请根据以上信息，严格按照系统提示中的 JSON 格式要求生成完整的 {city} {travel_days} 天旅行计划。")
        context_lines.append("【强制约束】行程中的景点名称只能从上方「景点候选」中选择，酒店只能从「酒店候选」中选择；请复制候选中的游览时长、坐标和价格。")
        context_lines.append("【参考攻略】只用于片区串线与节奏；不得新增候选外地点；禁止采用攻略中的票价、开放时间、酒店报价。")
        
        user_message = "\n".join(context_lines)
        
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

    def _get_guide_reference(
        self,
        city: str,
        travel_days: int = 1,
        preferences: Optional[List[str]] = None,
        top_k: int = 3,
        max_chars: int = 650,
    ) -> str:
        """从 travel_guides 取参考攻略，失败或无数据返回空串。"""
        if not city:
            return ""
        try:
            prefs = " ".join(preferences or [])
            query = f"{city} {travel_days}天 {prefs} 旅行攻略 路线建议".strip()
            hits = self.retriever.retrieve_guides(query=query, city=city, top_k=top_k)
        except Exception as e:
            print(f"  ⚠ 攻略检索跳过: {e}")
            return ""
        if not hits:
            return ""
        parts = [
            "仅供片区串线、每日节奏、餐饮时段参考。"
            "不得把下文出现但「景点信息」中没有的地点写入行程；"
            "禁止采用其中的票价、开放时间、酒店报价。"
        ]
        for i, hit in enumerate(hits, 1):
            meta = hit.get("metadata") or {}
            title = meta.get("title") or f"攻略{i}"
            section = meta.get("section") or meta.get("category") or "攻略"
            source = meta.get("source") or ""
            year = meta.get("source_year") or ""
            header = f"{i}. [{section}] {title}"
            extras = [x for x in (f"来源:{source}" if source else "", f"年份约{year}" if year else "") if x]
            if extras:
                header += "  (" + "，".join(extras) + ")"
            snippet = (hit.get("content") or "").replace("\n", " ").strip()
            if len(snippet) > max_chars:
                snippet = snippet[:max_chars] + "..."
            parts.append(header)
            parts.append(snippet)
        return "\n".join(parts)
    
    def _build_attraction_messages(self, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """基于 md 模板构建景点结果整理消息（attraction_formatter）"""
        system_prompt = self._get_agent_prompt("attraction_formatter")
        
        city = state.get("city", "")
        query = state.get("query", "")
        preferences = state.get("preferences", [])
        merged = state.get("merged_results", "")
        
        context = f"""请整理以下景点搜索结果，为用户推荐最合适的景点：

【搜索查询】{query}
【城市】{city}
【用户偏好】{', '.join(preferences) if preferences else '无'}

【景点搜索结果】
{merged}

请按照系统提示中的格式要求，整理出清晰的景点推荐列表。"""
        
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": context},
        ]
    
    # ==================== 景点检索节点 ====================
    
    def retrieve_knowledge(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """从知识库检索景点（Hybrid RAG：查询增强 + 检索验证）"""
        if self.enable_monitoring and self.logger:
            self.logger.log_info("RETRIEVE", msg=f"[景点节点] Hybrid RAG检索: {state.get('query', '')[:50]}")
        
        try:
            city = state.get("city", "")
            query = state.get("query", "")
            preferences = state.get("preferences", [])
            
            kb_docs, meta = self.hybrid_rag.retrieve_with_validation(
                query=query,
                city=city,
                preferences=preferences,
            )
            
            if self.enable_monitoring and self.logger:
                self.logger.log_info("RETRIEVE", msg=(
                    f"[景点节点] 检索完成: {meta['total_docs']}篇文档, "
                    f"{meta['retrieval_rounds']}轮, "
                    f"查询变体: {meta['query_variants']}"
                ))
            
            return {
                "kb_docs": kb_docs[:20],
                "rag_meta": meta,
                "iteration_count": state.get("iteration_count", 0)
            }
        except Exception as e:
            if self.logger:
                self.logger.log_error("RETRIEVE", msg=f"[景点节点] Hybrid RAG检索失败: {e}")
            return {"kb_docs": [], "iteration_count": state.get("iteration_count", 0)}
    
    def search_amap(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """调用高德地图 API 搜索景点（自动扩展搜索词）"""
        if self.enable_monitoring and self.logger:
            self.logger.log_info("AMAP_SEARCH", msg=f"[景点节点] 高德搜索: {state.get('city', '')}")
        
        try:
            city = state.get("city", "")
            query = state.get("query", "")
            preferences = state.get("preferences", [])
            
            search_keywords = self._expand_amap_keywords(query, preferences)
            
            all_results = []
            for keyword in search_keywords[:3]:
                results = self.amap_service.search_poi(
                    keywords=keyword,
                    city=city
                )
                all_results.extend(results)
            
            amap_results = self._format_amap_results(all_results)
            
            if self.logger:
                self.logger.log_info("AMAP_SEARCH", msg=f"[景点节点] 高德搜索完成: {len(all_results)} 条结果")
            
            return {"amap_results": amap_results}
        except Exception as e:
            if self.logger:
                self.logger.log_error("AMAP_SEARCH", msg=f"[景点节点] 高德搜索失败: {e}")
            return {"amap_results": ""}
    
    def _expand_amap_keywords(self, query: str, preferences: List[str]) -> List[str]:
        """将通用查询扩展为多个高德搜索关键词"""
        keyword_map = {
            "历史文化": ["古迹", "博物馆", "古镇", "故居", "纪念"],
            "自然风光": ["山水", "公园", "园林", "湖", "森林", "瀑布"],
            "主题乐园": ["乐园", "主题公园", "游乐园", "迪士尼", "海洋馆"],
            "艺术文化": ["艺术", "美术馆", "剧院", "文化", "展览"],
            "购物休闲": ["步行街", "商场", "美食", "小吃"],
        }
        
        matched_keywords = []
        text = f"{query} {' '.join(preferences)}"
        
        for category, kws in keyword_map.items():
            for kw in kws:
                if kw in text:
                    matched_keywords.append(kw)
                    break
        
        if not matched_keywords:
            if any(cat in text for cat in ["历史", "古迹", "文化"]):
                matched_keywords = ["古迹", "博物馆", "古镇"]
            elif any(cat in text for cat in ["自然", "风景", "山水"]):
                matched_keywords = ["公园", "园林", "山水"]
            else:
                matched_keywords = ["景点", "名胜", "热门"]
        
        return matched_keywords
    
    def merge_attraction_results(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """合并知识库和高德 API 结果（保留 Rerank 顺序，正确读取 metadata）"""
        kb_docs = state.get("kb_docs", [])
        amap_results = state.get("amap_results", "")
        
        merged = []
        used_names = set()
        
        for doc in kb_docs:
            metadata = doc.get("metadata") if isinstance(doc.get("metadata"), dict) else {}
            # 兼容扁平结构
            title = (
                metadata.get("title")
                or doc.get("title")
                or ""
            ).strip()
            if not title:
                content = doc.get("content", "") or ""
                title = content.split("\n")[0][:40].strip()
            if not title:
                continue
            
            location = metadata.get("location") or doc.get("location") or ""
            if isinstance(location, dict):
                lon = location.get("longitude")
                lat = location.get("latitude")
                location = f"{lon},{lat}" if lon is not None and lat is not None else ""
            
            rating = metadata.get("rating") or doc.get("rating") or 0
            try:
                rating = float(rating) if rating else 0
            except (TypeError, ValueError):
                rating = 0
            
            visit_duration = metadata.get("visit_duration") or doc.get("visit_duration")
            try:
                visit_duration = int(visit_duration) if visit_duration else None
            except (TypeError, ValueError):
                visit_duration = None
            
            item = {
                "name": title,
                "description": (metadata.get("description") or doc.get("content") or "")[:200],
                "category": metadata.get("category") or doc.get("category") or "",
                "tags": metadata.get("tags") or doc.get("tags") or "",
                "rating": rating,
                "address": metadata.get("address") or "",
                "location": location,
                "visit_duration": visit_duration,
                "price": metadata.get("price"),
                "open_time": metadata.get("open_time"),
                "rerank_score": doc.get("rerank_score"),
                "source": "知识库",
            }
            merged.append(item)
            used_names.add(title)
        
        # 解析高德文本补充未出现的景点
        if amap_results:
            for line in amap_results.split("\n"):
                stripped = line.strip()
                if stripped and stripped[0].isdigit() and ". " in stripped[:5]:
                    name = stripped.split(". ", 1)[1].strip()
                    if name and name not in used_names and not any(name in u or u in name for u in used_names):
                        merged.append({
                            "name": name,
                            "description": "",
                            "category": "",
                            "tags": "",
                            "rating": 0,
                            "address": "",
                            "location": "",
                            "visit_duration": None,
                            "rerank_score": None,
                            "source": "高德API",
                        })
                        used_names.add(name)
        
        visited = state.get("visited_attractions", [])
        if visited:
            merged = [m for m in merged if m["name"] not in visited]
        
        # 保留知识库 Rerank 顺序：有 rerank_score 的在前且按分数降序；其余靠后
        def _sort_key(x):
            score = x.get("rerank_score")
            if score is not None:
                try:
                    return (0, -float(score))
                except (TypeError, ValueError):
                    pass
            return (1, -float(x.get("rating") or 0))
        
        sorted_results = sorted(merged, key=_sort_key)
        formatted = self._format_attraction_response(sorted_results)
        
        return {
            "merged_results": formatted,
            "attraction_response": formatted,
        }
    
    def reflect_attractions(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """反思节点：检查景点结果质量"""
        merged = state.get("merged_results", "")
        kb_docs = state.get("kb_docs", [])
        amap_results = state.get("amap_results", "")
        iteration = state.get("iteration_count", 0)
        
        issues = []
        reflection_pass = True
        
        amap_poi_count = 0
        if amap_results:
            for line in amap_results.split("\n"):
                stripped = line.strip()
                if stripped and stripped[0].isdigit() and ". " in stripped[:5]:
                    amap_poi_count += 1
        
        total_results = len(kb_docs) + amap_poi_count
        
        if len(kb_docs) == 0 and amap_poi_count == 0:
            issues.append("知识库和高德API均未返回结果")
            reflection_pass = False
        
        if total_results < 3:
            issues.append(f"结果数量不足（知识库{len(kb_docs)}个，高德{amap_poi_count}个）")
            reflection_pass = False
        
        if not merged or len(merged.strip()) < 50:
            issues.append("合并结果为空或过短")
            reflection_pass = False
        
        if reflection_pass:
            return {
                "reflection_feedback": "景点结果充足，可进入生成阶段",
                "reflection_pass": True,
                "iteration_count": iteration
            }
        else:
            return {
                "reflection_feedback": "; ".join(issues),
                "reflection_pass": False,
                "iteration_count": iteration + 1
            }
    
    def generate_attraction_response(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """生成最终景点结果（使用 Hybrid RAG 答案验证 + md 模板渐进披露）"""
        merged = state.get("merged_results", "")
        query = state.get("query", "")
        kb_docs = state.get("kb_docs", [])
        
        if not merged:
            return {"attraction_response": "未找到符合条件的景点"}
        
        try:
            messages = self._build_attraction_messages(state)
            response = self.llm.invoke(messages)
            result_text = response.content if hasattr(response, 'content') else str(response)
            
            # Hybrid RAG 答案验证
            if self.hybrid_rag.config.enable_answer_validation and kb_docs:
                is_valid, validation_detail = self.hybrid_rag.validate_answer(
                    answer=result_text,
                    source_docs=kb_docs,
                    original_query=query
                )
                if self.logger:
                    self.logger.log_info("ANSWER_VALIDATION", msg=(
                        f"[景点节点] 答案验证: grounded={validation_detail['grounded']}, "
                        f"completeness={validation_detail['completeness']:.2f}, "
                        f"alignment={validation_detail['alignment_score']:.2f}"
                    ))
                    if validation_detail.get("issues"):
                        self.logger.log_info("ANSWER_VALIDATION", msg=f"[景点节点] 问题: {validation_detail['issues']}")
            
            return {"attraction_response": result_text}
        except Exception:
            response = f"""景点搜索结果：

{merged}

请根据以上景点信息，为用户推荐最合适的景点。"""
            return {"attraction_response": response}
    
    # ==================== 天气子图节点 ====================
    
    def fetch_weather(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """查询天气"""
        try:
            city = state.get("city", "")
            weather_info = self.amap_service.get_weather(city)
            
            if weather_info:
                weather_text = f"""天气信息：
城市: {weather_info.city}
天气: {weather_info.weather}
温度: {weather_info.temperature or '未知'}
风向: {weather_info.wind_direction or '未知'}
风力: {weather_info.wind_power or '未知'}
湿度: {weather_info.humidity or '未知'}"""
            else:
                weather_text = f"天气信息：城市 {city} 查询失败"
            
            return {"weather_response": weather_text}
        except Exception as e:
            return {"weather_response": f"天气查询失败: {str(e)}"}
    
    # ==================== 酒店子图节点 ====================
    
    def search_hotels(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """按城市、档次和入住日期直接搜索酒店，有报价的按价格从低到高排列。"""
        try:
            city = state.get("city", "")
            accommodation = state.get("accommodation") or "经济型"
            start_date = state.get("start_date") or ""
            travel_days = int(state.get("travel_days") or 1)
            stay_nights = max(1, travel_days - 1)

            from app.services.hotel_mcp_service import get_hotel_mcp_service

            service = get_hotel_mcp_service()
            if service.is_configured():
                hotels = service.search_hotels(
                    city=city,
                    keywords=accommodation,
                    accommodation=accommodation,
                    check_in_date=start_date or None,
                    stay_nights=stay_nights,
                )
                hotels = service.sort_by_price(hotels)
                if hotels:
                    return {"hotel_response": service.format_for_agent(hotels, city=city)}

            results = self.amap_service.search_poi(
                keywords=f"{accommodation}酒店",
                city=city,
            )
            return {"hotel_response": self._format_hotel_results(results)}
        except Exception as e:
            return {"hotel_response": f"酒店搜索失败: {str(e)}"}
    
    def reflect_hotels(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """反思酒店搜索结果"""
        hotel_response = state.get("hotel_response", "")
        iteration = state.get("iteration_count", 0)
        
        has_hotels = "酒店" in hotel_response and len(hotel_response) > 100
        
        if has_hotels:
            return {"reflection_pass": True, "reflection_feedback": "酒店结果充足", "iteration_count": iteration}
        else:
            return {"reflection_pass": False, "reflection_feedback": "酒店结果不足，需要重新搜索", "iteration_count": iteration + 1}
    
    # ==================== 主图节点 ====================
    
    def extract_trip_intent(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """从用户请求提取意图"""
        return {
            "current_phase": "init",
            "iteration_count": 0,
            "max_iterations": 3,
            "errors": []
        }
    
    def query_memory(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """查询历史记忆"""
        user_id = state.get("user_id", "")
        city = state.get("city", "")
        
        if not user_id or not city:
            return {
                "memory_info": "",
                "visited_attractions": [],
                "current_phase": "memory"
            }
        
        try:
            comparison = self.memory_manager.compare_plans(user_id, city)
            
            if not comparison.get("has_history", False):
                return {
                    "memory_info": f"用户是第一次访问{city}，推荐经典路线",
                    "visited_attractions": [],
                    "current_phase": "memory"
                }
            
            visited = comparison.get("visited_attractions", [])
            prefs = comparison.get("preferences") or []
            memory_info = f"用户之前去过{city}（{comparison.get('plan_count', 0)}次）\n"
            memory_info += f"已访问景点: {', '.join(visited)}\n"
            if prefs:
                memory_info += f"用户偏好: {', '.join(prefs)}\n"
            memory_info += f"请推荐不同的景点和路线"
            
            return {
                "memory_info": memory_info,
                "visited_attractions": visited,
                "current_phase": "memory"
            }
        except Exception as e:
            if self.logger:
                self.logger.log_error("MEMORY", msg=f"[记忆节点] 查询失败: {e}")
            return {
                "memory_info": "",
                "visited_attractions": [],
                "current_phase": "memory"
            }
    
    def build_attraction_query(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """构建景点搜索查询"""
        city = state.get("city", "")
        preferences = state.get("preferences", [])
        visited = state.get("visited_attractions", [])
        
        keywords = preferences[0] if preferences else "景点"
        
        visited_info = ""
        if visited:
            visited_info = f"\n注意：用户已访问过以下景点，请优先搜索其他景点：{', '.join(visited)}"
        
        query = f"{city}的{keywords}相关景点"
        
        return {
            "query": query,
            "current_phase": "attractions"
        }
    
    def build_weather_query(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """构建天气查询"""
        return {
            "current_phase": "weather"
        }
    
    def build_hotel_query(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """构建酒店查询"""
        return {
            "current_phase": "hotels"
        }
    
    def plan_routes(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """路线规划节点：调用高德真实路径规划，供后续 planner 使用"""
        attraction_response = state.get("attraction_response", "")
        city = state.get("city", "")
        transportation = state.get("transportation", "公共交通")
        travel_days = state.get("travel_days", 1)
        route_info = self._extract_route_info(
            attraction_response,
            city=city,
            transportation=transportation,
            travel_days=travel_days,
        )
        return {
            "route_info": route_info,
            "current_phase": "route"
        }
    
    def generate_trip_plan(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """生成旅行计划（使用 md 模板渐进披露式提示词）"""
        try:
            messages = self._build_planner_messages(state)
            city = state.get("city", "")
            transportation = state.get("transportation", "地铁")
            parsed_model = invoke_structured(self.llm, TripPlanOutput, messages)
            plan_data = as_dict(parsed_model)
            if isinstance(parsed_model, TripPlanOutput):
                parsed = self._normalize_trip_plan(plan_data or {}, city, transportation)
                plan_text = json.dumps(plan_data, ensure_ascii=False)
            elif plan_data:
                parsed = self._normalize_trip_plan(plan_data, city, transportation)
                plan_text = json.dumps(plan_data, ensure_ascii=False)
            else:
                parsed = None
                plan_text = ""
            
            if self.logger:
                days_count = len(parsed.get("days", [])) if parsed and isinstance(parsed, dict) else 0
                self.logger.log_info("PLAN", msg=f"[规划节点] md模板渐进披露, LLM响应长度: {len(plan_text)}, 解析结果: {type(parsed).__name__}, 天数: {days_count}")
            
            return {
                "trip_plan": plan_text,
                "trip_plan_parsed": parsed,
                "current_phase": "plan"
            }
        except Exception as e:
            if self.logger:
                self.logger.log_error("PLAN", msg=f"[规划节点] 生成失败: {e}")
            return {
                "trip_plan": f"旅行计划生成失败: {str(e)}",
                "trip_plan_parsed": None,
                "current_phase": "plan",
                "errors": [e]
            }
    
    def reflect_trip_plan(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """反思旅行计划"""
        trip_plan = state.get("trip_plan", "")
        parsed = state.get("trip_plan_parsed")
        attraction_response = state.get("attraction_response", "")
        weather_response = state.get("weather_response", "")
        travel_days = state.get("travel_days", 1)
        
        issues = []
        should_retry = False
        
        if not parsed:
            issues.append("旅行计划JSON解析失败")
            should_retry = True
        
        if parsed:
            days = parsed.get("days", [])
            if isinstance(days, int):
                days = []
            if not isinstance(days, list):
                days = []
            
            if len(days) == 0:
                issues.append("旅行计划缺少每日行程")
                should_retry = True
            elif len(days) < travel_days:
                issues.append(f"旅行计划天数({len(days)})少于要求天数({travel_days})")
                should_retry = True
            
            for day in days:
                if not isinstance(day, dict):
                    continue
                attractions = day.get("attractions", [])
                if not isinstance(attractions, list):
                    attractions = []
                if len(attractions) < 3 and self._day_end_hour(day) < self.DAY_END:
                    issues.append(f"第{day.get('day_index', 0)}天未排到21:00，且景点偏少（{len(attractions)}个）")
                    should_retry = True
                    break
        
        if len(attraction_response) < 100:
            issues.append("景点信息不足，可能需要重新搜索")
            should_retry = True
        
        if should_retry:
            iteration = state.get("iteration_count", 0)
            max_iterations = state.get("max_iterations", 3)
            
            if iteration < max_iterations:
                return {
                    "reflection_feedback": f"反思发现问题: {'; '.join(issues)}，将重试（{iteration + 1}/{max_iterations}）",
                    "iteration_count": iteration + 1,
                    "current_phase": "reflect"
                }
            else:
                return {
                    "reflection_feedback": f"达到最大重试次数，返回当前结果。问题: {'; '.join(issues)}",
                    "current_phase": "final"
                }
        else:
            return {
                "reflection_feedback": "旅行计划完整，进入最终阶段",
                "current_phase": "final"
            }
    
    def save_trip_plan(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """保存旅行计划（含预算计算）"""
        user_id = state.get("user_id", "")
        city = state.get("city", "")
        parsed = state.get("trip_plan_parsed")
        
        if not user_id or not parsed:
            return {"current_phase": "final"}
        
        try:
            attractions = []
            for day in parsed.get("days", []):
                for attr in day.get("attractions", []):
                    if isinstance(attr, dict):
                        attractions.append(attr.get("name", ""))
            
            daily_itinerary = {}
            for day in parsed.get("days", []):
                day_idx = day.get("day_index", 0)
                daily_itinerary[f"day{day_idx}"] = [
                    a.get("name", "") if isinstance(a, dict) else str(a)
                    for a in day.get("attractions", [])
                ]
            
            # 计算预算
            budget = self._calculate_budget(parsed, state)
            
            self.memory_manager.save_plan(
                user_id=user_id,
                city=city,
                duration_days=state.get("travel_days", 1),
                attractions=attractions,
                daily_itinerary=daily_itinerary,
                preferences=state.get("preferences", []),
                budget=budget,
                notes=f"{state.get('travel_days', 1)}天{city}旅行计划"
            )
            
            if self.logger:
                self.logger.log_info("SAVE", msg=(
                    f"[保存节点] 旅行计划已保存: {user_id}/{city}, "
                    f"预算总计: {budget['total']}元"
                ))
            
            return {
                "current_phase": "final",
                "budget": budget,
                "budget_total": budget.get("total", 0)
            }
        except Exception as e:
            if self.logger:
                self.logger.log_error("SAVE", msg=f"[保存节点] 保存失败: {e}")
            return {"current_phase": "final"}

    # ==================== 版本管理节点 ====================

    def save_version(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        保存初始版本到VersionManager

        在规划完成后，创建第一个版本快照，用于后续对话修改的版本追溯。
        """
        from app.agents.version_manager import VersionManager
        from app.agents.dialog_models import DialogSession

        user_id = state.get("user_id", "")
        city = state.get("city", "")
        parsed = state.get("trip_plan_parsed")
        session_id = state.get("session_id", "")

        if not user_id or not parsed:
            return {"current_phase": "final"}

        try:
            # 初始化版本管理器
            version_manager = VersionManager(use_redis=True)

            # 如果没有session_id，创建新会话
            if not session_id:
                session = version_manager.create_session(user_id=user_id, plan_id="")
                session_id = session.session_id
            else:
                session = version_manager.get_session(session_id)
                if not session:
                    session = version_manager.create_session(user_id=user_id, plan_id="")

            # 构建计划数据
            attractions = []
            for day in parsed.get("days", []):
                for attr in day.get("attractions", []):
                    if isinstance(attr, dict):
                        attractions.append(attr.get("name", ""))

            plan_data = {
                "city": city,
                "duration_days": state.get("travel_days", 1),
                "attractions": attractions,
                "budget": state.get("budget", {}),
                "daily_itinerary": parsed,
                "transportation": state.get("transportation", ""),
                "accommodation": state.get("accommodation", ""),
            }

            # 创建初始版本
            initial_version = version_manager.create_initial_version(
                session_id=session_id,
                plan_data=plan_data
            )

            if self.logger:
                self.logger.log_info("VERSION", msg=(
                    f"[版本节点] 初始版本已保存: v{initial_version.version_number}, "
                    f"session_id={session_id}"
                ))

            return {
                "session_id": session_id,
                "current_version_id": initial_version.version_id,
                "current_phase": "save_version"
            }

        except Exception as e:
            if self.logger:
                self.logger.log_error("VERSION", msg=f"[版本节点] 保存失败: {e}")
            return {"current_phase": "final"}

    # ==================== 对话Agent节点 ====================

    def _dialog_welcome(self, state: Dict[str, Any]) -> str:
        city = state.get("city", "")
        budget = state.get("budget", {})
        budget_total = budget.get("total", 0) if isinstance(budget, dict) else 0
        return f"""✅ 旅行计划已生成！

📍 城市：{city}
💰 预算总计：{budget_total}元

您可以输入修改指令，例如：
• "添加迪士尼"
• "降低酒店预算"
• "延长到5天"
• "对比一下变化"
• "回退到之前的版本"

或输入"结束"退出对话。"""

    def dialog(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        对话节点：用 interrupt 停住，等用户的下一条修改。

        第一次进入时暂停，把欢迎语交给调用方。
        Command(resume=用户输入) 之后从这里继续，改完计划再暂停，直到用户结束。
        """
        from langgraph.types import interrupt

        session_id = state.get("session_id", "")
        if not session_id:
            return {
                "dialog_reply": "错误：会话未初始化，请先规划旅行",
                "dialog_active": False,
                "current_phase": "final",
            }

        shown = state.get("dialog_reply") or self._dialog_welcome(state)
        user_text = interrupt({"reply": shown, "session_id": session_id})
        text = str(user_text or "").strip()

        if not text:
            return {
                "dialog_reply": shown,
                "dialog_input": "",
                "dialog_active": True,
                "current_phase": "dialog",
            }

        if text in ["结束", "退出", "完成", "不需要修改", "好的"]:
            return {
                "dialog_reply": "好的，旅行计划已确定。祝您旅途愉快！",
                "dialog_input": text,
                "dialog_changes": [],
                "dialog_active": False,
                "current_phase": "final",
            }

        from app.agents.dialog_agent import DialogAgent

        try:
            dialog_agent = DialogAgent(
                llm_provider="aliyun",
                use_redis=True,
                enable_rag=True,
                enable_amap=True,
                enable_memory=True,
            )
            result = dialog_agent.process_modification(
                session_id=session_id,
                user_input=text,
            )

            new_plan = None
            if result.get("new_version_id"):
                from app.agents.version_manager import VersionManager
                version_manager = VersionManager(use_redis=True)
                new_version = version_manager.get_version(result["new_version_id"])
                if new_version:
                    new_plan = new_version.plan_data

            if self.logger:
                self.logger.log_info("DIALOG", msg=(
                    f"[对话节点] 用户输入: {text}, "
                    f"新版本: {result.get('new_version_id', '无')}"
                ))

            updates = {
                "dialog_reply": result.get("reply", ""),
                "dialog_changes": result.get("changes", []),
                "dialog_input": text,
                "dialog_active": True,
                "current_phase": "dialog",
            }
            if new_plan:
                updates["current_version_id"] = result["new_version_id"]
                daily = new_plan.get("daily_itinerary")
                if isinstance(daily, dict) and daily.get("days"):
                    updates["trip_plan_parsed"] = daily
                elif isinstance(new_plan.get("days"), list):
                    updates["trip_plan_parsed"] = new_plan
                budget = new_plan.get("budget")
                if isinstance(budget, dict):
                    updates["budget"] = budget
                    updates["budget_total"] = budget.get("total", 0)
                if new_plan.get("duration_days"):
                    updates["travel_days"] = new_plan["duration_days"]
                if new_plan.get("accommodation"):
                    updates["accommodation"] = new_plan["accommodation"]
            return updates
        except Exception as e:
            if self.logger:
                self.logger.log_error("DIALOG", msg=f"[对话节点] 处理失败: {e}")
            return {
                "dialog_reply": f"抱歉，处理您的请求时出现错误：{str(e)}",
                "dialog_input": text,
                "dialog_active": True,
                "current_phase": "dialog",
            }

    # ==================== 条件路由函数 ====================
    
    def route_after_reflect_attractions(self, state: Dict[str, Any]) -> str:
        """景点反思后的条件路由（失败时回到 start，重新并行检索）"""
        if state.get("reflection_pass", False):
            return "generate"
        iteration = state.get("iteration_count", 0)
        if iteration < 2:
            return "retrieve"  # 映射到 start（见 builder 条件边）
        return "generate"
    
    def route_after_reflect_hotels(self, state: Dict[str, Any]) -> str:
        """酒店反思后的条件路由"""
        if state.get("reflection_pass", False):
            return "end"
        iteration = state.get("iteration_count", 0)
        if iteration < 2:
            return "search"
        return "end"
    
    def route_after_reflect_trip(self, state: Dict[str, Any]) -> str:
        """主图反思后只重跑规划，不再回到景点检索。"""
        if state.get("current_phase") == "reflect":
            return "plan"
        return "final"
    
    # ==================== 辅助方法 ====================
    
    def _calculate_budget(
        self,
        parsed_plan: Dict[str, Any],
        state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        计算旅行预算

        数据来源：
        - 景点门票：从 KB 文档的 metadata.price 提取
        - 酒店费用：从酒店搜索结果估算
        - 餐饮：按 60 元/人/餐 × 3 餐 × 天数
        - 交通：按景点间距离估算
        """
        days = parsed_plan.get("days", [])
        travel_days = state.get("travel_days", len(days) if days else 1)
        kb_docs = state.get("kb_docs", [])
        hotel_response = state.get("hotel_response", "")
        amap_results = state.get("amap_results", "")

        # 1. 景点门票费用
        attraction_prices = self._extract_attraction_prices(kb_docs, days)
        total_attractions = sum(attraction_prices.values())

        # 2. 酒店费用（每晚 300 元估算）
        hotel_count = max(1, travel_days - 1)
        hotel_price_per_night = self._estimate_hotel_price(hotel_response)
        total_hotels = hotel_price_per_night * hotel_count

        # 3. 餐饮费用（每天 3 餐 × 60 元/餐）
        meal_cost_per_day = 180
        total_meals = meal_cost_per_day * travel_days

        # 4. 交通费用
        total_transportation = self._estimate_transport_cost(days, amap_results)

        # 5. 汇总
        total = total_attractions + total_hotels + total_meals + total_transportation

        return {
            "per_day_breakdown": {
                f"day_{i+1}": {
                    "attractions": [
                        {"name": a.get("name", ""), "ticket_price": attraction_prices.get(a.get("name", ""), 0)}
                        for a in day.get("attractions", [])
                        if isinstance(a, dict)
                    ]
                }
                for i, day in enumerate(days)
                if isinstance(day, dict)
            },
            "attraction_tickets": attraction_prices,
            "total_attractions": total_attractions,
            "hotel_details": {
                "nights": hotel_count,
                "price_per_night": hotel_price_per_night,
            },
            "total_hotels": total_hotels,
            "meal_details": {
                "meals_per_day": 3,
                "price_per_meal": 60,
                "days": travel_days,
            },
            "total_meals": total_meals,
            "total_transportation": total_transportation,
            "total": total,
            "currency": "CNY",
            "people_count": 1,
        }

    def _extract_attraction_prices(
        self,
        kb_docs: List[Dict],
        days: List[Dict]
    ) -> Dict[str, int]:
        """从知识库文档中提取景点门票价格"""
        price_map = {}
        for doc in kb_docs:
            meta = doc.get("metadata", {})
            title = meta.get("title", "")
            price_str = meta.get("price", "0")
            try:
                price = int(re.sub(r'[^\d]', '', str(price_str)) or 0)
            except (ValueError, TypeError):
                price = 0
            if title:
                price_map[title] = price

        attraction_prices = {}
        for day in days:
            if not isinstance(day, dict):
                continue
            for attr in day.get("attractions", []):
                if isinstance(attr, dict):
                    name = attr.get("name", "")
                    if name:
                        attraction_prices[name] = price_map.get(name, 0)

        return attraction_prices

    def _estimate_hotel_price(self, hotel_response: str) -> int:
        """从酒店搜索结果估算平均房价"""
        if not hotel_response:
            return 300

        prices = []
        for match in re.finditer(r'(\d+)\s*元', hotel_response):
            try:
                price = int(match.group(1))
                if 50 <= price <= 3000:
                    prices.append(price)
            except ValueError:
                pass

        if prices:
            return max(200, sum(prices) // len(prices))
        return 300

    def _estimate_transport_cost(
        self,
        days: List[Dict],
        amap_results: str
    ) -> int:
        """估算交通费用"""
        base_cost_per_day = 30
        total = 0

        for day in days:
            if not isinstance(day, dict):
                continue
            attractions = day.get("attractions", [])
            route_segments = day.get("route_segments", [])

            day_cost = base_cost_per_day
            if route_segments and isinstance(route_segments, list):
                for seg in route_segments:
                    if isinstance(seg, dict):
                        distance = seg.get("distance", 0)
                        if distance:
                            day_cost += max(5, int(distance * 0.005))

            if len(attractions) > 2:
                day_cost += 20

            total += day_cost

        return total

    def _match_categories(self, query: str, preferences: List[str]) -> List[str]:
        """匹配分类"""
        category_keywords = {
            "历史文化": ["历史", "古迹", "遗址", "文物", "古城", "古镇", "古寺", "故居", "博物馆", "纪念", "革命"],
            "自然风光": ["自然", "风景", "山水", "湖", "海", "山", "森林", "公园", "生态", "瀑布"],
            "主题乐园": ["乐园", "主题公园", "游乐园", "迪士尼", "欢乐谷", "海洋馆", "水族馆"],
            "艺术文化": ["艺术", "美术馆", "剧院", "剧场", "音乐", "文化", "展览"],
            "购物休闲": ["购物", "商业", "步行街", "商场", "美食", "小吃"],
        }
        
        matched = []
        text = f"{query} {' '.join(preferences)}"
        
        for category, keywords in category_keywords.items():
            for kw in keywords:
                if kw in text:
                    matched.append(category)
                    break
        
        if not matched:
            matched = ["历史文化", "自然风光"]
        
        return matched
    
    def _format_amap_results(self, results) -> str:
        """格式化高德搜索结果（POIInfo 对象列表）"""
        if not results:
            return ""
        
        lines = []
        for i, r in enumerate(results[:10], 1):
            name = r.name if hasattr(r, 'name') else r.get('name', '未知')
            lines.append(f"{i}. {name}")
            
            address = r.address if hasattr(r, 'address') else r.get('address')
            if address:
                lines.append(f"   地址: {address}")
            
            location = r.location if hasattr(r, 'location') else r.get('location')
            if location:
                loc_str = str(location)
                lines.append(f"   坐标: {loc_str}")
            
            poi_type = r.type if hasattr(r, 'type') else r.get('type')
            if poi_type:
                lines.append(f"   类型: {poi_type}")
            
            lines.append("")
        
        return "\n".join(lines)
    
    def _format_attraction_response(self, results: List[Dict]) -> str:
        """格式化景点响应（含坐标、建议游览时长）"""
        lines = []
        for i, r in enumerate(results[:12], 1):
            name = r.get("name", "未知")
            lines.append(f"{i}. {name}")
            
            address = r.get("address", "")
            if address:
                lines.append(f"   地址: {address}")
            
            desc = r.get("description", "")
            if desc:
                lines.append(f"   简介: {desc[:150]}")
            
            category = r.get("category", "")
            if category:
                lines.append(f"   分类: {category}")
            
            rating = r.get("rating", 0)
            if rating:
                lines.append(f"   评分: {rating}")
            
            visit_duration = r.get("visit_duration")
            if visit_duration:
                lines.append(f"   建议游览: {visit_duration}分钟")
            
            location = r.get("location", "")
            if location:
                lines.append(f"   坐标: {location}")
            
            if r.get("rerank_score") is not None:
                try:
                    lines.append(f"   相关性: {float(r['rerank_score']):.2f}")
                except (TypeError, ValueError):
                    pass
            
            source = r.get("source", "")
            if source:
                lines.append(f"   来源: {source}")
            
            lines.append("")
        
        return "\n".join(lines)
    
    def _format_hotel_results(self, results) -> str:
        """格式化酒店结果（POIInfo 对象列表）"""
        if not results:
            return "未找到符合条件的酒店"
        
        lines = []
        for i, r in enumerate(results[:10], 1):
            name = r.name if hasattr(r, 'name') else r.get('name', '未知')
            lines.append(f"{i}. {name}")
            
            address = r.address if hasattr(r, 'address') else r.get('address')
            if address:
                lines.append(f"   地址: {address}")
            
            tel = r.tel if hasattr(r, 'tel') else r.get('tel')
            if tel:
                lines.append(f"   电话: {tel}")
            
            poi_type = r.type if hasattr(r, 'type') else r.get('type')
            if poi_type:
                lines.append(f"   类型: {poi_type}")
            
            lines.append("")
        
        return "\n".join(lines)
    
    def _extract_route_info(
        self,
        attraction_response: str,
        city: str = "",
        transportation: str = "公共交通",
        travel_days: int = 1,
    ) -> str:
        """从景点响应提取列表，并调用高德真实路径规划生成路线摘要"""
        try:
            attractions = []
            current = {}

            for line in attraction_response.split("\n"):
                line = line.strip()
                if line.startswith(tuple(f"{i}." for i in range(1, 16))):
                    if current:
                        attractions.append(current)
                    current = {"name": line.split(".", 1)[1].strip() if "." in line else line}
                elif line.startswith("坐标:"):
                    current["location"] = line[3:].strip()
                elif line.startswith("地址:"):
                    current["address"] = line[3:].strip()

            if current:
                attractions.append(current)

            max_n = max(8, (travel_days or 1) * 5)
            attractions = attractions[:max_n]
            if len(attractions) < 2:
                return ""

            # 生成相邻段真实路线
            segments = self._generate_route_segments_from_attractions(
                attractions, city=city, transportation=transportation
            )

            lines = [
                "以下为高德真实路径规划结果，规划行程时请优先采用这些距离与时长：",
                "",
                "景点列表:",
            ]
            for i, attr in enumerate(attractions, 1):
                loc = attr.get("location") or attr.get("address") or "未知"
                lines.append(f"  {i}. {attr.get('name', '')}: {loc}")

            lines.append("")
            lines.append("相邻景点高德路线:")
            for seg in segments:
                dist = seg.get("distance") or 0
                dist_txt = f"{dist/1000:.1f}公里" if dist >= 1000 else f"{dist}米"
                detail = seg.get("route_detail") or seg.get("description") or ""
                lines.append(
                    f"  {seg.get('start_point')} -> {seg.get('end_point')}: "
                    f"{seg.get('transport_mode')}，约{seg.get('duration')}分钟，{dist_txt}"
                    f"{(' · ' + detail) if detail else ''}（高德真实路线）"
                )
            return "\n".join(lines)
        except Exception as e:
            if self.logger:
                self.logger.log_info("ROUTE", msg=f"[路线规划] 提取失败: {e}")
            return ""
    
    def _parse_trip_plan(self, response: str, city: str = "", transportation: str = "地铁") -> Optional[Dict[str, Any]]:
        """解析旅行计划 JSON，自动归一化字段结构"""
        try:
            json_str = None
            
            if "```json" in response:
                start = response.find("```json") + 7
                end = response.find("```", start)
                json_str = response[start:end].strip()
            elif "```" in response:
                start = response.find("```") + 3
                end = response.find("```", start)
                if end != -1:
                    json_str = response[start:end].strip()
            elif "{" in response and "}" in response:
                start = response.find("{")
                end = response.rfind("}") + 1
                json_str = response[start:end]
            
            if json_str:
                parsed = json.loads(json_str)
                return self._normalize_trip_plan(parsed, city, transportation)
            return None
        except Exception:
            return None
    
    def _normalize_trip_plan(self, parsed: Dict[str, Any], city: str = "", transportation: str = "地铁") -> Dict[str, Any]:
        """将各种可能的 JSON 结构归一化为标准格式，修正 day_index，保留详细结构"""
        if not isinstance(parsed, dict):
            return parsed
        
        days = parsed.get("days")
        
        if days is None and isinstance(parsed.get("days"), int):
            parsed["days"] = []
            return parsed
        
        if not isinstance(days, list):
            for alt_key in ["itinerary", "daily_plan", "schedule", "day_by_day", "plans"]:
                if alt_key in parsed and isinstance(parsed[alt_key], list):
                    parsed["days"] = parsed[alt_key]
                    days = parsed["days"]
                    break
        
        if not isinstance(days, list):
            days = []
            for key, value in parsed.items():
                if isinstance(value, list) and key not in ["attractions", "hotels", "meals", "transport", "weather_info", "budget"]:
                    if value and isinstance(value[0], dict) and ("attractions" in value[0] or "spots" in value[0]):
                        days = value
                        break
        
        if not days:
            attractions_list = parsed.get("attractions", [])
            if isinstance(attractions_list, list) and attractions_list:
                days = [{
                    "day_index": i,
                    "attractions": attractions_list[i:i+3] if i < len(attractions_list) else attractions_list[-3:]
                } for i in range(0, len(attractions_list), 3)]
        
        if isinstance(days, list):
            used_names = set()
            for day in days:
                if not isinstance(day, dict):
                    continue
                for item in day.get("attractions") or []:
                    name = item.get("name", "") if isinstance(item, dict) else str(item)
                    if name:
                        used_names.add(name)
            hotel = parsed.get("hotel")
            if not (isinstance(hotel, dict) and str(hotel.get("name") or "").strip()):
                accommodation = parsed.get("accommodation")
                if isinstance(accommodation, str) and accommodation.strip():
                    hotel = {"name": accommodation.strip()}
                else:
                    hotel = None
            for i, day in enumerate(days):
                if isinstance(day, dict):
                    day["day_index"] = i
                    if isinstance(hotel, dict) and hotel.get("name"):
                        day["hotel"] = hotel
                    self._fill_day_until_close(day, city, transportation, used_names)
                    if "meals" not in day:
                        day["meals"] = [
                            {"type": "breakfast", "name": "酒店早餐", "description": "酒店或附近早餐", "estimated_cost": 30},
                            {"type": "lunch", "name": "当地特色午餐", "description": "附近特色餐厅", "estimated_cost": 60},
                            {"type": "dinner", "name": "当地特色晚餐", "description": "附近特色餐厅", "estimated_cost": 80}
                        ]
        if isinstance(days, list):
            parsed["days"] = days
            flat_names = []
            for day in days:
                if not isinstance(day, dict):
                    continue
                for item in day.get("attractions") or []:
                    name = item.get("name", "") if isinstance(item, dict) else str(item)
                    if name and name not in flat_names:
                        flat_names.append(name)
            if flat_names:
                parsed["attractions"] = flat_names
        
        return parsed
    
    @staticmethod
    def _visit_minutes(value: Any, default: int = 120) -> int:
        try:
            minutes = int(float(value))
        except (TypeError, ValueError):
            return default
        return minutes if minutes > 0 else default

    @staticmethod
    def _format_hhmm(minutes: int) -> str:
        minutes = max(0, int(minutes))
        return f"{minutes // 60:02d}:{minutes % 60:02d}"

    @staticmethod
    def _parse_hhmm(value: Any) -> int:
        text = str(value or "")
        if ":" not in text:
            return -1
        hour, minute = text.split(":", 1)
        try:
            return int(hour) * 60 + int(minute)
        except ValueError:
            return -1

    def _generate_time_slots_from_attractions(self, attractions: List[Dict], route_segments: List[Dict] = None) -> List[Dict]:
        """按游览分钟和高德路程分钟排时间。一段游览要能在 21:00 前结束，才写入时间轴。"""
        time_slots = []
        current = self.DAY_START_MIN
        route_segs = route_segments or []

        for i, attr in enumerate(attractions):
            if not isinstance(attr, dict):
                continue
            if current >= self.DAY_END_MIN:
                break
            name = attr.get("name", "景点")
            duration = self._visit_minutes(attr.get("visit_duration"), default=120)
            if current + duration > self.DAY_END_MIN:
                break
            end = current + duration
            time_slots.append({
                "start_time": self._format_hhmm(current),
                "end_time": self._format_hhmm(end),
                "activity_type": "景点游览",
                "activity_name": name,
                "description": attr.get("description", f"游览{name}")
            })
            current = end

            if i < len(attractions) - 1:
                travel_duration = 20
                travel_desc = "前往下一个景点"
                transport_mode = "交通"
                unknown = False

                if i < len(route_segs):
                    seg = route_segs[i]
                    try:
                        parsed = int(float(seg.get("duration") or 0))
                    except (TypeError, ValueError):
                        parsed = 0
                    travel_desc = seg.get("route_detail") or travel_desc
                    transport_mode = seg.get("transport_mode") or transport_mode
                    if parsed > 0 and "未取得" not in str(travel_desc):
                        travel_duration = parsed
                    else:
                        unknown = True
                        travel_duration = parsed or self.UNKNOWN_TRAVEL_MIN

                if unknown:
                    buffered = travel_duration
                    description = (
                        travel_desc if "未取得" in str(travel_desc)
                        else f"未取得高德路线，按{buffered}分钟估算"
                    )
                else:
                    buffered = travel_duration + self.TRAVEL_BUFFER_MIN
                    description = (
                        f"{transport_mode} · 约{travel_duration}分钟，"
                        f"另留{self.TRAVEL_BUFFER_MIN}分钟缓冲"
                    )

                next_item = attractions[i + 1]
                next_duration = self._visit_minutes(
                    next_item.get("visit_duration") if isinstance(next_item, dict) else None,
                    default=120,
                )
                if current + buffered + next_duration > self.DAY_END_MIN:
                    break
                travel_end = current + buffered
                time_slots.append({
                    "start_time": self._format_hhmm(current),
                    "end_time": self._format_hhmm(travel_end),
                    "activity_type": "交通",
                    "activity_name": travel_desc,
                    "description": description,
                })
                current = travel_end

        return time_slots

    @staticmethod
    def _day_end_hour(day: Dict[str, Any]) -> int:
        if not isinstance(day, dict):
            return GraphNodes.DAY_START
        slots = day.get("time_slots") or []
        if not slots:
            return GraphNodes.DAY_START
        end = str(slots[-1].get("end_time") or "")
        if ":" not in end:
            return GraphNodes.DAY_START
        try:
            return int(end.split(":", 1)[0])
        except ValueError:
            return GraphNodes.DAY_START

    @staticmethod
    def _location_text(data: Dict[str, Any]) -> str:
        if not isinstance(data, dict):
            return ""
        location = data.get("location")
        if isinstance(location, str) and "," in location:
            return location.strip()
        if isinstance(location, dict):
            lon = location.get("longitude")
            lat = location.get("latitude")
        else:
            lon = data.get("longitude")
            lat = data.get("latitude")
        if lon in (None, "") or lat in (None, ""):
            return ""
        return f"{lon},{lat}"

    def _coerce_attraction(self, item: Any) -> Optional[Dict[str, Any]]:
        if isinstance(item, str) and item.strip():
            return {
                "name": item.strip(),
                "address": "",
                "location": "",
                "visit_duration": 120,
            }
        if not isinstance(item, dict) or not item.get("name"):
            return None
        copied = dict(item)
        copied["location"] = self._location_text(copied)
        copied["visit_duration"] = self._visit_minutes(copied.get("visit_duration"), default=120)
        return copied

    def _lookup_kb_meta(self, name: str, city: str) -> Dict[str, Any]:
        store = getattr(getattr(self, "retriever", None), "vector_store", None)
        finder = getattr(store, "find_by_title", None)
        if finder is None or not name:
            return {}
        try:
            hits = finder(name, city=city, limit=1)
        except Exception:
            return {}
        if not hits:
            return {}
        meta = hits[0].get("metadata") or {}
        return meta if isinstance(meta, dict) else {}

    def _enrich_from_kb(self, attr: Dict[str, Any], city: str) -> Dict[str, Any]:
        """按景点标题从知识库补开放时间、坐标和游览分钟。"""
        cache = getattr(self, "_kb_cache", None)
        if cache is None:
            cache = {}
            self._kb_cache = cache
        key = (city or "", attr.get("name") or "")
        if key not in cache:
            cache[key] = self._lookup_kb_meta(attr.get("name") or "", city)
        meta = cache[key] or {}
        if not meta:
            return attr
        if not attr.get("open_time"):
            attr["open_time"] = meta.get("open_time") or ""
        if not attr.get("location"):
            attr["location"] = self._location_text(meta)
        if not attr.get("address"):
            attr["address"] = meta.get("address") or ""
        if meta.get("visit_duration"):
            attr["visit_duration"] = self._visit_minutes(
                meta.get("visit_duration"),
                default=attr.get("visit_duration") or 120,
            )
        return attr

    @staticmethod
    def _coords(item: Any) -> Optional[tuple]:
        if not isinstance(item, dict):
            return None
        text = item.get("location") or ""
        if not isinstance(text, str) or "," not in text:
            return None
        lon, lat = text.split(",", 1)
        try:
            return float(lon), float(lat)
        except ValueError:
            return None

    @staticmethod
    def _distance2(origin: tuple, dest: tuple) -> float:
        return (origin[0] - dest[0]) ** 2 + (origin[1] - dest[1]) ** 2

    def _order_nearest(self, items: List[Dict[str, Any]], origin: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """有坐标的点按最近邻排，没有坐标的保持原顺序跟在后面。"""
        located = [item for item in items if self._coords(item)]
        rest = [item for item in items if not self._coords(item)]
        ordered = []
        cursor = self._coords(origin) if origin else None
        pool = list(located)
        while pool:
            if cursor is None:
                pick = pool.pop(0)
            else:
                pick = min(pool, key=lambda item: self._distance2(cursor, self._coords(item)))
                pool.remove(pick)
            ordered.append(pick)
            cursor = self._coords(pick)
        ordered.extend(rest)
        return ordered

    @classmethod
    def _is_generic_stay(cls, name: str) -> bool:
        text = (name or "").strip()
        return not text or text in cls.GENERIC_STAYS

    def _resolve_hotel_anchor(self, day: Dict[str, Any], city: str) -> Optional[Dict[str, Any]]:
        """能定位的具体酒店作为当天的出发点和返回点。"""
        raw = day.get("hotel")
        if isinstance(raw, str):
            raw = {"name": raw}
        if not isinstance(raw, dict):
            return None
        name = str(raw.get("name") or "").strip()
        if self._is_generic_stay(name):
            return None
        location = self._location_text(raw)
        if location:
            return {"name": name, "location": location, "address": raw.get("address") or ""}
        cache = getattr(self, "_hotel_cache", None)
        if cache is None:
            cache = {}
            self._hotel_cache = cache
        key = (city or "", name)
        if key in cache:
            return cache[key]
        anchor = None
        amap = getattr(self, "amap_service", None)
        if amap is not None and city and hasattr(amap, "search_poi"):
            try:
                pois = amap.search_poi(name, city, True) or []
            except Exception:
                pois = []
            if pois:
                poi = pois[0]
                loc = getattr(poi, "location", None)
                if loc is not None and getattr(loc, "longitude", None) is not None:
                    anchor = {
                        "name": getattr(poi, "name", None) or name,
                        "location": f"{loc.longitude},{loc.latitude}",
                        "address": getattr(poi, "address", "") or "",
                    }
        cache[key] = anchor
        return anchor

    @staticmethod
    def _open_window(text: Any) -> tuple:
        """取开放时间里和 09:00–21:00 重叠最长的一段。原文没有钟点时，这一项不设开关门约束。"""
        spans = re.findall(
            r"(\d{1,2}):(\d{2})\s*[-~～—至到]+\s*(\d{1,2}):(\d{2})",
            str(text or ""),
        )
        best = None
        best_overlap = 0
        for oh, om, ch, cm in spans:
            try:
                open_m = int(oh) * 60 + int(om)
                close_m = int(ch) * 60 + int(cm)
            except ValueError:
                continue
            if close_m <= open_m:
                continue
            overlap = min(close_m, GraphNodes.DAY_END_MIN) - max(open_m, GraphNodes.DAY_START_MIN)
            if overlap > best_overlap:
                best_overlap = overlap
                best = (open_m, close_m)
        if not best:
            return None, None
        return best

    @staticmethod
    def _overlap(start: int, end: int, win_start: int, win_end: int) -> int:
        return max(0, min(end, win_end) - max(start, win_start))

    def _route_between(
        self,
        origin: Optional[Dict[str, Any]],
        dest: Optional[Dict[str, Any]],
        city: str,
        transportation: str,
    ) -> Optional[Dict[str, Any]]:
        if not origin or not dest:
            return None
        cache = getattr(self, "_route_cache", None)
        if cache is None:
            cache = {}
            self._route_cache = cache
        key = (origin.get("name"), dest.get("name"), city, transportation)
        if key in cache:
            return cache[key]
        try:
            segs = self._generate_route_segments_from_attractions(
                [origin, dest], city, transportation
            )
        except Exception:
            segs = []
        seg = segs[0] if segs else None
        cache[key] = seg
        return seg

    def _travel_block(self, seg: Optional[Dict[str, Any]]) -> tuple:
        """时间轴上的交通占用。真实路程加 10 分钟；没有时长时这一段按 30 分钟估算。"""
        if not isinstance(seg, dict):
            text = f"未取得高德路线，按{self.UNKNOWN_TRAVEL_MIN}分钟估算"
            return self.UNKNOWN_TRAVEL_MIN, text, text
        detail = str(seg.get("route_detail") or "")
        mode = str(seg.get("transport_mode") or "交通")
        try:
            minutes = int(float(seg.get("duration") or 0))
        except (TypeError, ValueError):
            minutes = 0
        if minutes <= 0 or "未取得" in detail:
            length = minutes or self.UNKNOWN_TRAVEL_MIN
            text = detail if "未取得" in detail else f"未取得高德路线，按{length}分钟估算"
            return length, text, text
        return (
            minutes + self.TRAVEL_BUFFER_MIN,
            detail or mode,
            f"{mode} · 约{minutes}分钟，另留{self.TRAVEL_BUFFER_MIN}分钟缓冲",
        )

    def _preview_visit(
        self,
        attr: Dict[str, Any],
        prev: Optional[Dict[str, Any]],
        current: int,
        city: str,
        transportation: str,
        hotel: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """游览结束和返回酒店都要落在关门时间与 21:00 之前，这一段才放入当天。"""
        duration = self._visit_minutes(attr.get("visit_duration"), default=120)
        travel_slot = 0
        travel_name = ""
        travel_desc = ""
        segment = None
        if prev is not None:
            segment = self._route_between(prev, attr, city, transportation)
            travel_slot, travel_name, travel_desc = self._travel_block(segment)
        arrival = current + travel_slot
        if arrival >= self.DAY_END_MIN:
            return None
        open_m, close_m = self._open_window(attr.get("open_time"))
        too_early = False
        if open_m is not None and arrival < open_m:
            if open_m - arrival <= 30:
                arrival = open_m
            else:
                too_early = True
        close_limit = self.DAY_END_MIN if close_m is None else min(self.DAY_END_MIN, close_m)
        if too_early:
            fits_later = open_m is not None and open_m + duration <= close_limit
            if not fits_later:
                return None
            return {"attr": attr, "too_early": True}
        end = arrival + duration
        if end > close_limit:
            return None
        if hotel is not None:
            back = self._route_between(attr, hotel, city, transportation)
            back_slot, _, _ = self._travel_block(back)
            if end + back_slot > self.DAY_END_MIN:
                return None
        return {
            "attr": attr,
            "too_early": False,
            "start": arrival,
            "end": end,
            "depart": current,
            "travel_slot": travel_slot,
            "travel_name": travel_name,
            "travel_desc": travel_desc,
            "segment": segment,
        }

    def _preview_wait(
        self,
        pending: List[Dict[str, Any]],
        prev: Optional[Dict[str, Any]],
        current: int,
        city: str,
        transportation: str,
        hotel: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """开放时间晚于当前时刻、且间隔在 3 小时以内时，出发时间对准开门。"""
        best = None
        best_open = None
        for attr in pending:
            open_m, _close_m = self._open_window(attr.get("open_time"))
            if open_m is None or open_m <= current or open_m - current > self.OPEN_WAIT_MAX:
                continue
            travel_slot = 0
            if prev is not None:
                seg = self._route_between(prev, attr, city, transportation)
                travel_slot, _, _ = self._travel_block(seg)
            depart = open_m - travel_slot
            if depart < current:
                continue
            preview = self._preview_visit(attr, prev, depart, city, transportation, hotel)
            if not preview or preview.get("too_early"):
                continue
            if best_open is None or open_m < best_open:
                best = preview
                best_open = open_m
        return best

    def _commit_visit(self, preview: Dict[str, Any], slots: List[Dict], segments: List[Dict], kept: List[Dict]) -> None:
        attr = preview["attr"]
        depart = preview["depart"]
        if preview["travel_slot"] > 0:
            if preview.get("segment"):
                segments.append(preview["segment"])
            slots.append({
                "start_time": self._format_hhmm(depart),
                "end_time": self._format_hhmm(depart + preview["travel_slot"]),
                "activity_type": "交通",
                "activity_name": preview["travel_name"],
                "description": preview["travel_desc"],
            })
        slots.append({
            "start_time": self._format_hhmm(preview["start"]),
            "end_time": self._format_hhmm(preview["end"]),
            "activity_type": "景点游览",
            "activity_name": attr.get("name", "景点"),
            "description": attr.get("description") or f"游览{attr.get('name', '景点')}",
        })
        kept.append(attr)

    def _note_meals(self, start: int, end: int, had_lunch: bool, had_dinner: bool) -> tuple:
        if self._overlap(start, end, self.LUNCH_START, self.LUNCH_END) >= 45:
            had_lunch = True
        if self._overlap(start, end, self.DINNER_START, self.DINNER_END) >= 45:
            had_dinner = True
        return had_lunch, had_dinner

    def _insert_due_meals(
        self,
        slots: List[Dict],
        current: int,
        had_lunch: bool,
        had_dinner: bool,
        reserve: int = 0,
    ) -> tuple:
        """钟点进入午饭或晚饭窗口时，先插入 60 分钟用餐再出发。正在进行的游览已经盖住饭点时，这一餐记在该游览里。"""
        if (
            not had_lunch
            and self.LUNCH_START <= current < self.LUNCH_END
            and current + self.MEAL_MINUTES + reserve <= self.DAY_END_MIN
        ):
            slots.append({
                "start_time": self._format_hhmm(current),
                "end_time": self._format_hhmm(current + self.MEAL_MINUTES),
                "activity_type": "用餐",
                "activity_name": "午餐",
                "description": "景点之间用餐",
            })
            current += self.MEAL_MINUTES
            had_lunch = True
        if (
            not had_dinner
            and self.DINNER_START <= current < self.DINNER_END
            and current + self.MEAL_MINUTES + reserve <= self.DAY_END_MIN
        ):
            slots.append({
                "start_time": self._format_hhmm(current),
                "end_time": self._format_hhmm(current + self.MEAL_MINUTES),
                "activity_type": "用餐",
                "activity_name": "晚餐",
                "description": "景点之间用餐",
            })
            current += self.MEAL_MINUTES
            had_dinner = True
        return current, had_lunch, had_dinner

    def _first_fit(
        self,
        pending: List[Dict[str, Any]],
        prev: Optional[Dict[str, Any]],
        current: int,
        city: str,
        transportation: str,
        hotel: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        for attr in pending:
            preview = self._preview_visit(attr, prev, current, city, transportation, hotel)
            if preview and not preview.get("too_early"):
                return preview
        return None

    @staticmethod
    def _take(pending: List[Dict[str, Any]], attr: Dict[str, Any]) -> None:
        for index, item in enumerate(pending):
            if item is attr or item.get("name") == attr.get("name"):
                pending.pop(index)
                return

    def _place_one(
        self,
        pending: List[Dict[str, Any]],
        slots: List[Dict],
        segments: List[Dict],
        kept: List[Dict],
        current: int,
        prev: Optional[Dict[str, Any]],
        city: str,
        transportation: str,
        hotel: Optional[Dict[str, Any]],
        had_lunch: bool,
        had_dinner: bool,
    ) -> tuple:
        preview = self._first_fit(pending, prev, current, city, transportation, hotel)
        covers = False
        if preview:
            covers = (
                (not had_lunch and self._overlap(preview["start"], preview["end"], self.LUNCH_START, self.LUNCH_END) >= 45)
                or (not had_dinner and self._overlap(preview["start"], preview["end"], self.DINNER_START, self.DINNER_END) >= 45)
            )
        meal_due = (
            (not had_lunch and self.LUNCH_START <= current < self.LUNCH_END)
            or (not had_dinner and self.DINNER_START <= current < self.DINNER_END)
        )
        if preview and (not meal_due or covers):
            self._commit_visit(preview, slots, segments, kept)
            had_lunch, had_dinner = self._note_meals(preview["start"], preview["end"], had_lunch, had_dinner)
            self._take(pending, preview["attr"])
            return preview["end"], preview["attr"], had_lunch, had_dinner, True
        if meal_due:
            current, had_lunch, had_dinner = self._insert_due_meals(
                slots, current, had_lunch, had_dinner
            )
            preview = self._first_fit(pending, prev, current, city, transportation, hotel)
            if preview:
                self._commit_visit(preview, slots, segments, kept)
                had_lunch, had_dinner = self._note_meals(preview["start"], preview["end"], had_lunch, had_dinner)
                self._take(pending, preview["attr"])
                return preview["end"], preview["attr"], had_lunch, had_dinner, True
        waited = self._preview_wait(pending, prev, current, city, transportation, hotel)
        if waited:
            self._commit_visit(waited, slots, segments, kept)
            had_lunch, had_dinner = self._note_meals(waited["start"], waited["end"], had_lunch, had_dinner)
            self._take(pending, waited["attr"])
            return waited["end"], waited["attr"], had_lunch, had_dinner, True
        return current, prev, had_lunch, had_dinner, False

    def _drain(
        self,
        pending: List[Dict[str, Any]],
        slots: List[Dict],
        segments: List[Dict],
        kept: List[Dict],
        current: int,
        prev: Optional[Dict[str, Any]],
        city: str,
        transportation: str,
        hotel: Optional[Dict[str, Any]],
        had_lunch: bool,
        had_dinner: bool,
    ) -> tuple:
        pending = list(pending)
        while pending:
            current, prev, had_lunch, had_dinner, placed = self._place_one(
                pending, slots, segments, kept, current, prev, city, transportation,
                hotel, had_lunch, had_dinner,
            )
            if not placed:
                break
        return current, prev, had_lunch, had_dinner, pending

    def _too_early_pinned(self, pending: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        kept = []
        for attr in pending:
            if attr.get("filler"):
                continue
            open_m, close_m = self._open_window(attr.get("open_time"))
            if open_m is None:
                continue
            close_limit = self.DAY_END_MIN if close_m is None else min(self.DAY_END_MIN, close_m)
            duration = self._visit_minutes(attr.get("visit_duration"), default=120)
            if open_m + duration <= close_limit and open_m > self.DAY_START_MIN:
                kept.append(attr)
        return kept

    def _assemble_day(
        self,
        attractions: List[Dict[str, Any]],
        city: str,
        transportation: str,
        hotel: Optional[Dict[str, Any]],
    ) -> tuple:
        """3 小时及以上的景点从上午排起，其余景点和补点按最近邻接在后面。"""
        pinned = [item for item in attractions if not item.get("filler")]
        fillers = [item for item in attractions if item.get("filler")]
        long_items = [
            item for item in pinned
            if self._visit_minutes(item.get("visit_duration"), default=120) >= self.LONG_VISIT_MIN
        ]
        long_names = {item.get("name") for item in long_items}
        short_items = [item for item in pinned if item.get("name") not in long_names]
        long_items = self._order_nearest(long_items, hotel)
        slots: List[Dict] = []
        segments: List[Dict] = []
        kept: List[Dict] = []
        current = self.DAY_START_MIN
        prev = hotel
        had_lunch = False
        had_dinner = False
        current, prev, had_lunch, had_dinner, left = self._drain(
            long_items, slots, segments, kept, current, prev, city, transportation,
            hotel, had_lunch, had_dinner,
        )
        deferred = self._too_early_pinned(left)
        short_items = self._order_nearest(short_items, prev or hotel)
        current, prev, had_lunch, had_dinner, left = self._drain(
            short_items, slots, segments, kept, current, prev, city, transportation,
            hotel, had_lunch, had_dinner,
        )
        deferred.extend(self._too_early_pinned(left))
        pending_fillers = self._order_nearest(fillers, prev or hotel)
        pending_deferred = deferred
        while pending_fillers or pending_deferred:
            if pending_deferred:
                current, prev, had_lunch, had_dinner, placed = self._place_one(
                    pending_deferred, slots, segments, kept, current, prev, city, transportation,
                    hotel, had_lunch, had_dinner,
                )
                if placed:
                    continue
            if not pending_fillers:
                break
            one = [pending_fillers.pop(0)]
            current, prev, had_lunch, had_dinner, left = self._drain(
                one, slots, segments, kept, current, prev, city, transportation,
                hotel, had_lunch, had_dinner,
            )
            if left:
                break
        back_slot = 0
        if hotel and kept and prev is not None:
            back = self._route_between(prev, hotel, city, transportation)
            back_slot, _, _ = self._travel_block(back)
        current, had_lunch, had_dinner = self._insert_due_meals(
            slots, current, had_lunch, had_dinner, reserve=back_slot
        )
        if hotel and kept and prev is not None:
            seg = self._route_between(prev, hotel, city, transportation)
            slot, name, desc = self._travel_block(seg)
            if current + slot <= self.DAY_END_MIN:
                if seg:
                    segments.append(seg)
                else:
                    segments.append({
                        "start_point": prev.get("name"),
                        "end_point": hotel.get("name"),
                        "transport_mode": "交通",
                        "distance": 0,
                        "duration": slot,
                        "description": desc,
                        "route_detail": name,
                    })
                slots.append({
                    "start_time": self._format_hhmm(current),
                    "end_time": self._format_hhmm(current + slot),
                    "activity_type": "交通",
                    "activity_name": name,
                    "description": desc,
                })
        still_early = [item for item in pending_deferred if item.get("name") not in {kept_item.get("name") for kept_item in kept}]
        return kept, slots, segments, self._too_early_pinned(still_early)

    def _schedule_day_clock(self, day: Dict[str, Any], city: str, transportation: str) -> None:
        attractions = []
        for item in day.get("attractions") or []:
            coerced = self._coerce_attraction(item)
            if coerced:
                attractions.append(self._enrich_from_kb(coerced, city))
        hotel = self._resolve_hotel_anchor(day, city)
        kept, slots, segments, deferred = self._assemble_day(
            attractions, city, transportation or "公共交通", hotel
        )
        day["attractions"] = kept + deferred
        day["route_segments"] = segments
        day["time_slots"] = slots
        names = [item["name"] for item in kept]
        if names:
            day["description"] = f"第{int(day.get('day_index', 0)) + 1}天行程：{', '.join(names)}"

    def _slot_start_minutes(self, day: Dict[str, Any], name: str) -> Optional[int]:
        for slot in day.get("time_slots") or []:
            if slot.get("activity_type") == "景点游览" and slot.get("activity_name") == name:
                start = self._parse_hhmm(slot.get("start_time"))
                return start if start >= 0 else None
        return None

    @staticmethod
    def _day_end_minutes(day: Dict[str, Any]) -> int:
        if not isinstance(day, dict):
            return GraphNodes.DAY_START_MIN
        slots = day.get("time_slots") or []
        if not slots:
            return GraphNodes.DAY_START_MIN
        end = GraphNodes._parse_hhmm(slots[-1].get("end_time"))
        return end if end >= 0 else GraphNodes.DAY_START_MIN

    def _fill_day_until_close(
        self,
        day: Dict[str, Any],
        city: str,
        transportation: str,
        used: set,
    ) -> None:
        """先排已有景点。结束早于 21:00 且还空出至少 1 小时时，再搜索一个知识库景点补上。"""
        if not isinstance(day, dict):
            return
        self._schedule_day_clock(day, city, transportation or "公共交通")
        if not city:
            return
        for _ in range(6):
            end = self._day_end_minutes(day)
            if end >= self.DAY_END_MIN:
                break
            existing = [
                item.get("name")
                for item in day.get("attractions") or []
                if isinstance(item, dict) and item.get("name")
            ]
            remaining = self.DAY_END_MIN - end
            if existing and remaining < 60:
                break
            slotted = [
                slot.get("activity_name")
                for slot in day.get("time_slots") or []
                if slot.get("activity_type") == "景点游览" and slot.get("activity_name")
            ]
            last_name = slotted[-1] if slotted else (existing[-1] if existing else "")
            try:
                record = self._suggest_filler_attraction(
                    city, list(used), max(1, remaining // 60), last_name
                )
            except Exception as e:
                if getattr(self, "logger", None):
                    self.logger.log_info("PLAN", msg=f"[补点] 失败: {e}")
                break
            if not isinstance(record, dict):
                break
            name = str(record.get("name") or "").strip()
            if not name or name in used:
                break
            record = dict(record)
            record["name"] = name
            record["filler"] = True
            record.setdefault("description", "补足当日空档")
            day.setdefault("attractions", []).append(record)
            used.add(name)
            self._schedule_day_clock(day, city, transportation or "公共交通")
            start = self._slot_start_minutes(day, name)
            if start is None or start >= self.DAY_END_MIN:
                used.discard(name)
                day["attractions"] = [
                    item for item in day.get("attractions") or []
                    if not (isinstance(item, dict) and item.get("name") == name)
                ]
                self._schedule_day_clock(day, city, transportation or "公共交通")
                break
        self._drop_unslotted(day)
        self._schedule_day_clock(day, city, transportation or "公共交通")

    def _drop_unslotted(self, day: Dict[str, Any]) -> None:
        slotted = {
            slot.get("activity_name")
            for slot in day.get("time_slots") or []
            if slot.get("activity_type") == "景点游览"
        }
        day["attractions"] = [
            item for item in day.get("attractions") or []
            if isinstance(item, dict) and item.get("name") in slotted
        ]

    def _suggest_filler_attraction(
        self,
        city: str,
        existing: List[str],
        remaining_hours: int,
        last_name: str,
    ) -> Optional[Dict[str, Any]]:
        """搜索一个知识库里标题能对上的景点，用来填 21:00 前的空档。"""
        llm = getattr(self, "llm", None)
        if llm is None or remaining_hours < 1:
            return None
        if getattr(self, "_fill_agent", None) is None:
            from app.agents.react_agent import ReActAgent
            from app.services.structured_output import FillNameOutput
            from app.tools.agent_tools import SmartAttractionTool

            self._fill_agent = ReActAgent(
                name="行程补点",
                tools=[SmartAttractionTool()],
                llm=llm,
                system_prompt=(
                    "你负责检查某一天是否排满。活动最晚到 21:00，不要排到 22:00。\n"
                    "若 21:00 前还有空档，调用 smart_attraction_search 补一个不在已有列表里的真实景点。\n"
                    "优先选游览时长能放进剩余时间、且离当天最后一个景点近的地点。\n"
                    "name 必须是工具结果中的完整标题，不要编造。"
                ),
                max_iterations=2,
                verbose=False,
                response_format=FillNameOutput,
            )
        query = (
            f"城市：{city}\n"
            f"已有景点：{', '.join(existing) or '无'}\n"
            f"当天最后一个景点：{last_name or '无'}\n"
            f"距离 21:00 还空着约 {remaining_hours} 小时。\n"
            "请调用 smart_attraction_search 补一个景点。"
        )
        try:
            text = self._fill_agent.run(query)
        except Exception as e:
            if getattr(self, "logger", None):
                self.logger.log_info("PLAN", msg=f"[补点] 检索失败: {e}")
            return None
        name = self._parse_fill_name(text)
        if not name or name in existing:
            return None
        return self._lookup_filler(name, city)

    @staticmethod
    def _parse_fill_name(text: str) -> str:
        if not text:
            return ""
        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            return ""
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return ""
        if not isinstance(data, dict):
            return ""
        return str(data.get("name") or "").strip()

    def _lookup_filler(self, name: str, city: str) -> Optional[Dict[str, Any]]:
        store = getattr(getattr(self, "retriever", None), "vector_store", None)
        finder = getattr(store, "find_by_title", None)
        if finder is None:
            return None
        try:
            hits = finder(name, city=city, limit=1)
        except Exception:
            return None
        if not hits:
            return None
        meta = hits[0].get("metadata") or {}
        price_str = meta.get("price", "0")
        try:
            price = int(re.sub(r"[^\d]", "", str(price_str)) or 0)
        except (TypeError, ValueError):
            price = 0
        return {
            "name": meta.get("title") or name,
            "visit_duration": self._visit_minutes(meta.get("visit_duration"), default=90),
            "address": meta.get("address") or "",
            "location": self._location_text(meta),
            "category": meta.get("category") or "景点",
            "ticket_price": price,
            "open_time": meta.get("open_time") or "",
            "filler": True,
            "description": "补足当日空档",
        }

    @staticmethod
    def _transit_line_names(route_info) -> List[str]:
        """收集一次公交换乘里每一段线路名。"""
        names = []
        transits = getattr(route_info, "transits", None) or []
        if not isinstance(transits, list) or not transits or not isinstance(transits[0], dict):
            return names
        for seg in transits[0].get("segments") or []:
            if not isinstance(seg, dict):
                continue
            bus = seg.get("bus") or {}
            if not isinstance(bus, dict):
                continue
            for line in bus.get("buslines") or []:
                if not isinstance(line, dict):
                    continue
                name = str(line.get("name") or "").strip()
                if name and name not in names:
                    names.append(name)
        return names

    @staticmethod
    def _qualify_route_leg(
        distance: int,
        duration_min: int,
        line_names: List[str],
        transportation: str,
        amap_mode: str,
    ) -> Dict[str, Any]:
        """500 米以内按步行计时。没有时长时记为 30 分钟估算。"""
        try:
            distance = int(distance or 0)
        except (TypeError, ValueError):
            distance = 0
        try:
            duration_min = int(duration_min or 0)
        except (TypeError, ValueError):
            duration_min = 0
        if 0 < distance < GraphNodes.WALK_MAX_METERS:
            minutes = max(5, int(round(distance / 70)))
            return {
                "distance": distance,
                "duration": minutes,
                "transport_mode": "步行",
                "route_detail": f"步行约{minutes}分钟，{distance}米",
            }
        if duration_min <= 0:
            return {
                "distance": distance,
                "duration": GraphNodes.UNKNOWN_TRAVEL_MIN,
                "transport_mode": transportation or "交通",
                "route_detail": f"未取得高德路线，按{GraphNodes.UNKNOWN_TRAVEL_MIN}分钟估算",
            }
        if line_names:
            detail = " → ".join(line_names) if len(line_names) > 1 else f"乘坐{line_names[0]}"
            mode = detail if len(line_names) > 1 else line_names[0]
        elif amap_mode == "driving":
            detail = f"驾车约{duration_min}分钟，{distance / 1000:.1f}公里"
            mode = "驾车"
        elif amap_mode == "walking":
            detail = f"步行约{duration_min}分钟，{distance}米"
            mode = "步行"
        else:
            detail = f"{transportation}约{duration_min}分钟"
            mode = transportation or "交通"
        return {
            "distance": distance,
            "duration": duration_min,
            "transport_mode": mode,
            "route_detail": detail,
        }
    
    def _generate_route_segments_from_attractions(self, attractions: List[Dict], city: str = "", transportation: str = "地铁") -> List[Dict]:
        """根据景点列表生成路线段（调用高德API获取真实路线信息）"""
        segments = []
        
        # 确定交通模式
        transport_mode_map = {
            "地铁": "transit",
            "公交": "transit",
            "公共交通": "transit",
            "打车": "driving",
            "驾车": "driving",
            "自驾": "driving",
            "步行": "walking",
            "骑行": "walking",
        }
        amap_mode = transport_mode_map.get(transportation, "transit")
        
        for i in range(len(attractions) - 1):
            a1 = attractions[i] if isinstance(attractions[i], dict) else {}
            a2 = attractions[i+1] if isinstance(attractions[i+1], dict) else {}
            
            name1 = a1.get("name", f"景点{i+1}")
            name2 = a2.get("name", f"景点{i+2}")
            
            segment = {
                "start_point": name1,
                "end_point": name2,
                "transport_mode": transportation,
                "distance": 0,
                "duration": 0,
                "description": f"从{name1}到{name2}",
                "route_detail": ""
            }
            
            # 调用高德路线规划API（优先坐标，其次城市+名称）
            try:
                loc1 = a1.get("location") or ""
                loc2 = a2.get("location") or ""
                if loc1 and "," in str(loc1):
                    origin = str(loc1).strip()
                else:
                    origin = a1.get("address") or (f"{city}{name1}" if city else name1)
                if loc2 and "," in str(loc2):
                    destination = str(loc2).strip()
                else:
                    destination = a2.get("address") or (f"{city}{name2}" if city else name2)

                route_info = self.amap_service.route_direction(
                    origin=origin,
                    destination=destination,
                    mode=amap_mode,
                    city=city
                )
                
                if route_info:
                    distance, duration_min = self._parse_amap_distance_duration(route_info)
                    line_names = self._transit_line_names(route_info) if amap_mode == "transit" else []
                    segment.update(self._qualify_route_leg(
                        distance, duration_min, line_names, transportation, amap_mode
                    ))
                else:
                    segment.update(self._qualify_route_leg(0, 0, [], transportation, amap_mode))
                segment["description"] = (
                    f"从{name1}到{name2}，{segment['transport_mode']}，约{segment['duration']}分钟"
                    + ("" if "未取得" in segment.get("route_detail", "") else "（高德路径规划）")
                )
                    
            except Exception as e:
                if getattr(self, "logger", None):
                    self.logger.log_info("ROUTE", msg=f"[路线规划] {name1}→{name2} 失败: {e}")
                segment.update(self._qualify_route_leg(0, 0, [], transportation, amap_mode))
            
            segments.append(segment)
        
        return segments

    def _parse_amap_distance_duration(self, route_info) -> tuple:
        """从 AmapRouteInfo 解析距离(米)与时长(分钟)"""
        import re

        distance = 0
        duration_sec = 0

        if route_info.paths and isinstance(route_info.paths, list) and route_info.paths:
            path = route_info.paths[0]
            try:
                distance = int(float(path.get("distance") or 0))
            except (TypeError, ValueError):
                distance = 0
            try:
                duration_sec = int(float(path.get("duration") or 0))
            except (TypeError, ValueError):
                duration_sec = 0
        elif route_info.transits and isinstance(route_info.transits, list) and route_info.transits:
            transit = route_info.transits[0]
            try:
                distance = int(float(transit.get("distance") or 0))
            except (TypeError, ValueError):
                distance = 0
            try:
                duration_sec = int(float(transit.get("duration") or 0))
            except (TypeError, ValueError):
                duration_sec = 0

        if not distance and route_info.distance:
            try:
                raw_d = str(route_info.distance)
                distance = int(float(re.sub(r"[^\d.]", "", raw_d) or 0))
                if "公里" in raw_d or "km" in raw_d.lower():
                    distance = int(distance * 1000)
            except (TypeError, ValueError):
                pass

        if duration_sec:
            return distance, max(1, duration_sec // 60)

        raw = str(route_info.duration or "")
        hours = re.search(r"(\d+)\s*小时", raw)
        mins = re.search(r"(\d+)\s*分钟", raw)
        if hours or mins:
            total = (int(hours.group(1)) if hours else 0) * 60 + (int(mins.group(1)) if mins else 0)
            return distance, max(1, total)
        try:
            m = re.search(r"(\d+)", raw)
            val = int(m.group(1)) if m else 0
            return distance, max(1, val // 60) if val > 180 else max(1, val) if val else 0
        except Exception:
            return distance, 0