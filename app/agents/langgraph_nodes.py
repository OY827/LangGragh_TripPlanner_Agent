"""
LangGraph 节点函数集合

每个节点是一个接受 State、返回 State 更新的函数。
集成 PromptManager 实现渐进披露式提示词（基于 md 模板）。
分层组织：
- 景点子图节点
- 天气子图节点
- 酒店子图节点
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


class GraphNodes:
    """
    LangGraph 节点函数集合
    
    所有节点函数都是无状态的（stateless），通过 State 传递数据。
    集成 PromptManager 加载渐进披露式 md 模板。
    """
    
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
        
        context_lines.append(f"\n## 景点信息")
        context_lines.append(attraction_response if attraction_response else "暂无景点信息")
        
        context_lines.append(f"\n## 天气信息（weather_skill）")
        context_lines.append(weather_response if weather_response else "暂无天气信息")
        
        context_lines.append(f"\n## 酒店信息")
        context_lines.append(hotel_response if hotel_response else "暂无酒店信息")
        
        if free_text:
            context_lines.append(f"\n## 补充要求")
            context_lines.append(free_text)
        
        context_lines.append(f"\n请根据以上信息，严格按照系统提示中的 JSON 格式要求生成完整的 {city} {travel_days} 天旅行计划。")
        context_lines.append("【强制约束】行程中的景点名称只能从上方「景点信息」候选中选择，禁止编造未列出的景点；请复制候选中的 visit_duration 与坐标。")
        
        user_message = "\n".join(context_lines)
        
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
    
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
    
    # ==================== 景点子图节点 ====================
    
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
        
        return {"merged_results": formatted}
    
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
        """搜索酒店"""
        try:
            city = state.get("city", "")
            accommodation = state.get("accommodation", "经济型")
            
            results = self.amap_service.search_poi(
                keywords=f"{accommodation}酒店",
                city=city
            )
            
            hotel_text = self._format_hotel_results(results)
            return {"hotel_response": hotel_text}
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
            memory_info = f"用户之前去过{city}（{comparison.get('plan_count', 0)}次）\n"
            memory_info += f"已访问景点: {', '.join(visited)}\n"
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
            response = self.llm.invoke(messages)
            plan_text = response.content if hasattr(response, 'content') else str(response)
            
            parsed = self._parse_trip_plan(plan_text, state.get("city", ""), state.get("transportation", "地铁"))
            
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
                if len(attractions) < 3:
                    issues.append(f"第{day.get('day_index', 0)}天景点偏少（{len(attractions)}个，建议≥3）")
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
            version_manager = VersionManager(use_redis=False)

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

    def dialog(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        对话节点：处理用户修改请求

        集成DialogAgent，支持多轮对话修改旅行计划。
        流程：
        1. 检查是否有用户输入
        2. 调用DialogAgent处理修改
        3. 返回修改结果和新版本
        """
        from app.agents.dialog_agent import DialogAgent

        session_id = state.get("session_id", "")
        dialog_input = state.get("dialog_input", "")

        if not session_id:
            return {
                "dialog_reply": "错误：会话未初始化，请先规划旅行",
                "dialog_active": False,
                "current_phase": "final"
            }

        # 如果没有用户输入，生成初始对话提示
        if not dialog_input:
            city = state.get("city", "")
            budget = state.get("budget", {})
            budget_total = budget.get("total", 0) if isinstance(budget, dict) else 0

            reply = f"""✅ 旅行计划已生成！

📍 城市：{city}
💰 预算总计：{budget_total}元

您可以输入修改指令，例如：
• "添加迪士尼"
• "降低酒店预算"
• "延长到5天"
• "对比一下变化"
• "回退到之前的版本"

或输入"结束"退出对话。"""

            return {
                "dialog_reply": reply,
                "dialog_active": True,
                "current_phase": "dialog"
            }

        # 检查是否结束对话
        if dialog_input.strip() in ["结束", "退出", "完成", "不需要修改", "好的"]:
            return {
                "dialog_reply": "好的，旅行计划已确定。祝您旅途愉快！",
                "dialog_active": False,
                "current_phase": "final"
            }

        try:
            # 初始化对话Agent
            dialog_agent = DialogAgent(
                llm_provider="aliyun",
                use_redis=False,
                enable_rag=True,
                enable_amap=True,
                enable_memory=False
            )

            # 处理修改请求
            result = dialog_agent.process_modification(
                session_id=session_id,
                user_input=dialog_input
            )

            # 提取修改后的计划
            new_plan = None
            if result.get("new_version_id"):
                from app.agents.version_manager import VersionManager
                version_manager = VersionManager(use_redis=False)
                new_version = version_manager.get_version(result["new_version_id"])
                if new_version:
                    new_plan = new_version.plan_data

            if self.logger:
                self.logger.log_info("DIALOG", msg=(
                    f"[对话节点] 用户输入: {dialog_input}, "
                    f"新版本: {result.get('new_version_id', '无')}"
                ))

            # 如果有新计划，更新state
            updates = {
                "dialog_reply": result.get("reply", ""),
                "dialog_changes": result.get("changes", []),
                "dialog_active": True,
                "current_phase": "dialog"
            }

            if new_plan:
                updates["current_version_id"] = result["new_version_id"]
                # 更新计划相关字段
                if "budget" in new_plan:
                    updates["budget"] = new_plan["budget"]
                    updates["budget_total"] = new_plan["budget"].get("total", 0)

            return updates

        except Exception as e:
            if self.logger:
                self.logger.log_error("DIALOG", msg=f"[对话节点] 处理失败: {e}")

            return {
                "dialog_reply": f"抱歉，处理您的请求时出现错误：{str(e)}",
                "dialog_active": True,
                "current_phase": "dialog"
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
        """主图反思后的条件路由"""
        phase = state.get("current_phase", "final")
        if phase == "reflect":
            iteration = state.get("iteration_count", 0)
            if iteration <= 1:
                return "build_attraction_query"
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
            for i, day in enumerate(days):
                if isinstance(day, dict):
                    day["day_index"] = i
                    # 始终用高德真实路线回填（覆盖 LLM 空/估算的 route_segments）
                    if "attractions" in day and len(day.get("attractions", [])) > 1:
                        day["route_segments"] = self._generate_route_segments_from_attractions(
                            day.get("attractions", []), city, transportation
                        )
                        day["time_slots"] = self._generate_time_slots_from_attractions(
                            day.get("attractions", []), day["route_segments"]
                        )
                    elif "time_slots" not in day and "attractions" in day:
                        day["time_slots"] = self._generate_time_slots_from_attractions(
                            day.get("attractions", []), []
                        )
                    if "meals" not in day:
                        day["meals"] = [
                            {"type": "breakfast", "name": "酒店早餐", "description": "酒店或附近早餐", "estimated_cost": 30},
                            {"type": "lunch", "name": "当地特色午餐", "description": "附近特色餐厅", "estimated_cost": 60},
                            {"type": "dinner", "name": "当地特色晚餐", "description": "附近特色餐厅", "estimated_cost": 80}
                        ]
                    if "description" not in day:
                        attractions_names = [a.get("name", "") for a in day.get("attractions", []) if isinstance(a, dict)]
                        day["description"] = f"第{day.get('day_index', i+1)}天行程：{', '.join(attractions_names)}"
        
        if isinstance(days, list):
            parsed["days"] = days
        
        return parsed
    
    def _generate_time_slots_from_attractions(self, attractions: List[Dict], route_segments: List[Dict] = None) -> List[Dict]:
        """根据景点列表生成时间段安排（使用真实交通耗时）"""
        time_slots = []
        current_time = 9
        route_segs = route_segments or []
        
        for i, attr in enumerate(attractions):
            if not isinstance(attr, dict):
                continue
            name = attr.get("name", "景点")
            duration = attr.get("visit_duration", 120)
            duration_hours = max(1, duration // 60)
            
            start_h = current_time
            end_h = min(22, current_time + duration_hours)
            time_slots.append({
                "start_time": f"{start_h:02d}:00",
                "end_time": f"{end_h:02d}:00",
                "activity_type": "景点游览",
                "activity_name": name,
                "description": attr.get("description", f"游览{name}")
            })
            current_time = end_h
            
            # 添加交通时间段
            if i < len(attractions) - 1:
                # 从路线段获取真实交通耗时
                travel_duration = 20  # 默认20分钟
                travel_desc = "前往下一个景点"
                transport_mode = "交通"
                
                if i < len(route_segs):
                    seg = route_segs[i]
                    travel_duration = seg.get("duration", 20)
                    travel_desc = seg.get("route_detail", "前往下一个景点")
                    transport_mode = seg.get("transport_mode", "交通")
                
                travel_hours = max(1, (travel_duration + 30) // 60)  # 分钟转小时，加30分钟缓冲
                travel_end = min(22, current_time + travel_hours)
                
                time_slots.append({
                    "start_time": f"{current_time:02d}:00",
                    "end_time": f"{travel_end:02d}:00",
                    "activity_type": "交通",
                    "activity_name": travel_desc,
                    "description": f"{transport_mode} · 约{travel_duration}分钟"
                })
                current_time = travel_end
        
        return time_slots
    
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
                    segment["distance"] = distance
                    segment["duration"] = duration_min or 20
                    
                    # 提取路线详情
                    if route_info.transits and amap_mode == "transit":
                        transit = route_info.transits[0] if route_info.transits else {}
                        segments_info = transit.get("segments", [])
                        if segments_info:
                            bus_info = segments_info[0].get("bus", {})
                            bus_name = bus_info.get("buslines", [{}])[0].get("name", "") if bus_info.get("buslines") else ""
                            if bus_name:
                                segment["route_detail"] = f"乘坐{bus_name}"
                                segment["transport_mode"] = bus_name
                            else:
                                segment["route_detail"] = f"公共交通约{segment['duration']}分钟"
                        else:
                            segment["route_detail"] = f"公共交通约{segment['duration']}分钟"
                    elif route_info.paths and amap_mode == "driving":
                        segment["route_detail"] = f"驾车约{segment['duration']}分钟，{segment['distance']/1000:.1f}公里"
                    elif amap_mode == "walking":
                        segment["route_detail"] = f"步行约{segment['duration']}分钟，{segment['distance']}米"
                        segment["transport_mode"] = "步行"
                    else:
                        segment["route_detail"] = f"{transportation}约{segment['duration']}分钟"
                    segment["description"] = (
                        f"从{name1}到{name2}，{segment['transport_mode']}，"
                        f"约{segment['duration']}分钟（高德路径规划）"
                    )
                else:
                    segment["route_detail"] = f"{transportation}约20分钟"
                    segment["duration"] = 20
                    
            except Exception as e:
                if self.logger:
                    self.logger.log_info("ROUTE", msg=f"[路线规划] {name1}→{name2} 失败: {e}")
                segment["route_detail"] = f"{transportation}约20分钟"
                segment["duration"] = 20
            
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
            return distance, max(1, val // 60) if val > 180 else max(1, val) if val else 20
        except Exception:
            return distance, 20