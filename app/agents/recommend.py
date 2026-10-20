"""景点推荐节点：高德搜索、ReAct 筛选、知识库与票价补全。"""

from typing import Any, Dict, List, Optional

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from app.services.structured_output import AttractionPickOutput, invoke_structured

class AmapSearchArgs(BaseModel):
    city: str = Field(..., description="城市名称")
    keywords: str = Field(default="景点", description="一个搜索关键词，例如博物馆、古镇、公园")


class AmapRestaurantArgs(BaseModel):
    city: str = Field(..., description="城市名称")
    keywords: str = Field(..., description="一个餐饮关键词，例如午餐、特色菜、火锅、老字号")
    meal: str = Field(..., description="lunch 表示午饭，dinner 表示晚饭")


class AmapPoolTool(BaseTool):
    """高德关键词搜索。每次调用的新景点累加到 pool，供推荐智能体换词再搜。"""

    name: str = "amap_attraction_search"
    description: str = (
        "用高德按一个关键词搜索景点，返回名称、类型和地址。"
        "数量不够或类型不符合用户偏好时，换一个关键词再次调用。"
    )
    args_schema: type = AmapSearchArgs
    search_poi: Any = None
    pool: List[Dict[str, Any]] = Field(default_factory=list)

    def _run(self, city: str, keywords: str = "景点") -> str:
        from app.services.attraction_recommend import search_keyword

        if self.search_poi is None:
            return "高德搜索不可用"
        batch = search_keyword(self.search_poi, city, keywords or "景点", limit=10)
        seen = {item.get("name") for item in self.pool}
        fresh = []
        for item in batch:
            if not item.get("name") or item["name"] in seen:
                continue
            self.pool.append(item)
            seen.add(item["name"])
            fresh.append(item)
        if not fresh:
            return "这次关键词没有新景点。请换一个更贴近用户偏好的关键词再搜。"
        lines = [
            f"- {item['name']} | {item.get('type') or '未知类型'} | {item.get('address') or '无地址'}"
            for item in fresh
        ]
        return f"新增 {len(fresh)} 个，当前共 {len(self.pool)} 个。\n" + "\n".join(lines)

    async def _arun(self, city: str, keywords: str = "景点") -> str:
        return self._run(city, keywords)


class AmapRestaurantTool(BaseTool):
    """高德餐饮搜索。午饭和晚饭分开累计，供推荐智能体换关键词再搜。"""

    name: str = "amap_restaurant_search"
    description: str = (
        "用高德按一个关键词搜索餐馆，返回名称、类型和地址。"
        "meal 只能是 lunch 或 dinner。"
        "数量不够或菜系不符合偏好时，换一个关键词再次调用。"
    )
    args_schema: type = AmapRestaurantArgs
    search_poi: Any = None
    pool: List[Dict[str, Any]] = Field(default_factory=list)

    def _run(self, city: str, keywords: str, meal: str) -> str:
        from app.services.attraction_recommend import search_keyword, tag_restaurant

        if self.search_poi is None:
            return "高德搜索不可用"
        slot = "dinner" if str(meal).strip().lower() == "dinner" else "lunch"
        label = "晚饭" if slot == "dinner" else "午饭"
        batch = search_keyword(self.search_poi, city, keywords or label, limit=8)
        seen = {item.get("name") for item in self.pool if item.get("meal") == slot}
        fresh = []
        for item in batch:
            if not item.get("name") or item["name"] in seen:
                continue
            tagged = tag_restaurant(item, slot)
            self.pool.append(tagged)
            seen.add(tagged["name"])
            fresh.append(tagged)
        if not fresh:
            return f"这次关键词没有新的{label}餐馆。请换一个更贴近餐饮偏好的关键词再搜。"
        lines = [
            f"- {item['name']} | {item.get('type') or '未知类型'} | {item.get('address') or '无地址'}"
            for item in fresh
        ]
        return f"{label}新增 {len(fresh)} 家，当前{label}共 {len(seen)} 家。\n" + "\n".join(lines)

    async def _arun(self, city: str, keywords: str, meal: str) -> str:
        return self._run(city, keywords, meal)


class RecommendMixin:
    """景点推荐。"""

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
            
            from app.services.attraction_recommend import gather_amap_pois, format_candidates

            amap_pois = gather_amap_pois(
                self.amap_service.search_poi,
                city=city,
                query=query,
                preferences=preferences,
                limit=24,
            )
            visited = set(state.get("visited_attractions") or [])
            if visited:
                amap_pois = [poi for poi in amap_pois if poi.get("name") not in visited]
            amap_results = format_candidates(amap_pois)
            
            if self.logger:
                self.logger.log_info("AMAP_SEARCH", msg=f"[景点节点] 高德搜索完成: {len(amap_pois)} 条结果")
            
            return {"amap_results": amap_results, "amap_pois": amap_pois}
        except Exception as e:
            if self.logger:
                self.logger.log_error("AMAP_SEARCH", msg=f"[景点节点] 高德搜索失败: {e}")
            return {"amap_results": "", "amap_pois": []}
    
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

    def recommend_attractions(
        self,
        state: Dict[str, Any],
        agent_runner: Optional[Any] = None,
        ticket_lookup: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """ReAct 按偏好搜景点和午饭晚饭，再补知识库介绍和飞猪门票。"""
        from app.services.attraction_recommend import (
            attraction_limits,
            dining_keywords,
            enrich_attractions,
            format_candidates,
            format_restaurants,
            gather_amap_pois,
            load_city_description,
            picks_from_agent_text,
            rank_by_preference,
            select_names,
            select_restaurants,
        )

        city = state.get("city", "")
        query = state.get("query", "")
        preferences = state.get("preferences") or []
        travel_days = int(state.get("travel_days") or 1)
        search = getattr(getattr(self, "amap_service", None), "search_poi", None)
        pool: List[Dict[str, Any]] = []
        restaurants: List[Dict[str, Any]] = []
        tool = AmapPoolTool(search_poi=search, pool=pool)
        restaurant_tool = AmapRestaurantTool(search_poi=search, pool=restaurants)
        low, high = attraction_limits(travel_days)
        if agent_runner is not None:
            try:
                agent_text = agent_runner(state, tool, restaurant_tool)
            except TypeError:
                agent_text = agent_runner(state, tool)
        else:
            agent_text = self._run_attraction_react(state, tool, restaurant_tool, low, high)
        if tool.pool:
            pool = list(tool.pool)
        if restaurant_tool.pool:
            restaurants = list(restaurant_tool.pool)
        if not pool and search is not None:
            pool = gather_amap_pois(
                search,
                city=city,
                query=query,
                preferences=preferences,
                limit=high,
            )
        if search is not None:
            lunch_words, dinner_words = dining_keywords(
                preferences, state.get("free_text_input") or query
            )
            self._fill_restaurant_pool(
                search, restaurants, city, "lunch", lunch_words[:1]
            )
            self._fill_restaurant_pool(
                search, restaurants, city, "dinner", dinner_words[:1]
            )
        visited = set(state.get("visited_attractions") or [])
        if visited:
            pool = [poi for poi in pool if poi.get("name") not in visited]
        ranked = rank_by_preference(pool, preferences, query)
        picks = picks_from_agent_text(agent_text or "")
        selected = select_names(ranked, picks["names"], travel_days)
        enriched = enrich_attractions(
            selected,
            city=city,
            depart_date=state.get("start_date") or "",
            ticket_lookup=ticket_lookup,
        )
        chosen_lunches = select_restaurants(restaurants, picks["lunch"], "lunch", travel_days)
        used_lunch = {item.get("name") for item in chosen_lunches}
        dinner_pool = [
            item for item in restaurants
            if item.get("meal") != "dinner" or item.get("name") not in used_lunch
        ]
        if not any(item.get("meal") == "dinner" for item in dinner_pool):
            dinner_pool = restaurants
        chosen_restaurants = chosen_lunches + select_restaurants(
            dinner_pool, picks["dinner"], "dinner", travel_days
        )
        text = format_candidates(enriched)
        restaurant_text = format_restaurants(chosen_restaurants)
        if self.logger:
            self.logger.log_info(
                "ATTRACTION_AGENT",
                msg=(
                    f"[景点推荐] 搜到{len(pool)}个景点，选出{len(enriched)}个；"
                    f"餐馆{len(restaurants)}家，选出{len(chosen_restaurants)}家"
                ),
            )
        return {
            "amap_pois": pool,
            "attraction_candidates": enriched,
            "restaurant_candidates": chosen_restaurants,
            "city_description": load_city_description(city),
            "merged_results": text,
            "attraction_response": text,
            "restaurant_response": restaurant_text,
        }

    def _fill_restaurant_pool(
        self,
        search,
        pool: List[Dict[str, Any]],
        city: str,
        meal: str,
        keywords: List[str],
    ) -> None:
        from app.services.attraction_recommend import search_keyword, tag_restaurant

        if any(item.get("meal") == meal for item in pool):
            return
        for word in keywords:
            batch = search_keyword(search, city, word, limit=8)
            for item in batch:
                if item.get("name"):
                    pool.append(tag_restaurant(item, meal))
            if any(item.get("meal") == meal for item in pool):
                return

    def _run_attraction_react(
        self,
        state: Dict[str, Any],
        tool: AmapPoolTool,
        restaurant_tool: AmapRestaurantTool,
        low: int,
        high: int,
    ) -> str:
        from app.agents.react_agent import ReActAgent
        from app.services.attraction_recommend import dining_keywords

        preferences = state.get("preferences") or []
        free_text = state.get("free_text_input") or ""
        lunch_words, dinner_words = dining_keywords(preferences, free_text)
        try:
            base = self._get_agent_prompt("attraction")
        except Exception:
            base = ""
        instruction = (
            f"景点候选要给够规划排满全天，一共选出 {low} 到 {high} 个，不要在每天 3 个时停止搜索。"
            "先调用 amap_attraction_search。数量不够或类型不符合偏好时，换一个关键词再搜。"
            "然后各搜一次午饭和一次晚饭，调用 amap_restaurant_search。"
            f"午饭可从这些词里选一个：{'、'.join(lunch_words)}。"
            f"晚饭可从这些词里选一个：{'、'.join(dinner_words)}。"
            "用户补充里如果写了具体菜系，午饭和晚饭都可以用那个词。"
            "不要安排早饭。names、lunch、dinner 只能使用对应工具返回过的原名。"
        )
        agent = ReActAgent(
            name="景点推荐",
            tools=[tool, restaurant_tool],
            llm=self.llm,
            system_prompt=f"{base}\n\n{instruction}".strip(),
            max_iterations=6,
            verbose=False,
            enable_monitoring=bool(getattr(self, "enable_monitoring", False)),
            response_format=AttractionPickOutput,
        )
        question = (
            f"城市：{state.get('city', '')}\n"
            f"偏好：{', '.join(preferences) if preferences else '无'}\n"
            f"餐饮偏好：{free_text or '、'.join(preferences) or '无'}\n"
            f"补充：{free_text or '无'}\n"
            f"天数：{state.get('travel_days') or 1}\n"
            "请搜索并选出景点和午饭晚饭。"
        )
        return agent.run(question)

    def select_and_enrich(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """按偏好从高德名单筛选，数量不够则补足，再补介绍和飞猪门票。"""
        from app.services.attraction_recommend import (
            enrich_attractions,
            format_candidates,
            load_city_description,
            rank_by_preference,
            select_names,
        )

        city = state.get("city", "")
        query = state.get("query", "")
        preferences = state.get("preferences") or []
        travel_days = int(state.get("travel_days") or 1)
        pois = list(state.get("amap_pois") or [])
        ranked = rank_by_preference(pois, preferences, query)
        chosen = self._pick_attraction_names(ranked, state)
        selected = select_names(ranked, chosen, travel_days)
        enriched = enrich_attractions(
            selected,
            city=city,
            depart_date=state.get("start_date") or "",
        )
        text = format_candidates(enriched)
        if self.logger:
            self.logger.log_info(
                "ATTRACTION_SELECT",
                msg=f"[景点节点] 高德{len(pois)}个，筛选后{len(enriched)}个",
            )
        return {
            "attraction_candidates": enriched,
            "city_description": load_city_description(city),
            "merged_results": text,
            "attraction_response": text,
        }

    def _pick_attraction_names(self, pois: List[Dict[str, Any]], state: Dict[str, Any]) -> List[str]:
        if not pois:
            return []
        from app.services.attraction_recommend import minimum_attraction_count

        minimum = minimum_attraction_count(int(state.get("travel_days") or 1), len(pois))
        names = [poi.get("name") or "" for poi in pois]
        preferences = state.get("preferences") or []
        prompt = (
            "从下面的高德景点原名中，按用户偏好挑选适合的行程候选。"
            f"至少选出 {minimum} 个，可以更多。只能使用名单里的原名，不要新造景点。\n"
            f"城市：{state.get('city', '')}\n"
            f"偏好：{', '.join(preferences) if preferences else '无'}\n"
            f"补充：{state.get('free_text_input') or '无'}\n"
            "名单：\n" + "\n".join(f"- {name}" for name in names)
        )
        try:
            parsed = invoke_structured(
                self.llm,
                AttractionPickOutput,
                [
                    {"role": "system", "content": "你只负责从给定名单点名，不写行程。"},
                    {"role": "user", "content": prompt},
                ],
            )
            picked = parsed.names if isinstance(parsed, AttractionPickOutput) else []
            if not picked and isinstance(parsed, dict):
                picked = parsed.get("names") or []
            return [str(name).strip() for name in picked if str(name).strip()]
        except Exception as exc:
            if self.logger:
                self.logger.log_error("ATTRACTION_SELECT", f"[景点节点] 筛选失败，改用偏好排序: {exc}")
            return []

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
    
