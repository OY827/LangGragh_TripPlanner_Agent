"""规划节点：排好每一天后调用路线工具。"""

import json
from typing import Any, Dict, List, Optional

from app.agents.route import RouteDistanceTool
from app.services.hotel_mcp_service import HotelMCPService
from app.services.structured_output import (
    PlanReflectOutput,
    TripPlanOutput,
    as_dict,
    invoke_structured,
)

class PlannerMixin:
    """行程规划与反思。"""

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
        
        attraction_candidates = state.get("merged_results") or attraction_response
        context_lines.append("\n## 景点候选")
        context_lines.append(attraction_candidates if attraction_candidates else "暂无景点信息")
        context_lines.append(
            "景点只写名称，按游览顺序排列。看建议游览分钟估算，不要把分钟数写进输出。"
            "90 分钟左右的景点一天写 4 到 6 个，午饭晚饭各 60 分钟，景点之间按大约 30 分钟交通估算，把当天排到 20:00。"
            "某个景点太远，或加上它会超过 21:00，就跳过并改写下一个候选，不要只留下 3 个就结束。"
            "午饭和晚饭各写一家餐馆名称，不要安排早饭。不要写 time_slots 和 route_segments。"
        )

        city_description = state.get("city_description") or ""
        context_lines.append("\n## 城市描述")
        if city_description:
            context_lines.append(city_description)
        else:
            context_lines.append("知识库没有该城市的攻略概述。")
        context_lines.append(
            "introduction 用 180 到 320 字介绍这一趟旅程，不要编造候选里没有的景点。"
            "overall_suggestions 不超过 120 字。其余说明留空。"
        )

        guide_info = self._get_guide_reference(city, travel_days, preferences)
        if guide_info:
            context_lines.append(f"\n## 参考攻略")
            context_lines.append(guide_info)
        
        context_lines.append(f"\n## 天气（只用于室内外安排，不新增景点）")
        context_lines.append(weather_response if weather_response else "暂无天气信息")
        
        hotel_budget = state.get("hotel_budget") or ""
        budget_label = HotelMCPService.budget_label(hotel_budget)
        context_lines.append("\n## 酒店候选（已按价格从低到高）")
        context_lines.append(
            "从酒店候选中选择一家，hotel 里只写名称。"
            "地址和价格由系统补上，不要把全部酒店写进每一天。"
        )
        if budget_label:
            context_lines.append(f"用户每晚预算是 {budget_label}。候选已按这个区间筛过。")
        context_lines.append(hotel_response if hotel_response else "暂无酒店信息")

        context_lines.append("\n## 餐饮候选")
        context_lines.append(
            "每天午饭和晚饭各选一家，只写名称。不要安排早饭，不要编造候选里没有的餐馆。"
        )
        restaurant_text = state.get("restaurant_response") or ""
        context_lines.append(restaurant_text if restaurant_text else "暂无餐饮信息")
        
        if free_text:
            context_lines.append(f"\n## 补充要求")
            context_lines.append(free_text)

        feedback = state.get("reflection_feedback") or ""
        if feedback and state.get("iteration_count", 0) > 0:
            context_lines.append("\n## 上一轮已排好的时间表")
            context_lines.append(
                "下面是系统按真实路程排好的名称、顺序和时间。以此判断是否达标，不要再按 30 分钟估算。"
                "收尾早于 20:00，或午饭、晚饭没有出现的那天，保留表里已经合适的景点和餐馆，只补缺口。"
                "已经达到 20:00 且午饭晚饭都在的天数、酒店名称保持不变。"
            )
            context_lines.append(feedback)
        
        context_lines.append(f"\n请生成 {city} {travel_days} 天计划。只做选择和排序，不要展开介绍。")
        context_lines.append("【强制约束】行程中的景点名称只能从上方「景点候选」中选择，酒店只能从「酒店候选」中选择，午饭和晚饭只能从「餐饮候选」中选择。按建议游览时长把每天排到 20:00，超过 20:00 可以，短景点一天 4 到 6 个。")
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

    def _calculate_routes_for_plan(self, parsed: Dict[str, Any], state: Dict[str, Any]) -> None:
        """日程确定后，按每一天的实际顺序调用路线工具。"""
        city = state.get("city", "") or parsed.get("city", "")
        transportation = state.get("transportation") or "公共交通"
        catalog = {}
        for item in state.get("attraction_candidates") or []:
            name = str(item.get("name") or "").strip()
            if name:
                catalog[name] = item
        tool = RouteDistanceTool(nodes=self, catalog=catalog)
        for day in parsed.get("days") or []:
            if not isinstance(day, dict):
                continue
            stops = self._ordered_stops(day, city, catalog)
            if len(stops) < 2:
                continue
            summary = tool._run(city, transportation, stops)
            day["route_segments"] = list(tool.last_segments)
            if summary:
                day["route_summary"] = summary

    def _ordered_stops(self, day: Dict[str, Any], city: str, catalog: Dict[str, Dict[str, Any]]) -> List[str]:
        """按时间表顺序走：酒店、已排上的景点、有坐标的餐馆，再回酒店。没排上的景点不进路线。"""
        names: List[str] = []

        def add(name: str, record: Optional[Dict[str, Any]] = None) -> None:
            name = str(name or "").strip()
            if not name or (names and names[-1] == name):
                return
            if isinstance(record, dict) and name not in catalog:
                catalog[name] = record
            names.append(name)

        anchor = self._resolve_hotel_anchor(day, city)
        if anchor and anchor.get("name"):
            add(anchor["name"], anchor)
        slots = day.get("time_slots") or []
        meals = {
            str(item.get("name") or ""): item
            for item in day.get("meals") or []
            if isinstance(item, dict) and item.get("name")
        }
        attractions = {
            str(item.get("name") or ""): item
            for item in day.get("attractions") or []
            if isinstance(item, dict) and item.get("name")
        }
        if slots:
            for slot in slots:
                kind = slot.get("activity_type")
                name = str(slot.get("activity_name") or "").strip()
                if kind == "景点游览":
                    add(name, attractions.get(name) or catalog.get(name))
                elif kind == "用餐":
                    meal = meals.get(name)
                    location = str((meal or {}).get("location") or "")
                    if meal and "," in location:
                        add(name, meal)
        else:
            for attr in day.get("attractions") or []:
                if isinstance(attr, dict):
                    add(str(attr.get("name") or ""), attr)
                else:
                    add(str(attr or ""))
        if anchor and anchor.get("name"):
            add(anchor["name"], anchor)
        return names

    def generate_trip_plan(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """生成旅行计划（使用 md 模板渐进披露式提示词）"""
        try:
            messages = self._build_planner_messages(state)
            city = state.get("city", "")
            transportation = state.get("transportation", "地铁")
            llm = self.llm.bind(max_tokens=2048) if hasattr(self.llm, "bind") else self.llm
            parsed_model = invoke_structured(llm, TripPlanOutput, messages)
            plan_data = as_dict(parsed_model)
            if isinstance(plan_data, dict):
                self._apply_candidate_facts(plan_data, state.get("attraction_candidates") or [])
            if isinstance(parsed_model, TripPlanOutput):
                parsed = self._normalize_trip_plan(
                    plan_data or {}, city, transportation, fill=False
                )
                plan_text = json.dumps(plan_data, ensure_ascii=False)
            elif plan_data:
                parsed = self._normalize_trip_plan(plan_data, city, transportation, fill=False)
                plan_text = json.dumps(plan_data, ensure_ascii=False)
            else:
                parsed = None
                plan_text = ""
            if isinstance(parsed, dict):
                from app.services.attraction_recommend import price_checks_for_plan

                self._attach_hotel_options(parsed, state.get("hotel_candidates") or [])
                self._attach_meals(parsed, state.get("restaurant_candidates") or [])
                pool = state.get("attraction_candidates") or []
                used_names = set()
                for day in parsed.get("days") or []:
                    if isinstance(day, dict):
                        self._fill_day_until_close(
                            day, city, transportation, used_names, pool
                        )
                self._calculate_routes_for_plan(parsed, state)
                parsed["price_checks"] = price_checks_for_plan(
                    parsed,
                    state.get("attraction_candidates") or [],
                )
            
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

    def _apply_candidate_facts(self, parsed: Dict[str, Any], candidates: List[Dict[str, Any]]) -> None:
        """模型只点名。简介、坐标、时长和票价从候选抄回来，避免再生成一遍。"""
        catalog = [item for item in candidates if isinstance(item, dict) and item.get("name")]
        for day in parsed.get("days") or []:
            if not isinstance(day, dict):
                continue
            hydrated = []
            for item in day.get("attractions") or []:
                name = item.get("name") if isinstance(item, dict) else str(item or "")
                name = str(name or "").strip()
                if not name:
                    continue
                source = PlannerMixin._candidate_named(catalog, name)
                if source is None:
                    hydrated.append(item if isinstance(item, dict) else {"name": name})
                    continue
                hydrated.append({
                    "name": source.get("name") or name,
                    "address": source.get("address") or "",
                    "location": source.get("location") or "",
                    "visit_duration": source.get("visit_duration") or 120,
                    "description": source.get("description") or "",
                    "category": source.get("category") or source.get("type") or "",
                    "ticket_price": source.get("ticket_price") or 0,
                    "open_time": source.get("open_time") or "",
                })
            day["attractions"] = hydrated

    @staticmethod
    def _candidate_named(candidates: List[Dict[str, Any]], name: str) -> Optional[Dict[str, Any]]:
        exact = next((item for item in candidates if item.get("name") == name), None)
        if exact is not None:
            return exact
        for item in candidates:
            title = str(item.get("name") or "")
            if len(name) >= 2 and (name in title or title in name):
                return item
        return None

    def _attach_hotel_options(self, parsed: Dict[str, Any], candidates: List[Dict[str, Any]]) -> None:
        """选中的酒店补上报价，其余候选留给用户自己选。"""
        options = [item for item in candidates if isinstance(item, dict) and item.get("name")]
        hotel = parsed.get("hotel") if isinstance(parsed.get("hotel"), dict) else {}
        chosen = str(hotel.get("name") or "").strip()
        if not chosen:
            for day in parsed.get("days") or []:
                stay = day.get("hotel") if isinstance(day, dict) else None
                if isinstance(stay, dict) and str(stay.get("name") or "").strip():
                    chosen = str(stay.get("name")).strip()
                    hotel = stay
                    break
        matched = next(
            (item for item in options if HotelMCPService.same_hotel(chosen, item.get("name") or "")),
            None,
        )
        if matched and isinstance(hotel, dict) and hotel.get("name"):
            for key in ("address", "price_range", "rating", "distance", "type", "estimated_cost"):
                if not hotel.get(key) and matched.get(key):
                    hotel[key] = matched[key]
            parsed["hotel"] = hotel
            for day in parsed.get("days") or []:
                if isinstance(day, dict):
                    day["hotel"] = dict(hotel)
        parsed["hotel_options"] = HotelMCPService.remaining_options(chosen, options)

    _PLACEHOLDER_MEALS = {
        "", "早餐", "晚饭", "晚餐", "午饭", "午餐",
        "酒店早餐", "当地特色午餐", "当地特色晚餐", "附近早餐", "附近晚餐",
    }

    def _attach_meals(self, parsed: Dict[str, Any], candidates: List[Dict[str, Any]]) -> None:
        """每天写入一家午饭和一家晚饭，地址和坐标从高德候选补上。"""
        lunches = [item for item in candidates if isinstance(item, dict) and item.get("meal") == "lunch" and item.get("name")]
        dinners = [item for item in candidates if isinstance(item, dict) and item.get("meal") == "dinner" and item.get("name")]
        for index, day in enumerate(parsed.get("days") or []):
            if not isinstance(day, dict):
                continue
            current = {}
            for meal in day.get("meals") or []:
                if isinstance(meal, dict) and meal.get("type") in {"lunch", "dinner"}:
                    current[meal["type"]] = meal
            day["meals"] = [
                self._resolve_meal(current.get("lunch"), lunches, index, "lunch"),
                self._resolve_meal(current.get("dinner"), dinners, index, "dinner"),
            ]
            lunch_name = day["meals"][0].get("name")
            dinner_name = day["meals"][1].get("name")
            if lunch_name and lunch_name == dinner_name:
                alternative = next((item for item in dinners if item.get("name") != lunch_name), None)
                if alternative:
                    day["meals"][1] = self._resolve_meal(
                        {"name": alternative.get("name"), "type": "dinner"},
                        dinners,
                        index,
                        "dinner",
                    )

    def _resolve_meal(
        self,
        existing: Optional[Dict[str, Any]],
        options: List[Dict[str, Any]],
        index: int,
        meal_type: str,
    ) -> Dict[str, Any]:
        name = str((existing or {}).get("name") or "").strip()
        matched = next((item for item in options if item.get("name") == name), None)
        if matched is None and name in self._PLACEHOLDER_MEALS and options:
            matched = options[index % len(options)]
        if matched is None and not name and options:
            matched = options[index % len(options)]
        if matched is not None:
            return {
                "type": meal_type,
                "name": matched.get("name") or name,
                "address": matched.get("address") or "",
                "location": matched.get("location") or "",
                "description": matched.get("type") or matched.get("description") or "",
                "estimated_cost": matched.get("estimated_cost") or 0,
                "duration_minutes": 60,
            }
        label = "午餐" if meal_type == "lunch" else "晚餐"
        return {
            "type": meal_type,
            "name": name or label,
            "address": str((existing or {}).get("address") or ""),
            "location": (existing or {}).get("location") or "",
            "description": str((existing or {}).get("description") or f"附近{label}"),
            "estimated_cost": (existing or {}).get("estimated_cost") or 0,
            "duration_minutes": 60,
        }

    def _schedule_brief(self, parsed: Dict[str, Any]) -> str:
        """把本地排好的时间表收成名称、顺序和时间，供模型反思。"""
        if not isinstance(parsed, dict):
            return ""
        lines = ["上一轮时间表:"]
        for day in parsed.get("days") or []:
            if not isinstance(day, dict):
                continue
            slots = [slot for slot in day.get("time_slots") or [] if isinstance(slot, dict)]
            end = str(slots[-1].get("end_time") or "无") if slots else "无"
            index = int(day.get("day_index") or 0) + 1
            lines.append(f"第{index}天 收尾 {end}")
            order = 0
            for slot in slots:
                kind = slot.get("activity_type")
                start = slot.get("start_time") or ""
                stop = slot.get("end_time") or ""
                if kind == "交通":
                    lines.append(f"  路程 {start}-{stop}")
                    continue
                if kind not in {"景点游览", "用餐"}:
                    continue
                order += 1
                label = "景点" if kind == "景点游览" else "用餐"
                lines.append(f"  {order}. {start}-{stop} {label} {slot.get('activity_name') or ''}")
            if order == 0:
                names = []
                for item in day.get("attractions") or []:
                    name = item.get("name") if isinstance(item, dict) else str(item or "")
                    if name:
                        names.append(str(name))
                meals = []
                for meal in day.get("meals") or []:
                    if isinstance(meal, dict) and meal.get("name"):
                        meals.append(f"{meal.get('type') or ''} {meal.get('name')}")
                if names:
                    lines.append("  景点 " + "、".join(names))
                if meals:
                    lines.append("  餐饮 " + "、".join(meals))
        return "\n".join(lines) if len(lines) > 1 else ""

    def _ask_plan_reflection(self, brief: str) -> Optional[PlanReflectOutput]:
        """把时间表交给模型，判断是否收到 20:00 且午饭晚饭都在。"""
        llm = getattr(self, "llm", None)
        if llm is None or not brief:
            return None
        try:
            bound = llm.bind(max_tokens=256) if hasattr(llm, "bind") else llm
            result = invoke_structured(
                bound,
                PlanReflectOutput,
                [
                    {
                        "role": "system",
                        "content": "你只根据给定时间表判断是否达标。不要编造表里没有的地点，不要重写整天行程。",
                    },
                    {
                        "role": "user",
                        "content": (
                            "达标标准：每一天最后一段时间达到或超过 20:00；每一天都有午饭和晚饭；"
                            "景点游览在 21:00 前结束。\n"
                            "passed 表示是否达标。未达标时 note 不超过 80 字，说明哪一天缺什么。"
                            "达标时 note 留空。\n\n"
                            + brief
                        ),
                    },
                ],
            )
        except Exception:
            return None
        if isinstance(result, PlanReflectOutput):
            return result
        if isinstance(result, dict):
            try:
                return PlanReflectOutput.model_validate(result)
            except Exception:
                return None
        return None

    def reflect_trip_plan(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """反思旅行计划。模型看到的是本地排好的名称、顺序和时间。"""
        parsed = state.get("trip_plan_parsed")
        attraction_response = state.get("attraction_response", "")
        travel_days = state.get("travel_days", 1)
        brief = self._schedule_brief(parsed) if isinstance(parsed, dict) else ""
        review = self._ask_plan_reflection(brief)
        model_note = ""
        if review is not None and not review.passed:
            model_note = review.note or "当前时间表未达标"
        
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
                if self._day_end_hour(day) < 20:
                    issues.append(
                        f"第{day.get('day_index', 0)}天收尾早于20:00，目前{len(attractions)}个景点。"
                        "从候选里继续加；太远或会超过21:00的跳过，换下一个。"
                    )
                    should_retry = True
                    break
        
        if len(attraction_response) < 100:
            issues.append("景点信息不足，重新生成计划无法补齐候选")
        if model_note:
            issues.append(model_note)
            should_retry = True

        detail = brief
        if should_retry:
            iteration = state.get("iteration_count", 0)
            max_iterations = state.get("max_iterations", 3)
            
            if iteration < max_iterations:
                feedback = f"反思发现问题: {'; '.join(issues)}，将重试（{iteration + 1}/{max_iterations}）"
                if detail:
                    feedback = f"{feedback}\n{detail}"
                return {
                    "reflection_feedback": feedback,
                    "iteration_count": iteration + 1,
                    "current_phase": "reflect"
                }
            else:
                feedback = f"达到最大重试次数，返回当前结果。问题: {'; '.join(issues)}"
                if detail:
                    feedback = f"{feedback}\n{detail}"
                return {
                    "reflection_feedback": feedback,
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
    
    def _normalize_trip_plan(
        self,
        parsed: Dict[str, Any],
        city: str = "",
        transportation: str = "地铁",
        fill: bool = True,
    ) -> Dict[str, Any]:
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
                    if fill:
                        self._fill_day_until_close(day, city, transportation, used_names)
                    if "meals" not in day or not day.get("meals"):
                        day["meals"] = [
                            {"type": "lunch", "name": "午餐", "description": "附近午餐", "estimated_cost": 60, "duration_minutes": 60},
                            {"type": "dinner", "name": "晚餐", "description": "附近晚餐", "estimated_cost": 80, "duration_minutes": 60},
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
    
