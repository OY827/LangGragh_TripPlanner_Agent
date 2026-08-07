"""
LangChain 版本的多智能体旅行规划系统
"""
import json
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

from .simple_agent import SimpleAgent
from .react_agent import ReActAgent
from app.models import (
    TripRequest, TripPlan, DayPlan, Attraction, Meal, 
    WeatherInfo, Location, Hotel, Budget, RouteSegment, TimeSlot
)
from app.llm import get_llm
from app.tools import get_basic_tools, get_all_tools, get_attraction_tools, get_weather_tools, get_hotel_tools, MCPToolManager, settings
from app.prompts import get_prompt_manager

try:
    from app.memory.memory_manager import TravelMemoryManager
except ImportError:
    TravelMemoryManager = None

class MultiAgentTripPlanner:
    """
    LangChain 版本的多智能体旅行规划系统
    
    使用 LangChain 框架实现，包含：
    - 景点搜索 Agent（ReAct）
    - 天气查询 Agent（ReAct）
    - 酒店推荐 Agent（ReAct）
    - 行程规划 Agent（Simple）
    
    特性：
    - 支持 MCP 工具集成
    - 工具调用采用 ReAct 模式
    - 支持 JSON 格式输出
    - 完整的错误处理和降级方案
    """
    
    def __init__(self, provider: str = "aliyun", model: str = None):
        """
        初始化多智能体系统
        
        Args:
            provider: LLM 提供商（默认阿里云）
            model: 模型名称（默认 qwen-turbo）
        """
        print("🔄 开始初始化 LangChain 多智能体旅行规划系统...")
        
        try:
            self.provider = provider
            self.model = model
            
            # 初始化 LLM
            print("  - 初始化 LLM...")
            self.llm = get_llm(provider=provider, model=model, temperature=0.7)
            
            # 获取基础工具
            print("  - 加载基础工具...")
            self.basic_tools = get_basic_tools()
            
            # 初始化 MCP 工具管理器
            print("  - 初始化 MCP 工具管理器...")
            self.mcp_manager = MCPToolManager()
            
            # 创建 MCP 工具（高德地图）
            self._setup_mcp_tools()
            
            # 初始化旅行记忆管理器
            print("  - 初始化旅行记忆管理器...")
            self.memory_manager = self._setup_memory_manager()

            # 高德服务（真实路线规划）
            print("  - 初始化高德地图服务...")
            try:
                from app.services.amap_service import get_amap_service
                self.amap_service = get_amap_service()
                print("  ✓ 高德地图服务已加载（用于真实路线规划）")
            except Exception as e:
                self.amap_service = None
                print(f"  ⚠ 高德地图服务加载失败，将使用直线距离估算: {e}")
            
            # 初始化提示词管理器（渐进式披露）
            print("  - 初始化提示词管理器...")
            self.prompt_manager = get_prompt_manager()
            
            # 创建景点搜索 Agent（使用 ReAct 模式）
            print("  - 创建景点搜索 Agent...")
            self.attraction_agent = self._create_attraction_agent()
            
            # 创建天气查询 Agent（使用 ReAct 模式）
            print("  - 创建天气查询 Agent...")
            self.weather_agent = self._create_weather_agent()
            
            # 创建酒店推荐 Agent（使用 ReAct 模式）
            print("  - 创建酒店推荐 Agent...")
            self.hotel_agent = self._create_hotel_agent()
            
            # 创建行程规划 Agent（使用 Simple 模式，纯推理）
            print("  - 创建行程规划 Agent...")
            self.planner_agent = self._create_planner_agent()
            
            print("✅ LangChain 多智能体系统初始化成功")
            print(f"   景点搜索Agent: ReAct 模式")
            print(f"   天气查询Agent: ReAct 模式")
            print(f"   酒店推荐Agent: ReAct 模式")
            print(f"   行程规划Agent: Simple 模式")
            print(f"   旅行记忆管理器: {'Redis' if self.memory_manager else '未启用'}")
            
        except Exception as e:
            print(f"❌ 多智能体系统初始化失败: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    
    def _setup_memory_manager(self):
        """设置旅行记忆管理器"""
        if TravelMemoryManager is None:
            print("  ⚠ 旅行记忆管理器未启用（缺少依赖）")
            return None
        
        try:
            memory_manager = TravelMemoryManager()
            storage_type = memory_manager.get_storage_type()
            print(f"  ✓ 旅行记忆管理器初始化成功（存储类型: {storage_type}）")
            return memory_manager
        except Exception as e:
            print(f"  ⚠ 旅行记忆管理器初始化失败: {str(e)}")
            return None
    
    def _setup_mcp_tools(self):
        """设置 MCP 工具"""
        try:
            if settings.enable_mcp_tools and settings.amap_api_key:
                print("  - 注册高德地图 MCP 工具...")
                self.mcp_manager.register_tool(
                    name="amap",
                    description="高德地图服务 - 提供地理编码、逆地理编码、POI 搜索、路径规划等功能",
                    server_command=["python", "-m", "amap_mcp_server"],
                    env={"AMAP_MAPS_API_KEY": settings.amap_api_key},
                    auto_expand=True
                )
                print("  ✓ 高德地图 MCP 工具注册成功")
            else:
                print("  ⚠ 高德地图 MCP 工具未启用（请检查 ENABLE_MCP_TOOLS 和 AMAP_MAPS_API_KEY）")
                self.amap_tool = None
        except Exception as e:
            print(f"  ⚠ MCP 工具设置失败: {str(e)}")
            self.amap_tool = None
    
    def _create_attraction_agent(self) -> ReActAgent:
        """创建景点搜索 Agent（使用渐进式提示词）"""
        attraction_tools = get_attraction_tools()
        prompt = self.prompt_manager.get_agent_prompt("attraction")
        return ReActAgent(
            name="景点搜索专家",
            tools=attraction_tools,
            llm=self.llm,
            system_prompt=prompt,
            max_iterations=5,
            verbose=True
        )
    
    def _create_weather_agent(self) -> ReActAgent:
        """创建天气查询 Agent（使用渐进式提示词）"""
        weather_tools = get_weather_tools()
        prompt = self.prompt_manager.get_agent_prompt("weather")
        return ReActAgent(
            name="天气查询专家",
            tools=weather_tools,
            llm=self.llm,
            system_prompt=prompt,
            max_iterations=5,
            verbose=True
        )
    
    def _create_hotel_agent(self) -> ReActAgent:
        """创建酒店推荐 Agent（使用渐进式提示词+记忆技能）"""
        hotel_tools = get_hotel_tools()
        prompt = self.prompt_manager.get_agent_prompt("hotel")
        return ReActAgent(
            name="酒店推荐专家",
            tools=hotel_tools,
            llm=self.llm,
            system_prompt=prompt,
            max_iterations=5,
            verbose=True
        )
    
    def _create_planner_agent(self) -> SimpleAgent:
        """创建行程规划 Agent（使用渐进式提示词+多项技能）"""
        prompt = self.prompt_manager.get_agent_prompt("planner")
        return SimpleAgent(
            name="行程规划专家",
            llm=self.llm,
            system_prompt=prompt,
            verbose=True
        )
    
    def plan_trip(self, request: TripRequest) -> TripPlan:
        """
        使用多智能体协作生成旅行计划
        
        Args:
            request: 旅行请求
            
        Returns:
            旅行计划
        """
        from app.monitoring import traced

        @traced(name="multi_agent_plan_trip", run_type="chain", tags=["multi-agent", "plan_trip"])
        def _run():
            return self._plan_trip_impl(request)

        return _run()

    def _plan_trip_impl(self, request: TripRequest) -> TripPlan:
        """多智能体规划实现（供 LangSmith 追踪包装）"""
        try:
            print(f"\n{'='*60}")
            print(f"🚀 开始 LangChain 多智能体协作规划旅行...")
            print(f"目的地: {request.city}")
            print(f"日期: {request.start_date} 至 {request.end_date}")
            print(f"天数: {request.travel_days}天")
            print(f"偏好: {', '.join(request.preferences) if request.preferences else '无'}")
            print(f"用户ID: {request.user_id}")
            print(f"{'='*60}\n")
            
            # 步骤0: 查询旅行记忆（如果启用）
            memory_info = ""
            visited_attractions = []
            if self.memory_manager and request.user_id:
                print("🧠 步骤0: 查询旅行记忆...")
                memory_info = self._get_memory_info(request.user_id, request.city)
                visited_attractions = self.memory_manager.get_visited_attractions(request.user_id, request.city)
                print(f"旅行记忆: {memory_info[:150]}...\n")
            
            # 步骤1: 景点搜索 Agent 搜索景点
            print("📍 步骤1: 景点搜索 Agent 搜索景点...")
            attraction_query = self._build_attraction_query(request, visited_attractions)
            attraction_response = self.attraction_agent.run(attraction_query)
            print(f"景点搜索结果: {attraction_response[:200]}...\n")
            
            # 步骤2: 天气查询 Agent 查询天气
            print("🌤️ 步骤2: 天气查询 Agent 查询天气...")
            weather_query = self._build_weather_query(request)
            weather_response = self.weather_agent.run(weather_query)
            print(f"天气查询结果: {weather_response[:200]}...\n")
            
            # 步骤3: 酒店推荐 Agent 搜索酒店
            print("🏨 步骤3: 酒店推荐 Agent 搜索酒店...")
            hotel_query = self._build_hotel_query(request)
            hotel_response = self.hotel_agent.run(hotel_query)
            print(f"酒店搜索结果: {hotel_response[:200]}...\n")
            
            # 步骤4: 路线规划（基于景点坐标计算路线）
            print("🗺️ 步骤4: 路线规划...")
            route_info = self._build_route_info(request, attraction_response)
            print(f"路线规划结果: {route_info[:200]}...\n")
            
            # 步骤5: 行程规划 Agent 整合信息生成详细计划
            print("📋 步骤5: 行程规划 Agent 生成详细计划...")
            planner_query = self._build_planner_query(
                request, attraction_response, weather_response, hotel_response, memory_info, route_info
            )
            planner_response = self.planner_agent.run(planner_query)
            print(f"行程规划结果: {planner_response[:300]}...\n")
            
            # 解析最终计划
            trip_plan = self._parse_response(planner_response, request)
            
            # 用高德真实路线回填 route_segments，并按 visit_duration 重算时间表
            trip_plan = self._enrich_plan_with_routes(
                trip_plan,
                city=request.city,
                transportation=request.transportation,
            )
            
            # 步骤6: 保存旅行计划到记忆（如果启用）
            if self.memory_manager and request.user_id:
                print("💾 步骤6: 保存旅行计划到记忆...")
                self._save_plan_to_memory(request, trip_plan)
                print("✅ 旅行计划已保存到记忆\n")
            
            print(f"{'='*60}")
            print(f"✅ 旅行计划生成完成!")
            print(f"{'='*60}\n")
            
            return trip_plan
            
        except Exception as e:
            print(f"❌ 生成旅行计划失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return self._create_fallback_plan(request)
    
    def _get_memory_info(self, user_id: str, city: str) -> str:
        """获取用户的旅行记忆信息"""
        if not self.memory_manager:
            return ""
        
        try:
            comparison = self.memory_manager.compare_plans(user_id, city)
            
            if not comparison["has_history"]:
                return f"用户是第一次访问{city}，推荐经典路线"
            
            visited = comparison["visited_attractions"]
            result = f"用户之前去过{city}（{comparison['plan_count']}次）\n"
            result += f"已访问景点: {', '.join(visited)}\n"
            result += f"请推荐不同的景点和路线"
            
            return result
        except Exception as e:
            print(f"⚠️ 获取旅行记忆失败: {str(e)}")
            return ""
    
    def _save_plan_to_memory(self, request: TripRequest, trip_plan: TripPlan):
        """将旅行计划保存到记忆"""
        if not self.memory_manager:
            return
        
        try:
            attractions = []
            for day in trip_plan.days:
                for attraction in day.attractions:
                    attractions.append(attraction.name)
            
            daily_itinerary = {}
            for day in trip_plan.days:
                daily_itinerary[f"day{day.day_index + 1}"] = [a.name for a in day.attractions]
            
            plan_id = self.memory_manager.save_plan(
                user_id=request.user_id,
                city=request.city,
                duration_days=request.travel_days,
                attractions=attractions,
                daily_itinerary=daily_itinerary,
                preferences=request.preferences,
                notes=f"{request.travel_days}天{request.city}旅行计划"
            )
            
            print(f"📝 旅行计划已保存，计划ID: {plan_id}")
        except Exception as e:
            print(f"⚠️ 保存旅行计划到记忆失败: {str(e)}")
    
    def _build_attraction_query(self, request: TripRequest, visited_attractions: List[str] = None) -> str:
        """构建景点搜索查询"""
        keywords = request.preferences[0] if request.preferences else "景点"
        
        visited_info = ""
        if visited_attractions and len(visited_attractions) > 0:
            visited_info = f"\n注意：用户已访问过以下景点，请优先搜索其他景点：{', '.join(visited_attractions)}"
        
        query = f"""请帮我搜索{request.city}的{keywords}相关景点。

用户需求：
- 城市：{request.city}
- 偏好：{', '.join(request.preferences) if request.preferences else '无'}
- 旅行天数：{request.travel_days}天
{visited_info}

请使用工具搜索景点，并返回景点的详细信息，包括名称、地址、坐标、建议游览时长(visit_duration)、简介等。
重要：需为{request.travel_days}天行程提供充足候选，建议至少返回 {max(8, (request.travel_days or 1) * 5)} 个不同景点（覆盖全城不同片区），以便后续按游玩时长排满每天 3-5 个景点。"""
        
        return query
    
    def _build_weather_query(self, request: TripRequest) -> str:
        """构建天气查询"""
        query = f"""请查询{request.city}的天气情况。
        
查询信息：
- 城市：{request.city}
- 日期范围：{request.start_date} 至 {request.end_date}
- 旅行天数：{request.travel_days}天

请使用工具查询天气，返回温度、湿度、风力等信息。"""
        
        return query
    
    def _build_hotel_query(self, request: TripRequest) -> str:
        """构建酒店搜索查询"""
        query = f"""请搜索{request.city}的{request.accommodation}酒店。
        
搜索信息：
- 城市：{request.city}
- 住宿类型：{request.accommodation}

要求：
1. 只搜索酒店，不要搜索景点或其他地点
2. 优先搜索有评分和价格信息的酒店
3. 返回酒店名称、地址、价格、评分等信息
4. 如果搜索结果中没有足够的有评分酒店，可以尝试不同的关键词"""
        
        return query
    
    def _build_route_info(self, request: TripRequest, attraction_response: str) -> str:
        """从景点搜索结果提取坐标，并用高德真实路径规划生成相邻景点路线摘要"""
        try:
            attractions = []
            current_attraction = {}

            for line in attraction_response.split("\n"):
                line = line.strip()
                if line.startswith(tuple(f"{i}." for i in range(1, 16))):
                    if current_attraction:
                        attractions.append(current_attraction)
                    # 兼容 "1. 名称" / "10. 名称"
                    name = line.split(".", 1)[1].strip() if "." in line else line
                    current_attraction = {"name": name}
                elif line.startswith("坐标:"):
                    current_attraction["location"] = line[3:].strip()
                elif line.startswith("地址:"):
                    current_attraction["address"] = line[3:].strip()

            if current_attraction:
                attractions.append(current_attraction)

            # 取足够候选供全天排程（每天 3-5 个 × 天数）
            max_n = max(8, (request.travel_days or 1) * 5)
            attractions = attractions[:max_n]

            if len(attractions) < 2:
                return "景点数量不足，无法规划路线"

            mode = self._map_transport_mode(request.transportation)
            mode_label = {"walking": "步行", "driving": "驾车", "transit": "公共交通"}.get(mode, mode)

            lines = [
                f"以下为高德真实路径规划结果（模式: {mode_label}），规划行程时请优先采用这些距离与时长：",
                "",
                "景点列表:",
            ]
            for i, attr in enumerate(attractions, 1):
                loc = attr.get("location") or attr.get("address") or "未知"
                lines.append(f"  {i}. {attr.get('name', '')}: {loc}")

            lines.append("")
            lines.append("相邻景点高德路线:")

            for i in range(len(attractions) - 1):
                a1, a2 = attractions[i], attractions[i + 1]
                origin = a1.get("location") or a1.get("address") or f"{request.city}{a1.get('name', '')}"
                dest = a2.get("location") or a2.get("address") or f"{request.city}{a2.get('name', '')}"
                summary = self._amap_route_summary(
                    origin=origin,
                    destination=dest,
                    mode=mode,
                    city=request.city,
                    name1=a1.get("name", ""),
                    name2=a2.get("name", ""),
                )
                lines.append(f"  {summary}")

            return "\n".join(lines)

        except Exception as e:
            print(f"⚠️ 构建路线信息失败: {str(e)}")
            return ""

    def _amap_route_summary(
        self,
        origin: str,
        destination: str,
        mode: str,
        city: str,
        name1: str,
        name2: str,
    ) -> str:
        """调用高德返回一行可读路线摘要；失败则回退估算"""
        if not getattr(self, "amap_service", None):
            return f"{name1} -> {name2}: 高德不可用，请按距离估算"

        try:
            route = self.amap_service.route_direction(
                origin=origin,
                destination=destination,
                mode=mode,
                city=city or None,
            )
            if not route:
                return f"{name1} -> {name2}: 无路线结果"

            distance, duration_min = self._parse_amap_distance_duration(route)
            mode_label = {"walking": "步行", "driving": "驾车", "transit": "公共交通"}.get(mode, mode)
            dist_txt = f"{distance/1000:.1f}公里" if distance >= 1000 else f"{distance}米"
            return (
                f"{name1} -> {name2}: {mode_label}，约{duration_min}分钟，{dist_txt}"
                f"（高德真实路线）"
            )
        except Exception as e:
            return f"{name1} -> {name2}: 路线查询失败({e})"

    def _parse_amap_distance_duration(self, route) -> tuple:
        """从 AmapRouteInfo 解析距离(米)与时长(分钟)"""
        import re

        distance = 0
        duration_sec = 0

        if route.paths and isinstance(route.paths, list) and route.paths:
            path = route.paths[0]
            try:
                distance = int(float(path.get("distance") or 0))
            except (TypeError, ValueError):
                distance = 0
            try:
                duration_sec = int(float(path.get("duration") or 0))
            except (TypeError, ValueError):
                duration_sec = 0
        elif route.transits and isinstance(route.transits, list) and route.transits:
            transit = route.transits[0]
            try:
                distance = int(float(transit.get("distance") or 0))
            except (TypeError, ValueError):
                distance = 0
            try:
                duration_sec = int(float(transit.get("duration") or 0))
            except (TypeError, ValueError):
                duration_sec = 0

        if not distance and route.distance:
            try:
                distance = int(float(str(route.distance).replace("米", "").replace("公里", "")))
                if "公里" in str(route.distance):
                    distance = int(distance * 1000)
            except (TypeError, ValueError):
                pass

        if duration_sec:
            return distance, max(1, duration_sec // 60)

        # duration 可能是 "25分钟" / "1小时30分钟" / 纯秒数字符串
        raw = str(route.duration or "")
        hours = re.search(r"(\d+)\s*小时", raw)
        mins = re.search(r"(\d+)\s*分钟", raw)
        if hours or mins:
            total = (int(hours.group(1)) if hours else 0) * 60 + (int(mins.group(1)) if mins else 0)
            return distance, max(1, total)
        try:
            val = int(float(re.search(r"(\d+)", raw).group(1))) if re.search(r"(\d+)", raw) else 0
            # 大于 180 更可能是秒
            return distance, max(1, val // 60) if val > 180 else max(1, val) if val else 20
        except Exception:
            return distance, 20
    
    def _build_planner_query(
        self, 
        request: TripRequest, 
        attractions: str, 
        weather: str, 
        hotels: str = "",
        memory_info: str = "",
        route_info: str = ""
    ) -> str:
        """构建行程规划查询"""
        memory_section = ""
        if memory_info:
            memory_section = f"\n**旅行记忆:**\n{memory_info}\n"
        
        route_section = ""
        if route_info:
            route_section = f"\n**路线信息:**\n{route_info}\n"
        
        query = f"""请根据以下信息生成{request.city}的{request.travel_days}天旅行计划:

**基本信息:**
- 城市: {request.city}
- 日期: {request.start_date} 至 {request.end_date}
- 天数: {request.travel_days}天
- 交通方式: {request.transportation}
- 住宿: {request.accommodation}
- 偏好: {', '.join(request.preferences) if request.preferences else '无'}

{memory_section}
**景点信息:**
{attractions}

**天气信息:**
{weather}

{route_section}
**酒店信息:**
{hotels if hotels else '暂无酒店信息，请根据城市推荐合适酒店'}

**要求:**
1. 【强制】行程景点名称必须来自上方「景点信息」候选列表，禁止编造未出现的景点；可筛选排序，不可新增列表外景点
2. 按可游玩时段排满全天（约09:00-21:00），根据景点visit_duration累加安排；每天建议3-5个景点（大型景区可2-3个），避免日程空档
3. 每天必须包含早中晚三餐，并安排合适的就餐时间
4. 每天推荐一个具体的酒店(从酒店信息中选择或合理推荐)
5. 考虑景点之间的距离和交通方式，规划合理的游览顺序
6. 必须为每个景点规划具体的时间安排（开始时间+visit_duration）；优先复制候选中的建议游览时长与坐标
7. 必须规划景点之间的交通方式和交通时间；若上文提供了高德真实路线数据，必须采用其距离与时长
8. 返回完整的JSON格式数据，包含time_slots和route_segments字段
9. 景点的经纬度坐标要真实准确（从候选复制）
10. 如果用户有历史旅行记录，尽量推荐不同的景点和路线

请严格按照提示词中的JSON格式返回完整的旅行计划，确保包含详细的时间安排和路线规划。"""
        
        if request.free_text_input:
            query += f"\n**额外要求:** {request.free_text_input}"
        
        return query
    
    def _enrich_plan_with_routes(
        self,
        trip_plan: TripPlan,
        city: str = "",
        transportation: str = "公共交通",
    ) -> TripPlan:
        """用高德真实路径规划回填路线，并按游玩时长生成全天时间表"""
        try:
            for day in trip_plan.days:
                day.route_segments = self._generate_route_segments(
                    day, city=city or trip_plan.city, transportation=transportation
                )
                day.time_slots = self._generate_time_slots(day)
            return trip_plan
        except Exception as e:
            print(f"⚠️ 补充路线信息失败: {str(e)}")
            return trip_plan
    
    def _generate_time_slots(self, day_plan: DayPlan) -> list:
        """按 visit_duration + 真实交通时长生成全天时间安排"""
        time_slots = []
        current_time = "09:00"
        route_by_pair = {}
        for seg in day_plan.route_segments or []:
            route_by_pair[(seg.start_point, seg.end_point)] = seg
        
        for i, attraction in enumerate(day_plan.attractions):
            duration = attraction.visit_duration or 120
            end_time = self._add_minutes(current_time, duration)
            time_slots.append(TimeSlot(
                start_time=current_time,
                end_time=end_time,
                activity_type="景点游览",
                activity_name=attraction.name,
                description=f"游览{attraction.name}（约{duration}分钟）"
            ))
            
            if i < len(day_plan.attractions) - 1:
                next_attr = day_plan.attractions[i + 1]
                seg = route_by_pair.get((attraction.name, next_attr.name))
                travel_time = seg.duration if seg and seg.duration else 30
                mode = seg.transport_mode if seg else (day_plan.transportation or "公共交通")
                current_time = end_time
                # 午餐空档：若跨越中午，插入午餐
                if current_time <= "12:00" < self._add_minutes(current_time, travel_time + 30):
                    lunch_end = self._add_minutes("12:00", 60)
                    time_slots.append(TimeSlot(
                        start_time="12:00",
                        end_time=lunch_end,
                        activity_type="餐饮",
                        activity_name="午餐",
                        description="午餐"
                    ))
                    current_time = lunch_end
                arrival_time = self._add_minutes(current_time, travel_time)
                time_slots.append(TimeSlot(
                    start_time=current_time,
                    end_time=arrival_time,
                    activity_type="交通",
                    activity_name=mode,
                    description=f"从{attraction.name}前往{next_attr.name}"
                    + (f"（约{travel_time}分钟/{seg.distance}米）" if seg else f"（约{travel_time}分钟）")
                ))
                current_time = arrival_time
        
        # 补充晚餐（若尚未覆盖）
        has_dinner = any(s.activity_name == "晚餐" for s in time_slots)
        if not has_dinner:
            time_slots.append(TimeSlot(
                start_time="18:00",
                end_time="19:00",
                activity_type="餐饮",
                activity_name="晚餐",
                description="晚餐"
            ))
        
        time_slots.sort(key=lambda x: x.start_time)
        return time_slots
    
    def _generate_route_segments(
        self,
        day_plan: DayPlan,
        city: str = "",
        transportation: str = "公共交通",
    ) -> list:
        """调用高德真实路径规划生成相邻景点路线段；失败时回退直线估算"""
        segments = []
        mode = self._map_transport_mode(transportation)

        for i in range(len(day_plan.attractions) - 1):
            from_attr = day_plan.attractions[i]
            to_attr = day_plan.attractions[i + 1]
            segment = self._plan_amap_segment(from_attr, to_attr, city=city, mode=mode)
            if segment is None:
                segment = self._estimate_segment_fallback(from_attr, to_attr, transportation)
            segments.append(segment)

        return segments

    def _map_transport_mode(self, transportation: str) -> str:
        text = transportation or ""
        if "步行" in text:
            return "walking"
        if "自驾" in text or "开车" in text or "驾车" in text:
            return "driving"
        return "transit"

    def _plan_amap_segment(self, from_attr, to_attr, city: str = "", mode: str = "transit"):
        """调用高德 route_direction，返回 RouteSegment 或 None"""
        if not getattr(self, "amap_service", None):
            return None

        origin = self._attraction_route_point(from_attr)
        destination = self._attraction_route_point(to_attr)
        if not origin or not destination:
            return None

        try:
            route = self.amap_service.route_direction(
                origin=origin,
                destination=destination,
                mode=mode,
                city=city or None,
            )
            if not route:
                return None

            distance, duration_min = self._parse_amap_distance_duration(route)

            mode_label = {"walking": "步行", "driving": "驾车", "transit": "公共交通"}.get(mode, mode)
            if not duration_min and distance:
                duration_min = max(5, distance // 80) if mode == "walking" else max(10, distance // 400)

            return RouteSegment(
                start_point=from_attr.name,
                end_point=to_attr.name,
                transport_mode=mode_label,
                distance=distance,
                duration=duration_min or 20,
                description=(
                    f"从{from_attr.name}到{to_attr.name}，{mode_label}，"
                    f"约{duration_min or 20}分钟"
                    + (f"/{distance}米" if distance else "")
                    + "（高德路径规划）"
                ),
            )
        except Exception as e:
            print(f"⚠️ 高德路线规划失败 {from_attr.name}->{to_attr.name}: {e}")
            return None

    def _attraction_route_point(self, attraction) -> str:
        """优先坐标，其次地址+名称，供高德路径规划使用"""
        if attraction.location and attraction.location.longitude and attraction.location.latitude:
            return f"{attraction.location.longitude},{attraction.location.latitude}"
        if attraction.address:
            return attraction.address
        return attraction.name or ""

    def _estimate_segment_fallback(self, from_attr, to_attr, transportation: str = "公共交通") -> RouteSegment:
        """无高德结果时的直线距离回退"""
        distance = 0
        duration = 30
        transport_mode = transportation or "公共交通"

        if from_attr.location and to_attr.location:
            try:
                import math
                lon1, lat1 = from_attr.location.longitude, from_attr.location.latitude
                lon2, lat2 = to_attr.location.longitude, to_attr.location.latitude
                R = 6371000
                phi1 = math.radians(lat1)
                phi2 = math.radians(lat2)
                delta_phi = math.radians(lat2 - lat1)
                delta_lambda = math.radians(lon2 - lon1)
                a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
                c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
                distance = int(R * c)
                if distance < 500:
                    transport_mode = "步行"
                    duration = max(5, int(distance / 60))
                elif distance < 3000:
                    transport_mode = "地铁"
                    duration = 20
                else:
                    transport_mode = "地铁/公交"
                    duration = 45
            except Exception:
                pass

        return RouteSegment(
            start_point=from_attr.name,
            end_point=to_attr.name,
            transport_mode=transport_mode,
            distance=distance,
            duration=duration,
            description=f"从{from_attr.name}到{to_attr.name}，{transport_mode}，约{duration}分钟（估算）",
        )    
    def _add_minutes(self, time_str: str, minutes: int) -> str:
        """在时间字符串上加上指定分钟数"""
        try:
            hours, mins = map(int, time_str.split(':'))
            total_mins = hours * 60 + mins + minutes
            new_hours = total_mins // 60
            new_mins = total_mins % 60
            return f"{new_hours:02d}:{new_mins:02d}"
        except:
            return time_str
    
    def _parse_response(self, response: str, request: TripRequest) -> TripPlan:
        """解析 Agent 响应"""
        try:
            # 查找 JSON 代码块
            if "```json" in response:
                json_start = response.find("```json") + 7
                json_end = response.find("```", json_start)
                json_str = response[json_start:json_end].strip()
            elif "```" in response:
                # 查找第一个 ``` 后的内容
                json_start = response.find("```") + 3
                json_end = response.find("```", json_start)
                if json_end != -1:
                    json_str = response[json_start:json_end].strip()
                else:
                    # 没有结束标记，直接取到末尾
                    json_str = response[json_start:].strip()
            elif "{" in response and "}" in response:
                # 直接查找 JSON 对象
                json_start = response.find("{")
                json_end = response.rfind("}") + 1
                json_str = response[json_start:json_end]
            else:
                raise ValueError("响应中未找到 JSON 数据")
            
            # 解析 JSON
            data = json.loads(json_str)
            
            # 转换为 TripPlan 对象，并修正 day_index（LLM 常输出 1-based）
            trip_plan = TripPlan(**data)
            trip_plan.normalize_day_indices()
            
            return trip_plan
            
        except json.JSONDecodeError as e:
            print(f"⚠️  JSON 解析失败: {str(e)}")
            print(f"   响应内容: {response[:500]}...")
            return self._create_fallback_plan(request)
        except Exception as e:
            print(f"⚠️  解析响应失败: {str(e)}")
            return self._create_fallback_plan(request)
    
    def _create_fallback_plan(self, request: TripRequest) -> TripPlan:
        """创建备用计划（当 Agent 失败时）"""
        # 解析日期
        start_date = datetime.strptime(request.start_date, "%Y-%m-%d")
        
        # 创建每日行程
        days = []
        for i in range(request.travel_days):
            current_date = start_date + timedelta(days=i)
            
            day_plan = DayPlan(
                date=current_date.strftime("%Y-%m-%d"),
                day_index=i,
                description=f"第{i+1}天探索{request.city}的精彩之旅",
                transportation=request.transportation,
                accommodation=request.accommodation,
                attractions=[
                    Attraction(
                        name=f"{request.city}标志性景点",
                        address=f"{request.city}市中心",
                        location=Location(
                            longitude=116.4 + i * 0.01 + j * 0.005,
                            latitude=39.9 + i * 0.01 + j * 0.005
                        ),
                        visit_duration=120,
                        description=f"这是{request.city}的著名景点",
                        category="景点",
                        ticket_price=60
                    )
                    for j in range(2)
                ],
                meals=[
                    Meal(
                        type="breakfast",
                        name=f"第{i+1}天早餐",
                        description="当地特色早餐",
                        estimated_cost=30
                    ),
                    Meal(
                        type="lunch",
                        name=f"第{i+1}天午餐",
                        description="午餐推荐当地美食",
                        estimated_cost=50
                    ),
                    Meal(
                        type="dinner",
                        name=f"第{i+1}天晚餐",
                        description="晚餐推荐特色餐厅",
                        estimated_cost=80
                    )
                ]
            )
            days.append(day_plan)
        
        return TripPlan(
            city=request.city,
            start_date=request.start_date,
            end_date=request.end_date,
            days=days,
            weather_info=[
                WeatherInfo(
                    date=(start_date + timedelta(days=i)).strftime("%Y-%m-%d"),
                    day_weather="晴",
                    night_weather="多云",
                    day_temp=25,
                    night_temp=15,
                    wind_direction="南风",
                    wind_power="1-3级"
                )
                for i in range(request.travel_days)
            ],
            overall_suggestions=f"这是为您规划的{request.city}{request.travel_days}日游行程，"
                              f"建议提前查看各景点的开放时间和门票信息。祝您旅途愉快！",
            budget=Budget(
                total_attractions=60 * 2 * request.travel_days,
                total_hotels=400 * request.travel_days,
                total_meals=(30 + 50 + 80) * request.travel_days,
                total_transportation=50 * request.travel_days,
                total=(60 * 2 * request.travel_days) + 
                      (400 * request.travel_days) + 
                      ((30 + 50 + 80) * request.travel_days) +
                      (50 * request.travel_days)
            )
        )


# 全局多智能体系统实例（单例）
_multi_agent_planner = None


def get_trip_planner_agent(provider: str = "aliyun", model: str = None) -> MultiAgentTripPlanner:
    """获取多智能体旅行规划系统实例（单例模式）"""
    global _multi_agent_planner
    
    if _multi_agent_planner is None:
        _multi_agent_planner = MultiAgentTripPlanner(provider=provider, model=model)
    
    return _multi_agent_planner


# 测试代码
if __name__ == "__main__":
    # 创建测试请求
    test_request = TripRequest(
        city="上海",
        start_date="2026-05-25",
        end_date="2026-05-26",
        travel_days=2,
        transportation="公共交通",
        accommodation="经济型",
        preferences=["历史文化", "古迹"],
        user_id="test_user_001"
    )
    
    print("="*60)
    print("测试 LangChain 版本的多智能体旅行规划系统")
    print("="*60)
    
    try:
        # 创建规划器
        planner = MultiAgentTripPlanner(provider="aliyun")
        
        # 生成旅行计划
        trip_plan = planner.plan_trip(test_request)
        
        # 打印结果
        print("\n" + "="*60)
        print("生成的旅行计划:")
        print("="*60)
        print(f"城市: {trip_plan.city}")
        print(f"日期: {trip_plan.start_date} 至 {trip_plan.end_date}")
        print(f"天数: {len(trip_plan.days)}天")
        print(f"建议: {trip_plan.overall_suggestions}")
        
        if trip_plan.budget:
            print(f"\n预算汇总:")
            print(f"  景点门票: {trip_plan.budget.total_attractions}元")
            print(f"  酒店住宿: {trip_plan.budget.total_hotels}元")
            print(f"  餐饮费用: {trip_plan.budget.total_meals}元")
            print(f"  交通费用: {trip_plan.budget.total_transportation}元")
            print(f"  总计: {trip_plan.budget.total}元")
        
        print("\n每日行程详情:")
        for day in trip_plan.days[:2]:
            print(f"\n{day.date} - {day.description}")
            print(f"  交通方式: {day.transportation}")
            print(f"  住宿: {day.accommodation}")
            print(f"  景点: {', '.join([a.name for a in day.attractions])}")
            print(f"  餐饮: {', '.join([m.name for m in day.meals])}")
            
            if day.time_slots:
                print(f"\n  时间安排:")
                for slot in day.time_slots:
                    print(f"    {slot.start_time}-{slot.end_time}: [{slot.activity_type}] {slot.activity_name}")
                    if slot.description:
                        print(f"      {slot.description}")
            
            if day.route_segments:
                print(f"\n  路线规划:")
                for seg in day.route_segments:
                    print(f"    {seg.start_point} -> {seg.end_point}")
                    print(f"      交通方式: {seg.transport_mode}")
                    print(f"      距离: {seg.distance}米")
                    print(f"      时长: {seg.duration}分钟")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
