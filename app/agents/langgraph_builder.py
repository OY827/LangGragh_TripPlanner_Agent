"""
LangGraph 构建器 - 构建子图和主图

架构：
- AttractionGraph: 景点子图（RAG检索 || 高德搜索 → 合并 → 反思 → 生成）
- WeatherGraph: 天气子图
- HotelGraph: 酒店子图（检索→反思→重试/生成）
- TripPlannerGraph: 主图（景点/天气/酒店异步并发 → 汇聚规划 + 反思循环）

并行阶段：
1. 主图 fan-out 三个异步节点，通过 ainvoke 同一步并发调度
2. 各子图在线程池中执行同步 I/O（RAG/高德），避免阻塞事件循环
3. 仅回写各子图专属字段，避免同一步并发写冲突
"""

from __future__ import annotations

import asyncio
from typing import Optional, Any, Dict
from langgraph.graph import StateGraph, END

from .langgraph_state import AttractionState, WeatherState, HotelState, TripState
from .langgraph_nodes import GraphNodes


def _passthrough(state: Dict[str, Any]) -> Dict[str, Any]:
    """空节点，仅用于 fan-out 入口"""
    return {}


class LangGraphBuilder:
    """
    LangGraph 构建器

    负责创建和编译所有 Agent 图。
    子图通过异步包装函数作为主图节点，实现状态字段映射与并发执行。
    """

    def __init__(self, nodes: Optional[GraphNodes] = None):
        self.nodes = nodes or GraphNodes()

        self._attraction_graph = None
        self._weather_graph = None
        self._hotel_graph = None
        self._trip_graph = None

        self._attraction_app = None
        self._weather_app = None
        self._hotel_app = None

    # ==================== 景点子图 ====================

    def build_attraction_graph(self) -> StateGraph:
        """构建景点子图（RAG检索与高德搜索异步并行，再合并）"""
        graph = StateGraph(AttractionState)

        graph.add_node("start", _passthrough)
        graph.add_node("retrieve", self.nodes.retrieve_knowledge)
        graph.add_node("search_amap", self.nodes.search_amap)
        graph.add_node("merge", self.nodes.merge_attraction_results)
        graph.add_node("reflect", self.nodes.reflect_attractions)
        graph.add_node("generate", self.nodes.generate_attraction_response)

        graph.set_entry_point("start")
        # 内部并行：知识库检索 || 高德搜索
        graph.add_edge("start", "retrieve")
        graph.add_edge("start", "search_amap")
        graph.add_edge("retrieve", "merge")
        graph.add_edge("search_amap", "merge")
        graph.add_edge("merge", "reflect")

        graph.add_conditional_edges(
            "reflect",
            self.nodes.route_after_reflect_attractions,
            {
                "generate": "generate",
                "retrieve": "start",  # 重试时重新并行检索
            }
        )

        graph.add_edge("generate", END)

        self._attraction_graph = graph
        return graph

    def _get_attraction_app(self):
        if self._attraction_app is None:
            if self._attraction_graph is None:
                self.build_attraction_graph()
            self._attraction_app = self._attraction_graph.compile()
        return self._attraction_app

    async def _run_attraction_subgraph(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """景点子图异步包装：在线程池中执行同步子图，不阻塞事件循环"""
        subgraph_input = {
            "query": state.get("query", ""),
            "city": state.get("city", ""),
            "preferences": state.get("preferences", []),
            "visited_attractions": state.get("visited_attractions", []),
            "iteration_count": 0,
        }

        app = self._get_attraction_app()
        result = await asyncio.to_thread(app.invoke, subgraph_input)

        return {
            "kb_docs": result.get("kb_docs", []),
            "amap_results": result.get("amap_results", ""),
            "merged_results": result.get("merged_results", ""),
            "attraction_response": result.get("attraction_response", ""),
        }

    # ==================== 天气子图 ====================

    def build_weather_graph(self) -> StateGraph:
        """构建天气子图"""
        graph = StateGraph(WeatherState)

        graph.add_node("fetch", self.nodes.fetch_weather)
        graph.set_entry_point("fetch")
        graph.add_edge("fetch", END)

        self._weather_graph = graph
        return graph

    def _get_weather_app(self):
        if self._weather_app is None:
            if self._weather_graph is None:
                self.build_weather_graph()
            self._weather_app = self._weather_graph.compile()
        return self._weather_app

    async def _run_weather_subgraph(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """天气子图异步包装"""
        subgraph_input = {
            "city": state.get("city", ""),
            "start_date": state.get("start_date", ""),
            "end_date": state.get("end_date", ""),
        }

        app = self._get_weather_app()
        result = await asyncio.to_thread(app.invoke, subgraph_input)

        return {
            "weather_response": result.get("weather_response", ""),
        }

    # ==================== 酒店子图 ====================

    def build_hotel_graph(self) -> StateGraph:
        """构建酒店子图"""
        graph = StateGraph(HotelState)

        graph.add_node("search", self.nodes.search_hotels)
        graph.add_node("reflect", self.nodes.reflect_hotels)

        graph.set_entry_point("search")
        graph.add_edge("search", "reflect")

        graph.add_conditional_edges(
            "reflect",
            self.nodes.route_after_reflect_hotels,
            {
                "search": "search",
                "end": END,
            }
        )

        self._hotel_graph = graph
        return graph

    def _get_hotel_app(self):
        if self._hotel_app is None:
            if self._hotel_graph is None:
                self.build_hotel_graph()
            self._hotel_app = self._hotel_graph.compile()
        return self._hotel_app

    async def _run_hotel_subgraph(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """酒店子图异步包装"""
        subgraph_input = {
            "city": state.get("city", ""),
            "accommodation": state.get("accommodation", ""),
            "iteration_count": 0,
        }

        app = self._get_hotel_app()
        result = await asyncio.to_thread(app.invoke, subgraph_input)

        return {
            "hotel_response": result.get("hotel_response", ""),
        }

    async def run_parallel_subgraphs(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        显式并发执行三个子图（供调试/单测使用）。

        主图通过 fan-out 异步节点实现同等并发；本方法用 gather 做对照验证。
        """
        attraction_result, weather_result, hotel_result = await asyncio.gather(
            self._run_attraction_subgraph(state),
            self._run_weather_subgraph(state),
            self._run_hotel_subgraph(state),
            return_exceptions=True,
        )

        merged: Dict[str, Any] = {"current_phase": "parallel_fetch"}

        for name, result in [
            ("attractions", attraction_result),
            ("weather", weather_result),
            ("hotels", hotel_result),
        ]:
            if isinstance(result, Exception):
                merged.setdefault("errors", []).append(f"{name}: {result}")
            elif isinstance(result, dict):
                merged.update(result)

        return merged

    # ==================== 主图 ====================

    def build_trip_planner_graph(self) -> StateGraph:
        """构建主旅行规划图（含异步并行检索与对话循环）

        流程：
            init → memory → prepare
                 ↘ attractions ↘
                  → weather   → route → plan → reflect → ...
                 ↗ hotels    ↗
        """
        graph = StateGraph(TripState)

        graph.add_node("init", self.nodes.extract_trip_intent)
        graph.add_node("memory", self.nodes.query_memory)
        graph.add_node("build_attraction_query", self.nodes.build_attraction_query)

        # 异步节点：主图 ainvoke 时同一步并发调度
        graph.add_node("attractions", self._run_attraction_subgraph)
        graph.add_node("weather", self._run_weather_subgraph)
        graph.add_node("hotels", self._run_hotel_subgraph)

        graph.add_node("route", self.nodes.plan_routes)
        graph.add_node("plan", self.nodes.generate_trip_plan)
        graph.add_node("reflect", self.nodes.reflect_trip_plan)
        graph.add_node("save", self.nodes.save_trip_plan)

        graph.add_node("save_version", self.nodes.save_version)
        graph.add_node("dialog", self.nodes.dialog)

        graph.set_entry_point("init")

        graph.add_edge("init", "memory")
        graph.add_edge("memory", "build_attraction_query")

        # 异步 fan-out
        graph.add_edge("build_attraction_query", "attractions")
        graph.add_edge("build_attraction_query", "weather")
        graph.add_edge("build_attraction_query", "hotels")

        # fan-in
        graph.add_edge("attractions", "route")
        graph.add_edge("weather", "route")
        graph.add_edge("hotels", "route")
        graph.add_edge("route", "plan")
        graph.add_edge("plan", "reflect")

        graph.add_conditional_edges(
            "reflect",
            self.nodes.route_after_reflect_trip,
            {
                "build_attraction_query": "build_attraction_query",
                "plan": "plan",
                "final": "save",
            }
        )

        graph.add_edge("save", "save_version")
        graph.add_edge("save_version", "dialog")

        graph.add_conditional_edges(
            "dialog",
            self._route_after_dialog,
            {
                "continue": "dialog",
                "final": END,
            }
        )

        self._trip_graph = graph
        return graph

    def _route_after_dialog(self, state: Dict[str, Any]) -> str:
        """对话后的条件路由"""
        if not state.get("dialog_active", False):
            return "final"
        if state.get("dialog_input"):
            return "continue"
        return "final"

    def get_attraction_graph(self):
        """获取编译后的景点子图"""
        return self._get_attraction_app()

    def get_weather_graph(self):
        """获取编译后的天气子图"""
        return self._get_weather_app()

    def get_hotel_graph(self):
        """获取编译后的酒店子图"""
        return self._get_hotel_app()

    def get_trip_planner_graph(self):
        if self._trip_graph is None:
            self.build_trip_planner_graph()
        return self._trip_graph.compile()

    def get_trip_planner_with_checkpointer(self, checkpointer):
        """获取带 Checkpointer 的旅行规划图"""
        if self._trip_graph is None:
            self.build_trip_planner_graph()
        return self._trip_graph.compile(checkpointer=checkpointer)


class TripPlannerAgent:
    """
    基于 LangGraph 的旅行规划 Agent（默认异步并发执行并行子图）
    """

    def __init__(
        self,
        llm_provider: str = "aliyun",
        llm_model: str = None,
        use_redis: bool = False,
        redis_url: str = "redis://localhost:6379",
    ):
        self.nodes = GraphNodes(llm_provider=llm_provider, llm_model=llm_model)
        self.builder = LangGraphBuilder(nodes=self.nodes)

        self.checkpointer = None
        if use_redis:
            try:
                from langgraph.checkpoint.redis import RedisSaver
                self.checkpointer = RedisSaver(redis_url)
                print(f"✅ Redis Checkpointer 已连接: {redis_url}")
            except (ImportError, Exception) as e:
                print(f"⚠️  Redis 连接失败: {e}")
                self._setup_memory_checkpointer()
        else:
            self._setup_memory_checkpointer()

        self.app = self._compile_app()

        # LangSmith：进程内幂等初始化
        try:
            from app.monitoring import setup_langsmith, is_tracing_enabled
            setup_langsmith()
            if is_tracing_enabled():
                print("✅ LangSmith 链路追踪已接入 TripPlannerAgent")
        except Exception as e:
            print(f"⚠️  LangSmith 初始化跳过: {e}")

        print("✅ LangGraph 旅行规划 Agent 初始化完成（异步并发已启用）")

    def _setup_memory_checkpointer(self):
        try:
            from langgraph.checkpoint.memory import MemorySaver
            self.checkpointer = MemorySaver()
            print("✅ 内存 Checkpointer 已启用")
        except ImportError:
            self.checkpointer = None
            print("⚠️  无 Checkpointer，会话状态将不会持久化")

    def _compile_app(self):
        if self.checkpointer:
            return self.builder.get_trip_planner_with_checkpointer(self.checkpointer)
        return self.builder.get_trip_planner_graph()

    def _ensure_thread_id(self, thread_id: Optional[str] = None):
        """确保 Checkpointer 始终有 thread_id"""
        if self.checkpointer:
            return thread_id or f"session_{id(self)}_{__import__('time').time()}"
        return None

    def _build_input_state(
        self,
        city: str,
        travel_days: int,
        start_date: str,
        end_date: str,
        transportation: str,
        accommodation: str,
        preferences: list,
        user_id: str,
        free_text_input: str,
    ) -> Dict[str, Any]:
        from datetime import datetime, timedelta

        if not start_date:
            start_date = datetime.now().strftime("%Y-%m-%d")
        if not end_date:
            end_date = (datetime.now() + timedelta(days=travel_days)).strftime("%Y-%m-%d")
        if not accommodation:
            accommodation = "经济型"

        return {
            "city": city,
            "start_date": start_date,
            "end_date": end_date,
            "travel_days": travel_days,
            "transportation": transportation,
            "accommodation": accommodation,
            "preferences": preferences or [],
            "user_id": user_id,
            "free_text_input": free_text_input,
        }

    async def aplan_trip(
        self,
        city: str,
        travel_days: int = 1,
        start_date: str = "",
        end_date: str = "",
        transportation: str = "公共交通",
        accommodation: str = "",
        preferences: list = None,
        user_id: str = "",
        free_text_input: str = "",
        thread_id: str = None,
    ) -> Dict[str, Any]:
        """异步规划旅行（并行子图通过 ainvoke 真正并发）"""
        from app.monitoring import merge_run_config

        input_state = self._build_input_state(
            city, travel_days, start_date, end_date,
            transportation, accommodation, preferences or [],
            user_id, free_text_input,
        )

        thread_id = self._ensure_thread_id(thread_id)
        config = merge_run_config(
            {"configurable": {"thread_id": thread_id}} if thread_id else None,
            run_name="langgraph_plan_trip",
            thread_id=thread_id,
            tags=["langgraph", "plan_trip", city or "unknown-city"],
            metadata={
                "user_id": user_id,
                "city": city,
                "travel_days": travel_days,
                "preferences": preferences or [],
                "transportation": transportation,
                "accommodation": accommodation,
            },
        )

        try:
            result = await self.app.ainvoke(input_state, config)
        except Exception as e:
            print(f"❌ 图异步执行失败: {e}")
            import traceback
            traceback.print_exc()
            result = self._create_fallback(
                city, travel_days,
                input_state["start_date"], input_state["end_date"],
                preferences,
            )

        return self._format_result(result)

    def plan_trip(
        self,
        city: str,
        travel_days: int = 1,
        start_date: str = "",
        end_date: str = "",
        transportation: str = "公共交通",
        accommodation: str = "",
        preferences: list = None,
        user_id: str = "",
        free_text_input: str = "",
        thread_id: str = None,
    ) -> Dict[str, Any]:
        """同步入口：内部走异步并发执行"""
        kwargs = dict(
            city=city,
            travel_days=travel_days,
            start_date=start_date,
            end_date=end_date,
            transportation=transportation,
            accommodation=accommodation,
            preferences=preferences,
            user_id=user_id,
            free_text_input=free_text_input,
            thread_id=thread_id,
        )

        try:
            asyncio.get_running_loop()
            in_running_loop = True
        except RuntimeError:
            in_running_loop = False

        if not in_running_loop:
            return asyncio.run(self.aplan_trip(**kwargs))

        # 已在事件循环中：避免 run_coroutine_threadsafe 同线程死锁
        import concurrent.futures

        def _run_in_fresh_loop():
            return asyncio.run(self.aplan_trip(**kwargs))

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(_run_in_fresh_loop).result()

    def _create_fallback(self, city, travel_days, start_date, end_date, preferences):
        """创建降级结果"""
        return {
            "city": city,
            "trip_plan": f"旅行规划失败，请重试。城市: {city}",
            "trip_plan_parsed": None,
            "attraction_response": "",
            "weather_response": "",
            "hotel_response": "",
            "reflection_feedback": "执行失败，已返回降级结果",
            "iteration_count": 0,
        }

    def _format_result(self, result: Dict) -> Dict:
        return {
            "city": result.get("city", ""),
            "trip_plan": result.get("trip_plan", ""),
            "trip_plan_parsed": result.get("trip_plan_parsed"),
            "attraction_response": result.get("attraction_response", ""),
            "weather_response": result.get("weather_response", ""),
            "hotel_response": result.get("hotel_response", ""),
            "reflection_feedback": result.get("reflection_feedback", ""),
            "iteration_count": result.get("iteration_count", 0),
            "budget": result.get("budget"),
            "budget_total": result.get("budget_total", 0),
            "session_id": result.get("session_id", ""),
            "current_version_id": result.get("current_version_id", ""),
        }

    def get_state_history(self, thread_id: str) -> list:
        """获取会话历史"""
        if not self.checkpointer:
            return []

        try:
            history = list(
                self.checkpointer.list(
                    self._trip_graph or self.builder.get_trip_planner_graph()
                )
            )
            return history
        except Exception:
            return []

    async def aresume_from_checkpoint(self, thread_id: str) -> Optional[Dict]:
        """异步从断点恢复会话"""
        if not self.checkpointer:
            return None

        try:
            config = {"configurable": {"thread_id": thread_id}}
            return await self.app.ainvoke(None, config)
        except Exception as e:
            print(f"恢复失败: {e}")
            return None

    def resume_from_checkpoint(self, thread_id: str) -> Optional[Dict]:
        """从断点恢复会话"""
        if not self.checkpointer:
            return None

        try:
            return asyncio.run(self.aresume_from_checkpoint(thread_id))
        except Exception as e:
            print(f"恢复失败: {e}")
            return None


_agent_instance = None


def get_langgraph_agent(**kwargs) -> TripPlannerAgent:
    """获取 LangGraph Agent 实例（单例）"""
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = TripPlannerAgent(**kwargs)
    return _agent_instance
