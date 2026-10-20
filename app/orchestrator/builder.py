"""编排层：只决定主图怎么连、按什么顺序跑。

景点、天气、酒店、规划、排程这些具体工作仍在 app.agents。
流程是推荐、天气、酒店三路同时出发，到齐后规划；反思不通过时只重生成行程。
"""

from __future__ import annotations

import asyncio
import os
import uuid
from typing import Optional, Any, Dict
from urllib.parse import quote

from dotenv import load_dotenv
from langgraph.graph import StateGraph, END

from app.agents.langgraph_nodes import GraphNodes
from app.agents.langgraph_state import TripState


def build_redis_url(
    host: str = "localhost",
    port: int = 6379,
    password: str = "",
) -> str:
    """拼出和版本管理同一套主机、端口、密码的 Redis 地址。"""
    if password:
        return f"redis://:{quote(password, safe='')}@{host}:{port}/0"
    return f"redis://{host}:{port}/0"


def redis_url_from_env() -> str:
    load_dotenv()
    host = os.getenv("REDIS_HOST", "localhost")
    port = int(os.getenv("REDIS_PORT", "6379"))
    password = os.getenv("REDIS_PASSWORD", "") or ""
    return build_redis_url(host, port, password)


def read_interrupt(result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """取出图暂停时交给调用方的内容。没有暂停则返回 None。"""
    raw = result.get("__interrupt__") if isinstance(result, dict) else None
    if not raw:
        return None
    item = raw[0]
    value = getattr(item, "value", None)
    if value is None and isinstance(item, dict):
        value = item.get("value")
    if isinstance(value, dict):
        return value
    return {"reply": "" if value is None else str(value)}


class LangGraphBuilder:
    """
    LangGraph 构建器

    只编译主图。知识库、高德、天气、酒店都是主图上的节点。
    """

    def __init__(self, nodes: Optional[GraphNodes] = None):
        self.nodes = nodes or GraphNodes()
        self._trip_graph = None

    # ==================== 并行节点 ====================

    async def _run_retrieve(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """知识库检索。"""
        return await asyncio.to_thread(self.nodes.retrieve_knowledge, state)

    async def _run_amap(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """调试用的高德搜索。主图景点推荐走 recommend，不走这里。"""
        return await asyncio.to_thread(self.nodes.search_amap, state)

    async def _run_recommend(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """景点推荐智能体：按偏好反复搜高德，再补介绍和门票。"""
        return await asyncio.to_thread(self.nodes.recommend_attractions, state)

    # ==================== 天气 / 酒店节点 ====================

    async def _run_weather(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """直接查高德天气，不经过模型。"""
        return await asyncio.to_thread(
            self.nodes.fetch_weather,
            {
                "city": state.get("city", ""),
                "start_date": state.get("start_date", ""),
                "end_date": state.get("end_date", ""),
            },
        )

    async def _run_hotels(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """按主图里的日期和天数直接搜酒店。"""
        return await asyncio.to_thread(
            self.nodes.search_hotels,
            {
                "city": state.get("city", ""),
                "accommodation": state.get("accommodation", ""),
                "hotel_budget": state.get("hotel_budget", ""),
                "start_date": state.get("start_date", ""),
                "travel_days": state.get("travel_days", 1),
            },
        )

    async def run_parallel_fetch(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        显式并发执行知识库、高德、天气、酒店（供调试/单测使用）。

        主图通过 fan-out 实现同等并发；本方法用 gather 做对照验证。
        知识库和高德都返回后，再合并成景点候选。
        """
        retrieve_result, amap_result, weather_result, hotel_result = await asyncio.gather(
            self._run_retrieve(state),
            self._run_amap(state),
            self._run_weather(state),
            self._run_hotels(state),
            return_exceptions=True,
        )

        merged: Dict[str, Any] = {"current_phase": "parallel_fetch"}

        for name, result in [
            ("retrieve", retrieve_result),
            ("amap", amap_result),
            ("weather", weather_result),
            ("hotels", hotel_result),
        ]:
            if isinstance(result, Exception):
                merged.setdefault("errors", []).append(f"{name}: {result}")
            elif isinstance(result, dict):
                merged.update(result)

        if not isinstance(amap_result, Exception):
            merged.update(self.nodes.select_and_enrich({**state, **merged}))

        return merged

    # ==================== 主图 ====================

    def build_trip_planner_graph(self) -> StateGraph:
        """构建主旅行规划图。

        流程：
            init → memory → prepare
                 ├─ recommend（ReAct 反复调高德并按偏好点名）─┐
                 ├─ weather                                   ├→ plan → reflect
                 └─ hotels ───────────────────────────────────┘
            规划节点先排好每一天，再调用路线工具按这个顺序算距离。
        """
        graph = StateGraph(TripState)

        graph.add_node("init", self.nodes.extract_trip_intent)
        graph.add_node("memory", self.nodes.query_memory)
        graph.add_node("build_attraction_query", self.nodes.build_attraction_query)

        graph.add_node("recommend", self._run_recommend)
        graph.add_node("weather", self._run_weather)
        graph.add_node("hotels", self._run_hotels)

        graph.add_node("plan", self.nodes.generate_trip_plan)
        graph.add_node("reflect", self.nodes.reflect_trip_plan)
        graph.add_node("save", self.nodes.save_trip_plan)

        graph.add_node("save_version", self.nodes.save_version)
        graph.add_node("dialog", self.nodes.dialog)

        graph.set_entry_point("init")

        graph.add_edge("init", "memory")
        graph.add_edge("memory", "build_attraction_query")

        graph.add_edge("build_attraction_query", "recommend")
        graph.add_edge("build_attraction_query", "weather")
        graph.add_edge("build_attraction_query", "hotels")

        graph.add_edge("recommend", "plan")
        graph.add_edge("weather", "plan")
        graph.add_edge("hotels", "plan")
        graph.add_edge("plan", "reflect")

        graph.add_conditional_edges(
            "reflect",
            self.nodes.route_after_reflect_trip,
            {
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
        """改完计划后回到对话节点，再次 interrupt 等待下一条。"""
        if state.get("dialog_active", False):
            return "continue"
        return "final"

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
    基于 LangGraph 的旅行规划 Agent（知识库、高德、天气、酒店异步并发）
    """

    def __init__(
        self,
        llm_provider: str = "aliyun",
        llm_model: str = None,
        use_redis: bool = True,
        redis_url: str = None,
    ):
        self.nodes = GraphNodes(llm_provider=llm_provider, llm_model=llm_model)
        self.builder = LangGraphBuilder(nodes=self.nodes)
        self.redis_url = redis_url
        self.checkpointer_backend = "none"

        self.checkpointer = None
        if use_redis and not self._setup_redis_checkpointer():
            self._setup_memory_checkpointer()
        elif not use_redis:
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

    def _setup_redis_checkpointer(self) -> bool:
        """把主图暂停点写到 Redis。连不上或无法建索引时返回 False。"""
        url = self.redis_url or redis_url_from_env()
        try:
            import redis
            from redis.backoff import NoBackoff
            from redis.retry import Retry
            from langgraph.checkpoint.redis import RedisSaver

            client = redis.Redis.from_url(
                url,
                socket_timeout=2,
                socket_connect_timeout=2,
                retry=Retry(NoBackoff(), 0),
            )
            client.ping()
            saver = RedisSaver(redis_client=client)
            saver.setup()
            self.checkpointer = saver
            self.checkpointer_backend = "redis"
            print(f"✅ Redis Checkpointer 已连接: {url.split('@')[-1]}")
            return True
        except Exception as e:
            print(f"⚠️  Redis Checkpointer 连接失败，改用内存: {e}")
            return False

    def _setup_memory_checkpointer(self):
        try:
            from langgraph.checkpoint.memory import MemorySaver
            self.checkpointer = MemorySaver()
            self.checkpointer_backend = "memory"
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
            return thread_id or uuid.uuid4().hex
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
        hotel_budget: str = "",
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
            "hotel_budget": hotel_budget or "",
            "preferences": preferences or [],
            "user_id": user_id,
            "free_text_input": free_text_input,
            "dialog_active": False,
            "max_iterations": 3,
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
        hotel_budget: str = "",
        thread_id: str = None,
    ) -> Dict[str, Any]:
        """异步规划旅行（并行子图通过 ainvoke 真正并发）"""
        from app.monitoring import merge_run_config

        input_state = self._build_input_state(
            city, travel_days, start_date, end_date,
            transportation, accommodation, preferences or [],
            user_id, free_text_input, hotel_budget,
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

        from app.monitoring.token_usage import track_llm_usage

        try:
            with track_llm_usage() as usage:
                result = await self.app.ainvoke(input_state, config)
                if isinstance(result, dict):
                    result["llm_usage"] = dict(usage)
        except Exception as e:
            print(f"❌ 图异步执行失败: {e}")
            import traceback
            traceback.print_exc()
            result = self._create_fallback(
                city, travel_days,
                input_state["start_date"], input_state["end_date"],
                preferences,
            )

        return self._format_result(result, thread_id)

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
        hotel_budget: str = "",
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
            hotel_budget=hotel_budget,
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

    def _format_result(self, result: Dict, thread_id: str = None) -> Dict:
        if not isinstance(result, dict):
            result = {}
        paused = read_interrupt(result)
        reply = ""
        if paused and paused.get("reply"):
            reply = str(paused["reply"])
        elif result.get("dialog_reply"):
            reply = str(result.get("dialog_reply"))
        return {
            "city": result.get("city", ""),
            "trip_plan": result.get("trip_plan", ""),
            "trip_plan_parsed": result.get("trip_plan_parsed"),
            "attraction_response": result.get("attraction_response", ""),
            "attraction_candidates": result.get("attraction_candidates") or [],
            "restaurant_candidates": result.get("restaurant_candidates") or [],
            "weather_response": result.get("weather_response", ""),
            "hotel_response": result.get("hotel_response", ""),
            "hotel_candidates": result.get("hotel_candidates") or [],
            "reflection_feedback": result.get("reflection_feedback", ""),
            "iteration_count": result.get("iteration_count", 0),
            "budget": result.get("budget"),
            "budget_total": result.get("budget_total", 0),
            "travel_days": result.get("travel_days", 1),
            "session_id": result.get("session_id") or (paused or {}).get("session_id", ""),
            "current_version_id": result.get("current_version_id", ""),
            "thread_id": thread_id or "",
            "dialog_reply": reply,
            "dialog_changes": result.get("dialog_changes") or [],
            "dialog_active": bool(result.get("dialog_active")),
            "awaiting_input": paused is not None,
            "llm_usage": result.get("llm_usage") or {},
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

    async def aresume_dialog(self, thread_id: str, user_text: str) -> Dict[str, Any]:
        """用同一 thread_id 把用户的修改送回暂停中的主图。"""
        from langgraph.types import Command
        from app.monitoring import merge_run_config

        if not self.checkpointer or not thread_id:
            return {"error": "没有可恢复的规划会话", "thread_id": thread_id or ""}

        config = merge_run_config(
            {"configurable": {"thread_id": thread_id}},
            run_name="langgraph_resume_dialog",
            thread_id=thread_id,
            tags=["langgraph", "dialog"],
        )
        from app.monitoring.token_usage import track_llm_usage

        try:
            with track_llm_usage() as usage:
                result = await self.app.ainvoke(Command(resume=user_text), config)
                if isinstance(result, dict):
                    result["llm_usage"] = dict(usage)
        except Exception as e:
            print(f"❌ 对话恢复失败: {e}")
            return {"error": str(e), "thread_id": thread_id}
        return self._format_result(result, thread_id)

    def resume_dialog(self, thread_id: str, user_text: str) -> Dict[str, Any]:
        """同步恢复对话。已在事件循环中时改到新线程执行。"""
        try:
            asyncio.get_running_loop()
            in_running_loop = True
        except RuntimeError:
            in_running_loop = False

        if not in_running_loop:
            return asyncio.run(self.aresume_dialog(thread_id, user_text))

        import concurrent.futures

        def _run():
            return asyncio.run(self.aresume_dialog(thread_id, user_text))

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(_run).result()

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
