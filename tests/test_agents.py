"""
主图节点、预算和反思循环。不调用模型，不访问高德、飞猪和向量库。

运行: python -m pytest tests/test_agents.py -v
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.orchestrator import LangGraphBuilder
from app.agents.langgraph_nodes import GraphNodes


class _Memory:
    def __init__(self, history=None):
        self.history = history
        self.saved = None

    def compare_plans(self, user_id, city):
        if self.history is not None:
            return self.history
        return {"has_history": False}

    def save_plan(self, **kwargs):
        self.saved = kwargs
        return "plan-1"


def _nodes(history=None) -> GraphNodes:
    nodes = GraphNodes.__new__(GraphNodes)
    nodes.logger = None
    nodes.memory_manager = _Memory(history)
    return nodes


def _state(**overrides) -> dict:
    state = {
        "city": "上海",
        "start_date": "2026-06-01",
        "end_date": "2026-06-02",
        "travel_days": 2,
        "transportation": "公共交通",
        "accommodation": "经济型酒店",
        "preferences": ["历史文化"],
        "user_id": "zhiyuan",
        "free_text_input": "",
        "iteration_count": 0,
        "max_iterations": 3,
    }
    state.update(overrides)
    return state


def _day(index: int, names: list[str], end: str = "21:00") -> dict:
    return {
        "day_index": index,
        "attractions": [{"name": name} for name in names],
        "time_slots": [{"end_time": end}],
    }


def _complete_plan() -> dict:
    return {
        "days": [
            _day(0, ["豫园", "外滩", "南京路步行街"]),
            _day(1, ["上海博物馆", "人民广场", "静安寺"]),
        ]
    }


# ━━━━━━ 意图与查询 ━━━━━━


def test_extract_trip_intent_resets_iteration():
    result = _nodes().extract_trip_intent(_state(iteration_count=2))
    assert result["current_phase"] == "init"
    assert result["iteration_count"] == 0
    assert result["max_iterations"] == 3
    assert result["errors"] == []


def test_build_attraction_query_uses_first_preference():
    result = _nodes().build_attraction_query(_state(preferences=["历史文化", "美食"]))
    assert result["query"] == "上海的历史文化相关景点"
    assert result["current_phase"] == "attractions"


def test_build_attraction_query_defaults_when_preference_missing():
    result = _nodes().build_attraction_query(_state(preferences=[]))
    assert result["query"] == "上海的景点相关景点"


@pytest.mark.parametrize("preference", ["历史文化", "自然风光", "美食"])
def test_query_keeps_each_preference(preference):
    result = _nodes().build_attraction_query(_state(preferences=[preference]))
    assert preference in result["query"]


# ━━━━━━ 记忆 ━━━━━━


def test_memory_skips_lookup_without_user():
    result = _nodes().query_memory(_state(user_id=""))
    assert result["visited_attractions"] == []
    assert result["memory_info"] == ""


def test_memory_marks_first_visit():
    result = _nodes().query_memory(_state())
    assert "第一次访问上海" in result["memory_info"]
    assert result["visited_attractions"] == []


def test_memory_keeps_visited_attractions():
    history = {
        "has_history": True,
        "plan_count": 2,
        "visited_attractions": ["外滩", "豫园"],
        "preferences": ["历史文化"],
    }
    result = _nodes(history).query_memory(_state())
    assert result["visited_attractions"] == ["外滩", "豫园"]
    assert "外滩" in result["memory_info"]


# ━━━━━━ 预算 ━━━━━━


def test_budget_uses_candidate_prices():
    nodes = _nodes()
    state = _state(
        attraction_candidates=[
            {"name": "上海动物园", "ticket_price": 40},
            {"name": "东方明珠", "ticket_price": 199},
        ],
        hotel_response="静安酒店 480元/晚",
        travel_days=2,
    )
    budget = nodes._calculate_budget(_complete_plan() | {
        "days": [
            _day(0, ["上海动物园"]),
            _day(1, ["东方明珠"]),
        ]
    }, state)
    assert budget["total_attractions"] == 239
    assert budget["total_hotels"] == 480
    assert budget["total_meals"] == 240
    assert budget["total"] == (
        budget["total_attractions"]
        + budget["total_hotels"]
        + budget["total_meals"]
        + budget["total_transportation"]
    )
    assert budget["currency"] == "CNY"


def test_hotel_price_falls_back_when_text_has_no_price():
    assert _nodes()._estimate_hotel_price("") == 300
    assert _nodes()._estimate_hotel_price("评分 4.8，暂无报价") == 300


def test_save_trip_plan_writes_budget():
    nodes = _nodes()
    state = _state(
        trip_plan_parsed=_complete_plan(),
        attraction_candidates=[{"name": "豫园", "ticket_price": 40}],
        hotel_response="300元",
    )
    result = nodes.save_trip_plan(state)
    assert result["budget_total"] == result["budget"]["total"]
    assert result["budget_total"] > 0
    assert nodes.memory_manager.saved["city"] == "上海"
    assert "豫园" in nodes.memory_manager.saved["attractions"]


# ━━━━━━ 反思循环 ━━━━━━


def test_reflect_accepts_complete_plan():
    state = _state(
        trip_plan_parsed=_complete_plan(),
        attraction_response="候选景点" * 20,
    )
    result = _nodes().reflect_trip_plan(state)
    assert result["current_phase"] == "final"
    assert "完整" in result["reflection_feedback"]


def test_reflect_retries_short_plan_within_limit():
    state = _state(
        travel_days=2,
        trip_plan_parsed={"days": [_day(0, ["外滩", "豫园", "南京路"])]},
        attraction_response="候选景点" * 20,
        iteration_count=0,
    )
    result = _nodes().reflect_trip_plan(state)
    assert result["current_phase"] == "reflect"
    assert result["iteration_count"] == 1
    assert _nodes().route_after_reflect_trip({**state, **result}) == "plan"


def test_reflect_stops_at_max_iterations():
    state = _state(
        travel_days=2,
        trip_plan_parsed={"days": []},
        attraction_response="候选景点" * 20,
        iteration_count=3,
        max_iterations=3,
    )
    result = _nodes().reflect_trip_plan(state)
    assert result["current_phase"] == "final"
    assert "最大重试次数" in result["reflection_feedback"]
    assert _nodes().route_after_reflect_trip({**state, **result}) == "final"


def test_reflect_retry_includes_scheduled_names_and_times():
    state = _state(
        travel_days=1,
        trip_plan_parsed={
            "days": [{
                "day_index": 0,
                "attractions": [{"name": "外滩"}],
                "meals": [
                    {"type": "lunch", "name": "南翔馒头店"},
                    {"type": "dinner", "name": "老正兴"},
                ],
                "time_slots": [
                    {"activity_type": "景点游览", "activity_name": "外滩", "start_time": "09:00", "end_time": "10:30"},
                    {"activity_type": "交通", "activity_name": "地铁", "start_time": "10:30", "end_time": "11:00"},
                    {"activity_type": "用餐", "activity_name": "南翔馒头店", "start_time": "11:00", "end_time": "12:00"},
                ],
            }]
        },
        attraction_response="候选景点" * 20,
        iteration_count=0,
    )
    result = _nodes().reflect_trip_plan(state)
    feedback = result["reflection_feedback"]
    assert result["current_phase"] == "reflect"
    assert "1. 09:00-10:30 景点 外滩" in feedback
    assert "2. 11:00-12:00 用餐 南翔馒头店" in feedback
    assert "路程 10:30-11:00" in feedback


class _ReflectLLM:
    def __init__(self, passed: bool, note: str):
        self.passed = passed
        self.note = note
        self.prompts = []

    def bind(self, **_kwargs):
        return self

    def with_structured_output(self, schema, **_kwargs):
        self.schema = schema
        return self

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return {"parsed": self.schema(passed=self.passed, note=self.note), "raw": None}


def test_model_reflection_sees_the_timetable_and_can_reject():
    nodes = _nodes()
    nodes.llm = _ReflectLLM(False, "第1天没有晚饭")
    state = _state(
        travel_days=1,
        trip_plan_parsed={"days": [_day(0, ["外滩", "豫园"], end="21:00")]},
        attraction_response="候选景点" * 20,
    )
    result = nodes.reflect_trip_plan(state)
    assert result["current_phase"] == "reflect"
    assert "第1天没有晚饭" in result["reflection_feedback"]
    assert "外滩" in result["reflection_feedback"]
    sent = nodes.llm.prompts[0][1]["content"]
    assert "外滩" in sent
    assert "收尾 21:00" in sent


def test_reflect_does_not_retry_when_candidates_are_short():
    state = _state(
        trip_plan_parsed=_complete_plan(),
        attraction_response="太短",
    )
    result = _nodes().reflect_trip_plan(state)
    assert result["current_phase"] == "final"
    assert "iteration_count" not in result
    assert _nodes().route_after_reflect_trip({**state, **result}) == "final"


# ━━━━━━ 主图 ━━━━━━


class _FetchStub:
    def retrieve_knowledge(self, state):
        raise RuntimeError("知识库不可用")

    def search_amap(self, state):
        return {"amap_pois": [{"name": "外滩"}]}

    def fetch_weather(self, state):
        return {"weather_response": "晴"}

    def search_hotels(self, state):
        return {"hotel_response": "200元"}

    def select_and_enrich(self, state):
        return {"attraction_candidates": state.get("amap_pois") or []}


def test_parallel_fetch_keeps_other_results_when_one_fails():
    builder = LangGraphBuilder(nodes=_FetchStub())
    merged = asyncio.run(builder.run_parallel_fetch(_state()))
    assert any("retrieve" in item for item in merged["errors"])
    assert merged["weather_response"] == "晴"
    assert merged["hotel_response"] == "200元"
    assert merged["attraction_candidates"] == [{"name": "外滩"}]


def test_trip_graph_fans_out_and_loops_reflect_to_plan():
    builder = LangGraphBuilder(nodes=_nodes())
    compiled = builder.build_trip_planner_graph().compile()
    edges = {(edge.source, edge.target) for edge in compiled.get_graph().edges}
    assert ("build_attraction_query", "recommend") in edges
    assert ("build_attraction_query", "weather") in edges
    assert ("build_attraction_query", "hotels") in edges
    assert ("recommend", "plan") in edges
    assert ("weather", "plan") in edges
    assert ("hotels", "plan") in edges
    assert ("plan", "reflect") in edges
    assert ("reflect", "plan") in edges
    assert ("reflect", "save") in edges


def test_pipeline_reaches_save_without_llm():
    nodes = _nodes()
    state = _state()
    state.update(nodes.extract_trip_intent(state))
    state.update(nodes.query_memory(state))
    state.update(nodes.build_attraction_query(state))
    state["trip_plan_parsed"] = _complete_plan()
    state["attraction_response"] = "候选景点" * 20
    state["attraction_candidates"] = [{"name": "豫园", "ticket_price": 40}]
    state["hotel_response"] = "360元"
    state.update(nodes.reflect_trip_plan(state))
    assert nodes.route_after_reflect_trip(state) == "final"
    saved = nodes.save_trip_plan(state)
    assert saved["current_phase"] == "final"
    assert saved["budget"]["total_attractions"] == 40


class _Poi:
    def __init__(self, name, address, location, poi_type):
        self.name = name
        self.address = address
        self.location = location
        self.type = poi_type


def test_recommend_agent_searches_again_and_keeps_structured_fields():
    nodes = _nodes()

    def search_poi(keywords, city):
        if keywords == "博物馆":
            return [_Poi("上海博物馆", "人民大道", "121.47,31.23", "博物馆")]
        if keywords == "古镇":
            return [_Poi("新场古镇", "浦东", "121.64,31.05", "古镇")]
        if keywords == "午餐":
            return [_Poi("南翔馒头店", "豫园路", "121.49,31.22", "餐饮服务;中餐厅")]
        if keywords == "老字号":
            return [_Poi("老正兴", "福州路", "121.48,31.23", "餐饮服务;中餐厅")]
        return []

    nodes.amap_service = type("Amap", (), {"search_poi": staticmethod(search_poi)})()

    def runner(state, tool, restaurant_tool):
        first = tool._run("上海", "博物馆")
        assert "上海博物馆" in first
        assert "不够" not in first or "新增" in first
        second = tool._run("上海", "古镇")
        assert "新场古镇" in second
        same = tool._run("上海", "博物馆")
        assert "没有新景点" in same
        lunch = restaurant_tool._run("上海", "午餐", "lunch")
        assert "南翔馒头店" in lunch
        dinner = restaurant_tool._run("上海", "老字号", "dinner")
        assert "老正兴" in dinner
        return '{"names": ["上海博物馆", "新场古镇"], "lunch": ["南翔馒头店"], "dinner": ["老正兴"]}'

    result = nodes.recommend_attractions(
        _state(preferences=["历史文化"], travel_days=1),
        agent_runner=runner,
        ticket_lookup=lambda city, name, depart_date: {"price": None, "jump_url": ""},
    )
    by_name = {item["name"]: item for item in result["attraction_candidates"]}
    assert "上海博物馆" in by_name
    assert "新场古镇" in by_name
    assert by_name["上海博物馆"]["description"]
    assert by_name["上海博物馆"]["ticket_source"] == "knowledge_base"
    meals = {item["meal"]: item for item in result["restaurant_candidates"]}
    assert meals["lunch"]["name"] == "南翔馒头店"
    assert meals["lunch"]["address"] == "豫园路"
    assert meals["lunch"]["location"] == "121.49,31.22"
    assert meals["lunch"]["duration_minutes"] == 60
    assert meals["dinner"]["name"] == "老正兴"
    assert "老正兴" in result["restaurant_response"]


def test_route_tool_uses_the_day_order():
    nodes = _nodes()
    seen = []

    def fake(attractions, city="", transportation=""):
        seen.append([item.get("name") for item in attractions])
        return [{
            "start_point": attractions[0]["name"],
            "end_point": attractions[1]["name"],
            "duration": 15,
            "distance": 1200,
            "transport_mode": "地铁",
            "route_detail": "地铁2号线",
        }]

    nodes._generate_route_segments_from_attractions = fake
    parsed = {
        "days": [{
            "attractions": [
                {"name": "外滩", "location": "121.49,31.24"},
                {"name": "豫园", "location": "121.49,31.23"},
            ]
        }]
    }
    nodes._calculate_routes_for_plan(parsed, _state())
    assert seen == [["外滩", "豫园"]]
    assert parsed["days"][0]["route_segments"][0]["duration"] == 15
    assert "外滩 -> 豫园" in parsed["days"][0]["route_summary"]


def test_route_follows_timeline_and_skips_unplaced_spots():
    nodes = _nodes()
    seen = []

    def fake(attractions, city="", transportation=""):
        seen.append([item.get("name") for item in attractions])
        return [{
            "start_point": attractions[0]["name"],
            "end_point": attractions[-1]["name"],
            "duration": 15,
            "distance": 1200,
            "transport_mode": "地铁",
            "route_detail": "地铁",
        }]

    nodes._generate_route_segments_from_attractions = fake
    parsed = {
        "days": [{
            "hotel": {"name": "全季酒店外滩", "location": "121.49,31.24"},
            "attractions": [
                {"name": "外滩", "location": "121.49,31.24"},
                {"name": "没排上", "location": "121.50,31.20"},
            ],
            "meals": [
                {"type": "lunch", "name": "南翔馒头店", "location": "121.49,31.22"},
                {"type": "dinner", "name": "晚餐"},
            ],
            "time_slots": [
                {"activity_type": "景点游览", "activity_name": "外滩"},
                {"activity_type": "用餐", "activity_name": "南翔馒头店"},
                {"activity_type": "用餐", "activity_name": "晚餐"},
            ],
        }]
    }
    nodes._calculate_routes_for_plan(parsed, _state())
    assert seen[0] == ["全季酒店外滩", "外滩", "南翔馒头店", "全季酒店外滩"]
