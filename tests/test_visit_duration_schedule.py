"""当天时间表：知识库游览分钟、10 分钟路程缓冲、酒店进出、用餐和开放时间。"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.agents.dialog_agent import DialogAgent
from app.agents.langgraph_nodes import GraphNodes
from app.rag.hybrid_rag import HybridRAG, HybridRAGConfig


def test_string_price_does_not_break_free_intent_boost():
    rag = HybridRAG.__new__(HybridRAG)
    rag.config = HybridRAGConfig()
    docs = [
        {"metadata": {"price": "30元", "category": "", "hot": ""}, "score": 1.0},
        {"metadata": {"price": "免费", "category": "", "hot": ""}, "score": 1.0},
    ]
    ranked = rag._boost_by_intent(docs, {"categories": [], "price_filter": 0})
    by_price = {item["metadata"]["price"]: item["_boosted_score"] for item in ranked}
    assert by_price["30元"] == 0.5
    assert by_price["免费"] == 2.0


def test_four_hour_visit_spans_four_hours():
    nodes = GraphNodes.__new__(GraphNodes)
    slots = nodes._generate_time_slots_from_attractions(
        [
            {"name": "南京路步行街", "visit_duration": 60},
            {"name": "上海迪士尼度假区", "visit_duration": 240},
        ],
        [{"duration": 62, "route_detail": "地铁", "transport_mode": "地铁2号线"}],
    )
    travel = next(slot for slot in slots if slot["activity_type"] == "交通")
    disney = next(slot for slot in slots if slot["activity_name"] == "上海迪士尼度假区")
    assert travel["start_time"] == "10:00"
    assert travel["end_time"] == "11:12"
    assert disney["start_time"] == "11:12"
    assert disney["end_time"] == "15:12"


def test_short_transit_keeps_real_minutes():
    nodes = GraphNodes.__new__(GraphNodes)
    slots = nodes._generate_time_slots_from_attractions(
        [
            {"name": "浦江游览", "visit_duration": 90},
            {"name": "外滩", "visit_duration": 60},
        ],
        [{"duration": 20, "route_detail": "公共交通约20分钟", "transport_mode": "公共交通"}],
    )
    assert slots[0]["start_time"] == "09:00"
    assert slots[0]["end_time"] == "10:30"
    assert slots[1]["start_time"] == "10:30"
    assert slots[1]["end_time"] == "11:00"
    assert "约20分钟" in slots[1]["description"]
    assert slots[2]["start_time"] == "11:00"
    assert slots[2]["end_time"] == "12:00"


def test_visit_that_cannot_finish_is_omitted():
    nodes = GraphNodes.__new__(GraphNodes)
    slots = nodes._generate_time_slots_from_attractions(
        [{"name": "超长景点", "visit_duration": 24 * 60}],
        [],
    )
    assert slots == []
    slots = nodes._generate_time_slots_from_attractions(
        [{"name": "正好到关门", "visit_duration": 12 * 60}],
        [],
    )
    assert slots[0]["start_time"] == "09:00"
    assert slots[0]["end_time"] == "21:00"


def test_short_day_is_filled_until_21():
    nodes = GraphNodes.__new__(GraphNodes)
    nodes.logger = None

    def suggest(city, existing, remaining_hours, last_name):
        # 剩余不到 1 小时时补一段 20 分钟的游览，让当天在 21:00 结束
        minutes = 20 if remaining_hours <= 1 else 120
        return {
            "name": f"补点{len(existing)}",
            "visit_duration": minutes,
            "address": "",
            "location": "",
        }

    nodes._suggest_filler_attraction = suggest
    nodes._generate_route_segments_from_attractions = lambda *args, **kwargs: [
        {"duration": 30, "route_detail": "地铁", "transport_mode": "地铁"}
    ]
    day = {
        "day_index": 0,
        "attractions": [
            {"name": "南京路步行街", "visit_duration": 60},
            {"name": "上海迪士尼度假区", "visit_duration": 240},
        ],
    }
    used = {"南京路步行街", "上海迪士尼度假区"}
    nodes._fill_day_until_close(day, "上海", "公共交通", used)
    assert len([item for item in day["attractions"] if item.get("name")]) <= 6
    assert any(item["name"].startswith("补点") for item in day["attractions"])
    assert all(slot["end_time"] <= "21:00" for slot in day["time_slots"])


def test_pool_filler_uses_unused_candidates_without_another_search():
    nodes = GraphNodes.__new__(GraphNodes)
    nodes.logger = None

    def suggest(*_args, **_kwargs):
        raise AssertionError("候选池还有景点时不再搜索")

    nodes._suggest_filler_attraction = suggest
    day = {
        "day_index": 0,
        "attractions": [{"name": "外滩", "visit_duration": 90}],
        "meals": [
            {"type": "lunch", "name": "午饭", "duration_minutes": 60},
            {"type": "dinner", "name": "晚饭", "duration_minutes": 60},
        ],
    }
    pool = [
        {"name": "外滩", "visit_duration": 90},
        {"name": "排不下", "visit_duration": 180, "open_time": "09:00-10:00"},
        {"name": "豫园", "visit_duration": 90},
        {"name": "人民广场", "visit_duration": 90},
        {"name": "新天地", "visit_duration": 90},
        {"name": "武康路", "visit_duration": 90},
    ]
    nodes._fill_day_until_close(day, "上海", "公共交通", {"外滩"}, pool)
    names = [item["name"] for item in day["attractions"]]
    assert "豫园" in names
    assert len(names) >= 4
    assert all(slot["end_time"] <= "21:00" for slot in day["time_slots"])


def test_exact_kb_lookup_keeps_disney_duration():
    from app.rag.chroma_store import ChromaVectorStore

    store = ChromaVectorStore()
    hits = store.find_by_title("上海迪士尼度假区", city="上海")
    assert hits, "知识库里应有上海迪士尼度假区"
    assert hits[0]["metadata"]["visit_duration"] == 240

    class Retriever:
        def __init__(self, vector_store):
            self.vector_store = vector_store

    class Rag:
        def __init__(self, vector_store):
            self.retriever = Retriever(vector_store)

    agent = DialogAgent.__new__(DialogAgent)
    agent.rag = Rag(store)
    info = agent._lookup_attraction_exact("上海迪士尼度假区", "上海")
    assert info["visit_duration"] == 240
    assert info["location"] == "121.667662,31.146785"

    class Schedule:
        def _generate_route_segments_from_attractions(self, attractions, city, transportation):
            return [{"duration": 62, "route_detail": "地铁", "transport_mode": "地铁2号线"}]

        def _generate_time_slots_from_attractions(self, attractions, segments):
            return GraphNodes._generate_time_slots_from_attractions(
                GraphNodes.__new__(GraphNodes), attractions, segments
            )

        def _fill_day_until_close(self, day, city, transportation, used):
            attractions = day.get("attractions") or []
            segments = self._generate_route_segments_from_attractions(
                attractions, city, transportation
            )
            day["route_segments"] = segments
            day["time_slots"] = self._generate_time_slots_from_attractions(attractions, segments)

    agent._schedule_nodes = Schedule()
    day = {
        "attractions": [
            {"name": "南京路步行街", "visit_duration": 60},
            {
                "name": info["name"],
                "visit_duration": info["visit_duration"],
                "location": info["location"],
                "address": info["address"],
            },
        ]
    }
    agent._replan_day_schedule(day, "上海", "公共交通")
    disney = next(slot for slot in day["time_slots"] if slot["activity_name"] == "上海迪士尼度假区")
    assert disney["start_time"] == "11:12"
    assert disney["end_time"] == "15:12"


def test_short_distance_is_walk_and_failed_route_is_not_a_bus():
    walk = GraphNodes._qualify_route_leg(22, 20, ["地铁2号线"], "公共交通", "transit")
    assert walk["transport_mode"] == "步行"
    assert walk["duration"] == 5
    assert "22米" in walk["route_detail"]

    chain = GraphNodes._qualify_route_leg(
        26216, 78, ["地铁2号线", "地铁11号线"], "公共交通", "transit"
    )
    assert chain["route_detail"] == "地铁2号线 → 地铁11号线"
    assert chain["duration"] == 78

    unknown = GraphNodes._qualify_route_leg(0, 0, [], "公共交通", "transit")
    assert unknown["duration"] == 5
    assert unknown["transport_mode"] == "步行"
    assert "步行" in unknown["route_detail"]


def test_nearest_neighbor_orders_from_hotel():
    nodes = GraphNodes.__new__(GraphNodes)
    ordered = nodes._order_nearest(
        [
            {"name": "A", "location": "121.50,31.20"},
            {"name": "B", "location": "121.51,31.20"},
            {"name": "C", "location": "121.80,31.20"},
        ],
        {"name": "酒店", "location": "121.80,31.20"},
    )
    assert [item["name"] for item in ordered] == ["C", "B", "A"]


def _nodes_with_fixed_route(minutes=10):
    nodes = GraphNodes.__new__(GraphNodes)
    nodes.logger = None

    def route(attractions, city="", transportation=""):
        return [{
            "duration": minutes,
            "distance": 2000,
            "route_detail": "地铁2号线",
            "transport_mode": "地铁2号线",
            "start_point": attractions[0].get("name"),
            "end_point": attractions[1].get("name"),
        }]

    nodes._generate_route_segments_from_attractions = route
    return nodes


def test_day_drops_visit_that_cannot_finish():
    nodes = _nodes_with_fixed_route()
    day = {
        "day_index": 0,
        "attractions": [
            {"name": "上午", "visit_duration": 600},
            {"name": "下午", "visit_duration": 200},
        ],
    }
    nodes._schedule_day_clock(day, "上海", "公共交通")
    names = [item["name"] for item in day["attractions"]]
    assert names == ["上午"]
    assert day["time_slots"][0]["end_time"] == "19:00"
    assert all(slot["end_time"] <= "21:00" for slot in day["time_slots"])


def test_lunch_is_inserted_between_attractions():
    nodes = _nodes_with_fixed_route(10)
    day = {
        "day_index": 0,
        "meals": [
            {"type": "lunch", "name": "南翔馒头店", "address": "豫园路"},
            {"type": "dinner", "name": "老正兴", "address": "福州路"},
        ],
        "attractions": [
            {"name": "上午街", "visit_duration": 150},
            {"name": "小店", "visit_duration": 30},
        ],
    }
    nodes._schedule_day_clock(day, "上海", "公共交通")
    lunch = next(slot for slot in day["time_slots"] if slot["activity_name"] == "南翔馒头店")
    shop = next(slot for slot in day["time_slots"] if slot["activity_name"] == "小店")
    assert lunch["start_time"] == "11:30"
    assert lunch["end_time"] == "12:30"
    assert lunch["description"] == "豫园路"
    assert shop["start_time"] == "12:50"
    assert shop["end_time"] == "13:20"
    assert all(slot["activity_name"] != "早餐" for slot in day["time_slots"])


def test_located_lunch_is_a_stop_on_the_path():
    nodes = _nodes_with_fixed_route(10)
    day = {
        "day_index": 0,
        "meals": [
            {"type": "lunch", "name": "南翔馒头店", "address": "豫园路", "location": "121.49,31.22"},
        ],
        "attractions": [
            {"name": "上午街", "visit_duration": 150, "location": "121.48,31.23"},
            {"name": "小店", "visit_duration": 30, "location": "121.49,31.23"},
        ],
    }
    nodes._schedule_day_clock(day, "上海", "公共交通")
    lunch = next(slot for slot in day["time_slots"] if slot["activity_name"] == "南翔馒头店")
    shop = next(slot for slot in day["time_slots"] if slot["activity_name"] == "小店")
    assert lunch["start_time"] == "11:50"
    assert lunch["end_time"] == "12:50"
    assert shop["start_time"] == "13:10"
    assert shop["end_time"] == "13:40"
    assert [seg["end_point"] for seg in day["route_segments"]] == ["南翔馒头店", "小店"]


def test_long_visit_still_gets_meals_around_it():
    nodes = _nodes_with_fixed_route(10)
    day = {
        "day_index": 0,
        "attractions": [
            {"name": "上海迪士尼度假区", "visit_duration": 240},
        ],
    }
    nodes._schedule_day_clock(day, "上海", "公共交通")
    assert day["time_slots"][0]["start_time"] == "09:00"
    assert day["time_slots"][0]["end_time"] == "13:00"
    lunch = next(slot for slot in day["time_slots"] if slot["activity_name"] == "午餐")
    dinner = next(slot for slot in day["time_slots"] if slot["activity_name"] == "晚餐")
    assert lunch["start_time"] == "13:00"
    assert lunch["end_time"] == "14:00"
    assert dinner["start_time"] == "17:00"
    assert dinner["end_time"] == "18:00"


def test_visit_that_covers_lunch_eats_first():
    nodes = _nodes_with_fixed_route(10)
    day = {
        "day_index": 0,
        "meals": [
            {"type": "lunch", "name": "南翔馒头店", "address": "豫园路"},
            {"type": "dinner", "name": "老正兴", "address": "福州路"},
        ],
        "attractions": [
            {"name": "上午街", "visit_duration": 90},
            {"name": "豫园", "visit_duration": 120},
        ],
    }
    nodes._schedule_day_clock(day, "上海", "公共交通")
    lunch = next(slot for slot in day["time_slots"] if slot["activity_name"] == "南翔馒头店")
    garden = next(slot for slot in day["time_slots"] if slot["activity_name"] == "豫园")
    dinner = next(slot for slot in day["time_slots"] if slot["activity_name"] == "老正兴")
    assert lunch["start_time"] == "11:30"
    assert lunch["end_time"] == "12:30"
    assert garden["start_time"] == "12:50"
    assert garden["end_time"] == "14:50"
    assert dinner["start_time"] == "17:00"
    assert dinner["end_time"] == "18:00"


def test_adjacent_places_are_a_five_minute_walk():
    nodes = GraphNodes.__new__(GraphNodes)
    nodes.logger = None

    def explode(*_args, **_kwargs):
        raise AssertionError("直线距离很近时不应再请求高德")

    nodes.amap_service = type("Amap", (), {"route_direction": staticmethod(explode)})()
    segments = nodes._generate_route_segments_from_attractions(
        [
            {"name": "上海城隍庙", "location": "121.492000,31.226000"},
            {"name": "上海豫园", "location": "121.492300,31.227200"},
        ],
        city="上海",
        transportation="公共交通",
    )
    assert segments[0]["transport_mode"] == "步行"
    assert segments[0]["duration"] == 5
    assert 0 < segments[0]["distance"] < 500


def test_hotel_legs_bookend_the_day():
    nodes = _nodes_with_fixed_route(20)
    day = {
        "day_index": 0,
        "hotel": {"name": "全季酒店外滩", "location": "121.49,31.24"},
        "attractions": [
            {"name": "外滩", "visit_duration": 60, "location": "121.50,31.24"},
        ],
    }
    nodes._schedule_day_clock(day, "上海", "公共交通")
    assert day["time_slots"][0]["activity_type"] == "交通"
    assert day["time_slots"][0]["start_time"] == "09:00"
    assert day["time_slots"][0]["end_time"] == "09:30"
    visit = next(slot for slot in day["time_slots"] if slot["activity_name"] == "外滩")
    assert visit["start_time"] == "09:30"
    assert visit["end_time"] == "10:30"
    assert day["time_slots"][-1]["activity_type"] == "交通"
    assert day["time_slots"][-1]["start_time"] == "10:30"
    assert day["time_slots"][-1]["end_time"] == "11:00"
    assert day["route_segments"][0]["start_point"] == "全季酒店外滩"
    assert day["route_segments"][-1]["end_point"] == "全季酒店外滩"


def test_night_spot_is_not_scheduled_in_the_morning():
    nodes = _nodes_with_fixed_route(10)
    day = {
        "day_index": 0,
        "attractions": [
            {"name": "白天", "visit_duration": 480},
            {"name": "外滩夜景", "visit_duration": 60, "open_time": "19:00-22:00"},
        ],
    }
    nodes._schedule_day_clock(day, "上海", "公共交通")
    night = next(slot for slot in day["time_slots"] if slot["activity_name"] == "外滩夜景")
    assert night["start_time"] == "19:00"
    assert night["end_time"] == "20:00"

    only_night = {
        "day_index": 0,
        "attractions": [
            {"name": "外滩夜景", "visit_duration": 60, "open_time": "19:00-22:00"},
        ],
    }
    nodes._schedule_day_clock(only_night, "上海", "公共交通")
    assert not any(
        slot.get("activity_type") == "景点游览" and slot.get("start_time", "99:99") < "19:00"
        for slot in only_night["time_slots"]
    )


def test_closes_before_visit_finishes_is_omitted():
    nodes = GraphNodes.__new__(GraphNodes)
    nodes.logger = None
    day = {
        "day_index": 0,
        "attractions": [
            {"name": "早关门", "visit_duration": 420, "open_time": "09:00-15:00"},
        ],
    }
    nodes._schedule_day_clock(day, "上海", "公共交通")
    assert day["time_slots"] == []
    assert day["attractions"] == []
