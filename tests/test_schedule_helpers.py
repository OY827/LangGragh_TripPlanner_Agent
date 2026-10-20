"""日程、开放时间、路线解析等本地工具函数。不调用模型和外部接口。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.agents.langgraph_nodes import GraphNodes


class _Route:
    def __init__(self, paths=None, transits=None, distance="", duration=""):
        self.paths = paths
        self.transits = transits
        self.distance = distance
        self.duration = duration


class _Amap:
    def __init__(self, route=None, pois=None, error=False):
        self.route = route
        self.pois = pois or []
        self.error = error
        self.calls = []

    def route_direction(self, origin, destination, mode, city):
        self.calls.append((origin, destination, mode, city))
        if self.error:
            raise RuntimeError("route down")
        return self.route

    def search_poi(self, name, city, nearby):
        if self.error:
            raise RuntimeError("poi down")
        return self.pois


class _Poi:
    def __init__(self, name, lon, lat, address=""):
        self.name = name
        self.address = address
        self.location = type("Loc", (), {"longitude": lon, "latitude": lat})()


def _nodes() -> GraphNodes:
    nodes = GraphNodes.__new__(GraphNodes)
    nodes.logger = None
    nodes.amap_service = _Amap()
    return nodes


def _fixed_route(nodes: GraphNodes, minutes: int = 10) -> None:
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


# ━━━━━━ 时间与坐标 ━━━━━━


def test_visit_minutes_rejects_blank_and_zero():
    assert GraphNodes._visit_minutes(None) == 120
    assert GraphNodes._visit_minutes("0") == 120
    assert GraphNodes._visit_minutes("90.0") == 90


def test_clock_helpers_round_trip():
    assert GraphNodes._format_hhmm(9 * 60 + 5) == "09:05"
    assert GraphNodes._parse_hhmm("21:00") == 21 * 60
    assert GraphNodes._parse_hhmm("bad") == -1
    assert GraphNodes._day_end_hour({"time_slots": [{"end_time": "18:40"}]}) == 18
    assert GraphNodes._day_end_minutes({}) == GraphNodes.DAY_START_MIN


def test_location_text_accepts_string_and_pair():
    assert GraphNodes._location_text({"location": " 121.5,31.2 "}) == "121.5,31.2"
    assert GraphNodes._location_text({"longitude": 121.4, "latitude": 31.1}) == "121.4,31.1"
    assert GraphNodes._location_text({"location": "外滩"}) == ""
    assert GraphNodes._coords({"location": "121.5,31.2"}) == (121.5, 31.2)
    assert GraphNodes._coords({"location": "nope"}) is None


def test_coerce_attraction_fills_defaults():
    nodes = _nodes()
    assert nodes._coerce_attraction("  外滩 ")["visit_duration"] == 120
    coerced = nodes._coerce_attraction({
        "name": "豫园",
        "visit_duration": 0,
        "longitude": 121.49,
        "latitude": 31.23,
    })
    assert coerced["visit_duration"] == 120
    assert coerced["location"] == "121.49,31.23"
    assert nodes._coerce_attraction({"address": "无名称"}) is None


def test_enrich_from_kb_fills_only_missing_fields():
    nodes = _nodes()

    class Store:
        def find_by_title(self, name, city="", limit=1):
            return [{
                "metadata": {
                    "open_time": "09:00-17:00",
                    "location": "121.48,31.22",
                    "address": "人民大道",
                    "visit_duration": 150,
                }
            }]

    nodes.retriever = type("R", (), {"vector_store": Store()})()
    filled = nodes._enrich_from_kb({"name": "上海博物馆", "location": "1,2"}, "上海")
    assert filled["open_time"] == "09:00-17:00"
    assert filled["location"] == "1,2"
    assert filled["visit_duration"] == 150
    assert nodes._lookup_kb_meta("", "上海") == {}


def test_lookup_kb_returns_empty_when_finder_fails():
    nodes = _nodes()

    class Store:
        def find_by_title(self, name, city="", limit=1):
            raise RuntimeError("down")

    nodes.retriever = type("R", (), {"vector_store": Store()})()
    assert nodes._lookup_kb_meta("外滩", "上海") == {}


# ━━━━━━ 酒店锚点与开放时间 ━━━━━━


@pytest.mark.parametrize("name", ["", "经济型", "酒店", "舒适型"])
def test_generic_stay_is_not_an_anchor(name):
    assert GraphNodes._is_generic_stay(name) is True
    assert _nodes()._resolve_hotel_anchor({"hotel": {"name": name}}, "上海") is None


def test_hotel_anchor_uses_existing_coordinates():
    anchor = _nodes()._resolve_hotel_anchor(
        {"hotel": "全季酒店外滩"},
        "上海",
    )
    assert anchor is None
    anchor = _nodes()._resolve_hotel_anchor(
        {"hotel": {"name": "全季酒店外滩", "location": "121.49,31.24", "address": "中山东路"}},
        "上海",
    )
    assert anchor["location"] == "121.49,31.24"


def test_hotel_anchor_geocodes_once_and_caches():
    nodes = _nodes()
    nodes.amap_service = _Amap(pois=[_Poi("全季酒店(外滩店)", 121.49, 31.24, "中山东路")])
    first = nodes._resolve_hotel_anchor({"hotel": {"name": "全季酒店外滩"}}, "上海")
    nodes.amap_service.pois = []
    second = nodes._resolve_hotel_anchor({"hotel": {"name": "全季酒店外滩"}}, "上海")
    assert first["location"] == "121.49,31.24"
    assert second == first


def test_open_window_picks_longest_overlap_with_day():
    assert GraphNodes._open_window("全天开放") == (None, None)
    assert GraphNodes._open_window("22:00-23:30") == (None, None)
    assert GraphNodes._open_window("10:00-12:00，13:00-20:00") == (13 * 60, 20 * 60)
    assert GraphNodes._overlap(11 * 60, 12 * 60, GraphNodes.LUNCH_START, GraphNodes.LUNCH_END) == 30


def test_travel_block_adds_buffer_or_estimates_a_short_walk():
    nodes = _nodes()
    assert nodes._travel_block(None)[0] == 5
    minutes, _name, desc = nodes._travel_block({
        "duration": 18,
        "transport_mode": "地铁2号线",
        "route_detail": "乘坐地铁2号线",
    })
    assert minutes == 28
    assert "另留10分钟缓冲" in desc
    unknown, name, text = nodes._travel_block({"duration": 0, "route_detail": ""})
    assert unknown == 5
    assert name == "步行"
    assert "步行" in text


# ━━━━━━ 排序、用餐、当天组装 ━━━━━━


def test_nearest_neighbor_keeps_points_without_coordinates_last():
    ordered = _nodes()._order_nearest(
        [
            {"name": "无坐标"},
            {"name": "远", "location": "121.80,31.20"},
            {"name": "近", "location": "121.51,31.20"},
        ],
        {"name": "酒店", "location": "121.50,31.20"},
    )
    assert [item["name"] for item in ordered] == ["近", "远", "无坐标"]


def test_meals_count_when_overlap_reaches_45_minutes():
    nodes = _nodes()
    lunch, dinner = nodes._note_meals(11 * 60 + 30, 12 * 60 + 20, False, False)
    assert lunch is True and dinner is False
    short, _ = nodes._note_meals(11 * 60 + 30, 12 * 60 + 10, False, False)
    assert short is False
    slots = []
    current, had_lunch, had_dinner, _prev = nodes._insert_due_meals(
        slots, 17 * 60, True, False, dinner={"name": "老正兴", "address": "福州路"}
    )
    assert had_lunch is True and had_dinner is True
    assert current == 18 * 60
    assert slots[0]["activity_name"] == "老正兴"
    assert slots[0]["description"] == "福州路"


def test_preview_waits_up_to_thirty_minutes_for_opening():
    nodes = _nodes()
    preview = nodes._preview_visit(
        {"name": "博物馆", "visit_duration": 60, "open_time": "09:20-17:00"},
        None, 9 * 60, "上海", "公共交通", None,
    )
    assert preview["start"] == 9 * 60 + 20
    assert preview["end"] == 10 * 60 + 20
    assert preview["too_early"] is False


def test_preview_defers_when_opening_is_more_than_thirty_minutes_away():
    nodes = _nodes()
    preview = nodes._preview_visit(
        {"name": "夜景", "visit_duration": 60, "open_time": "19:00-21:00"},
        None, 9 * 60, "上海", "公共交通", None,
    )
    assert preview == {"attr": preview["attr"], "too_early": True}


def test_preview_drops_visit_that_passes_closing_or_return():
    nodes = _nodes()
    assert nodes._preview_visit(
        {"name": "早关门", "visit_duration": 180, "open_time": "09:00-11:00"},
        None, 9 * 60, "上海", "公共交通", None,
    ) is None
    _fixed_route(nodes, 20)
    hotel = {"name": "酒店", "location": "121.49,31.24"}
    late = nodes._preview_visit(
        {"name": "外滩", "visit_duration": 50, "location": "121.50,31.24"},
        None, 20 * 60, "上海", "公共交通", hotel,
    )
    assert late is None


def test_preview_wait_picks_the_earliest_opening_inside_three_hours():
    nodes = _nodes()
    _fixed_route(nodes, 10)
    pending = [
        {"name": "更晚", "visit_duration": 60, "open_time": "18:30-21:00"},
        {"name": "较早", "visit_duration": 60, "open_time": "17:30-21:00"},
        {"name": "太远", "visit_duration": 60, "open_time": "21:00-22:00"},
    ]
    preview = nodes._preview_wait(pending, None, 16 * 60, "上海", "公共交通", None)
    assert preview["attr"]["name"] == "较早"
    assert preview["start"] == 17 * 60 + 30


def test_long_visit_is_placed_before_short_ones():
    nodes = _nodes()
    _fixed_route(nodes, 10)
    kept, slots, _segments, _deferred = nodes._assemble_day(
        [
            {"name": "小店", "visit_duration": 60, "location": "121.50,31.20"},
            {"name": "博物馆", "visit_duration": 180, "location": "121.51,31.20"},
        ],
        "上海",
        "公共交通",
        None,
    )
    assert [item["name"] for item in kept][:2] == ["博物馆", "小店"]
    assert slots[0]["activity_name"] == "博物馆"
    assert slots[0]["start_time"] == "09:00"
    assert slots[0]["end_time"] == "12:00"


def test_drop_unslotted_removes_attractions_missing_from_the_clock():
    nodes = _nodes()
    day = {
        "attractions": [{"name": "留下"}, {"name": "丢掉"}],
        "time_slots": [
            {"activity_type": "景点游览", "activity_name": "留下", "end_time": "12:00"},
        ],
    }
    nodes._drop_unslotted(day)
    assert [item["name"] for item in day["attractions"]] == ["留下"]


# ━━━━━━ 文本解析与格式化 ━━━━━━


def test_match_categories_and_formatters():
    nodes = _nodes()
    assert nodes._match_categories("想看博物馆", []) == ["历史文化"]
    assert "主题乐园" in nodes._match_categories("", ["迪士尼"])
    assert nodes._match_categories("随便", []) == ["历史文化", "自然风光"]

    class Poi:
        name = "外滩"
        address = "中山东路"
        location = "121.49,31.24"
        type = "风景名胜"
        tel = "021-0000"

    text = nodes._format_amap_results([Poi()])
    assert "1. 外滩" in text and "坐标:" in text
    assert nodes._format_hotel_results([]) == "未找到符合条件的酒店"
    hotel = nodes._format_hotel_results([Poi()])
    assert "电话: 021-0000" in hotel
    attraction = nodes._format_attraction_response([{
        "name": "豫园",
        "description": "园林",
        "visit_duration": 90,
        "location": "121.49,31.23",
        "rerank_score": 0.82,
        "source": "高德",
    }])
    assert "建议游览: 90分钟" in attraction
    assert "相关性: 0.82" in attraction


def test_parse_trip_plan_normalizes_alternate_day_key():
    nodes = _nodes()
    nodes._fill_day_until_close = lambda day, city, transportation, used: None
    parsed = nodes._parse_trip_plan(
        '说明\n```json\n{"itinerary": [{"spots": ["外滩"]}], "accommodation": "全季酒店外滩"}\n```',
        city="上海",
    )
    assert parsed["days"][0]["day_index"] == 0
    assert parsed["days"][0]["hotel"]["name"] == "全季酒店外滩"
    assert parsed["days"][0]["meals"][0]["type"] == "lunch"
    assert parsed["days"][0]["meals"][1]["type"] == "dinner"
    assert nodes._parse_trip_plan("没有 json") is None


def test_parse_fill_name_and_lookup_filler():
    assert GraphNodes._parse_fill_name('结果 {"name": " 上海博物馆 "}') == "上海博物馆"
    assert GraphNodes._parse_fill_name("没有对象") == ""
    nodes = _nodes()

    class Store:
        def find_by_title(self, name, city="", limit=1):
            return [{
                "metadata": {
                    "title": "上海博物馆",
                    "visit_duration": 120,
                    "price": "40元",
                    "address": "人民大道",
                    "location": "121.47,31.23",
                    "open_time": "09:00-17:00",
                }
            }]

    nodes.retriever = type("R", (), {"vector_store": Store()})()
    filler = nodes._lookup_filler("上海博物馆", "上海")
    assert filler["ticket_price"] == 40
    assert filler["filler"] is True
    assert nodes._suggest_filler_attraction("上海", [], 2, "外滩") is None


def test_transport_cost_uses_distance_and_extra_stops():
    nodes = _nodes()
    quiet = nodes._estimate_transport_cost([{"attractions": [{}, {}]}], "")
    busy = nodes._estimate_transport_cost(
        [{
            "attractions": [{}, {}, {}],
            "route_segments": [{"distance": 2000}, {"distance": 100}],
        }],
        "",
    )
    assert quiet == 30
    assert busy == 30 + 10 + 5 + 20


def test_attraction_price_matches_contained_name():
    prices = _nodes()._extract_attraction_prices(
        [],
        [{"attractions": [{"name": "上海博物馆"}]}],
        [{"name": "上海博物馆(人民广场)", "ticket_price": 40}],
    )
    assert prices["上海博物馆"] == 40


# ━━━━━━ 高德结果解析（注入假路线对象） ━━━━━━


def test_transit_line_names_keep_unique_order():
    route = _Route(transits=[{
        "segments": [
            {"bus": {"buslines": [{"name": "地铁2号线"}, {"name": "地铁2号线"}]}},
            {"bus": {"buslines": [{"name": "地铁10号线"}]}},
            "bad",
        ]
    }])
    assert GraphNodes._transit_line_names(route) == ["地铁2号线", "地铁10号线"]
    assert GraphNodes._transit_line_names(_Route()) == []


def test_parse_amap_distance_from_path_transit_and_text():
    nodes = _nodes()
    assert nodes._parse_amap_distance_duration(
        _Route(paths=[{"distance": "3000", "duration": "1250"}])
    ) == (3000, 20)
    assert nodes._parse_amap_distance_duration(
        _Route(transits=[{"distance": 800, "duration": 600}])
    ) == (800, 10)
    distance, minutes = nodes._parse_amap_distance_duration(
        _Route(distance="3公里", duration="1小时5分钟")
    )
    assert distance == 3000
    assert minutes == 65


def test_route_segments_use_driving_leg_and_survive_api_error():
    nodes = _nodes()
    nodes.amap_service = _Amap(route=_Route(paths=[{"distance": 3000, "duration": 1200}]))
    segments = nodes._generate_route_segments_from_attractions(
        [
            {"name": "外滩", "location": "121.49,31.24"},
            {"name": "豫园", "location": "121.49,31.23"},
        ],
        city="上海",
        transportation="驾车",
    )
    assert segments[0]["transport_mode"] == "驾车"
    assert segments[0]["duration"] == 20
    assert segments[0]["distance"] == 3000
    assert "121.49,31.24" == nodes.amap_service.calls[0][0]

    nodes.amap_service = _Amap(error=True)
    failed = nodes._generate_route_segments_from_attractions(
        [{"name": "外滩"}, {"name": "豫园"}],
        city="上海",
        transportation="公共交通",
    )
    assert failed[0]["duration"] == 5
    assert failed[0]["transport_mode"] == "步行"
    assert "步行" in failed[0]["route_detail"]


def test_extract_route_info_formats_stubbed_legs():
    nodes = _nodes()
    _fixed_route(nodes, 18)
    text = nodes._extract_route_info(
        "1. 外滩\n坐标: 121.49,31.24\n2. 豫园\n地址: 福佑路\n",
        city="上海",
        travel_days=1,
    )
    assert "外滩 -> 豫园" in text
    assert "约18分钟" in text
    assert "2.0公里" in text
    assert nodes._extract_route_info("1. 只有一个\n坐标: 1,2\n") == ""


def test_qualify_route_leg_driving_and_single_bus_line():
    drive = GraphNodes._qualify_route_leg(3200, 15, [], "驾车", "driving")
    assert drive["transport_mode"] == "驾车"
    assert "3.2公里" in drive["route_detail"]
    bus = GraphNodes._qualify_route_leg(5000, 25, ["71路"], "公共交通", "transit")
    assert bus["transport_mode"] == "71路"
    assert bus["route_detail"] == "乘坐71路"
