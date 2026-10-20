"""高德名单筛选、知识库介绍和飞猪票价补全。不访问高德和飞猪。"""

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load():
    name = "attraction_recommend_under_test"
    path = ROOT / "app" / "services" / "attraction_recommend.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


mod = _load()


def test_selection_backfills_to_minimum():
    pois = [{"name": f"景点{i}", "type": "公园" if i < 2 else "商场"} for i in range(10)]
    ranked = mod.rank_by_preference(pois, ["自然风光"], "公园")
    selected = mod.select_names(ranked, ["景点0"], travel_days=1)
    assert len(selected) == 6
    assert selected[0]["name"] == "景点0"
    assert {item["name"] for item in selected}.issubset({item["name"] for item in pois})


def test_lookup_shanghai_zoo_intro_and_city_overview():
    profile = mod.lookup_profile("上海", "上海动物园")
    assert "动物" in profile["description"]
    assert profile["visit_duration"]
    overview = mod.load_city_description("上海")
    assert "上海" in overview


def test_enrich_uses_ticket_price_and_keeps_description():
    pois = [{"name": "上海动物园", "address": "虹桥路", "location": "121.36,31.19", "type": "动物园"}]

    def prices(city, name, depart_date):
        assert city == "上海"
        assert name == "上海动物园"
        return 40

    enriched = mod.enrich_attractions(pois, "上海", "2026-10-20", ticket_lookup=prices)
    assert enriched[0]["ticket_price"] == 40
    assert "动物" in enriched[0]["description"]
    text = mod.format_candidates(enriched)
    assert "40元" in text
    assert "坐标: 121.36,31.19" in text


def test_missing_flyai_price_uses_knowledge_base_and_keeps_link():
    pois = [{"name": "上海动物园", "address": "虹桥路", "location": "121.36,31.19", "type": "动物园"}]

    def quotes(city, name, depart_date):
        return {"price": None, "jump_url": "https://example.com/zoo"}

    enriched = mod.enrich_attractions(pois, "上海", "2026-10-20", ticket_lookup=quotes)
    assert enriched[0]["ticket_price"] == 40
    assert enriched[0]["ticket_source"] == "knowledge_base"
    assert enriched[0]["needs_user_check"] is True
    checks = mod.price_checks_for_plan(
        {"days": [{"attractions": [{"name": "上海动物园"}]}]},
        enriched,
    )
    assert checks == [{
        "name": "上海动物园",
        "reference_price": 40,
        "jump_url": "https://example.com/zoo",
    }]


def test_dining_keywords_follow_preference_and_cuisine():
    lunch, dinner = mod.dining_keywords(["美食"], "想吃火锅")
    assert lunch[0] == "火锅"
    assert dinner[0] == "火锅"
    plain_lunch, plain_dinner = mod.dining_keywords(["历史文化"], "")
    assert plain_lunch[0] == "老字号"
    assert "老字号" in plain_dinner


def test_restaurant_selection_keeps_named_meal_and_backfills():
    pois = [
        {"name": "甲", "meal": "lunch", "address": "甲路", "location": "1,2"},
        {"name": "乙", "meal": "lunch", "address": "乙路"},
        {"name": "丙", "meal": "dinner", "address": "丙路"},
    ]
    selected = mod.select_restaurants(pois, ["乙"], "lunch", travel_days=2)
    assert [item["name"] for item in selected] == ["乙", "甲"]
    text = mod.format_restaurants(selected)
    assert "午饭" in text
    assert "乙路" in text


if __name__ == "__main__":
    test_selection_backfills_to_minimum()
    test_lookup_shanghai_zoo_intro_and_city_overview()
    test_enrich_uses_ticket_price_and_keeps_description()
    test_missing_flyai_price_uses_knowledge_base_and_keeps_link()
    print("ok")
