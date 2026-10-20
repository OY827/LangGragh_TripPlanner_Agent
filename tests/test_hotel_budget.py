"""每晚预算只保留落在区间内、并且有报价的酒店。"""

from app.services.hotel_mcp_service import HotelMCPService


def test_budget_keeps_prices_inside_the_band_and_drops_missing_quotes():
    hotels = [
        {"name": "甲", "price": 80},
        {"name": "乙", "price": 100},
        {"name": "丙", "price": 180},
        {"name": "丁", "price": 200},
        {"name": "戊", "price": 260},
        {"name": "无报价"},
    ]
    kept = HotelMCPService.filter_by_budget(hotels, "100-200")
    assert [item["name"] for item in kept] == ["丙", "丁"]


def test_lowest_band_includes_zero_and_one_hundred():
    hotels = [{"name": "甲", "price": 0}, {"name": "乙", "price": 100}, {"name": "丙", "price": 101}]
    kept = HotelMCPService.filter_by_budget(hotels, "0-100")
    assert [item["name"] for item in kept] == ["甲", "乙"]


def test_open_ended_budget_starts_above_five_hundred():
    hotels = [{"name": "甲", "price": 500}, {"name": "乙", "price": 680}]
    kept = HotelMCPService.filter_by_budget(hotels, "500+")
    assert [item["name"] for item in kept] == ["乙"]


def test_blank_budget_keeps_every_hotel():
    hotels = [{"name": "甲"}, {"name": "乙", "price": 10}]
    assert HotelMCPService.filter_by_budget(hotels, "") == hotels


def test_remaining_options_drop_the_hotel_the_planner_chose():
    candidates = [
        {"name": "外滩酒店", "estimated_cost": 180},
        {"name": "南京路酒店", "estimated_cost": 220},
        {"name": "豫园酒店", "estimated_cost": 260},
    ]
    left = HotelMCPService.remaining_options("南京路酒店", candidates)
    assert [item["name"] for item in left] == ["外滩酒店", "豫园酒店"]


def test_option_keeps_nightly_price_and_address():
    option = HotelMCPService.as_option({
        "hotelName": "外滩酒店",
        "address": "中山东一路",
        "price": 268,
        "score": 4.6,
    })
    assert option["name"] == "外滩酒店"
    assert option["address"] == "中山东一路"
    assert option["estimated_cost"] == 268
    assert option["rating"] == "4.6"
    text = HotelMCPService().format_for_agent([], city="上海", budget="200-300")
    assert text == "未找到上海每晚 200~300元 的酒店"
