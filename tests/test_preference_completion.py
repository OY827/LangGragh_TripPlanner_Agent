"""城市和偏好的完成率分组。使用写死的规划结果，不调用模型。

运行: python -m pytest tests/test_preference_completion.py -q
真实跑分: python evaluation/run_preference_completion.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.preference_completion import case_from_result, format_matrix_report, summarize_matrix
from tests.test_task_pass_rate import _ok_case


def _live_like(city: str, preferences: list[str], *, early_lunch: bool = False) -> dict:
    base = _ok_case()
    if early_lunch:
        lunch = base["plan"]["days"][0]["time_slots"][1]
        lunch["start_time"] = "10:30"
        lunch["end_time"] = "11:30"
    result = {
        "attraction_candidates": [{"name": name} for name in base["attraction_pool"]],
        "restaurant_candidates": [
            {"name": "南翔馒头店", "meal": "lunch"},
            {"name": "老正兴", "meal": "dinner"},
        ],
        "hotel_candidates": [{"name": "全季酒店外滩", "estimated_cost": 300, "price_range": "300元/晚"}],
        "weather_response": f"天气信息：\n城市: {city}\n天气: 晴",
        "trip_plan_parsed": {
            "hotel": {"name": "全季酒店外滩", "price_range": "300元/晚"},
            "days": base["plan"]["days"],
        },
        "budget": base["budget"],
    }
    return case_from_result(result, city=city, preferences=preferences, travel_days=1)


def test_completion_splits_by_city_and_preference():
    cases = [
        _live_like("上海", ["夜景", "步行街"]),
        _live_like("上海", ["历史文化"], early_lunch=True),
        _live_like("杭州", ["自然风光"]),
    ]
    summary = summarize_matrix(cases)
    assert summary["overall"]["full_chain"]["rate"] == 2 / 3
    assert summary["by_city"]["上海"]["full_chain"]["rate"] == 0.5
    assert summary["by_city"]["杭州"]["full_chain"]["rate"] == 1
    assert summary["by_preference"]["夜景+步行街"]["full_chain"]["rate"] == 1
    assert summary["by_preference"]["历史文化"]["full_chain"]["rate"] == 0
    assert summary["by_preference"]["自然风光"]["full_chain"]["rate"] == 1
    text = format_matrix_report(summary)
    assert "按城市（全链路）" in text
    assert "上海: 1/2 (50%)" in text
    assert "杭州: 1/1 (100%)" in text
    assert "历史文化: 0/1 (0%)" in text


def test_hotel_price_comes_from_the_candidate_quote():
    case = _live_like("北京", ["亲子"])
    assert case["hotel"]["name"] == "全季酒店外滩"
    assert case["hotel"]["price_per_night"] == 300
    assert case["city"] == "北京"
    assert case["preference"] == "亲子"
