"""任务通过率：各 agent 按任务计，全链路按一整次行程计。不调用模型。

运行: python -m pytest tests/test_task_pass_rate.py -q
"""

from __future__ import annotations

import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.task_pass import format_pass_report, score_task_suite


def _ok_case() -> dict:
    return {
        "id": "shanghai_1d",
        "city": "上海",
        "travel_days": 1,
        "attraction_pool": ["外滩", "豫园", "南京路步行街", "上海城隍庙", "上海新天地", "人民广场"],
        "hotel_pool": ["全季酒店外滩"],
        "recommend": {
            "names": ["外滩", "豫园", "南京路步行街", "上海城隍庙"],
            "lunch": ["南翔馒头店"],
            "dinner": ["老正兴"],
        },
        "weather": {"weather_response": "天气信息：\n城市: 上海\n天气: 晴"},
        "hotel": {"name": "全季酒店外滩", "price_per_night": 300},
        "plan": {
            "days": [{
                "day_index": 0,
                "attractions": [
                    {"name": "外滩"},
                    {"name": "豫园"},
                    {"name": "南京路步行街"},
                    {"name": "上海城隍庙"},
                ],
                "meals": [
                    {"type": "lunch", "name": "南翔馒头店"},
                    {"type": "dinner", "name": "老正兴"},
                ],
                "time_slots": [
                    {"activity_type": "景点游览", "activity_name": "外滩", "start_time": "09:00", "end_time": "10:30"},
                    {"activity_type": "用餐", "activity_name": "南翔馒头店", "start_time": "11:30", "end_time": "12:30"},
                    {"activity_type": "景点游览", "activity_name": "豫园", "start_time": "12:40", "end_time": "14:10"},
                    {"activity_type": "景点游览", "activity_name": "南京路步行街", "start_time": "14:30", "end_time": "16:00"},
                    {"activity_type": "用餐", "activity_name": "老正兴", "start_time": "17:00", "end_time": "18:00"},
                    {"activity_type": "景点游览", "activity_name": "上海城隍庙", "start_time": "18:10", "end_time": "20:10"},
                ],
                "route_segments": [
                    {
                        "start_point": "南京路步行街",
                        "end_point": "上海城隍庙",
                        "duration": 5,
                        "transport_mode": "步行",
                        "route_detail": "未取得高德路线，按步行5分钟估算",
                    },
                ],
            }]
        },
        "budget": {
            "total_attractions": 40,
            "total_hotels": 300,
            "total_meals": 120,
            "total_transportation": 20,
            "total": 480,
            "meal_details": {"meals_per_day": 2, "price_per_meal": 60, "days": 1},
        },
    }


def _early_lunch_and_outside_name() -> dict:
    case = copy.deepcopy(_ok_case())
    case["id"] = "early_lunch"
    case["recommend"]["names"] = ["外滩", "豫园", "南京路步行街", "迪士尼"]
    lunch = case["plan"]["days"][0]["time_slots"][1]
    lunch["start_time"] = "10:30"
    lunch["end_time"] = "11:30"
    return case


def test_agent_rates_stay_separate_from_the_full_chain():
    summary = score_task_suite([_ok_case(), _early_lunch_and_outside_name()])
    agents = summary["agents"]
    assert agents["recommend"]["rate"] == 7 / 8
    assert agents["schedule"]["rate"] == 7 / 8
    assert agents["weather"]["rate"] == 1
    assert agents["hotel"]["rate"] == 1
    assert agents["planner"]["rate"] == 1
    assert agents["route"]["rate"] == 1
    assert agents["budget"]["rate"] == 1
    assert summary["full_chain"]["rate"] == 0.5
    assert summary["full_chain"]["passed"] == 1
    assert summary["full_chain"]["total"] == 2

    failed = {item["task"] for item in agents["recommend"]["failed"]}
    assert failed == {"recommend.names_from_pool"}
    schedule_failed = {item["task"] for item in agents["schedule"]["failed"]}
    assert schedule_failed == {"schedule.meal_windows"}
    chain = {item["id"]: item for item in summary["full_chain"]["cases"]}
    assert chain["shanghai_1d"]["passed"] is True
    assert chain["early_lunch"]["passed"] is False
    assert "recommend.names_from_pool" in chain["early_lunch"]["failed"]
    assert "schedule.meal_windows" in chain["early_lunch"]["failed"]


def test_one_clean_trip_passes_every_agent_and_the_chain():
    summary = score_task_suite([_ok_case()])
    assert summary["full_chain"]["rate"] == 1
    assert all(row["rate"] == 1 for row in summary["agents"].values())
    text = format_pass_report(summary)
    assert "recommend: 4/4 (100%)" in text
    assert "全链路: 1/1 (100%)" in text


def test_unknown_route_must_be_a_five_minute_walk():
    case = _ok_case()
    case["id"] = "slow_unknown"
    case["plan"]["days"][0]["route_segments"][0]["duration"] = 30
    case["plan"]["days"][0]["route_segments"][0]["transport_mode"] = "公共交通"
    summary = score_task_suite([case])
    assert summary["agents"]["route"]["rate"] == 0.5
    assert summary["agents"]["schedule"]["rate"] == 1
    assert summary["full_chain"]["rate"] == 0
    assert summary["agents"]["route"]["failed"][0]["task"] == "route.unknown_route_is_five_minute_walk"
