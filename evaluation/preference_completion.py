"""按城市和用户偏好汇总真实规划的任务完成率。

每条任务是一次真实 plan_trip。打分口径与 evaluation.task_pass 相同。
样例只能核对分组，完成率要以跑出来的结果为准。
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Sequence

from evaluation.task_pass import format_pass_report, score_task_suite

# 知识库里有景点数据的城市，配上会改变检索和餐饮关键词的偏好。
TASKS = [
    ("上海", ["夜景", "步行街"]),
    ("上海", ["历史文化"]),
    ("北京", ["历史文化"]),
    ("北京", ["亲子"]),
    ("杭州", ["自然风光"]),
    ("成都", ["美食"]),
]


def case_from_result(
    result: Dict[str, Any],
    *,
    city: str,
    preferences: Sequence[str],
    travel_days: int,
) -> Dict[str, Any]:
    """把一次 plan_trip 的返回收成任务打分用的记录。"""
    attractions = _candidate_names(
        result.get("attraction_candidates"),
        result.get("attraction_response"),
    )
    restaurants = [
        item for item in result.get("restaurant_candidates") or []
        if isinstance(item, dict)
    ]
    hotels = [item for item in result.get("hotel_candidates") or [] if isinstance(item, dict)]
    plan = result.get("trip_plan_parsed") if isinstance(result.get("trip_plan_parsed"), dict) else {}
    label = "+".join(preferences)
    return {
        "id": f"{city}/{label}",
        "city": city,
        "preference": label,
        "travel_days": travel_days,
        "attraction_pool": attractions,
        "hotel_pool": [str(item.get("name") or "").strip() for item in hotels if item.get("name")],
        "recommend": {
            "names": attractions,
            "lunch": _meal_names(restaurants, "lunch"),
            "dinner": _meal_names(restaurants, "dinner"),
        },
        "weather": {"weather_response": result.get("weather_response") or ""},
        "hotel": _chosen_hotel(plan, hotels),
        "plan": plan,
        "budget": result.get("budget") if isinstance(result.get("budget"), dict) else {},
    }


def summarize_matrix(cases: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """总体、按城市、按偏好各算一遍通过率。"""
    return {
        "overall": score_task_suite(cases),
        "by_city": {
            city: score_task_suite([item for item in cases if item.get("city") == city])
            for city in _unique(item.get("city") for item in cases)
        },
        "by_preference": {
            label: score_task_suite([item for item in cases if item.get("preference") == label])
            for label in _unique(item.get("preference") for item in cases)
        },
    }


def format_matrix_report(summary: Dict[str, Any]) -> str:
    lines = ["按城市和偏好的任务完成率", ""]
    lines.append(format_pass_report(summary["overall"]))
    lines.append("")
    lines.append("按城市（全链路）")
    for city, block in summary["by_city"].items():
        chain = block["full_chain"]
        lines.append(f"  {city}: {chain['passed']}/{chain['total']} ({_pct(chain['rate'])})")
    lines.append("按偏好（全链路）")
    for label, block in summary["by_preference"].items():
        chain = block["full_chain"]
        lines.append(f"  {label}: {chain['passed']}/{chain['total']} ({_pct(chain['rate'])})")
    return "\n".join(lines)


def _candidate_names(records: Any, response: Any) -> List[str]:
    names = []
    for item in records or []:
        if isinstance(item, dict) and str(item.get("name") or "").strip():
            names.append(str(item["name"]).strip())
    if names:
        return names
    for line in str(response or "").splitlines():
        matched = re.match(r"\s*\d+\.\s+(.+)", line)
        if matched:
            names.append(matched.group(1).strip())
    return names


def _meal_names(restaurants: Sequence[Dict[str, Any]], meal: str) -> List[str]:
    return [
        str(item.get("name")).strip()
        for item in restaurants
        if item.get("meal") == meal and str(item.get("name") or "").strip()
    ]


def _chosen_hotel(plan: Dict[str, Any], hotels: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    hotel = plan.get("hotel") if isinstance(plan.get("hotel"), dict) else {}
    name = str(hotel.get("name") or "").strip()
    if not name:
        for day in plan.get("days") or []:
            stay = day.get("hotel") if isinstance(day, dict) else None
            if isinstance(stay, dict) and str(stay.get("name") or "").strip():
                hotel = stay
                name = str(stay.get("name")).strip()
                break
    matched = next((item for item in hotels if str(item.get("name") or "").strip() == name), None)
    price = _nightly_price(hotel) or _nightly_price(matched or {})
    return {"name": name, "price_per_night": price}


def _nightly_price(hotel: Dict[str, Any]) -> float:
    for key in ("price_per_night", "estimated_cost"):
        try:
            value = float(hotel.get(key) or 0)
        except (TypeError, ValueError):
            value = 0
        if value > 0:
            return value
    matched = re.search(r"(\d+(?:\.\d+)?)", str(hotel.get("price_range") or ""))
    return float(matched.group(1)) if matched else 0.0


def _unique(values: Iterable[Any]) -> List[str]:
    seen = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in seen:
            seen.append(text)
    return seen


def _pct(value: float) -> str:
    return f"{value * 100:.0f}%"
