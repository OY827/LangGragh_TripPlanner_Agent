"""按 agent 统计任务通过率，并单独统计整次行程的全链路通过率。

一个 agent 任务只检查该步自己的输出。全链路看同一次行程的全部任务：
有一项失败，这次行程就不通过。两边的分母不同，不能混成一个比例。
"""

from typing import Any, Dict, List, Optional, Sequence

from app.services.attraction_recommend import attraction_limits
from evaluation.metrics import rate

LUNCH_START = 11 * 60 + 30
LUNCH_END = 14 * 60
DINNER_START = 17 * 60
DINNER_END = 19 * 60 + 30
EVENING = 20 * 60
DAY_END = 21 * 60
AGENTS = ("recommend", "weather", "hotel", "planner", "schedule", "route", "budget")


def score_task_suite(cases: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """返回各 agent 的任务通过率，以及全链路按行程计的通过率。"""
    rows = []
    chain_rows = []
    for case in cases:
        tasks = score_chain(case)
        rows.extend(tasks)
        failed = [item["task"] for item in tasks if not item["passed"]]
        chain_rows.append({
            "id": case.get("id", ""),
            "passed": not failed,
            "failed": failed,
        })
    agents = {}
    for agent in AGENTS:
        owned = [item for item in rows if item["agent"] == agent]
        passed = sum(1 for item in owned if item["passed"])
        agents[agent] = {
            "passed": passed,
            "total": len(owned),
            "rate": rate(passed, len(owned)),
            "failed": [item for item in owned if not item["passed"]],
        }
    chain_passed = sum(1 for item in chain_rows if item["passed"])
    return {
        "agents": agents,
        "full_chain": {
            "passed": chain_passed,
            "total": len(chain_rows),
            "rate": rate(chain_passed, len(chain_rows)),
            "cases": chain_rows,
        },
    }


def score_chain(case: Dict[str, Any]) -> List[Dict[str, Any]]:
    case_id = str(case.get("id") or "")
    return [_row(case_id, agent, task, ok, reason) for agent, task, ok, reason in _checks(case)]


def format_pass_report(summary: Dict[str, Any]) -> str:
    lines = ["任务通过率"]
    for agent in AGENTS:
        row = summary["agents"][agent]
        lines.append(f"{agent}: {row['passed']}/{row['total']} ({_pct(row['rate'])})")
        for item in row["failed"]:
            lines.append(f"  {item['case_id']} {item['task']}: {item['reason']}")
    chain = summary["full_chain"]
    lines.append(f"全链路: {chain['passed']}/{chain['total']} ({_pct(chain['rate'])})")
    for item in chain["cases"]:
        if item["passed"]:
            continue
        lines.append(f"  {item['id']}: {', '.join(item['failed'])}")
    return "\n".join(lines)


def _checks(case: Dict[str, Any]):
    yield from _recommend(case)
    yield from _weather(case)
    yield from _hotel(case)
    yield from _planner(case)
    yield from _schedule(case)
    yield from _route(case)
    yield from _budget(case)


def _recommend(case: Dict[str, Any]):
    days = max(1, int(case.get("travel_days") or 1))
    low, high = attraction_limits(days)
    pool = _names(case.get("attraction_pool"))
    picked = _names((case.get("recommend") or {}).get("names"))
    if len(pool) < low:
        bounds_ok = len(picked) == len(pool) and bool(pool or not picked)
        bounds_reason = f"候选只有 {len(pool)} 个，应全部留下"
    else:
        bounds_ok = low <= len(picked) <= min(high, len(pool))
        bounds_reason = f"应选出 {low} 到 {min(high, len(pool))} 个，实际 {len(picked)} 个"
    yield "recommend", "count_in_range", bounds_ok, "" if bounds_ok else bounds_reason

    outside = [name for name in picked if name not in pool]
    yield "recommend", "names_from_pool", not outside, "" if not outside else "不在候选池: " + "、".join(outside)

    lunch = _names((case.get("recommend") or {}).get("lunch"))
    dinner = _names((case.get("recommend") or {}).get("dinner"))
    meals_ok = bool(lunch) and bool(dinner)
    yield "recommend", "lunch_and_dinner", meals_ok, "" if meals_ok else "午饭或晚饭为空"

    shared = sorted(set(lunch) & set(dinner))
    distinct = not shared
    yield "recommend", "lunch_dinner_distinct", distinct, "" if distinct else "午饭晚饭是同一家: " + "、".join(shared)


def _weather(case: Dict[str, Any]):
    text = str((case.get("weather") or {}).get("weather_response") or "")
    ok = "天气" in text and "失败" not in text and bool(case.get("city") or "") and str(case.get("city")) in text
    reason = "" if ok else "天气结果里没有城市和天气，或查询失败"
    yield "weather", "forecast_ready", ok, reason


def _hotel(case: Dict[str, Any]):
    hotel = case.get("hotel") or {}
    name = str(hotel.get("name") or "").strip()
    candidates = _names(case.get("hotel_pool"))
    named = bool(name) and name in candidates
    yield "hotel", "chosen_from_pool", named, "" if named else "酒店不在候选里"
    try:
        price = float(hotel.get("price_per_night") or 0)
    except (TypeError, ValueError):
        price = 0
    priced = price > 0
    yield "hotel", "has_nightly_price", priced, "" if priced else "没有每晚价格"


def _planner(case: Dict[str, Any]):
    days = _days(case)
    expected = max(1, int(case.get("travel_days") or 1))
    count_ok = len(days) == expected
    yield "planner", "day_count", count_ok, "" if count_ok else f"天数 {len(days)}，要求 {expected}"

    pool = set(_names(case.get("attraction_pool")))
    outside = [name for name in _plan_attractions(days) if name not in pool]
    yield "planner", "attractions_from_pool", not outside and bool(_plan_attractions(days)), (
        "" if not outside and _plan_attractions(days) else "景点不在候选里: " + "、".join(outside)
    )

    missing = []
    same = []
    for index, day in enumerate(days, 1):
        lunch = _meal_name(day, "lunch")
        dinner = _meal_name(day, "dinner")
        if not lunch or not dinner:
            missing.append(str(index))
        elif lunch == dinner:
            same.append(str(index))
    meals_ok = not missing and not same and bool(days)
    reason = ""
    if missing:
        reason = "第 " + "、".join(missing) + " 天缺少午饭或晚饭"
    elif same:
        reason = "第 " + "、".join(same) + " 天午饭晚饭是同一家"
    yield "planner", "each_day_has_two_meals", meals_ok, reason


def _schedule(case: Dict[str, Any]):
    days = _days(case)
    meal_problems = []
    window_problems = []
    late_visits = []
    reached = False
    for index, day in enumerate(days, 1):
        slots = [slot for slot in day.get("time_slots") or [] if isinstance(slot, dict)]
        if slots and _hhmm(slots[-1].get("end_time")) >= EVENING:
            reached = True
        lunch = _meal_slot(day, _meal_name(day, "lunch"))
        dinner = _meal_slot(day, _meal_name(day, "dinner"))
        if not _is_hour(lunch) or not _is_hour(dinner):
            meal_problems.append(str(index))
        else:
            if not _in_window(lunch, LUNCH_START, LUNCH_END):
                window_problems.append(f"第{index}天午饭 {_span(lunch)}")
            if not _in_window(dinner, DINNER_START, DINNER_END):
                window_problems.append(f"第{index}天晚饭 {_span(dinner)}")
        for slot in slots:
            if slot.get("activity_type") != "景点游览":
                continue
            if _hhmm(slot.get("end_time")) > DAY_END:
                late_visits.append(str(slot.get("activity_name") or index))
    yield "schedule", "meals_on_timeline", not meal_problems and bool(days), (
        "" if not meal_problems else "第 " + "、".join(meal_problems) + " 天的午饭或晚饭没有 60 分钟用餐"
    )
    yield "schedule", "meal_windows", not window_problems and bool(days), (
        "" if not window_problems else "；".join(window_problems)
    )
    yield "schedule", "visits_finish_by_21", not late_visits, (
        "" if not late_visits else "超过 21:00: " + "、".join(late_visits)
    )
    yield "schedule", "one_day_reaches_20", reached, "" if reached else "没有任何一天收到 20:00"


def _route(case: Dict[str, Any]):
    days = _days(case)
    missing = []
    bad_unknown = []
    for index, day in enumerate(days, 1):
        names = _plan_attractions([day])
        segments = [seg for seg in day.get("route_segments") or [] if isinstance(seg, dict)]
        if len(names) > 1 and not segments:
            missing.append(str(index))
        for seg in segments:
            detail = str(seg.get("route_detail") or "")
            if "未取得" not in detail:
                continue
            try:
                duration = int(seg.get("duration") or 0)
            except (TypeError, ValueError):
                duration = 0
            if duration != 5 or seg.get("transport_mode") != "步行":
                bad_unknown.append(f"{seg.get('start_point')}→{seg.get('end_point')}")
    yield "route", "segments_for_multi_stop_day", not missing, (
        "" if not missing else "第 " + "、".join(missing) + " 天有多个景点但没有路程"
    )
    yield "route", "unknown_route_is_five_minute_walk", not bad_unknown, (
        "" if not bad_unknown else "未取得路线时应为步行 5 分钟: " + "、".join(bad_unknown)
    )


def _budget(case: Dict[str, Any]):
    budget = case.get("budget") or {}
    parts = ["total_attractions", "total_hotels", "total_meals", "total_transportation"]
    try:
        numbers = [float(budget.get(key) or 0) for key in parts]
        total = float(budget.get("total") or 0)
    except (TypeError, ValueError):
        numbers = []
        total = -1
    summed = sum(numbers) if numbers else -1
    adds = abs(summed - total) < 0.01 and total >= 0
    yield "budget", "parts_sum_to_total", adds, "" if adds else "分项之和与总价不一致"
    days = max(1, int(case.get("travel_days") or 1))
    details = budget.get("meal_details") or {}
    meals_ok = details.get("meals_per_day") == 2 and float(budget.get("total_meals") or 0) == 120 * days
    yield "budget", "lunch_and_dinner_cost", meals_ok, "" if meals_ok else "餐饮应按每天午饭晚饭共 120 元"


def _row(case_id: str, agent: str, task: str, passed: bool, reason: str) -> Dict[str, Any]:
    return {
        "case_id": case_id,
        "agent": agent,
        "task": f"{agent}.{task}",
        "passed": passed,
        "reason": reason,
    }


def _days(case: Dict[str, Any]) -> List[Dict[str, Any]]:
    plan = case.get("plan") or {}
    days = plan.get("days") if isinstance(plan, dict) else None
    if not isinstance(days, list):
        return []
    return [day for day in days if isinstance(day, dict)]


def _names(value: Any) -> List[str]:
    names = []
    for item in value or []:
        if isinstance(item, dict):
            name = str(item.get("name") or "").strip()
        else:
            name = str(item or "").strip()
        if name:
            names.append(name)
    return names


def _plan_attractions(days: Sequence[Dict[str, Any]]) -> List[str]:
    names = []
    for day in days:
        names.extend(_names(day.get("attractions")))
    return names


def _meal_name(day: Dict[str, Any], meal_type: str) -> str:
    for meal in day.get("meals") or []:
        if isinstance(meal, dict) and meal.get("type") == meal_type:
            return str(meal.get("name") or "").strip()
    return ""


def _meal_slot(day: Dict[str, Any], name: str) -> Optional[Dict[str, Any]]:
    if not name:
        return None
    for slot in day.get("time_slots") or []:
        if (
            isinstance(slot, dict)
            and slot.get("activity_type") == "用餐"
            and slot.get("activity_name") == name
        ):
            return slot
    return None


def _is_hour(slot: Optional[Dict[str, Any]]) -> bool:
    if not slot:
        return False
    start = _hhmm(slot.get("start_time"))
    end = _hhmm(slot.get("end_time"))
    return start >= 0 and end - start == 60


def _in_window(slot: Dict[str, Any], start: int, end: int) -> bool:
    begin = _hhmm(slot.get("start_time"))
    return start <= begin < end


def _span(slot: Dict[str, Any]) -> str:
    return f"{slot.get('start_time')}-{slot.get('end_time')}"


def _hhmm(value: Any) -> int:
    text = str(value or "")
    if ":" not in text:
        return -1
    hour, minute = text.split(":", 1)
    try:
        return int(hour) * 60 + int(minute[:2])
    except ValueError:
        return -1


def _pct(value: float) -> str:
    return f"{value * 100:.0f}%"
