"""
上海完整链路手工测试。

链路和 Web 一致：
  知识库检索 → LangGraph 主图（景点 / 天气 / 酒店并行）
  → 生成行程 → 按完整游览分钟排到 21:00，高德路程加 10 分钟缓冲，空档用知识库景点补上
  → 对话把一个景点换成上海迪士尼（重排当天时间并重算路程）
  → 对话换成带每晚价格的酒店

用法（项目根目录，会打真实模型、Embedding、高德和酒店接口，大约数分钟）:
    .\\.venv\\Scripts\\python.exe tests/test_full_chain.py
    .\\.venv\\Scripts\\python.exe tests/test_full_chain.py --days 2
    .\\.venv\\Scripts\\python.exe tests/test_full_chain.py --skip-dialog
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from datetime import datetime, timedelta

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

try:
    from dotenv import load_dotenv

    load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
except ImportError:
    pass

os.environ.setdefault("VECTOR_BACKEND", "chroma")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

USER_ID = "shanghai_full_chain_user"
CITY = "上海"
PREFERENCES = ["夜景", "步行街"]


def _banner(title: str) -> None:
    print("\n" + "=" * 68)
    print(title)
    print("=" * 68)


def _clip(text, n: int = 500) -> str:
    text = (text or "").strip()
    if len(text) <= n:
        return text
    return text[:n] + "\n    ...[截断]..."


def _ok(name: str, passed: bool, detail: str = "") -> bool:
    extra = f"  {detail}" if detail else ""
    print(f"  [{'通过' if passed else '失败'}] {name}{extra}")
    return passed


def _hour(value) -> int:
    text = str(value or "")
    if ":" not in text:
        return -1
    try:
        return int(text.split(":", 1)[0])
    except ValueError:
        return -1


def _attraction_name(item) -> str:
    if isinstance(item, dict):
        return str(item.get("name") or "").strip()
    return str(item or "").strip()


def _days_of(plan: dict) -> list:
    if not isinstance(plan, dict):
        return []
    daily = plan.get("daily_itinerary")
    if isinstance(daily, dict) and isinstance(daily.get("days"), list):
        return daily["days"]
    days = plan.get("days")
    return days if isinstance(days, list) else []


def _print_days(days: list) -> None:
    for day in days:
        if not isinstance(day, dict):
            continue
        names = [_attraction_name(item) for item in day.get("attractions") or []]
        names = [name for name in names if name]
        label = day.get("date") or f"第{int(day.get('day_index') or 0) + 1}天"
        print(f"  {label}: {names}")
        for slot in day.get("time_slots") or []:
            print(
                f"    {slot.get('start_time')}-{slot.get('end_time')} "
                f"[{slot.get('activity_type')}] {slot.get('activity_name')}"
            )
        for seg in day.get("route_segments") or []:
            print(
                f"    路程 {seg.get('start_point')} → {seg.get('end_point')} "
                f"{seg.get('duration')}分钟 {seg.get('distance')}米 "
                f"{seg.get('route_detail') or seg.get('transport_mode') or ''}"
            )
        if not day.get("time_slots"):
            print("    (没有时间表)")


def _schedule_checks(days: list, label: str) -> bool:
    has_slot = False
    within_close = True
    reached_close = False
    routes_ok = True
    late_detail = ""
    for day in days:
        if not isinstance(day, dict):
            continue
        names = [_attraction_name(item) for item in day.get("attractions") or []]
        names = [name for name in names if name]
        slots = day.get("time_slots") or []
        if slots:
            has_slot = True
            end = _hour(slots[-1].get("end_time"))
            if end > 21:
                within_close = False
                late_detail = slots[-1].get("end_time")
            if end == 21:
                reached_close = True
            for slot in slots:
                if _hour(slot.get("end_time")) > 21:
                    within_close = False
                    late_detail = slot.get("end_time")
        if len(names) > 1 and not day.get("route_segments"):
            routes_ok = False
    checks = [
        _ok(f"{label} 有时间表", has_slot),
        _ok(f"{label} 没有晚于 21:00 的时段", within_close, late_detail),
        _ok(f"{label} 至少有一天排到 21:00", reached_close),
        _ok(f"{label} 多个景点的那天有路程", routes_ok),
    ]
    return all(checks)


def step_preflight(days: int, start: str, end: str) -> None:
    _banner("0) 环境")
    print(f"  city={CITY}  days={days}  {start} ~ {end}")
    print(f"  user_id={USER_ID}")
    print(f"  DASHSCOPE_API_KEY={'已配置' if os.getenv('DASHSCOPE_API_KEY') else '缺失'}")
    print(f"  DASHSCOPE_MODEL={os.getenv('DASHSCOPE_MODEL') or '(默认)'}")
    print(f"  AMAP_MAPS_API_KEY={'已配置' if os.getenv('AMAP_MAPS_API_KEY') else '缺失'}")
    print(f"  HOTEL_API_KEY={'已配置' if os.getenv('HOTEL_API_KEY') else '缺失'}")
    print(f"  REDIS={os.getenv('REDIS_HOST', 'localhost')}:{os.getenv('REDIS_PORT', '6379')}")


def step_plan(days: int, start: str, end: str) -> tuple[bool, dict]:
    _banner("1) LangGraph 规划（景点 / 天气 / 酒店 → 时间表和路程）")
    from app.agents.langgraph_builder import get_langgraph_agent

    print("  正在初始化 Agent ...")
    t0 = time.time()
    agent = get_langgraph_agent(llm_provider="aliyun")
    print(f"  初始化耗时 {time.time() - t0:.1f}s")

    print("  正在 plan_trip ...")
    t1 = time.time()
    result = agent.plan_trip(
        city=CITY,
        travel_days=days,
        start_date=start,
        end_date=end,
        transportation="公共交通",
        accommodation="经济型",
        preferences=PREFERENCES,
        user_id=USER_ID,
        free_text_input="想看外滩和夜景，景点之间用公共交通",
        thread_id=USER_ID + "_full_chain",
    )
    print(f"  规划耗时 {time.time() - t1:.1f}s")
    print("\n  --- 景点 ---")
    print("   ", _clip(result.get("attraction_response"), 400))
    print("\n  --- 天气 ---")
    print("   ", _clip(result.get("weather_response"), 240))
    print("\n  --- 酒店 ---")
    print("   ", _clip(result.get("hotel_response"), 240))
    print(f"\n  session_id={result.get('session_id')}")
    print(f"  budget_total={result.get('budget_total')}")
    print(f"  reflect={result.get('reflection_feedback')}")

    parsed = result.get("trip_plan_parsed") or {}
    print("\n  --- 解析后的日程 ---")
    _print_days(parsed.get("days") or [])

    parsed_ok = isinstance(parsed, dict) and bool(parsed.get("days"))
    checks = [
        _ok("景点检索有输出", bool((result.get("attraction_response") or "").strip())),
        _ok("天气节点有输出", bool((result.get("weather_response") or "").strip())),
        _ok("酒店节点有输出", bool((result.get("hotel_response") or "").strip())),
        _ok("生成了可解析行程", parsed_ok),
        _ok("写入了会话", bool(result.get("session_id"))),
        _schedule_checks(parsed.get("days") or [], "第一轮"),
    ]
    return all(checks), result


def _bind_session(result: dict):
    from app.agents.version_manager import VersionManager

    vm = VersionManager(use_redis=True)
    session_id = result.get("session_id") or ""
    if session_id and vm.get_session(session_id):
        print(f"  使用规划写入的会话 {session_id}  后端={'redis' if vm.use_redis else 'file'}")
        return vm, session_id

    print("  规划没有留下可用会话，按解析结果补建一份")
    session = vm.create_session(user_id=USER_ID, plan_id="full_chain")
    parsed = result.get("trip_plan_parsed") or {}
    names = []
    for day in parsed.get("days") or []:
        for item in day.get("attractions") or []:
            name = _attraction_name(item)
            if name and name not in names:
                names.append(name)
    vm.create_initial_version(session.session_id, {
        "city": CITY,
        "duration_days": len(parsed.get("days") or []) or 1,
        "attractions": names,
        "budget": result.get("budget") or {},
        "daily_itinerary": parsed,
        "transportation": "公共交通",
        "accommodation": "经济型",
    })
    return vm, session.session_id


def _print_dialog(vm, dialog_result: dict) -> dict:
    print("\n  回复:")
    print(_clip(dialog_result.get("reply") or "", 700))
    print("  变更:")
    for line in dialog_result.get("changes") or []:
        print(f"    - {line}")
    version_id = dialog_result.get("new_version_id")
    plan = {}
    if not version_id:
        print("  没有产生新版本")
        return plan
    version = vm.get_version(version_id)
    plan = version.plan_data if version else {}
    intent = getattr(version, "change_intent", None) or {}
    if not isinstance(intent, dict):
        intent = getattr(intent, "__dict__", {}) or {}
    search = (intent.get("details") or {}).get("search_result") or {}
    if search:
        print("  工具搜索:")
        print(f"    attractions={search.get('attractions')}")
        print(f"    hotel={search.get('hotel')}  hotel_price={search.get('hotel_price')}")
        print(f"    summary={search.get('summary')}")
    print(f"  attractions={plan.get('attractions')}")
    hotel = plan.get("hotel") or {}
    if hotel:
        print(f"  hotel={hotel}")
    budget = plan.get("budget") or {}
    print(f"  budget.total_hotels={budget.get('total_hotels')}  total={budget.get('total')}")
    print("  日程:")
    _print_days(_days_of(plan))
    return plan


def _replace_target(plan: dict) -> str:
    for day in _days_of(plan):
        for item in day.get("attractions") or []:
            name = _attraction_name(item)
            if name and "迪士尼" not in name:
                return name
    names = plan.get("attractions") or []
    for name in names:
        if name and "迪士尼" not in str(name):
            return str(name)
    return "外滩"


def step_dialog(result: dict) -> bool:
    _banner("2) 对话：换成迪士尼，再换成有每晚价格的酒店")
    from app.agents.dialog_agent import DialogAgent

    vm, session_id = _bind_session(result)
    current = vm.get_current_version(session_id)
    current_plan = current.plan_data if current else {}
    target = _replace_target(current_plan)
    print(f"  将替换的景点: {target}")

    agent = DialogAgent(
        llm_provider="aliyun",
        use_redis=vm.use_redis,
        enable_rag=True,
        enable_amap=True,
        enable_memory=False,
        context_window_size=6,
    )
    agent.version_manager = vm

    replace_text = f"把{target}换成上海迪士尼度假区"
    print(f"\n  用户: {replace_text}")
    t0 = time.time()
    replace_result = agent.process_modification(session_id, replace_text)
    print(f"  耗时 {time.time() - t0:.1f}s")
    replace_plan = _print_dialog(vm, replace_result)

    disney_slots = []
    for day in _days_of(replace_plan):
        for slot in day.get("time_slots") or []:
            if "迪士尼" in str(slot.get("activity_name") or "") and slot.get("activity_type") == "景点游览":
                disney_slots.append(slot)
    disney_span_ok = False
    disney_detail = "没有游玩时段"
    if disney_slots:
        slot = disney_slots[0]
        span = _hour(slot.get("end_time")) - _hour(slot.get("start_time"))
        disney_detail = f"{slot.get('start_time')}-{slot.get('end_time')} 共{span}小时"
        disney_span_ok = span == 4 or _hour(slot.get("end_time")) == 21

    hotel_text = "酒店换成便宜一点的经济型，要有每晚价格"
    print(f"\n  用户: {hotel_text}")
    t1 = time.time()
    hotel_result = agent.process_modification(session_id, hotel_text)
    print(f"  耗时 {time.time() - t1:.1f}s")
    hotel_plan = _print_dialog(vm, hotel_result)
    hotel = hotel_plan.get("hotel") or {}
    price = hotel.get("price_per_night")
    budget = hotel_plan.get("budget") or {}

    checks = [
        _ok("替换产生了新版本", bool(replace_result.get("new_version_id"))),
        _ok("计划里出现迪士尼", any("迪士尼" in str(name) for name in (replace_plan.get("attractions") or []))),
        _ok("迪士尼游览时段是 4 小时，或被 21:00 截断", disney_span_ok, disney_detail),
        _schedule_checks(_days_of(replace_plan), "换景点后"),
        _ok("酒店搜索带回名称和每晚价格", bool(hotel.get("name")) and isinstance(price, (int, float)) and price > 0, str(hotel)),
        _ok("酒店总价按晚数写入预算", isinstance(budget.get("total_hotels"), (int, float)) and budget.get("total_hotels", 0) > 0),
        _schedule_checks(_days_of(hotel_plan), "换酒店后"),
    ]
    return all(checks)


def main() -> int:
    parser = argparse.ArgumentParser(description="上海规划到对话的完整链路测试")
    parser.add_argument("--days", type=int, default=2, help="行程天数，默认 2")
    parser.add_argument("--skip-dialog", action="store_true", help="只跑第一轮规划")
    args = parser.parse_args()

    days = max(1, args.days)
    start = (datetime.now() + timedelta(days=14)).strftime("%Y-%m-%d")
    end = (datetime.now() + timedelta(days=14 + days - 1)).strftime("%Y-%m-%d")
    step_preflight(days, start, end)

    results = []
    try:
        plan_ok, plan_result = step_plan(days, start, end)
        results.append(("LangGraph 规划", plan_ok))
        if not args.skip_dialog:
            results.append(("对话修改", step_dialog(plan_result)))
    except Exception:
        traceback.print_exc()
        results.append(("未捕获异常", False))

    _banner("结果")
    passed = 0
    for name, ok in results:
        print(f"  {'✅' if ok else '❌'} {name}")
        if ok:
            passed += 1
    print(f"  {passed}/{len(results)} 通过")
    return 0 if results and passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
