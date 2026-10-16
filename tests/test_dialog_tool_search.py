"""
对话修改的工具调用测试（上海）

plan 第一轮编排不在这个脚本里重跑。这里用一份现成的上海计划，
检查对话 Agent 在改景点、改酒店时会不会调用搜索工具，并把结果写进新版本。

用法（项目根目录）:
    .\\.venv\\Scripts\\python.exe tests/test_dialog_tool_search.py

会调用通义千问，以及景点检索 / 酒店搜索。版本写在本地文件，不依赖 Redis。
"""

from __future__ import annotations

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
except ImportError:
    pass

os.environ.setdefault("PYTHONIOENCODING", "utf-8")

USER_ID = "dialog_tool_search_user"
CITY = "上海"


def _banner(title: str) -> None:
    print("\n" + "=" * 68)
    print(title)
    print("=" * 68)


def _clip(text: str, n: int = 800) -> str:
    text = (text or "").strip()
    if len(text) <= n:
        return text
    return text[:n] + "\n    ...[截断]..."


def _seed_plan() -> str:
    from app.agents.version_manager import VersionManager

    vm = VersionManager(use_redis=False)
    session = vm.create_session(user_id=USER_ID, plan_id="")
    vm.create_initial_version(
        session_id=session.session_id,
        plan_data={
            "city": CITY,
            "duration_days": 2,
            "start_date": "2026-10-08",
            "attractions": ["外滩", "豫园", "南京路步行街"],
            "budget": {
                "total_attractions": 80,
                "total_hotels": 600,
                "total_meals": 360,
                "total_transportation": 80,
                "total": 1120,
            },
            "transportation": "公共交通",
            "accommodation": "经济型",
            "daily_itinerary": {
                "city": CITY,
                "start_date": "2026-10-08",
                "days": [
                    {
                        "day_index": 0,
                        "date": "2026-10-08",
                        "attractions": [
                            {"name": "外滩", "visit_duration": 90},
                            {"name": "南京路步行街", "visit_duration": 60},
                        ],
                    },
                    {
                        "day_index": 1,
                        "date": "2026-10-09",
                        "attractions": [
                            {"name": "豫园", "visit_duration": 120},
                        ],
                    },
                ],
            },
        },
    )
    print(f"  session_id={session.session_id}")
    print("  初始景点: 外滩、豫园、南京路步行街")
    print("  初始住宿: 经济型")
    return session.session_id


def _print_turn(vm, result: dict) -> None:
    print("\n  回复:")
    print(_clip(result.get("reply") or ""))
    print("\n  变更:")
    for line in result.get("changes") or []:
        print(f"    - {line}")
    if not result.get("changes"):
        print("    (无)")

    version_id = result.get("new_version_id")
    if not version_id:
        print("\n  没有产生新版本")
        return

    version = vm.get_version(version_id)
    plan = version.plan_data if version else {}
    intent = getattr(version, "change_intent", None) or {}
    if not isinstance(intent, dict):
        intent = getattr(intent, "__dict__", {}) or {}
    details = intent.get("details") or {}
    search = details.get("search_result") or {}
    print("\n  工具搜索结果:")
    if search:
        print(f"    attractions={search.get('attractions')}")
        print(f"    hotel={search.get('hotel')}")
        print(f"    hotel_price={search.get('hotel_price')}")
        print(f"    summary={search.get('summary')}")
    else:
        print("    (这次修改没有记下 search_result，可能意图不是改景点/酒店，或工具结果没解析成 JSON)")

    print("\n  新版本计划:")
    print(f"    attractions={plan.get('attractions')}")
    print(f"    accommodation={plan.get('accommodation')}")
    hotel = plan.get("hotel") or {}
    if hotel:
        print(f"    hotel={hotel}")
    budget = plan.get("budget") or {}
    print(f"    budget.total_hotels={budget.get('total_hotels')}  total={budget.get('total')}")
    daily = (plan.get("daily_itinerary") or {}).get("days") or []
    for day in daily:
        names = []
        for item in day.get("attractions") or []:
            names.append(item.get("name") if isinstance(item, dict) else str(item))
        label = day.get("date") or f"第{(day.get('day_index') or 0) + 1}天"
        print(f"    {label}: {names}")
        for slot in day.get("time_slots") or []:
            print(f"      {slot.get('start_time')}-{slot.get('end_time')} {slot.get('activity_name')}")
        for seg in day.get("route_segments") or []:
            print(
                f"      路程 {seg.get('start_point')} → {seg.get('end_point')} "
                f"{seg.get('duration')}分钟 {seg.get('distance')}米"
            )


def main() -> None:
    _banner("0) 环境")
    print(f"  DASHSCOPE_API_KEY={'已配置' if os.getenv('DASHSCOPE_API_KEY') else '缺失'}")
    print(f"  HOTEL_API_KEY={'已配置' if os.getenv('HOTEL_API_KEY') else '缺失'}")
    print(f"  AMAP_MAPS_API_KEY={'已配置' if os.getenv('AMAP_MAPS_API_KEY') else '缺失'}")
    if not os.getenv("DASHSCOPE_API_KEY"):
        print("  缺少 DASHSCOPE_API_KEY，无法让模型发起工具调用。")
        return

    _banner("1) 写入一份上海计划（模拟 plan 第一轮的结果）")
    from app.agents.dialog_agent import DialogAgent
    from app.agents.version_manager import VersionManager

    session_id = _seed_plan()
    agent = DialogAgent(
        llm_provider="aliyun",
        use_redis=False,
        enable_rag=True,
        enable_amap=True,
        enable_memory=False,
    )
    vm = VersionManager(use_redis=False)

    turns = [
        "把外滩换成迪士尼",
        "酒店换成便宜一点的经济型，要有每晚价格",
    ]
    for index, text in enumerate(turns, 1):
        _banner(f"{index + 1}) 对话: {text}")
        result = agent.process_modification(session_id=session_id, user_input=text)
        _print_turn(vm, result)

    _banner("完成")
    print("  期望:")
    print("    第一轮 search_result.attractions 里出现迪士尼或同类乐园，计划景点不再只有外滩。")
    print("    第二轮 search_result.hotel 有酒店名；若带回 hotel_price，total_hotels 按晚数重算。")


if __name__ == "__main__":
    main()
