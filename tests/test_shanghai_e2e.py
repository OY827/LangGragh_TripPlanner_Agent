"""
上海完整链路手工测试（和 Web /api/trip/plan 同一条 LangGraph 主路径）

链路：
  知识库检索 → 长期记忆查询 → 景点/天气/酒店并行子图
  → 路线 → 生成行程 → 反思 → 写入 Chroma 长期记忆
  → Redis 计划版本 → DialogAgent 改计划（滑动窗口）→ 回退

用法（项目根目录，会打真实 LLM / Embedding / 高德，大约数分钟）:
  .\\.venv\\Scripts\\python.exe tests/test_shanghai_e2e.py
  .\\.venv\\Scripts\\python.exe tests/test_shanghai_e2e.py --days 2
  .\\.venv\\Scripts\\python.exe tests/test_shanghai_e2e.py --skip-dialog
  .\\.venv\\Scripts\\python.exe tests/test_shanghai_e2e.py --cleanup
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

USER_ID = "shanghai_e2e_user"
CITY = "上海"
PREFERENCES = ["夜景", "少走路"]


def _banner(title: str) -> None:
    print("\n" + "=" * 68)
    print(title)
    print("=" * 68)


def _clip(text, n: int = 600) -> str:
    text = (text or "").strip()
    if len(text) <= n:
        return text
    return text[:n] + "\n    ...[截断]..."


def _ok(name: str, passed: bool, detail: str = "") -> bool:
    extra = f"  {detail}" if detail else ""
    print(f"  [{'通过' if passed else '失败'}] {name}{extra}")
    return passed


def _print_plan_days(parsed: dict | None) -> list[str]:
    names: list[str] = []
    if not isinstance(parsed, dict):
        print("  (没有 trip_plan_parsed)")
        return names
    print(f"  城市={parsed.get('city')}  {parsed.get('start_date')} ~ {parsed.get('end_date')}")
    for day in parsed.get("days") or []:
        idx = day.get("day_index", "?")
        desc = day.get("description") or ""
        print(f"  Day {idx}: {desc}")
        for attr in day.get("attractions") or []:
            if isinstance(attr, dict):
                name = attr.get("name") or ""
                price = attr.get("ticket_price", "")
                print(f"    - {name}  票价={price}")
            else:
                name = str(attr)
                print(f"    - {name}")
            if name:
                names.append(name)
    budget = parsed.get("budget") or {}
    if budget:
        print(f"  parsed.budget={budget}")
    return names


def step_preflight(days: int, start: str, end: str) -> None:
    _banner("0) 环境")
    print(f"  city={CITY}  days={days}  {start} ~ {end}")
    print(f"  user_id={USER_ID}  preferences={PREFERENCES}")
    print(f"  DASHSCOPE_API_KEY={'已配置' if os.getenv('DASHSCOPE_API_KEY') else '缺失'}")
    print(f"  AMAP_MAPS_API_KEY={'已配置' if os.getenv('AMAP_MAPS_API_KEY') else '缺失'}")
    print(f"  REDIS={os.getenv('REDIS_HOST', 'localhost')}:{os.getenv('REDIS_PORT', '6379')}")
    print(f"  VECTOR_BACKEND={os.getenv('VECTOR_BACKEND', 'chroma')}")
    print(f"  CHROMA_PATH={os.getenv('CHROMA_PATH', 'app/data/chroma')}")


def step_knowledge_base() -> bool:
    _banner("1) 知识库 RAG（上海夜景）")
    from app.rag.vector_store import get_vector_store

    store = get_vector_store()
    stats = store.get_collection_stats()
    print(f"  collection={stats.get('collection_name')}  vectors={stats.get('vector_count')}")
    if stats.get("error") or not stats.get("vector_count"):
        print("  集合为空时规划仍会走高德，但 RAG 召回会弱。可先: python rebuild_kb.py")
        return _ok("知识库有数据", False)

    hits = store.search("夜景 外滩", filters={"city": CITY}, k=5)
    for i, item in enumerate(hits, 1):
        meta = item.get("metadata") or {}
        print(f"    {i}. {meta.get('title')}  score={item.get('score')}")
    leaked = [h for h in hits if (h.get("metadata") or {}).get("city") not in (None, CITY)]
    return all([
        _ok("检索到上海景点", len(hits) > 0, f"{len(hits)} 条"),
        _ok("未串到其他城市", not leaked),
    ])


def step_plan(days: int, start: str, end: str) -> tuple[bool, dict]:
    _banner("2) LangGraph 主图规划（并行景点/天气/酒店）")
    from app.agents.langgraph_builder import get_langgraph_agent

    print("  正在初始化 Agent（LLM + Chroma + 高德）...")
    t0 = time.time()
    agent = get_langgraph_agent(llm_provider="aliyun")
    print(f"  初始化耗时 {time.time() - t0:.1f}s")

    print("  正在 aplan_trip / plan_trip ...")
    t1 = time.time()
    result = agent.plan_trip(
        city=CITY,
        travel_days=days,
        start_date=start,
        end_date=end,
        transportation="地铁",
        accommodation="外滩附近中档酒店",
        preferences=PREFERENCES,
        user_id=USER_ID,
        free_text_input="想看夜景，白天少走路，不要安排太满",
        thread_id=USER_ID,
    )
    print(f"  规划耗时 {time.time() - t1:.1f}s")

    print("\n  --- 景点检索 ---")
    print("   ", _clip(result.get("attraction_response"), 800))
    print("\n  --- 天气节点 ---")
    print("   ", _clip(result.get("weather_response"), 500))
    print("\n  --- 酒店节点 ---")
    print("   ", _clip(result.get("hotel_response"), 500))
    print("\n  --- 行程原文 ---")
    print("   ", _clip(result.get("trip_plan"), 800))
    print("\n  --- 解析后的日程 ---")
    names = _print_plan_days(result.get("trip_plan_parsed"))
    print(f"\n  budget_total={result.get('budget_total')}  budget={result.get('budget')}")
    print(f"  session_id={result.get('session_id')}")
    print(f"  current_version_id={result.get('current_version_id')}")
    print(f"  reflect={result.get('reflection_feedback')}")

    parsed_ok = isinstance(result.get("trip_plan_parsed"), dict) and bool(
        (result.get("trip_plan_parsed") or {}).get("days")
    )
    ok = all([
        _ok("景点检索有输出", bool((result.get("attraction_response") or "").strip())),
        _ok("天气节点有输出", bool((result.get("weather_response") or "").strip())),
        _ok("酒店节点有输出", bool((result.get("hotel_response") or "").strip())),
        _ok("生成了可解析行程", parsed_ok, f"景点 {len(names)} 个"),
        _ok("写入了会话版本", bool(result.get("session_id"))),
    ])
    return ok, result


def step_long_term_memory(plan_names: list[str]) -> bool:
    _banner("3) Chroma 长期记忆（规划后应已写入）")
    from app.memory.memory_manager import TravelMemoryManager

    manager = TravelMemoryManager()
    print(f"  后端={manager.get_storage_type()}")
    if manager.get_storage_type() != "chroma":
        return _ok("Chroma 后端", False, manager.get_storage_type())

    visited = manager.get_visited_attractions(USER_ID, CITY)
    prefs = manager.get_preferences(USER_ID, CITY)
    comparison = manager.compare_plans(USER_ID, CITY)
    print(f"  已游: {visited}")
    print(f"  偏好: {prefs}")
    print(f"  同城历史: {comparison.get('plan_count')} 次")

    hits = manager.search_semantic(USER_ID, "上海夜景 少走路", k=5)
    for i, hit in enumerate(hits, 1):
        meta = hit.get("metadata") or {}
        print(f"    {i}. {meta.get('memory_type')} {meta.get('title')}  score={hit.get('score', 0):.3f}")

    overlap = set(visited) & set(plan_names)
    return all([
        _ok("有同城历史", comparison.get("has_history") is True),
        _ok("已游景点非空", len(visited) > 0, str(visited[:8])),
        _ok("行程景点进入长期记忆", bool(overlap) if plan_names else len(visited) > 0),
        _ok("语义检索有结果", len(hits) > 0),
    ])


def step_query_memory_node() -> bool:
    _banner("4) 再次进入 query_memory（应避开已游）")
    from app.agents.langgraph_builder import get_langgraph_agent

    agent = get_langgraph_agent(llm_provider="aliyun")
    result = agent.nodes.query_memory({"user_id": USER_ID, "city": CITY})
    info = result.get("memory_info") or ""
    visited = result.get("visited_attractions") or []
    print(f"  visited={visited}")
    for line in info.splitlines():
        print(f"    {line}")
    return all([
        _ok("读到已游", len(visited) > 0),
        _ok("提示推荐不同路线", "已访问" in info or "去过" in info or "不同" in info),
    ])


def _ensure_session(result: dict):
    from app.agents.version_manager import VersionManager

    vm = VersionManager(use_redis=True)
    session_id = result.get("session_id") or ""
    if session_id and vm.get_session(session_id):
        print(f"  使用图写入的 session  后端={'redis' if vm.use_redis else 'file'}  id={session_id}")
        return vm, session_id

    print("  图未返回可用 session，补建一个")
    session = vm.create_session(user_id=USER_ID, plan_id="shanghai_e2e")
    parsed = result.get("trip_plan_parsed") or {}
    names = []
    for day in parsed.get("days") or []:
        for attr in day.get("attractions") or []:
            name = attr.get("name") if isinstance(attr, dict) else str(attr)
            if name:
                names.append(name)
    vm.create_initial_version(session.session_id, {
        "city": CITY,
        "duration_days": parsed.get("days") and len(parsed.get("days")) or 1,
        "attractions": names,
        "budget": result.get("budget") or {},
        "daily_itinerary": parsed,
    })
    return vm, session.session_id


def step_dialog(result: dict) -> tuple[bool, object, str]:
    _banner("5) DialogAgent 改计划（滑动窗口 + Redis 版本）")
    from app.agents.dialog_agent import DialogAgent

    vm, session_id = _ensure_session(result)
    before = vm.get_current_version(session_id)
    before_attr = (before.plan_data or {}).get("attractions") if before else []
    print(f"  修改前景点: {before_attr}")

    agent = DialogAgent(
        llm_provider="aliyun",
        use_redis=vm.use_redis,
        enable_rag=True,
        enable_amap=True,
        enable_memory=True,
        context_window_size=6,
    )
    agent.version_manager = vm

    user_input = "把行程里加一个东方明珠，如果已经有就换成豫园夜景，白天不要排太满"
    print(f"  用户: {user_input}")
    t0 = time.time()
    dialog_result = agent.process_modification(session_id=session_id, user_input=user_input)
    print(f"  耗时 {time.time() - t0:.1f}s")
    print(f"  reply:\n    {_clip(dialog_result.get('reply'), 500)}")
    print(f"  changes={dialog_result.get('changes')}")
    print(f"  new_version_id={dialog_result.get('new_version_id')}")

    after = vm.get_current_version(session_id)
    after_attr = (after.plan_data or {}).get("attractions") if after else []
    print(f"  修改后景点: {after_attr}")

    history = vm.get_conversation_history(session_id, limit=100)
    window = agent.chat_memory.compress_records(history)
    print(f"  完整对话 {len(history)} 条 / 滑窗 {len(window)} 条")
    for item in window:
        role = "用户" if item.get("role") == "user" else "助手"
        print(f"    {role}: {_clip(item.get('content'), 80)}")

    versions = vm.format_versions_for_llm(session_id)
    print("  版本摘要:")
    for line in versions.splitlines():
        print(f"    {line}")

    ok = all([
        _ok("有对话回复", bool(dialog_result.get("reply"))),
        _ok("版本链可查询", "版本" in versions or "v1" in versions.lower() or "v" in versions),
        _ok("滑窗 <= 6", len(window) <= 6, str(len(window))),
    ])
    return ok, vm, session_id


def step_rollback(vm, session_id: str) -> bool:
    _banner("6) 回退到初始版本")
    history = vm.get_version_history(session_id)
    if len(history) < 2:
        return _ok("有可回退版本", False, f"只有 {len(history)} 个版本")

    target = history[0]
    print(f"  回退到 v{target.version_number} {target.version_id}")
    print(f"  目标景点: {(target.plan_data or {}).get('attractions')}")
    rolled = vm.rollback_to_version(session_id, target.version_id)
    current = vm.get_current_version(session_id)
    current_attr = (current.plan_data or {}).get("attractions") if current else []
    print(f"  回退后当前: {current_attr}")
    return all([
        _ok("回退成功", rolled is not None),
        _ok(
            "当前计划回到初始景点",
            current_attr == (target.plan_data or {}).get("attractions"),
        ),
    ])


def cleanup() -> None:
    _banner("清理")
    from app.memory.memory_manager import TravelMemoryManager

    manager = TravelMemoryManager()
    n = manager.clear_user_plans(USER_ID)
    print(f"  已删 {USER_ID} 长期记忆 {n} 条（{manager.get_storage_type()}）")
    print("  会话版本在 Redis/dialog_sessions，未自动删除。")


def main() -> int:
    parser = argparse.ArgumentParser(description="上海完整规划链路测试")
    parser.add_argument("--days", type=int, default=3, help="行程天数，默认 3")
    parser.add_argument("--skip-dialog", action="store_true", help="只跑规划+记忆，不改计划")
    parser.add_argument("--cleanup", action="store_true", help="只清理演示用户的 Chroma 记忆")
    args = parser.parse_args()

    if args.cleanup:
        cleanup()
        return 0

    start = (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d")
    end = (datetime.now() + timedelta(days=7 + max(args.days, 1) - 1)).strftime("%Y-%m-%d")
    step_preflight(args.days, start, end)

    results = []
    plan_result: dict = {}
    try:
        results.append(("知识库 RAG", step_knowledge_base()))
        plan_ok, plan_result = step_plan(args.days, start, end)
        results.append(("LangGraph 规划", plan_ok))

        parsed = plan_result.get("trip_plan_parsed") or {}
        names = []
        for day in parsed.get("days") or []:
            for attr in day.get("attractions") or []:
                name = attr.get("name") if isinstance(attr, dict) else str(attr)
                if name:
                    names.append(name)
        results.append(("Chroma 长期记忆", step_long_term_memory(names)))
        results.append(("query_memory 回读", step_query_memory_node()))

        if not args.skip_dialog:
            dialog_ok, vm, session_id = step_dialog(plan_result)
            results.append(("DialogAgent 改计划", dialog_ok))
            results.append(("版本回退", step_rollback(vm, session_id)))
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
    print("\n清理长期记忆: python tests/test_shanghai_e2e.py --cleanup")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
