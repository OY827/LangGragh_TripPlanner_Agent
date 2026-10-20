"""量推荐、天气、酒店三路：串行相加，对照同一输入下的并行墙钟。"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from evaluation.metrics import latency_reduction


def _state() -> dict:
    start = datetime.now() + timedelta(days=14)
    end = start + timedelta(days=1)
    return {
        "city": "上海",
        "start_date": start.strftime("%Y-%m-%d"),
        "end_date": end.strftime("%Y-%m-%d"),
        "travel_days": 1,
        "transportation": "公共交通",
        "accommodation": "经济型",
        "hotel_budget": "",
        "preferences": ["历史文化"],
        "user_id": "fanout_latency",
        "free_text_input": "",
    }


def _calls(nodes, state):
    weather = {
        "city": state.get("city", ""),
        "start_date": state.get("start_date", ""),
        "end_date": state.get("end_date", ""),
    }
    hotels = {
        "city": state.get("city", ""),
        "accommodation": state.get("accommodation", ""),
        "hotel_budget": state.get("hotel_budget", ""),
        "start_date": state.get("start_date", ""),
        "travel_days": state.get("travel_days", 1),
    }
    return (
        ("recommend", nodes.recommend_attractions, dict(state)),
        ("weather", nodes.fetch_weather, weather),
        ("hotels", nodes.search_hotels, hotels),
    )


def _run_serial(nodes, state) -> dict:
    branches = []
    started = time.perf_counter()
    for name, fn, payload in _calls(nodes, state):
        t0 = time.perf_counter()
        fn(payload)
        branches.append({"name": name, "seconds": round(time.perf_counter() - t0, 3)})
    wall = time.perf_counter() - started
    return {"wall_s": round(wall, 3), "branches": branches}


async def _run_parallel(nodes, state) -> dict:
    async def one(name, fn, payload):
        t0 = time.perf_counter()
        await asyncio.to_thread(fn, payload)
        return {"name": name, "seconds": round(time.perf_counter() - t0, 3)}

    started = time.perf_counter()
    branches = await asyncio.gather(
        *(one(name, fn, payload) for name, fn, payload in _calls(nodes, state))
    )
    wall = time.perf_counter() - started
    return {"wall_s": round(wall, 3), "branches": list(branches)}


def main() -> None:
    from app.agents.langgraph_nodes import GraphNodes

    print("初始化节点（不计入对比，不进入规划）...", flush=True)
    nodes = GraphNodes(llm_provider="aliyun")
    state = _state()
    prepared = nodes.build_attraction_query(state)
    state.update(prepared)

    print("串行：推荐 → 天气 → 酒店", flush=True)
    serial = _run_serial(nodes, state)
    print("并行：三路同时出发", flush=True)
    parallel = asyncio.run(_run_parallel(nodes, state))

    serial_s = serial["wall_s"]
    parallel_s = parallel["wall_s"]
    report = {
        "city": state["city"],
        "preferences": state["preferences"],
        "travel_days": state["travel_days"],
        "serial": serial,
        "parallel": parallel,
        "saved_s": round(serial_s - parallel_s, 3),
        "wait_reduction": round(latency_reduction(serial_s, parallel_s), 4),
    }
    out = Path("evaluation/results/fanout_latency.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    print(f"串行合计: {serial_s:.2f} 秒")
    for item in serial["branches"]:
        print(f"  {item['name']}: {item['seconds']:.2f} 秒")
    print(f"并行墙钟: {parallel_s:.2f} 秒")
    for item in parallel["branches"]:
        print(f"  {item['name']}: {item['seconds']:.2f} 秒")
    print(f"少等: {report['saved_s']:.2f} 秒")
    print(f"降幅: {report['wait_reduction'] * 100:.1f}%")
    print(f"结果已写入 {out.resolve()}")


if __name__ == "__main__":
    main()
