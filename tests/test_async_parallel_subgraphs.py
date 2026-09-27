"""
验证景点/天气/酒店子图异步并发

对比：
1. 串行 await 三个子图
2. asyncio.gather 并发三个子图

期望：并发总耗时明显小于串行之和（接近最慢子图耗时）。
"""

import sys
import os
import time
import asyncio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()


async def main():
    print("=" * 70)
    print("异步并发子图验证")
    print("=" * 70)

    from app.agents.langgraph_builder import LangGraphBuilder

    builder = LangGraphBuilder()
    state = {
        "city": "上海",
        "query": "上海的历史文化相关景点",
        "preferences": ["历史文化"],
        "visited_attractions": [],
        "start_date": "2026-07-20",
        "end_date": "2026-07-22",
        "accommodation": "经济型",
        "iteration_count": 0,
    }

    # 1) 串行
    print("\n[1] 串行执行三个子图...")
    t0 = time.time()
    r1 = await builder._run_attraction_subgraph(state)
    t_attr = time.time()
    r2 = await builder._run_weather_subgraph(state)
    t_weather = time.time()
    r3 = await builder._run_hotel_subgraph(state)
    t_hotel = time.time()

    serial_total = t_hotel - t0
    print(f"  attractions: {t_attr - t0:.2f}s, response_len={len(r1.get('attraction_response', ''))}")
    print(f"  weather:     {t_weather - t_attr:.2f}s, response_len={len(r2.get('weather_response', ''))}")
    print(f"  hotels:      {t_hotel - t_weather:.2f}s, response_len={len(r3.get('hotel_response', ''))}")
    print(f"  串行合计:    {serial_total:.2f}s")

    # 2) 并发 gather
    print("\n[2] asyncio.gather 并发执行...")
    t0 = time.time()
    merged = await builder.run_parallel_subgraphs(state)
    parallel_total = time.time() - t0
    print(f"  attraction_len={len(merged.get('attraction_response', ''))}")
    print(f"  weather_len={len(merged.get('weather_response', ''))}")
    print(f"  hotel_len={len(merged.get('hotel_response', ''))}")
    print(f"  并发合计:    {parallel_total:.2f}s")

    speedup = serial_total / parallel_total if parallel_total > 0 else 0
    print(f"\n加速比: {speedup:.2f}x  (串行 {serial_total:.2f}s → 并发 {parallel_total:.2f}s)")

    if parallel_total < serial_total * 0.85:
        print("✅ 并发生效：总耗时明显低于串行")
    else:
        print("⚠️ 并发加速不明显（可能受模型冷启动/限流影响，可再跑一次）")

    # 3) 主图 ainvoke 冒烟（只跑到并行阶段较重，这里做编译+短路径检查）
    print("\n[3] 主图异步节点注册检查...")
    app = builder.get_trip_planner_graph()
    print(f"  ✅ 主图编译成功: {type(app).__name__}")
    print("  节点 attractions/weather/hotels 为 async，需通过 agent.aplan_trip / ainvoke 触发并发")

    print("\n" + "=" * 70)
    print("验证完成")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
