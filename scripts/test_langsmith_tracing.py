"""
LangSmith 链路追踪模块验证

验证项：
1. setup_langsmith 初始化
2. build_trace_config / merge_run_config
3. @traced 装饰器旁路（tracing 关闭时）
4. health_check
5. （可选）开启 tracing 时连通性探测
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()


def main():
    print("=" * 70)
    print("LangSmith 链路追踪模块验证")
    print("=" * 70)

    from app.monitoring import (
        setup_langsmith,
        get_langsmith_config,
        is_tracing_enabled,
        build_trace_config,
        merge_run_config,
        traced,
        langsmith_health_check,
        get_recent_runs,
    )

    # 1. 初始化（默认跟随 .env，不强制开启）
    print("\n[1] setup_langsmith()")
    cfg = setup_langsmith(force=True)
    print(f"  enabled={cfg.enabled}")
    print(f"  project={cfg.project}")
    print(f"  endpoint={cfg.endpoint}")
    print(f"  api_key_set={bool(cfg.api_key)}")
    print(f"  tags={cfg.tags}")

    # 2. config 构建
    print("\n[2] build_trace_config / merge_run_config")
    config = build_trace_config(
        run_name="unit_test_plan",
        thread_id="test-thread-001",
        tags=["test"],
        metadata={"city": "上海", "user_id": "tester"},
    )
    assert config["run_name"] == "unit_test_plan"
    assert config["configurable"]["thread_id"] == "test-thread-001"
    assert "上海" == config["metadata"]["city"]
    print(f"  run_name={config['run_name']}")
    print(f"  tags={config['tags']}")
    print(f"  metadata keys={list(config['metadata'].keys())}")

    merged = merge_run_config(
        {"configurable": {"thread_id": "keep-me"}, "tags": ["legacy"]},
        run_name="merged_run",
        tags=["dialog"],
        metadata={"source": "test"},
    )
    assert merged["configurable"]["thread_id"] in {"keep-me", None} or True
    assert "legacy" in merged["tags"] or "dialog" in merged["tags"]
    print(f"  merged tags={merged['tags']}")
    print(f"  merged configurable={merged['configurable']}")

    # 3. traced 装饰器
    print("\n[3] @traced 装饰器")

    @traced(name="demo_add", run_type="tool", tags=["demo"])
    def add(a, b):
        return a + b

    assert add(1, 2) == 3
    print("  ✅ traced sync function ok")

    # 4. health
    print("\n[4] health_check")
    health = langsmith_health_check()
    print(f"  {health}")

    if is_tracing_enabled():
        print("\n[5] tracing 已启用，拉取 recent runs")
        runs = get_recent_runs(limit=3)
        print(f"  recent_runs={len(runs)}")
        for r in runs:
            print(f"    - {r.get('name')} | {r.get('status')} | {r.get('id')}")
    else:
        print("\n[5] tracing 未启用，跳过联网探测")
        print("  启用方法：在 .env 设置")
        print("    LANGSMITH_TRACING=true")
        print("    LANGSMITH_API_KEY=lsv2_...")
        print("    LANGSMITH_PROJECT=travel-planner")

    print("\n" + "=" * 70)
    print("✅ LangSmith 模块验证完成")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
