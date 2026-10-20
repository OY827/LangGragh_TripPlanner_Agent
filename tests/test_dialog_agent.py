"""
测试对话Agent与LangGraph集成

测试流程：
1. 规划旅行（上海，2天）
2. 进入对话模式
3. 模拟用户修改请求（添加迪士尼、降低酒店预算）
4. 验证版本管理和差异计算
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.orchestrator import TripPlannerAgent
from app.agents.version_manager import VersionManager


def test_dialog_integration():
    """测试对话Agent集成到LangGraph"""
    print("\n" + "=" * 70)
    print("测试对话Agent集成到LangGraph")
    print("=" * 70)

    # 使用固定的thread_id
    TEST_THREAD_ID = "test_dialog_001"

    # 初始化Agent
    print("\n📦 初始化TripPlannerAgent...")
    agent = TripPlannerAgent(
        llm_provider="aliyun",
        use_redis=False
    )

    # 规划旅行（传入固定的thread_id）
    print("\n🗺️  开始规划旅行...")
    result = agent.plan_trip(
        city="上海",
        travel_days=2,
        transportation="公共交通",
        accommodation="经济型",
        preferences=["历史文化"],
        user_id="test_user_001",
        thread_id=TEST_THREAD_ID  # 传入固定的thread_id
    )

    print("\n📋 规划结果：")
    print(f"  城市: {result.get('city', '')}")
    print(f"  预算: {result.get('budget_total', 0)}元")
    print(f"  session_id: {result.get('session_id', '')}")
    print(f"  version_id: {result.get('current_version_id', '')}")

    budget = result.get('budget', {})
    if isinstance(budget, dict):
        print(f"  景点门票: {budget.get('total_attractions', 0)}元")
        print(f"  酒店住宿: {budget.get('total_hotels', 0)}元")
        print(f"  餐饮费用: {budget.get('total_meals', 0)}元")
        print(f"  交通费用: {budget.get('total_transportation', 0)}元")

    trip_plan = result.get('trip_plan', '')
    if trip_plan:
        print(f"\n行程安排:\n{trip_plan[:500]}...")

    # 从结果中获取session_id
    session_id = result.get('session_id', '')
    current_version_id = result.get('current_version_id', '')

    if not session_id:
        # 如果结果中没有session_id，尝试从state中获取
        config = {"configurable": {"thread_id": TEST_THREAD_ID}}
        state_values = agent.app.get_state(config).values
        session_id = state_values.get("session_id", "")
        current_version_id = state_values.get("current_version_id", "")
        print(f"\n⚠️  从state获取: session_id={session_id}")

    print(f"\n✅ 规划完成，thread_id: {TEST_THREAD_ID}")
    print(result.get("dialog_reply") or "")

    user_inputs = [
        "添加迪士尼",
        "降低酒店预算",
        "结束",
    ]

    for i, user_input in enumerate(user_inputs, 1):
        print(f"\n[轮次 {i}] 用户输入: {user_input}")
        dialog_result = agent.resume_dialog(TEST_THREAD_ID, user_input)
        if dialog_result.get("error"):
            print(f"⚠️  恢复失败: {dialog_result['error']}")
            break

        print("\n🤖 Agent回复：")
        print(dialog_result.get("dialog_reply", ""))

        if dialog_result.get("dialog_changes"):
            print("\n变更内容：")
            for change in dialog_result["dialog_changes"]:
                print(f"  • {change}")

        if dialog_result.get("current_version_id"):
            print(f"\n新版本ID: {dialog_result['current_version_id']}")

        if not dialog_result.get("awaiting_input", True):
            print("\n✅ 对话结束")
            break

    # 测试版本历史
    if session_id:
        print("\n" + "=" * 70)
        print("版本历史")
        print("=" * 70)

        version_manager = VersionManager(use_redis=False)
        history = version_manager.get_version_history(session_id)
        print(f"\n共有 {len(history)} 个版本：")
        for v in history:
            print(f"  v{v.version_number} - {v.created_by} - {v.created_at}")
            if v.changes:
                print(f"    变更: {'; '.join(v.changes)}")

    print("\n" + "=" * 70)
    print("✅ 测试完成")
    print("=" * 70)


if __name__ == "__main__":
    test_dialog_integration()