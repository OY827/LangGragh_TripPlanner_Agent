"""
简化测试：直接测试对话Agent和版本管理器
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.agents.dialog_agent import DialogAgent
from app.agents.version_manager import VersionManager
from app.agents.dialog_models import PlanVersion


def test_dialog_agent():
    """测试对话Agent核心功能"""
    print("\n" + "=" * 70)
    print("测试对话Agent核心功能")
    print("=" * 70)

    # 1. 初始化版本管理器
    print("\n📦 初始化版本管理器...")
    version_manager = VersionManager(use_redis=False)

    # 2. 创建会话和初始版本
    print("\n📝 创建会话和初始版本...")
    session = version_manager.create_session(user_id="test_user_001", plan_id="")

    initial_plan = {
        "city": "上海",
        "duration_days": 2,
        "attractions": ["外滩", "东方明珠", "豫园"],
        "budget": {
            "total_attractions": 300,
            "total_hotels": 400,
            "total_meals": 360,
            "total_transportation": 100,
            "total": 1160
        },
        "transportation": "公共交通",
        "accommodation": "经济型"
    }

    initial_version = version_manager.create_initial_version(
        session_id=session.session_id,
        plan_data=initial_plan
    )

    print(f"✅ 初始版本已创建: v{initial_version.version_number}")
    print(f"  Session ID: {session.session_id}")
    print(f"  Version ID: {initial_version.version_id}")

    # 3. 初始化对话Agent
    print("\n🤖 初始化对话Agent...")
    dialog_agent = DialogAgent(
        llm_provider="aliyun",
        use_redis=False,
        enable_rag=False,  # 禁用知识库加速测试
        enable_amap=False,
        enable_memory=False
    )

    # 4. 测试对话修改
    print("\n" + "=" * 70)
    print("测试对话修改")
    print("=" * 70)

    test_cases = [
        "添加迪士尼",
        "降低酒店预算",
        "延长到3天",
        "结束"
    ]

    for i, user_input in enumerate(test_cases, 1):
        print(f"\n[轮次 {i}] 用户输入: {user_input}")

        result = dialog_agent.process_modification(
            session_id=session.session_id,
            user_input=user_input
        )

        print(f"\n🤖 Agent回复：")
        print(result.get("reply", ""))

        if result.get("dialog_changes"):
            print(f"\n变更内容：")
            for change in result["dialog_changes"]:
                print(f"  • {change}")

        if result.get("new_version_id"):
            print(f"\n新版本ID: {result['new_version_id']}")

            # 获取新版本详情
            new_version = version_manager.get_version(result["new_version_id"])
            if new_version:
                plan = new_version.plan_data
                print(f"\n新计划：")
                print(f"  景点: {', '.join(plan.get('attractions', []))}")
                budget = plan.get('budget', {})
                if isinstance(budget, dict):
                    print(f"  预算: {budget.get('total', 0)}元")

    # 5. 测试版本历史
    print("\n" + "=" * 70)
    print("版本历史")
    print("=" * 70)

    history = version_manager.get_version_history(session.session_id)
    print(f"\n共有 {len(history)} 个版本：")
    for v in history:
        print(f"  v{v.version_number} - {v.created_by} - {v.created_at}")
        if v.changes:
            print(f"    变更: {'; '.join(v.changes)}")

    # 6. 测试版本对比
    if len(history) >= 2:
        print("\n" + "=" * 70)
        print("版本对比（v1 vs v2）")
        print("=" * 70)

        diff = version_manager.compare_versions(
            history[0].version_id,
            history[1].version_id
        )

        if diff:
            print(f"\n差异摘要: {diff.summary}")
            print(f"新增景点: {diff.added_attractions}")
            print(f"移除景点: {diff.removed_attractions}")
            if diff.budget_change:
                print(f"预算变化: {diff.budget_change}")

    print("\n" + "=" * 70)
    print("✅ 测试完成")
    print("=" * 70)


if __name__ == "__main__":
    test_dialog_agent()