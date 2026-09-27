"""
对话 Agent 数据模型测试脚本
验证 PlanVersion、DialogSession、ChangeIntent、PlanDiff 等数据模型的正确性
"""

import os
import sys
import json
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# 直接导入模块，避免触发 app/__init__.py
import importlib.machinery
loader = importlib.machinery.SourceFileLoader(
    "dialog_models",
    os.path.join(ROOT, "app", "agents", "dialog_models.py"),
)
dialog_models = loader.load_module()

from dialog_models import (
    PlanVersion, DialogSession, DialogMessage,
    ChangeIntent, PlanDiff,
    ChangeType, ModifyScope
)


def test_change_intent():
    """测试修改意图模型"""
    print("=" * 60)
    print("测试1：修改意图模型 (ChangeIntent)")
    print("=" * 60)
    
    # 创建预算修改意图
    intent = ChangeIntent(
        change_type=ChangeType.MODIFY_BUDGET.value,
        scope=ModifyScope.INCREMENTAL.value,
        description="用户希望将预算降低到1000元以内",
        target_field="budget",
        details={"max_budget": 1000, "action": "reduce"},
        confidence=0.95
    )
    
    print(f"\n  意图ID: {intent.intent_id}")
    print(f"  类型: {intent.change_type}")
    print(f"  范围: {intent.scope}")
    print(f"  描述: {intent.description}")
    print(f"  目标字段: {intent.target_field}")
    print(f"  细节: {intent.details}")
    print(f"  置信度: {intent.confidence}")
    
    # 测试序列化/反序列化
    intent_dict = intent.to_dict()
    intent_restored = ChangeIntent.from_dict(intent_dict)
    
    status = "通过" if intent_restored.change_type == intent.change_type else "失败"
    print(f"\n  序列化/反序列化: {status}")
    print(f"\n测试结果: {status}")
    return status


def test_plan_diff():
    """测试计划差异模型"""
    print("\n" + "=" * 60)
    print("测试2：计划差异模型 (PlanDiff)")
    print("=" * 60)
    
    diff = PlanDiff(
        old_version_id="v001",
        new_version_id="v002",
        summary="减少了预算并替换了一个景点"
    )
    
    # 添加变化
    diff.added_attractions = ["迪士尼"]
    diff.removed_attractions = ["科技馆"]
    diff.budget_change = {"old": 1850, "new": 1200, "delta": -650}
    diff.change_descriptions = [
        "新增景点: 迪士尼",
        "移除景点: 科技馆",
        "预算变化: 1850元 → 1200元 (节省650元)"
    ]
    
    print(f"\n  差异ID: {diff.diff_id}")
    print(f"  旧版本: {diff.old_version_id}")
    print(f"  新版本: {diff.new_version_id}")
    print(f"  摘要: {diff.summary}")
    print(f"  新增景点: {diff.added_attractions}")
    print(f"  移除景点: {diff.removed_attractions}")
    print(f"  预算变化: {diff.budget_change}")
    print(f"  变化描述:")
    for desc in diff.change_descriptions:
        print(f"    - {desc}")
    
    # 测试 has_changes
    has_changes = diff.has_changes()
    print(f"\n  有实际变化: {has_changes}")
    
    # 测试序列化
    diff_dict = diff.to_dict()
    diff_restored = PlanDiff.from_dict(diff_dict)
    status = "通过" if diff_restored.has_changes() == has_changes else "失败"
    print(f"\n测试结果: {status}")
    return status


def test_plan_version():
    """测试计划版本模型"""
    print("\n" + "=" * 60)
    print("测试3：计划版本模型 (PlanVersion)")
    print("=" * 60)
    
    # 创建初始版本
    v1 = PlanVersion(
        plan_id="plan_001",
        version_number=1,
        plan_data={
            "city": "上海",
            "duration_days": 3,
            "attractions": ["外滩", "豫园", "南京路"],
            "budget": {"total": 1500}
        },
        created_by="planner"
    )
    
    print(f"\n  版本1 (初始版本):")
    print(f"    版本ID: {v1.version_id}")
    print(f"    计划ID: {v1.plan_id}")
    print(f"    版本号: {v1.version_number}")
    print(f"    父版本: {v1.parent_version_id}")
    print(f"    是否初始版本: {v1.is_initial_version()}")
    print(f"    城市: {v1.get_plan_field('city')}")
    print(f"    预算: {v1.get_plan_field('budget')}")
    
    # 创建修改版本
    v2 = PlanVersion(
        plan_id="plan_001",
        parent_version_id=v1.version_id,
        version_number=2,
        plan_data={
            "city": "上海",
            "duration_days": 3,
            "attractions": ["外滩", "豫园", "南京路", "迪士尼"],
            "budget": {"total": 2000}
        },
        changes=["新增景点: 迪士尼", "预算调整: 1500→2000"],
        created_by="dialog"
    )
    
    print(f"\n  版本2 (修改版本):")
    print(f"    版本ID: {v2.version_id}")
    print(f"    父版本: {v2.parent_version_id}")
    print(f"    版本号: {v2.version_number}")
    print(f"    是否初始版本: {v2.is_initial_version()}")
    print(f"    变更描述: {v2.changes}")
    print(f"    创建者: {v2.created_by}")
    
    # 测试序列化
    v1_dict = v1.to_dict()
    v1_restored = PlanVersion.from_dict(v1_dict)
    status = "通过" if v1_restored.is_initial_version() and not v2.is_initial_version() else "失败"
    print(f"\n测试结果: {status}")
    return status


def test_dialog_message():
    """测试对话消息模型"""
    print("\n" + "=" * 60)
    print("测试4：对话消息模型 (DialogMessage)")
    print("=" * 60)
    
    # 用户消息
    user_msg = DialogMessage(
        role="user",
        content="帮我加上迪士尼乐园"
    )
    
    # 助手回复消息
    assistant_msg = DialogMessage(
        role="assistant",
        content="好的，我已将迪士尼乐园添加到您的行程中。新版本预算调整为2000元。",
        version_id="v002",
        intent={"change_type": "add_item", "description": "添加迪士尼"},
        diff={"added_attractions": ["迪士尼"], "budget_change": {"old": 1500, "new": 2000}}
    )
    
    print(f"\n  用户消息:")
    print(f"    ID: {user_msg.message_id}")
    print(f"    角色: {user_msg.role}")
    print(f"    内容: {user_msg.content}")
    print(f"    时间: {user_msg.timestamp}")
    
    print(f"\n  助手消息:")
    print(f"    ID: {assistant_msg.message_id}")
    print(f"    角色: {assistant_msg.role}")
    print(f"    内容: {assistant_msg.content}")
    print(f"    关联版本: {assistant_msg.version_id}")
    print(f"    关联意图: {assistant_msg.intent}")
    
    # 测试序列化
    msg_dict = user_msg.to_dict()
    msg_restored = DialogMessage.from_dict(msg_dict)
    status = "通过" if msg_restored.role == "user" else "失败"
    print(f"\n测试结果: {status}")
    return status


def test_dialog_session():
    """测试对话会话模型"""
    print("\n" + "=" * 60)
    print("测试5：对话会话模型 (DialogSession)")
    print("=" * 60)
    
    # 创建会话
    session = DialogSession(
        user_id="user_001",
        plan_id="plan_001"
    )
    
    print(f"\n  会话信息:")
    print(f"    会话ID: {session.session_id}")
    print(f"    用户ID: {session.user_id}")
    print(f"    计划ID: {session.plan_id}")
    print(f"    是否活跃: {session.is_active}")
    print(f"    创建时间: {session.created_at}")
    
    # 创建初始版本
    v1 = PlanVersion(
        plan_id="plan_001",
        version_number=1,
        plan_data={"city": "上海", "budget": {"total": 1500}},
        created_by="planner"
    )
    session.add_version(v1)
    
    # 添加用户消息
    user_msg = DialogMessage(role="user", content="帮我加上迪士尼")
    session.add_message(user_msg)
    
    # 创建修改版本
    v2 = PlanVersion(
        plan_id="plan_001",
        parent_version_id=v1.version_id,
        version_number=2,
        plan_data={"city": "上海", "budget": {"total": 2000}},
        changes=["新增景点: 迪士尼"],
        created_by="dialog"
    )
    session.add_version(v2)
    
    # 添加助手回复
    assistant_msg = DialogMessage(
        role="assistant",
        content="已添加迪士尼，预算调整为2000元",
        version_id=v2.version_id
    )
    session.add_message(assistant_msg)
    
    print(f"\n  会话状态 (添加2个版本和2条消息后):")
    print(f"    当前版本: {session.current_version_id}")
    print(f"    版本数量: {session.get_version_count()}")
    print(f"    消息数量: {session.get_message_count()}")
    print(f"    版本链: {session.get_version_chain()}")
    print(f"    最近消息: {len(session.get_recent_messages(5))} 条")
    print(f"    可以回退: {session.can_rollback()}")
    
    # 测试序列化
    session_dict = session.to_dict()
    session_restored = DialogSession.from_dict(session_dict)
    
    checks = [
        session_restored.get_version_count() == 2,
        session_restored.get_message_count() == 2,
        session_restored.can_rollback() == True,
        session_restored.current_version_id == v2.version_id
    ]
    
    status = "通过" if all(checks) else "失败"
    print(f"\n  序列化/反序列化验证:")
    print(f"    版本数量一致: {checks[0]}")
    print(f"    消息数量一致: {checks[1]}")
    print(f"    可回退状态一致: {checks[2]}")
    print(f"    当前版本一致: {checks[3]}")
    print(f"\n测试结果: {status}")
    return status


def test_full_workflow():
    """测试完整工作流"""
    print("\n" + "=" * 60)
    print("测试6：完整工作流模拟")
    print("=" * 60)
    
    # 1. 创建会话
    session = DialogSession(user_id="user_001", plan_id="plan_001")
    print(f"\n  [1] 创建会话: {session.session_id}")
    
    # 2. 创建初始版本（由规划Agent生成）
    v1 = PlanVersion(
        plan_id="plan_001",
        version_number=1,
        plan_data={
            "city": "上海",
            "duration_days": 3,
            "attractions": ["外滩", "豫园", "南京路"],
            "budget": {"total": 1500, "total_attractions": 200, "total_hotels": 500}
        },
        created_by="planner"
    )
    session.add_version(v1)
    
    # 3. 用户提出修改需求
    user_msg1 = DialogMessage(role="user", content="酒店太贵了，换便宜点的")
    session.add_message(user_msg1)
    print(f"  [2] 用户请求: {user_msg1.content}")
    
    # 4. 对话Agent分析意图
    intent = ChangeIntent(
        change_type=ChangeType.MODIFY_HOTEL.value,
        scope=ModifyScope.INCREMENTAL.value,
        description="降低酒店预算",
        target_field="budget",
        details={"action": "reduce_hotel_cost", "target_reduction": 200}
    )
    print(f"  [3] 意图识别: {intent.change_type} - {intent.description}")
    
    # 5. 生成新版本
    v2 = PlanVersion(
        plan_id="plan_001",
        parent_version_id=v1.version_id,
        version_number=2,
        plan_data={
            "city": "上海",
            "duration_days": 3,
            "attractions": ["外滩", "豫园", "南京路"],
            "budget": {"total": 1300, "total_attractions": 200, "total_hotels": 300}
        },
        changes=["酒店预算从500元降至300元"],
        change_intent=intent.to_dict(),
        created_by="dialog"
    )
    session.add_version(v2)
    
    # 6. 生成差异
    diff = PlanDiff(
        old_version_id=v1.version_id,
        new_version_id=v2.version_id,
        summary="降低酒店预算200元",
        budget_change={"old": 1500, "new": 1300, "delta": -200},
        change_descriptions=["酒店预算: 500元 → 300元 (节省200元)", "总预算: 1500元 → 1300元"]
    )
    
    # 7. 助手回复
    assistant_msg1 = DialogMessage(
        role="assistant",
        content="已为您调整酒店预算，从500元降至300元，总预算现为1300元。",
        version_id=v2.version_id,
        intent=intent.to_dict(),
        diff=diff.to_dict()
    )
    session.add_message(assistant_msg1)
    print(f"  [4] 助手回复: {assistant_msg1.content}")
    
    # 8. 用户再次修改
    user_msg2 = DialogMessage(role="user", content="加上迪士尼乐园")
    session.add_message(user_msg2)
    print(f"  [5] 用户请求: {user_msg2.content}")
    
    # 9. 生成第三版
    v3 = PlanVersion(
        plan_id="plan_001",
        parent_version_id=v2.version_id,
        version_number=3,
        plan_data={
            "city": "上海",
            "duration_days": 3,
            "attractions": ["外滩", "豫园", "南京路", "迪士尼"],
            "budget": {"total": 2100, "total_attractions": 800, "total_hotels": 300}
        },
        changes=["新增景点: 迪士尼", "景点预算从200元增至800元"],
        created_by="dialog"
    )
    session.add_version(v3)
    
    assistant_msg2 = DialogMessage(
        role="assistant",
        content="已添加迪士尼乐园，景点预算调整为800元，总预算现为2100元。",
        version_id=v3.version_id
    )
    session.add_message(assistant_msg2)
    print(f"  [6] 助手回复: {assistant_msg2.content}")
    
    # 10. 输出会话总结
    print(f"\n  📊 会话总结:")
    print(f"    总版本数: {session.get_version_count()}")
    print(f"    总消息数: {session.get_message_count()}")
    print(f"    版本链: {' → '.join(session.get_version_chain())}")
    print(f"    当前版本: {session.current_version_id}")
    print(f"    可以回退: {session.can_rollback()}")
    
    # 验证版本链
    version_chain = session.get_version_chain()
    checks = [
        len(version_chain) == 3,
        version_chain[0] == v1.version_id,
        version_chain[1] == v2.version_id,
        version_chain[2] == v3.version_id,
        v2.parent_version_id == v1.version_id,
        v3.parent_version_id == v2.version_id,
    ]
    
    status = "通过" if all(checks) else "失败"
    print(f"\n  版本链验证: {status}")
    print(f"\n测试结果: {status}")
    return status


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("对话 Agent 数据模型测试")
    print("=" * 60)
    
    tests = [
        ("修改意图模型", test_change_intent),
        ("计划差异模型", test_plan_diff),
        ("计划版本模型", test_plan_version),
        ("对话消息模型", test_dialog_message),
        ("对话会话模型", test_dialog_session),
        ("完整工作流模拟", test_full_workflow),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            status = test_func()
            if isinstance(status, str):
                results.append((name, status))
            else:
                results.append((name, "通过"))
        except Exception as e:
            results.append((name, f"异常: {str(e)}"))
            import traceback
            traceback.print_exc()
    
    # 打印测试总结
    print("\n" + "=" * 60)
    print("测试结果总结")
    print("=" * 60)
    
    passed = sum(1 for _, status in results if status == "通过")
    failed = sum(1 for _, status in results if status != "通过")
    
    for name, status in results:
        icon = "✅" if status == "通过" else "❌"
        print(f"{icon} {name}: {status}")
    
    print("-" * 60)
    print(f"总计: {passed}/{len(results)} 测试通过")
    print(f"通过率: {passed/len(results)*100:.1f}%")
    
    if failed == 0:
        print("\n🎉 所有测试通过！数据模型设计正确。")
    else:
        print(f"\n⚠️ 有 {failed} 个测试失败，请检查。")


if __name__ == "__main__":
    main()
