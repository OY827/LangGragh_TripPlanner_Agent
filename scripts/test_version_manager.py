"""
版本管理器测试脚本
验证版本创建、查询、对比、回退等核心功能
"""

import sys
import os
import json
import shutil
from datetime import datetime

sys.path.insert(0, 'd:/Projects/LangChain')

from app.agents.dialog_models import (
    PlanVersion, DialogSession, ChangeIntent, PlanDiff,
    ChangeType, ModifyScope
)
from app.agents.version_manager import VersionManager


def cleanup_storage():
    """清理测试存储"""
    storage_path = "app/data/dialog_sessions"
    if os.path.exists(storage_path):
        shutil.rmtree(storage_path)


def test_create_session():
    """测试创建会话"""
    print("=" * 60)
    print("测试1：创建会话")
    print("=" * 60)
    
    vm = VersionManager(use_redis=False)
    session = vm.create_session(user_id="test_user_001", plan_id="plan_001")
    
    print(f"\n  会话ID: {session.session_id}")
    print(f"  用户ID: {session.user_id}")
    print(f"  计划ID: {session.plan_id}")
    print(f"  是否活跃: {session.is_active}")
    print(f"  版本数: {session.get_version_count()}")
    
    # 验证可以重新获取
    restored = vm.get_session(session.session_id)
    status = "通过" if restored and restored.user_id == "test_user_001" else "失败"
    print(f"\n测试结果: {status}")
    return vm, session


def test_create_initial_version(vm, session):
    """测试创建初始版本"""
    print("\n" + "=" * 60)
    print("测试2：创建初始版本")
    print("=" * 60)
    
    plan_data = {
        "city": "上海",
        "duration_days": 3,
        "attractions": ["外滩", "豫园", "南京路"],
        "budget": {"total": 1500, "total_attractions": 200, "total_hotels": 500},
        "daily_itinerary": {"Day 1": ["外滩", "豫园"]}
    }
    
    version = vm.create_initial_version(session.session_id, plan_data)
    
    print(f"\n  版本ID: {version.version_id}")
    print(f"  版本号: {version.version_number}")
    print(f"  父版本: {version.parent_version_id}")
    print(f"  创建者: {version.created_by}")
    print(f"  是否初始版本: {version.is_initial_version()}")
    print(f"  计划数据: {version.plan_data.get('city')}")
    
    # 验证重新获取
    restored = vm.get_version(version.version_id)
    status = "通过" if restored and restored.plan_data["city"] == "上海" else "失败"
    print(f"\n测试结果: {status}")
    return version


def test_create_modified_version(vm, session):
    """测试创建修改版本"""
    print("\n" + "=" * 60)
    print("测试3：创建修改版本")
    print("=" * 60)
    
    # 模拟修改后的计划数据
    modified_plan = {
        "city": "上海",
        "duration_days": 3,
        "attractions": ["外滩", "豫园", "南京路", "迪士尼"],
        "budget": {"total": 2000, "total_attractions": 800, "total_hotels": 500},
        "daily_itinerary": {"Day 1": ["外滩", "豫园"], "Day 2": ["迪士尼"]}
    }
    
    # 创建修改意图
    intent = ChangeIntent(
        change_type=ChangeType.ADD_ITEM.value,
        scope=ModifyScope.INCREMENTAL.value,
        description="添加迪士尼乐园",
        target_field="attractions",
        details={"added": ["迪士尼"]}
    )
    
    # 创建差异
    diff = PlanDiff(
        old_version_id=session.current_version_id,
        new_version_id="",  # 将在创建时填充
        summary="新增迪士尼乐园",
        added_attractions=["迪士尼"],
        budget_change={"old": 1500, "new": 2000, "delta": 500},
        change_descriptions=["新增景点: 迪士尼", "预算变化: 1500元 → 2000元 (增加500元)"]
    )
    
    version = vm.create_modified_version(
        session_id=session.session_id,
        plan_data=modified_plan,
        change_intent=intent,
        diff=diff,
        changes=["新增景点: 迪士尼", "预算增加500元"]
    )
    
    print(f"\n  版本ID: {version.version_id}")
    print(f"  版本号: {version.version_number}")
    print(f"  父版本: {version.parent_version_id}")
    print(f"  创建者: {version.created_by}")
    print(f"  变更描述: {version.changes}")
    print(f"  修改意图: {version.change_intent.get('change_type')}")
    
    status = "通过" if version.version_number == 2 and version.parent_version_id else "失败"
    print(f"\n测试结果: {status}")
    return version


def test_version_history(vm, session):
    """测试版本历史查询"""
    print("\n" + "=" * 60)
    print("测试4：版本历史查询")
    print("=" * 60)
    
    history = vm.get_version_history(session.session_id)
    
    print(f"\n  版本数量: {len(history)}")
    for v in history:
        print(f"  - v{v.version_number} ({v.version_id[:8]}): {v.created_by}, 父版本={v.parent_version_id or 'None'}")
    
    # 获取版本链
    chain = vm.get_version_chain(session.session_id)
    print(f"\n  版本链:")
    for item in chain:
        current = " ← 当前" if item["is_current"] else ""
        print(f"  - v{item['version_number']}: {item['version_id'][:8]}{current}")
    
    status = "通过" if len(history) == 2 else "失败"
    print(f"\n测试结果: {status}")
    return history


def test_compare_versions(vm, v1, v2):
    """测试版本对比"""
    print("\n" + "=" * 60)
    print("测试5：版本对比")
    print("=" * 60)
    
    diff = vm.compare_versions(v1.version_id, v2.version_id)
    
    print(f"\n  差异摘要: {diff.summary}")
    print(f"  新增景点: {diff.added_attractions}")
    print(f"  移除景点: {diff.removed_attractions}")
    print(f"  预算变化: {diff.budget_change}")
    print(f"  变化描述:")
    for desc in diff.change_descriptions:
        print(f"    - {desc}")
    
    checks = [
        len(diff.added_attractions) == 1,
        "迪士尼" in diff.added_attractions,
        diff.budget_change is not None,
        diff.budget_change["delta"] == 500
    ]
    
    status = "通过" if all(checks) else "失败"
    print(f"\n测试结果: {status}")
    return diff


def test_rollback(vm, session, target_version):
    """测试版本回退"""
    print("\n" + "=" * 60)
    print("测试6：版本回退")
    print("=" * 60)
    
    # 记录回退前的版本数
    before_count = len(vm.get_version_history(session.session_id))
    
    # 回退到目标版本
    rollback_version = vm.rollback_to_version(session.session_id, target_version.version_id)
    
    print(f"\n  回退到: v{target_version.version_number}")
    print(f"  新版本ID: {rollback_version.version_id}")
    print(f"  新版本号: {rollback_version.version_number}")
    print(f"  变更描述: {rollback_version.changes}")
    
    # 验证回退后的计划数据
    after_count = len(vm.get_version_history(session.session_id))
    current = vm.get_current_version(session.session_id)
    
    print(f"\n  回退前版本数: {before_count}")
    print(f"  回退后版本数: {after_count}")
    print(f"  当前版本号: {current.version_number}")
    print(f"  当前景点: {current.plan_data.get('attractions')}")
    
    checks = [
        rollback_version is not None,
        after_count == before_count + 1,  # 回退创建了新版本
        current.version_id == rollback_version.version_id,
        current.plan_data.get("attractions") == target_version.plan_data.get("attractions")
    ]
    
    status = "通过" if all(checks) else "失败"
    print(f"\n测试结果: {status}")
    return rollback_version


def test_add_message(vm, session):
    """测试消息管理"""
    print("\n" + "=" * 60)
    print("测试7：消息管理")
    print("=" * 60)
    
    # 添加用户消息
    user_msg = vm.add_message(
        session_id=session.session_id,
        role="user",
        content="帮我加上迪士尼乐园"
    )
    print(f"\n  用户消息: {user_msg.content}")
    
    # 添加助手回复
    assistant_msg = vm.add_message(
        session_id=session.session_id,
        role="assistant",
        content="已添加迪士尼乐园，预算调整为2000元。",
        version_id=session.current_version_id
    )
    print(f"  助手回复: {assistant_msg.content}")
    
    # 获取对话历史
    history = vm.get_conversation_history(session.session_id)
    print(f"\n  对话历史 ({len(history)} 条):")
    for msg in history:
        print(f"    [{msg['role']}] {msg['content']}")
    
    status = "通过" if len(history) == 2 else "失败"
    print(f"\n测试结果: {status}")
    return history


def test_user_sessions(vm):
    """测试用户会话列表"""
    print("\n" + "=" * 60)
    print("测试8：用户会话列表")
    print("=" * 60)
    
    sessions = vm.get_user_sessions("test_user_001")
    
    print(f"\n  用户会话数量: {len(sessions)}")
    for s in sessions:
        print(f"  - 会话ID: {s['session_id'][:12]}")
        print(f"    版本数: {s['version_count']}")
        print(f"    消息数: {s['message_count']}")
        print(f"    活跃: {s['is_active']}")
    
    status = "通过" if len(sessions) >= 1 else "失败"
    print(f"\n测试结果: {status}")
    return sessions


def test_session_summary(vm, session):
    """测试会话摘要"""
    print("\n" + "=" * 60)
    print("测试9：会话摘要")
    print("=" * 60)
    
    summary = vm.get_session_summary(session.session_id)
    
    print(f"\n  会话ID: {summary['session_id']}")
    print(f"  版本数: {summary['version_count']}")
    print(f"  消息数: {summary['message_count']}")
    print(f"  当前版本: {summary['current_version_id'][:8] if summary['current_version_id'] else 'None'}")
    print(f"  当前计划城市: {summary['current_plan'].get('city') if summary['current_plan'] else 'None'}")
    
    status = "通过" if summary and summary['version_count'] >= 2 else "失败"
    print(f"\n测试结果: {status}")
    return summary


def test_close_session(vm, session):
    """测试关闭会话"""
    print("\n" + "=" * 60)
    print("测试10：关闭会话")
    print("=" * 60)
    
    result = vm.close_session(session.session_id)
    
    restored = vm.get_session(session.session_id)
    print(f"\n  关闭结果: {result}")
    print(f"  会话状态: {'活跃' if restored.is_active else '已关闭'}")
    
    status = "通过" if result and not restored.is_active else "失败"
    print(f"\n测试结果: {status}")
    return result


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("版本管理器测试")
    print("=" * 60)
    
    # 清理旧数据
    cleanup_storage()
    
    results = []
    
    # 测试1: 创建会话
    try:
        vm, session = test_create_session()
        results.append(("创建会话", "通过"))
    except Exception as e:
        results.append(("创建会话", f"异常: {e}"))
        import traceback; traceback.print_exc()
        return
    
    # 测试2: 创建初始版本
    try:
        v1 = test_create_initial_version(vm, session)
        results.append(("创建初始版本", "通过"))
    except Exception as e:
        results.append(("创建初始版本", f"异常: {e}"))
        import traceback; traceback.print_exc()
        return
    
    # 测试3: 创建修改版本
    try:
        v2 = test_create_modified_version(vm, session)
        results.append(("创建修改版本", "通过"))
    except Exception as e:
        results.append(("创建修改版本", f"异常: {e}"))
        import traceback; traceback.print_exc()
        return
    
    # 测试4: 版本历史
    try:
        test_version_history(vm, session)
        results.append(("版本历史查询", "通过"))
    except Exception as e:
        results.append(("版本历史查询", f"异常: {e}"))
        import traceback; traceback.print_exc()
    
    # 测试5: 版本对比
    try:
        test_compare_versions(vm, v1, v2)
        results.append(("版本对比", "通过"))
    except Exception as e:
        results.append(("版本对比", f"异常: {e}"))
        import traceback; traceback.print_exc()
    
    # 测试6: 版本回退
    try:
        test_rollback(vm, session, v1)
        results.append(("版本回退", "通过"))
    except Exception as e:
        results.append(("版本回退", f"异常: {e}"))
        import traceback; traceback.print_exc()
    
    # 测试7: 消息管理
    try:
        test_add_message(vm, session)
        results.append(("消息管理", "通过"))
    except Exception as e:
        results.append(("消息管理", f"异常: {e}"))
        import traceback; traceback.print_exc()
    
    # 测试8: 用户会话列表
    try:
        test_user_sessions(vm)
        results.append(("用户会话列表", "通过"))
    except Exception as e:
        results.append(("用户会话列表", f"异常: {e}"))
        import traceback; traceback.print_exc()
    
    # 测试9: 会话摘要
    try:
        test_session_summary(vm, session)
        results.append(("会话摘要", "通过"))
    except Exception as e:
        results.append(("会话摘要", f"异常: {e}"))
        import traceback; traceback.print_exc()
    
    # 测试10: 关闭会话
    try:
        test_close_session(vm, session)
        results.append(("关闭会话", "通过"))
    except Exception as e:
        results.append(("关闭会话", f"异常: {e}"))
        import traceback; traceback.print_exc()
    
    # 打印总结
    print("\n" + "=" * 60)
    print("测试结果总结")
    print("=" * 60)
    
    passed = sum(1 for _, s in results if s == "通过")
    failed = sum(1 for _, s in results if s != "通过")
    
    for name, status in results:
        icon = "✅" if status == "通过" else "❌"
        print(f"{icon} {name}: {status}")
    
    print("-" * 60)
    print(f"总计: {passed}/{len(results)} 测试通过")
    print(f"通过率: {passed/len(results)*100:.1f}%")
    
    if failed == 0:
        print("\n🎉 所有测试通过！版本管理器工作正常。")
    else:
        print(f"\n⚠️ 有 {failed} 个测试失败，请检查。")


if __name__ == "__main__":
    main()
