"""
旅行记忆查询功能测试脚本（简化版，不依赖langchain）
验证日期筛选、预算筛选、景点筛选和统计分析功能
"""

import sys
import os
import json
from datetime import datetime, timedelta

# 直接导入 memory_manager，避免触发 app/__init__.py 的依赖
sys.path.insert(0, 'd:/Projects/LangChain')

# 手动导入模块，避免触发 app/__init__.py
module_path = 'd:/Projects/LangChain/app/memory/memory_manager.py'
import importlib.machinery
loader = importlib.machinery.SourceFileLoader("memory_manager", module_path)
memory_manager_module = loader.load_module()
TravelMemoryManager = memory_manager_module.TravelMemoryManager


def create_test_data(manager, user_id):
    """创建测试数据"""
    plans = [
        {
            "city": "上海",
            "duration_days": 3,
            "attractions": ["外滩", "豫园", "南京路", "陆家嘴"],
            "daily_itinerary": {
                "Day 1": ["外滩观光", "豫园游览"],
                "Day 2": ["南京路购物", "陆家嘴金融区"],
                "Day 3": ["新天地", "田子坊"]
            },
            "preferences": ["历史文化", "现代都市"],
            "budget": {
                "total": 1850,
                "total_attractions": 200,
                "total_hotels": 443,
                "total_meals": 360,
                "total_transportation": 100
            },
            "created_at": (datetime.now() - timedelta(days=30)).isoformat()
        },
        {
            "city": "杭州",
            "duration_days": 2,
            "attractions": ["西湖", "灵隐寺", "雷峰塔"],
            "daily_itinerary": {
                "Day 1": ["西湖游船", "雷峰塔"],
                "Day 2": ["灵隐寺", "河坊街"]
            },
            "preferences": ["自然风光", "历史文化"],
            "budget": {
                "total": 1200,
                "total_attractions": 150,
                "total_hotels": 300,
                "total_meals": 240,
                "total_transportation": 80
            },
            "created_at": (datetime.now() - timedelta(days=60)).isoformat()
        },
        {
            "city": "上海",
            "duration_days": 5,
            "attractions": ["外滩", "迪士尼", "科技馆", "海洋馆", "东方明珠"],
            "daily_itinerary": {
                "Day 1": ["外滩", "南京路"],
                "Day 2": ["迪士尼乐园"],
                "Day 3": ["科技馆", "海洋馆"],
                "Day 4": ["东方明珠", "陆家嘴"],
                "Day 5": ["朱家角古镇"]
            },
            "preferences": ["主题乐园", "亲子游"],
            "budget": {
                "total": 3500,
                "total_attractions": 800,
                "total_hotels": 1200,
                "total_meals": 900,
                "total_transportation": 200
            },
            "created_at": (datetime.now() - timedelta(days=90)).isoformat()
        },
        {
            "city": "北京",
            "duration_days": 4,
            "attractions": ["故宫", "长城", "颐和园", "天安门", "天坛"],
            "daily_itinerary": {
                "Day 1": ["天安门", "故宫"],
                "Day 2": ["长城一日游"],
                "Day 3": ["颐和园", "天坛"],
                "Day 4": ["王府井", "三里屯"]
            },
            "preferences": ["历史文化", "名胜古迹"],
            "budget": {
                "total": 2800,
                "total_attractions": 300,
                "total_hotels": 800,
                "total_meals": 720,
                "total_transportation": 150
            },
            "created_at": (datetime.now() - timedelta(days=180)).isoformat()
        },
        {
            "city": "成都",
            "duration_days": 3,
            "attractions": ["大熊猫基地", "宽窄巷子", "锦里古街", "都江堰"],
            "daily_itinerary": {
                "Day 1": ["大熊猫基地", "宽窄巷子"],
                "Day 2": ["都江堰", "青城山"],
                "Day 3": ["锦里古街", "武侯祠"]
            },
            "preferences": ["美食", "文化体验"],
            "budget": {
                "total": 1500,
                "total_attractions": 100,
                "total_hotels": 450,
                "total_meals": 450,
                "total_transportation": 100
            },
            "created_at": (datetime.now() - timedelta(days=15)).isoformat()
        }
    ]
    
    plan_ids = []
    for plan_data in plans:
        plan_id = manager.save_plan(
            user_id=user_id,
            city=plan_data["city"],
            duration_days=plan_data["duration_days"],
            attractions=plan_data["attractions"],
            daily_itinerary=plan_data["daily_itinerary"],
            preferences=plan_data.get("preferences", []),
            budget=plan_data.get("budget"),
            notes=f"测试计划 - {plan_data['city']}"
        )
        plan_ids.append(plan_id)
    
    return plan_ids


def test_date_range_query(manager, user_id):
    """测试日期范围查询"""
    print("=" * 60)
    print("测试1：日期范围查询")
    print("=" * 60)
    
    # 查询最近30天的计划
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    
    plans = manager.get_plans_by_date_range(user_id, start_date, end_date)
    print(f"\n最近30天的行程（{start_date} 到 {end_date}）:")
    print(f"  找到 {len(plans)} 条计划")
    for plan in plans:
        print(f"  - {plan.city} ({plan.duration_days}天) - {plan.created_at[:10]}")
    
    # 查询最近90天的计划
    start_date_90 = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
    plans_90 = manager.get_plans_by_date_range(user_id, start_date_90, end_date)
    print(f"\n最近90天的行程（{start_date_90} 到 {end_date}）:")
    print(f"  找到 {len(plans_90)} 条计划")
    
    status = "通过" if len(plans) >= 1 and len(plans_90) >= len(plans) else "失败"
    print(f"\n测试结果: {status}")
    return status


def test_budget_range_query(manager, user_id):
    """测试预算范围查询"""
    print("\n" + "=" * 60)
    print("测试2：预算范围查询")
    print("=" * 60)
    
    # 查询预算在1000-2000元之间的计划
    plans = manager.get_plans_by_budget_range(user_id, min_budget=1000, max_budget=2000)
    print("\n预算在1000-2000元之间的行程:")
    print(f"  找到 {len(plans)} 条计划")
    for plan in plans:
        budget_total = 0
        if isinstance(plan.budget, dict):
            budget_total = plan.budget.get("total", 0)
        elif isinstance(plan.budget, (int, float)):
            budget_total = plan.budget
        print(f"  - {plan.city} ({plan.duration_days}天) - 预算: {budget_total}元")
    
    # 查询预算大于3000元的计划
    plans_high = manager.get_plans_by_budget_range(user_id, min_budget=3000)
    print(f"\n预算大于3000元的行程:")
    print(f"  找到 {len(plans_high)} 条计划")
    
    status = "通过" if len(plans) >= 1 else "失败"
    print(f"\n测试结果: {status}")
    return status


def test_attraction_query(manager, user_id):
    """测试景点查询"""
    print("\n" + "=" * 60)
    print("测试3：景点查询")
    print("=" * 60)
    
    # 查询包含"外滩"的计划
    plans = manager.get_plans_by_attraction(user_id, "外滩")
    print("\n包含'外滩'的行程:")
    print(f"  找到 {len(plans)} 条计划")
    for plan in plans:
        print(f"  - {plan.city} ({plan.duration_days}天) - 景点: {plan.attractions}")
    
    # 查询包含"大熊猫"的计划
    plans_panda = manager.get_plans_by_attraction(user_id, "大熊猫")
    print(f"\n包含'大熊猫'的行程:")
    print(f"  找到 {len(plans_panda)} 条计划")
    
    status = "通过" if len(plans) >= 2 and len(plans_panda) == 1 else "失败"
    print(f"\n测试结果: {status}")
    return status


def test_combined_search(manager, user_id):
    """测试组合搜索"""
    print("\n" + "=" * 60)
    print("测试4：组合条件搜索")
    print("=" * 60)
    
    # 搜索预算在1000-2000元之间的上海行程
    plans = manager.search_plans(user_id, city="上海", min_budget=1000, max_budget=2000)
    print("\n预算1000-2000元的上海行程:")
    print(f"  找到 {len(plans)} 条计划")
    for plan in plans:
        print(f"  - {plan.city} ({plan.duration_days}天)")
    
    # 搜索3天以上的行程
    plans_long = manager.search_plans(user_id, min_days=3)
    print(f"\n3天以上的行程:")
    print(f"  找到 {len(plans_long)} 条计划")
    for plan in plans_long:
        print(f"  - {plan.city} ({plan.duration_days}天)")
    
    status = "通过" if len(plans) >= 1 else "失败"
    print(f"\n测试结果: {status}")
    return status


def test_statistics(manager, user_id):
    """测试统计分析"""
    print("\n" + "=" * 60)
    print("测试5：用户旅行统计")
    print("=" * 60)
    
    stats = manager.get_user_statistics(user_id)
    
    print(f"\n📊 旅行统计数据:")
    print("-" * 40)
    print(f"  总行程数: {stats['total_trips']}")
    print(f"  去过城市数: {stats['total_cities']}")
    print(f"  总旅行天数: {stats['total_days']}天")
    
    if stats.get('has_budget_data'):
        print(f"  预算总计: {stats['total_budget']}元")
        print(f"  平均预算: {stats['avg_budget']}元")
        print(f"  最低预算: {stats['min_budget']}元")
        print(f"  最高预算: {stats['max_budget']}元")
    
    print(f"\n  去过的城市: {', '.join(stats['cities_visited'])}")
    
    if stats['popular_attractions']:
        print("\n  热门景点 Top5:")
        for i, (name, count) in enumerate(stats['popular_attractions'][:5], 1):
            print(f"    {i}. {name} (去过{count}次)")
    
    if stats['budget_distribution']:
        print("\n  预算分布:")
        for range_name, count in stats['budget_distribution'].items():
            bar = '█' * count if count > 0 else '░'
            print(f"    {range_name}: {bar} ({count}次)")
    
    status = "通过" if stats['total_trips'] >= 5 and stats['total_cities'] >= 4 else "失败"
    print(f"\n测试结果: {status}")
    return status


def test_duration_query(manager, user_id):
    """测试天数查询"""
    print("\n" + "=" * 60)
    print("测试6：按天数查询")
    print("=" * 60)
    
    # 查询3天以上的行程
    plans = manager.get_plans_by_duration(user_id, min_days=3)
    print(f"\n3天以上的行程:")
    print(f"  找到 {len(plans)} 条计划")
    for plan in plans:
        print(f"  - {plan.city} ({plan.duration_days}天)")
    
    # 查询4-5天的行程
    plans_mid = manager.get_plans_by_duration(user_id, min_days=4, max_days=5)
    print(f"\n4-5天的行程:")
    print(f"  找到 {len(plans_mid)} 条计划")
    
    status = "通过" if len(plans) >= 2 else "失败"
    print(f"\n测试结果: {status}")
    return status


def test_memory_query_tool_simple(manager, user_id):
    """测试简单的查询接口（不依赖langchain）"""
    print("\n" + "=" * 60)
    print("测试7：高级查询接口")
    print("=" * 60)
    
    # 测试城市筛选
    plans_shanghai = manager.get_user_plans_by_city(user_id, "上海")
    print(f"\n7.1 上海行程: 找到 {len(plans_shanghai)} 条计划")
    
    # 测试历史对比
    comparison = manager.compare_plans(user_id, "上海")
    print(f"7.2 上海历史对比:")
    print(f"    有历史记录: {comparison['has_history']}")
    print(f"    历史行程数: {comparison['plan_count']}")
    print(f"    已游览景点: {comparison['visited_attractions']}")
    
    status = "通过" if comparison['has_history'] and comparison['plan_count'] >= 2 else "失败"
    print(f"\n测试结果: {status}")
    return status


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("旅行记忆查询功能测试")
    print("=" * 60)
    
    # 初始化管理器（使用文件存储，不需要Redis）
    storage_path = "d:/Projects/LangChain/app/data/travel_plans"
    os.makedirs(storage_path, exist_ok=True)
    
    manager = TravelMemoryManager(use_redis=False)
    user_id = "test_query_user_001"
    
    # 清空旧数据
    manager.clear_user_plans(user_id)
    
    # 创建测试数据
    print("\n📝 创建测试数据...")
    plan_ids = create_test_data(manager, user_id)
    print(f"  创建了 {len(plan_ids)} 条测试计划")
    for pid in plan_ids:
        print(f"    - {pid}")
    
    # 运行测试
    tests = [
        ("日期范围查询", test_date_range_query),
        ("预算范围查询", test_budget_range_query),
        ("景点查询", test_attraction_query),
        ("组合条件搜索", test_combined_search),
        ("用户旅行统计", test_statistics),
        ("天数查询", test_duration_query),
        ("高级查询接口", test_memory_query_tool_simple),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            status = test_func(manager, user_id)
            results.append((name, status))
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
        print("\n🎉 所有测试通过！旅行记忆查询功能正常工作。")
    else:
        print(f"\n⚠️ 有 {failed} 个测试失败，请检查。")


if __name__ == "__main__":
    main()
