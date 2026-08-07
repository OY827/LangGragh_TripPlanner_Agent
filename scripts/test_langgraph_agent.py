"""
LangGraph Agent 测试脚本

测试内容：
1. 导入验证
2. 景点子图独立测试
3. 天气/酒店子图测试
4. 主图端到端测试
5. Checkpointer 会话持久化测试
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()


def test_imports():
    """测试所有模块导入"""
    print("="*60)
    print("测试1: 模块导入")
    print("="*60)
    
    try:
        from app.agents.langgraph_state import (
            AttractionState, WeatherState, HotelState, TripState
        )
        print("  ✅ State 定义导入成功")
    except Exception as e:
        print(f"  ❌ State 导入失败: {e}")
        return False
    
    try:
        from app.agents.langgraph_nodes import GraphNodes
        print("  ✅ GraphNodes 导入成功")
    except Exception as e:
        print(f"  ❌ GraphNodes 导入失败: {e}")
        return False
    
    try:
        from app.agents.langgraph_builder import (
            LangGraphBuilder, TripPlannerAgent, get_langgraph_agent
        )
        print("  ✅ LangGraphBuilder 导入成功")
    except Exception as e:
        print(f"  ❌ LangGraphBuilder 导入失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    try:
        from app.agents.langgraph_hitl import (
            HumanInTheLoopManager, console_approver
        )
        print("  ✅ HITL 模块导入成功")
    except Exception as e:
        print(f"  ❌ HITL 导入失败: {e}")
        return False
    
    print("\n  🎉 所有模块导入成功！")
    return True


def test_attraction_subgraph():
    """测试景点子图"""
    print("\n" + "="*60)
    print("测试2: 景点子图（独立运行）")
    print("="*60)
    
    try:
        from app.agents.langgraph_builder import LangGraphBuilder
        
        builder = LangGraphBuilder()
        app = builder.get_attraction_graph()
        
        print("  正在运行景点子图...")
        start_time = time.time()
        
        result = app.invoke({
            "query": "历史文化古迹",
            "city": "上海",
            "preferences": ["历史文化"],
            "visited_attractions": [],
            "iteration_count": 0,
        })
        
        elapsed = time.time() - start_time
        
        print(f"  ✅ 子图运行完成（耗时: {elapsed:.1f}s）")
        print(f"     知识库文档数: {len(result.get('kb_docs', []))}")
        print(f"     高德结果长度: {len(result.get('amap_results', ''))}")
        print(f"     合并结果长度: {len(result.get('merged_results', ''))}")
        print(f"     反思结果: {result.get('reflection_feedback', '')}")
        print(f"     最终响应长度: {len(result.get('attraction_response', ''))}")
        
        if result.get('kb_docs'):
            print("\n  📋 知识库景点:")
            for doc in result['kb_docs'][:5]:
                meta = doc.get('metadata', {})
                title = meta.get('title', '未知')
                rating = meta.get('rating', 'N/A')
                print(f"    - {title} (评分: {rating})")
        
        return True
    except Exception as e:
        print(f"  ❌ 景点子图测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_weather_subgraph():
    """测试天气子图"""
    print("\n" + "="*60)
    print("测试3: 天气子图")
    print("="*60)
    
    try:
        from app.agents.langgraph_builder import LangGraphBuilder
        
        builder = LangGraphBuilder()
        app = builder.get_weather_graph()
        
        print("  正在查询天气...")
        result = app.invoke({
            "city": "上海",
            "start_date": "2026-07-29",
            "end_date": "2026-07-30",
        })
        
        print(f"  ✅ 天气查询完成")
        print(f"     天气响应: {result.get('weather_response', 'N/A')[:100]}...")
        return True
    except Exception as e:
        print(f"  ❌ 天气子图测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_hotel_subgraph():
    """测试酒店子图"""
    print("\n" + "="*60)
    print("测试4: 酒店子图")
    print("="*60)
    
    try:
        from app.agents.langgraph_builder import LangGraphBuilder
        
        builder = LangGraphBuilder()
        app = builder.get_hotel_graph()
        
        print("  正在搜索酒店...")
        result = app.invoke({
            "city": "上海",
            "accommodation": "经济型",
            "iteration_count": 0,
        })
        
        print(f"  ✅ 酒店搜索完成")
        print(f"     酒店响应: {result.get('hotel_response', 'N/A')[:100]}...")
        return True
    except Exception as e:
        print(f"  ❌ 酒店子图测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_full_trip_planner():
    """测试完整旅行规划"""
    print("\n" + "="*60)
    print("测试5: 完整旅行规划（主图端到端）")
    print("="*60)
    
    try:
        from app.agents.langgraph_builder import TripPlannerAgent
        
        print("  初始化 TripPlannerAgent...")
        agent = TripPlannerAgent()
        
        print("  开始规划旅行...")
        start_time = time.time()
        
        result = agent.plan_trip(
            city="上海",
            travel_days=2,
            start_date="2026-08-01",
            end_date="2026-08-02",
            transportation="地铁",
            accommodation="经济型",
            preferences=["历史文化", "古迹"],
            user_id="test_langgraph_user",
            free_text_input="希望安排博物馆和古镇",
        )
        
        elapsed = time.time() - start_time
        
        print(f"\n  ✅ 旅行规划完成（耗时: {elapsed:.1f}s）")
        print(f"     城市: {result.get('city', '')}")
        print(f"     迭代次数: {result.get('iteration_count', 0)}")
        print(f"     反思反馈: {result.get('reflection_feedback', '')}")
        print(f"     景点响应长度: {len(result.get('attraction_response', ''))}")
        print(f"     天气响应长度: {len(result.get('weather_response', ''))}")
        print(f"     酒店响应长度: {len(result.get('hotel_response', ''))}")
        
        parsed = result.get('trip_plan_parsed')
        if parsed:
            days = parsed.get('days', [])
            print(f"     行程天数: {len(days)}")
            for day in days:
                attractions = day.get('attractions', [])
                print(f"\n     📋 Day {day.get('day_index', '?')}: {len(attractions)} 个景点")
                
                # 打印景点
                for attr in attractions[:5]:
                    if isinstance(attr, dict):
                        name = attr.get('name', '?')
                        price = attr.get('ticket_price', 'N/A')
                        duration = attr.get('visit_duration', 120)
                        print(f"        🏛️  {name} (门票: {price}元, 建议游览: {duration}分钟)")
                    else:
                        print(f"        🏛️  {attr}")
                
                # 打印时间段
                time_slots = day.get('time_slots', [])
                if time_slots:
                    print(f"        ⏰ 时间安排:")
                    for ts in time_slots:
                        activity_type = ts.get('activity_type', '')
                        icon = "🏛️" if "景点" in activity_type else "🚶" if "交通" in activity_type else "🍽️"
                        print(f"           {ts.get('start_time', '')}-{ts.get('end_time', '')} {icon} {ts.get('activity_name', '')}")
                        if ts.get('description'):
                            print(f"              └ {ts['description']}")
                
                # 打印路线段
                route_segments = day.get('route_segments', [])
                if route_segments:
                    print(f"        🚇 交通路线:")
                    for seg in route_segments:
                        mode = seg.get('transport_mode', '')
                        detail = seg.get('route_detail', '')
                        dist = seg.get('distance', 0)
                        dur = seg.get('duration', 0)
                        if dist >= 1000:
                            dist_str = f"{dist/1000:.1f}公里"
                        else:
                            dist_str = f"{dist}米"
                        print(f"           {seg.get('start_point', '')} → {seg.get('end_point', '')}")
                        print(f"              └ {mode} | {dist_str} | 约{dur}分钟")
                        if detail:
                            print(f"              └ {detail}")
                
                # 打印餐饮
                meals = day.get('meals', [])
                if meals:
                    print(f"        🍽️ 餐饮安排:")
                    for meal in meals:
                        print(f"           {meal.get('type', '')}: {meal.get('name', '')} ({meal.get('estimated_cost', 0)}元)")
            
            # 预算输出
            budget = result.get('budget', {})
            if budget and isinstance(budget, dict):
                print(f"\n     💰 预算明细:")
                print(f"       景点门票: {budget.get('total_attractions', 0)} 元")
                print(f"       酒店住宿: {budget.get('total_hotels', 0)} 元 ({budget.get('hotel_details', {}).get('nights', 0)}晚)")
                print(f"       餐饮费用: {budget.get('total_meals', 0)} 元")
                print(f"       交通费用: {budget.get('total_transportation', 0)} 元")
                print(f"       总计: {budget.get('total', 0)} 元")
        
        return True
    except Exception as e:
        print(f"  ❌ 旅行规划测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_checkpointer_session():
    """测试 Checkpointer 会话持久化"""
    print("\n" + "="*60)
    print("测试6: Checkpointer 会话持久化")
    print("="*60)
    
    try:
        from langgraph.checkpoint.memory import MemorySaver
        from app.agents.langgraph_builder import LangGraphBuilder
        
        checkpointer = MemorySaver()
        builder = LangGraphBuilder()
        app = builder.get_trip_planner_with_checkpointer(checkpointer)
        
        thread_id = "test_session_001"
        config = {"configurable": {"thread_id": thread_id}}
        
        print("  首次调用（创建会话）...")
        result1 = app.invoke({
            "city": "杭州",
            "travel_days": 1,
            "preferences": ["自然风光"],
            "user_id": "test_user",
        }, config)
        
        print(f"  ✅ 首次调用完成")
        print(f"     迭代次数: {result1.get('iteration_count', 0)}")
        
        print("  尝试恢复会话（同一 thread_id）...")
        result2 = app.invoke(None, config)
        
        print(f"  ✅ 会话恢复成功")
        print(f"     阶段: {result2.get('current_phase', 'N/A')}")
        
        return True
    except Exception as e:
        print(f"  ❌ Checkpointer 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_hitl():
    """测试 Human-in-the-loop"""
    print("\n" + "="*60)
    print("测试7: Human-in-the-loop")
    print("="*60)
    
    try:
        from app.agents.langgraph_hitl import HumanInTheLoopManager
        
        hitl = HumanInTheLoopManager()
        
        def mock_approver(state):
            print(f"    审批状态: phase={state.get('current_phase')}")
            return {}  # 通过
        
        hitl.register_approver("test_review", mock_approver)
        
        result = hitl.wait_for_approval("test_review", {"current_phase": "plan"})
        print(f"  ✅ HITL 审批通过: {result}")
        
        result2 = hitl.wait_for_approval("nonexistent", {})
        print(f"  ✅ 未注册审批点默认通过: {result2}")
        
        return True
    except Exception as e:
        print(f"  ❌ HITL 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("\n" + "="*60)
    print("🚀 LangGraph Agent 完整测试")
    print("="*60)
    
    tests = [
        ("模块导入", test_imports),
        ("景点子图", test_attraction_subgraph),
        ("天气子图", test_weather_subgraph),
        ("酒店子图", test_hotel_subgraph),
        ("完整旅行规划", test_full_trip_planner),
        ("Checkpointer 持久化", test_checkpointer_session),
        ("Human-in-the-loop", test_hitl),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n  ❌ 测试 '{name}' 崩溃: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    print("\n" + "="*60)
    print("📊 测试结果汇总")
    print("="*60)
    
    for name, passed in results:
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"  {status}  {name}")
    
    passed_count = sum(1 for _, p in results if p)
    total_count = len(results)
    
    print(f"\n  通过率: {passed_count}/{total_count} ({passed_count*100//total_count}%)")
    
    return all(p for _, p in results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
