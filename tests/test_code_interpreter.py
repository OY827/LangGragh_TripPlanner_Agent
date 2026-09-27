"""
代码解释器工具测试脚本
验证 CodeInterpreterTool 的各项功能
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.tools.agent_tools import CodeInterpreterTool


def test_basic_math():
    """测试基本数学运算"""
    print("=" * 60)
    print("测试1：基本数学运算")
    print("=" * 60)
    
    tool = CodeInterpreterTool()
    
    code = """
# 计算圆的面积和周长
import math
radius = 5
area = math.pi * radius ** 2
circumference = 2 * math.pi * radius
print(f"半径: {radius}")
print(f"面积: {area:.2f}")
print(f"周长: {circumference:.2f}")
"""
    
    result = tool._run(code)
    print(f"结果:\n{result}\n")
    return "通过" if "面积" in result else "失败"


def test_budget_calculation():
    """测试预算计算场景"""
    print("=" * 60)
    print("测试2：旅行预算计算")
    print("=" * 60)
    
    tool = CodeInterpreterTool()
    
    code = """
# 计算3天上海行程预算
days = 3
attractions = 200  # 景点门票
hotel_per_night = 400  # 酒店每晚
meal_per_day = 150  # 餐饮每天
transport_per_day = 50  # 交通每天

total_hotels = hotel_per_night * (days - 1)  # 2晚
total_meals = meal_per_day * days
total_transport = transport_per_day * days

total = attractions + total_hotels + total_meals + total_transport

print("💰 上海3天行程预算明细:")
print("-" * 30)
print(f"景点门票: {attractions}元")
print(f"酒店住宿: {total_hotels}元 ({days-1}晚)")
print(f"餐饮费用: {total_meals}元 ({days}天)")
print(f"交通费用: {total_transport}元 ({days}天)")
print("-" * 30)
print(f"总计: {total}元")
"""
    
    result = tool._run(code)
    print(f"结果:\n{result}\n")
    return "通过" if "总计" in result and "1850" in result else "失败"


def test_data_processing():
    """测试数据处理"""
    print("=" * 60)
    print("测试3：数据处理与转换")
    print("=" * 60)
    
    tool = CodeInterpreterTool()
    
    code = """
# 处理酒店搜索结果，计算平均价格
import json

hotel_data = [
    {"name": "如家酒店", "price": "288元/晚", "rating": 4.5},
    {"name": "锦江之星", "price": "356元/晚", "rating": 4.3},
    {"name": "7天连锁", "price": "198元/晚", "rating": 4.0},
    {"name": "汉庭酒店", "price": "420元/晚", "rating": 4.7},
    {"name": "全季酒店", "price": "588元/晚", "rating": 4.8},
]

# 提取价格并计算
prices = [int(h["price"].replace("元/晚", "")) for h in hotel_data]
avg_price = sum(prices) / len(prices)
min_price = min(prices)
max_price = max(prices)

# 按评分排序
sorted_hotels = sorted(hotel_data, key=lambda x: x["rating"], reverse=True)

print("📊 酒店数据分析结果:")
print("-" * 30)
print(f"酒店数量: {len(hotel_data)}")
print(f"价格范围: {min_price}-{max_price}元/晚")
print(f"平均价格: {avg_price:.0f}元/晚")
print("-" * 30)
print("按评分排序:")
for i, h in enumerate(sorted_hotels, 1):
    print(f"{i}. {h['name']} - {h['price']} (评分: {h['rating']})")
"""
    
    result = tool._run(code)
    print(f"结果:\n{result}\n")
    return "通过" if "平均价格" in result else "失败"


def test_statistical_analysis():
    """测试统计分析"""
    print("=" * 60)
    print("测试4：统计分析")
    print("=" * 60)
    
    tool = CodeInterpreterTool()
    
    code = """
# 分析各景点评分分布
import statistics
from collections import Counter

ratings = [4.5, 4.3, 4.0, 4.7, 4.8, 4.2, 4.6, 4.1, 4.9, 4.4]

mean_rating = statistics.mean(ratings)
median_rating = statistics.median(ratings)
stdev_rating = statistics.stdev(ratings)
max_rating = max(ratings)
min_rating = min(ratings)

# 评分分布
rating_dist = Counter()
for r in ratings:
    bucket = int(r)
    rating_dist[f"{bucket}.x分"] += 1

print("📈 景点评分统计分析:")
print("-" * 30)
print(f"样本数量: {len(ratings)}")
print(f"平均评分: {mean_rating:.2f}")
print(f"中位数: {median_rating:.2f}")
print(f"标准差: {stdev_rating:.2f}")
print(f"最高分: {max_rating}")
print(f"最低分: {min_rating}")
print("-" * 30)
print("评分分布:")
for bucket, count in sorted(rating_dist.items()):
    print(f"  {bucket}: {'█' * count} ({count}个)")
"""
    
    result = tool._run(code)
    print(f"结果:\n{result}\n")
    return "通过" if "平均评分" in result else "失败"


def test_security_restriction():
    """测试安全限制 - 验证危险操作被阻止"""
    print("=" * 60)
    print("测试5：安全限制验证")
    print("=" * 60)
    
    tool = CodeInterpreterTool()
    
    # 测试1：尝试导入os模块
    print("5.1 尝试导入os模块:")
    result = tool._run("import os; print(os.getcwd())")
    print(f"结果: {result}")
    blocked1 = "错误" in result or "NameError" in result or "ModuleNotFoundError" in result
    
    # 测试2：尝试打开文件
    print("\n5.2 尝试打开文件:")
    result = tool._run("with open('test.txt', 'r') as f: print(f.read())")
    print(f"结果: {result}")
    blocked2 = "错误" in result or "NameError" in result
    
    # 测试3：尝试使用eval
    print("\n5.3 尝试使用eval:")
    result = tool._run("eval('__import__(\"os\").system(\"ls\")')")
    print(f"结果: {result}")
    blocked3 = "错误" in result or "NameError" in result
    
    all_blocked = blocked1 and blocked2 and blocked3
    print(f"\n安全限制: {'✅ 所有危险操作已阻止' if all_blocked else '❌ 部分限制失效'}")
    return "通过" if all_blocked else "失败"


def test_timeout():
    """测试超时控制"""
    print("=" * 60)
    print("测试6：超时控制")
    print("=" * 60)
    
    tool = CodeInterpreterTool()
    
    # 尝试执行一个耗时操作
    code = """
# 计算大量数据（应该在10秒内完成，但测试超时机制）
import time
start = time.time()
total = 0
for i in range(1000000):
    total += i * i
elapsed = time.time() - start
print(f"计算完成，耗时: {elapsed:.2f}秒")
print(f"结果: {total}")
"""
    
    result = tool._run(code)
    print(f"结果:\n{result}\n")
    
    # 测试真正的超时（一个无限循环）
    print("测试无限循环超时:")
    code = """
while True:
    pass
"""
    result = tool._run(code)
    print(f"结果: {result}")
    timeout_worked = "超时" in result or "Timeout" in result
    print(f"超时控制: {'✅ 正常工作' if timeout_worked else '❌ 失效'}")
    return "通过" if timeout_worked else "失败"


def test_error_handling():
    """测试错误处理"""
    print("=" * 60)
    print("测试7：错误处理")
    print("=" * 60)
    
    tool = CodeInterpreterTool()
    
    # 测试语法错误
    print("7.1 语法错误:")
    result = tool._run("if True print('error')")
    print(f"结果: {result}")
    has_syntax_error = "语法" in result or "SyntaxError" in result or "错误" in result
    
    # 测试运行时错误
    print("\n7.2 运行时错误:")
    result = tool._run("1 / 0")
    print(f"结果: {result}")
    has_runtime_error = "除零" in result or "ZeroDivisionError" in result or "错误" in result
    
    # 测试空输出
    print("\n7.3 无输出:")
    result = tool._run("x = 42")
    print(f"结果: {result}")
    has_no_output = "没有输出" in result
    
    all_errors = has_syntax_error and has_runtime_error and has_no_output
    print(f"\n错误处理: {'✅ 正常' if all_errors else '❌ 异常'}")
    return "通过" if all_errors else "失败"


def test_optimization_problem():
    """测试优化问题"""
    print("=" * 60)
    print("测试8：简单优化问题")
    print("=" * 60)
    
    tool = CodeInterpreterTool()
    
    code = """
# 景点路线优化 - 找出最优游览顺序
# 景点间的距离（公里）
distances = {
    ("外滩", "豫园"): 2,
    ("外滩", "南京路"): 3,
    ("外滩", "陆家嘴"): 5,
    ("豫园", "南京路"): 1,
    ("豫园", "陆家嘴"): 4,
    ("南京路", "陆家嘴"): 3,
}

# 使用贪心算法找出从外滩出发的最短路线
from itertools import permutations

attractions = ["外滩", "豫园", "南京路", "陆家嘴"]

def get_distance(from_a, to_b):
    if from_a == to_b:
        return 0
    if (from_a, to_b) in distances:
        return distances[(from_a, to_b)]
    if (to_b, from_a) in distances:
        return distances[(to_b, from_a)]
    return 10  # 默认大距离

# 找出最短路线（从外滩出发）
start = "外滩"
other_attractions = [a for a in attractions if a != start]

best_route = None
min_distance = float('inf')

for perm in permutations(other_attractions):
    route = [start] + list(perm)
    total_dist = sum(get_distance(route[i], route[i+1]) for i in range(len(route)-1))
    if total_dist < min_distance:
        min_distance = total_dist
        best_route = route

print("🗺️ 景点路线优化结果:")
print("-" * 30)
print(f"起点: {start}")
print(f"最优路线: {' → '.join(best_route)}")
print(f"总距离: {min_distance}公里")
print("-" * 30)
print("各段距离:")
for i in range(len(best_route)-1):
    d = get_distance(best_route[i], best_route[i+1])
    print(f"  {best_route[i]} → {best_route[i+1]}: {d}公里")
"""
    
    result = tool._run(code)
    print(f"结果:\n{result}\n")
    return "通过" if "最优路线" in result else "失败"


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("代码解释器工具功能测试")
    print("=" * 60)
    
    tests = [
        ("基本数学运算", test_basic_math),
        ("预算计算", test_budget_calculation),
        ("数据处理", test_data_processing),
        ("统计分析", test_statistical_analysis),
        ("安全限制", test_security_restriction),
        ("超时控制", test_timeout),
        ("错误处理", test_error_handling),
        ("优化问题", test_optimization_problem),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            status = test_func()
            results.append((name, status))
        except Exception as e:
            results.append((name, f"异常: {str(e)}"))
    
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
        print("\n🎉 所有测试通过！代码解释器工具功能正常。")
    else:
        print(f"\n⚠️ 有 {failed} 个测试失败，请检查。")


if __name__ == "__main__":
    main()