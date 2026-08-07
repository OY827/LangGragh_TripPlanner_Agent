"""
代码解释器工具简单测试
"""

import sys
sys.path.insert(0, 'd:/Projects/LangChain')

from app.tools.agent_tools import CodeInterpreterTool

def main():
    tool = CodeInterpreterTool()
    
    print("=" * 50)
    print("代码解释器工具测试")
    print("=" * 50)
    
    # 测试1：基本数学运算
    print("\n测试1：基本数学运算")
    result = tool._run("import math; print(math.pi * 5**2)")
    print(f"结果: {result}")
    
    # 测试2：预算计算
    print("\n测试2：预算计算")
    code = """
days = 3
attractions = 200
hotel_per_night = 400
meal_per_day = 150
transport_per_day = 50
total_hotels = hotel_per_night * (days - 1)
total_meals = meal_per_day * days
total_transport = transport_per_day * days
total = attractions + total_hotels + total_meals + total_transport
print("预算总计:", total, "元")
"""
    result = tool._run(code)
    print(f"结果: {result}")
    
    # 测试3：数据处理
    print("\n测试3：数据处理")
    code = """
import json
hotel_data = [
    {"name": "如家", "price": "288元/晚"},
    {"name": "锦江", "price": "356元/晚"},
    {"name": "汉庭", "price": "420元/晚"},
]
prices = [int(h["price"].replace("元/晚", "")) for h in hotel_data]
avg_price = sum(prices) / len(prices)
print(f"平均价格: {avg_price:.0f}元/晚")
"""
    result = tool._run(code)
    print(f"结果: {result}")
    
    # 测试4：安全限制 - 尝试导入os模块
    print("\n测试4：安全限制 - 尝试导入os模块")
    result = tool._run("import os; print(os.getcwd())")
    print(f"结果: {result[:100]}")
    is_blocked = "错误" in result or "NameError" in result or "ModuleNotFoundError" in result
    print(f"已阻止: {'是' if is_blocked else '否'}")
    
    # 测试5：安全限制 - 尝试使用open
    print("\n测试5：安全限制 - 尝试使用open")
    result = tool._run("f = open('test.txt', 'r'); print(f.read())")
    print(f"结果: {result[:100]}")
    is_blocked = "错误" in result or "NameError" in result
    print(f"已阻止: {'是' if is_blocked else '否'}")
    
    # 测试6：超时控制 - 无限循环
    print("\n测试6：超时控制 - 无限循环")
    result = tool._run("while True: pass")
    print(f"结果: {result[:100]}")
    timeout_worked = "超时" in result
    print(f"超时生效: {'是' if timeout_worked else '否'}")
    
    # 测试7：错误处理
    print("\n测试7：错误处理 - 语法错误")
    result = tool._run("if True print('error')")
    print(f"结果: {result[:100]}")
    
    # 测试8：测试支持的模块
    print("\n测试8：测试支持的模块")
    code = """
import math, json, itertools, datetime, re, statistics, collections

# 测试statistics模块
data = [4.5, 4.3, 4.0, 4.7, 4.8]
avg = statistics.mean(data)
print(f"统计模块: 平均评分为 {avg:.2f}")

# 测试collections模块
counter = collections.Counter([1, 2, 2, 3, 3, 3])
print(f"Collections模块: {dict(counter)}")

# 测试datetime模块
now = datetime.datetime.now()
print(f"Datetime模块: 当前时间 {now.strftime('%Y-%m-%d %H:%M')}")
"""
    result = tool._run(code)
    print(f"结果:\n{result}")
    
    print("\n" + "=" * 50)
    print("测试完成！")
    print("=" * 50)


if __name__ == "__main__":
    main()