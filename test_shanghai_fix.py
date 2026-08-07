"""测试上海历史文化景点推荐"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.tools.agent_tools import SmartAttractionTool

def test_shanghai_historical():
    """测试上海历史文化景点搜索"""
    print("=== 测试上海历史文化景点 ===")
    
    smart_tool = SmartAttractionTool()
    
    print("\n📍 查询：上海历史文化景点")
    result = smart_tool._run("上海", "历史文化")
    
    print("\n" + "="*70)
    print(result)
    print("="*70)

def test_shanghai_specific():
    """测试上海特定景点搜索"""
    print("\n=== 测试上海特定景点 ===")
    
    smart_tool = SmartAttractionTool()
    
    test_cases = [
        ("上海", "田子坊"),
        ("上海", "静安寺"),
        ("上海", "城隍庙"),
        ("上海", "历史建筑"),
    ]
    
    for city, keywords in test_cases:
        print(f"\n📍 查询：{city} {keywords}")
        result = smart_tool._run(city, keywords)
        
        lines = result.split('\n')
        attraction_names = []
        for line in lines:
            if line.strip() and line[0].isdigit() and '.' in line[:5]:
                attraction_names.append(line.split('.')[1].strip())
        
        print(f"推荐景点: {attraction_names}")

if __name__ == "__main__":
    test_shanghai_historical()
    test_shanghai_specific()