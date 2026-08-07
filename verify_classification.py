"""验证重新分类的效果"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.rag.data_collector import AttractionDataCollector

def verify_classification():
    """验证重新分类效果"""
    collector = AttractionDataCollector()
    
    cities = ["上海", "北京", "西安"]
    
    for city in cities:
        print(f"\n=== {city} ===")
        
        data = collector.load_from_json(city)
        
        category_count = {}
        reclassified_count = 0
        
        for item in data:
            metadata = item['metadata']
            category = metadata.get('category', '其他')
            tags = metadata.get('tags', '')
            
            category_count[category] = category_count.get(category, 0) + 1
            
            if '历史' in tags and category == '历史文化':
                reclassified_count += 1
        
        print("分类统计:")
        for cat, count in sorted(category_count.items(), key=lambda x: x[1], reverse=True):
            print(f"  {cat}: {count}")
        
        print(f"\n从tags中识别出历史类景点: {reclassified_count}")
        
        print("\n历史文化类景点列表:")
        for item in data:
            if item['metadata'].get('category') == '历史文化':
                tags = item['metadata'].get('tags', '')
                print(f"  - {item['metadata']['title']} (tags: {tags[:50]})")

def test_search():
    """测试检索效果"""
    print("\n=== 测试检索效果 ===")
    
    from app.tools.agent_tools import SmartAttractionTool
    
    smart_tool = SmartAttractionTool()
    
    print("\n📍 查询：上海历史文化景点")
    result = smart_tool._run("上海", "历史文化")
    
    print("\n" + "="*70)
    print(result)
    print("="*70)

if __name__ == "__main__":
    verify_classification()
    test_search()