"""测试上海景点推荐问题"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def check_shanghai_kb():
    """检查上海知识库中的景点"""
    kb_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app", "knowledge_base", "attractions", "上海.json")
    
    with open(kb_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"上海知识库共 {len(data)} 条数据")
    print("\n景点列表：")
    
    for item in data:
        title = item['metadata'].get('title', '')
        category = item['metadata'].get('category', '')
        print(f"  - {title} (分类: {category})")
    
    print("\n=== 搜索特定景点 ===")
    target_attractions = ["田子坊", "静安寺", "城隍庙", "豫园", "外滩", "陆家嘴"]
    
    for target in target_attractions:
        found = [item for item in data if target in item['metadata'].get('title', '')]
        if found:
            print(f"✅ {target}: 找到 {len(found)} 条")
            for item in found:
                print(f"   - {item['metadata']['title']}")
        else:
            print(f"❌ {target}: 未找到")

def test_amap_search():
    """测试高德地图搜索上海景点"""
    print("\n=== 测试高德地图搜索 ===")
    
    from app.tools.mcp_tools import MCPToolManager, settings
    
    manager = MCPToolManager()
    
    if "amap" not in manager:
        if settings.amap_api_key and settings.enable_mcp_tools:
            manager.register_tool(
                name="amap",
                description="高德地图服务",
                server_command=["python", "-m", "amap_mcp_server"],
                env={"AMAP_MAPS_API_KEY": settings.amap_api_key},
                auto_expand=True
            )
        else:
            print("⚠️ 未配置高德服务")
            return
    
    amap_tool = manager.get_tool("amap")
    result = amap_tool.invoke(
        name="maps_text_search",
        arguments={
            "keywords": "景点",
            "city": "上海",
            "citylimit": "true"
        }
    )
    
    if result and "pois" in result:
        print("高德地图搜索到的景点:")
        for poi in result["pois"][:10]:
            print(f"  - {poi.get('name')}")

def test_kb_retrieval():
    """测试知识库检索"""
    print("\n=== 测试知识库检索 ===")
    
    from app.rag.retriever import KnowledgeRetriever
    
    retriever = KnowledgeRetriever()
    
    test_queries = ["田子坊", "静安寺", "历史建筑", "上海历史文化"]
    
    for query in test_queries:
        results = retriever.retrieve(
            query=query,
            filters={"city": "上海"},
            top_k=5,
            search_type="hybrid"
        )
        
        print(f"\n🔍 搜索 '{query}':")
        if results:
            for r in results:
                title = r['metadata'].get('title', '未知')
                score = r.get('score', 0)
                print(f"   - {title} (相似度: {score:.4f})")
        else:
            print("   未找到")

if __name__ == "__main__":
    check_shanghai_kb()
    test_amap_search()
    test_kb_retrieval()