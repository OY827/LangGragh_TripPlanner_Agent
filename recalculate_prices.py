"""使用真实市场价数据库更新景点票价"""

import json
import os
from app.rag.price_database import get_attraction_price

def update_city_prices(city):
    """更新指定城市的景点票价"""
    file_path = f"app/knowledge_base/attractions/{city}.json"
    
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        return
    
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    updated_count = 0
    
    for item in data:
        title = item["metadata"].get("title", "")
        category = item["metadata"].get("category", "其他")
        poi_type = item["metadata"].get("poi_type", "")
        old_price = item["metadata"].get("price", "N/A")
        
        # 使用票价数据库获取真实票价
        price, duration = get_attraction_price(city, title, category, poi_type)
        
        if old_price != price:
            item["metadata"]["price"] = price
            item["metadata"]["duration"] = duration
            print(f"   更新: {title} - {old_price} → {price}")
            updated_count += 1
    
    # 保存更新后的数据
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"  ✅ 更新了 {updated_count} 条记录")

if __name__ == "__main__":
    print("=== 使用真实市场价数据库更新票价 ===")
    
    # 更新北京和上海的景点数据
    for city in ["上海", "北京"]:
        print(f"\n处理 {city}...")
        update_city_prices(city)
    
    print("\n✅ 所有文件更新完成！")