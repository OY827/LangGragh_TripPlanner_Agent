"""知识图谱数据加载器"""

import json
import os
import sys
import importlib.util
from typing import List, Dict, Any

try:
    from .graph_store import Neo4jGraphStore
except ImportError:
    try:
        from app.kg.graph_store import Neo4jGraphStore
    except ImportError:
        graph_store_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "graph_store.py"
        )
        spec = importlib.util.spec_from_file_location("graph_store", graph_store_path)
        graph_store_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(graph_store_module)
        Neo4jGraphStore = graph_store_module.Neo4jGraphStore


class GraphDataLoader:
    """从本地知识库加载数据到Neo4j"""

    def __init__(self, graph_store: Neo4jGraphStore):
        self.graph_store = graph_store
        self.knowledge_base_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "app", "knowledge_base", "attractions"
        )

    def load_from_local_files(self) -> int:
        """从本地JSON文件加载所有景点数据"""
        files = [f for f in os.listdir(self.knowledge_base_path) if f.endswith(".json")]
        total_count = 0

        for file in files:
            city = file.replace(".json", "")
            file_path = os.path.join(self.knowledge_base_path, file)
            
            print(f"📥 加载 {city} 的景点数据...")
            
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            count = self._load_city_attractions(data, city)
            total_count += count
            print(f"  ✅ 已加载 {count} 条数据")

        return total_count

    def _load_city_attractions(self, attractions_data: List[Dict[str, Any]], city: str) -> int:
        """加载单个城市的景点数据"""
        count = 0
        
        for item in attractions_data:
            try:
                attraction = self._convert_to_attraction(item)
                
                if attraction["name"]:
                    self.graph_store.create_city_node(city)
                    self.graph_store.create_attraction_node(attraction)
                    self.graph_store.create_has_city_relation(attraction["name"], city)
                    
                    if attraction.get("category"):
                        self.graph_store.create_category_node(attraction["category"])
                        self.graph_store.create_has_category_relation(attraction["name"], attraction["category"])
                    
                    count += 1
            except Exception as e:
                print(f"  ❌ 加载失败: {item.get('name', '未知')} - {e}")

        return count

    def _convert_to_attraction(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """将本地数据格式转换为景点节点格式"""
        metadata = item.get("metadata", item)
        
        name = metadata.get("title", metadata.get("name", ""))
        address = metadata.get("address", "")
        city = metadata.get("city", "")
        category = metadata.get("category", "景点")
        rating = metadata.get("rating", 0)
        price = metadata.get("price", "")
        duration = int(metadata.get("duration", metadata.get("visit_duration", 120)))
        poi_id = metadata.get("poi_id", metadata.get("id", ""))
        
        location_str = metadata.get("location", "")
        latitude, longitude = self._parse_location(location_str)
        
        description = item.get("content", "") or metadata.get("description", "")
        
        return {
            "name": name,
            "address": address,
            "latitude": latitude,
            "longitude": longitude,
            "rating": float(rating) if rating else 0,
            "price": self._parse_price(price),
            "duration": duration,
            "category": category,
            "description": description,
            "city": city,
            "poi_id": poi_id
        }
    
    def _parse_location(self, location_str: str) -> tuple:
        """解析位置字符串，格式如 'longitude=116.397029 latitude=39.917839'"""
        if not location_str:
            return 0, 0
        
        try:
            parts = location_str.split()
            longitude = 0
            latitude = 0
            
            for part in parts:
                if part.startswith("longitude="):
                    longitude = float(part.replace("longitude=", ""))
                elif part.startswith("latitude="):
                    latitude = float(part.replace("latitude=", ""))
                elif "," in part:
                    lon_str, lat_str = part.split(",")
                    longitude = float(lon_str)
                    latitude = float(lat_str)
            
            return latitude, longitude
        except (ValueError, TypeError):
            return 0, 0

    def _parse_price(self, price_str: str) -> int:
        """解析价格字符串为整数"""
        if not price_str or price_str == "免费" or price_str == "N/A":
            return 0
        
        try:
            price_str = str(price_str).replace("元", "").replace("¥", "").replace(",", "").strip()
            if price_str:
                return int(float(price_str))
        except (ValueError, TypeError):
            pass
        
        return 0

    def build_nearby_relations(self, max_distance_km: float = 5.0) -> int:
        """构建邻近关系"""
        print(f"\n🔗 构建邻近关系（距离 <= {max_distance_km}km）...")
        
        query = """
        MATCH (a1:Attraction), (a2:Attraction)
        WHERE a1.name <> a2.name
          AND a1.city = a2.city
        WITH a1, a2, 
             6371 * ACOS(
               COS(RADIANS(a1.latitude)) * COS(RADIANS(a2.latitude)) * 
               COS(RADIANS(a1.longitude) - RADIANS(a2.longitude)) + 
               SIN(RADIANS(a1.latitude)) * SIN(RADIANS(a2.latitude))
             ) AS distance
        WHERE distance <= $max_distance
        MERGE (a1)-[:NEARBY {distance: ROUND(distance * 100) / 100}]->(a2)
        RETURN COUNT(*) as count
        """
        
        params = {"max_distance": max_distance_km}
        result = self.graph_store.execute_query(query, params)
        
        if result:
            count = result[0].get("count", 0)
            print(f"  ✅ 已创建 {count} 条邻近关系")
            return count
        
        return 0

    def build_similarity_relations(self, threshold: float = 0.5) -> int:
        """构建相似关系（基于类别）"""
        print(f"\n🔗 构建相似关系（基于类别）...")
        
        query = """
        MATCH (a1:Attraction)-[:BELONGS_TO]->(cat:Category)<-[:BELONGS_TO]-(a2:Attraction)
        WHERE a1.name <> a2.name
        WITH a1, a2, COUNT(DISTINCT cat) as common_categories
        WITH a1, a2, common_categories,
             (common_categories * 0.3 + 
              CASE WHEN a1.city = a2.city THEN 0.3 ELSE 0 END + 
              CASE WHEN ABS(a1.rating - a2.rating) <= 0.5 THEN 0.2 ELSE 0 END +
              CASE WHEN ABS(a1.price - a2.price) <= 50 THEN 0.2 ELSE 0 END) as similarity
        WHERE similarity >= $threshold
        MERGE (a1)-[:SIMILAR_TO {score: ROUND(similarity * 100) / 100}]->(a2)
        RETURN COUNT(*) as count
        """
        
        params = {"threshold": threshold}
        result = self.graph_store.execute_query(query, params)
        
        if result:
            count = result[0].get("count", 0)
            print(f"  ✅ 已创建 {count} 条相似关系")
            return count
        
        return 0

    def rebuild_graph(self, clear_first: bool = True) -> Dict[str, int]:
        """重建整个知识图谱"""
        print("=== 重建知识图谱 ===")
        
        stats = {}
        
        if clear_first:
            self.graph_store.clear_database()
        
        stats["attractions"] = self.load_from_local_files()
        stats["nearby_relations"] = self.build_nearby_relations()
        stats["similarity_relations"] = self.build_similarity_relations()
        
        print("\n=== 重建完成 ===")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        return stats