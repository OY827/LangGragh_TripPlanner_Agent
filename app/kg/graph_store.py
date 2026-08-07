"""Neo4j图数据库操作类"""

import os
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv
from neo4j import GraphDatabase, Driver, Session, Record

load_dotenv()


class Neo4jGraphStore:
    """Neo4j图数据库存储管理器"""

    def __init__(self):
        """初始化Neo4j连接"""
        self.uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.username = os.getenv("NEO4J_USERNAME", "neo4j")
        self.password = os.getenv("NEO4J_PASSWORD", "password")
        self.driver: Optional[Driver] = None
        self._connect()

    def _connect(self):
        """建立Neo4j连接"""
        try:
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.username, self.password),
                max_connection_lifetime=30 * 60,
                max_connection_pool_size=50,
                connection_acquisition_timeout=20,
                connection_timeout=30
            )

            with self.driver.session() as session:
                session.run("RETURN 1")
            print(f"✅ Neo4j连接成功: {self.uri}")
        except Exception as e:
            print(f"❌ Neo4j连接失败: {e}")
            self.driver = None

    def close(self):
        """关闭连接"""
        if self.driver:
            self.driver.close()
            print("✅ Neo4j连接已关闭")

    def execute_query(self, query: str, params: Dict[str, Any] = None) -> List[Record]:
        """执行Cypher查询"""
        if not self.driver:
            return []

        params = params or {}
        try:
            with self.driver.session() as session:
                result = session.run(query, params)
                return list(result)
        except Exception as e:
            print(f"❌ 查询执行失败: {e}")
            return []

    def create_attraction_node(self, attraction: Dict[str, Any]) -> bool:
        """创建景点节点"""
        query = """
        MERGE (a:Attraction {name: $name})
        SET a += $properties
        RETURN a.name as name
        """
        properties = {
            "address": attraction.get("address", ""),
            "latitude": attraction.get("latitude", 0),
            "longitude": attraction.get("longitude", 0),
            "rating": attraction.get("rating", 0),
            "price": attraction.get("price", 0),
            "duration": attraction.get("duration", 120),
            "category": attraction.get("category", "景点"),
            "description": attraction.get("description", ""),
            "city": attraction.get("city", ""),
            "poi_id": attraction.get("poi_id", "")
        }

        result = self.execute_query(query, {"name": attraction["name"], "properties": properties})
        return len(result) > 0

    def create_city_node(self, city: str) -> bool:
        """创建城市节点"""
        query = """
        MERGE (c:City {name: $name})
        RETURN c.name as name
        """
        result = self.execute_query(query, {"name": city})
        return len(result) > 0

    def create_category_node(self, category: str) -> bool:
        """创建类别节点"""
        query = """
        MERGE (cat:Category {name: $name})
        RETURN cat.name as name
        """
        result = self.execute_query(query, {"name": category})
        return len(result) > 0

    def create_has_city_relation(self, attraction_name: str, city_name: str) -> bool:
        """创建景点-城市关系"""
        query = """
        MATCH (a:Attraction {name: $attraction_name}), (c:City {name: $city_name})
        MERGE (a)-[:LOCATED_IN]->(c)
        RETURN a.name, c.name
        """
        result = self.execute_query(query, {
            "attraction_name": attraction_name,
            "city_name": city_name
        })
        return len(result) > 0

    def create_has_category_relation(self, attraction_name: str, category_name: str) -> bool:
        """创建景点-类别关系"""
        query = """
        MATCH (a:Attraction {name: $attraction_name}), (cat:Category {name: $category_name})
        MERGE (a)-[:BELONGS_TO]->(cat)
        RETURN a.name, cat.name
        """
        result = self.execute_query(query, {
            "attraction_name": attraction_name,
            "category_name": category_name
        })
        return len(result) > 0

    def create_nearby_relation(self, attraction_name1: str, attraction_name2: str, distance: float) -> bool:
        """创建景点-邻近关系"""
        query = """
        MATCH (a1:Attraction {name: $name1}), (a2:Attraction {name: $name2})
        MERGE (a1)-[:NEARBY {distance: $distance}]->(a2)
        RETURN a1.name, a2.name
        """
        result = self.execute_query(query, {
            "name1": attraction_name1,
            "name2": attraction_name2,
            "distance": distance
        })
        return len(result) > 0

    def create_similar_relation(self, attraction_name1: str, attraction_name2: str, similarity: float) -> bool:
        """创建景点-相似关系"""
        query = """
        MATCH (a1:Attraction {name: $name1}), (a2:Attraction {name: $name2})
        MERGE (a1)-[:SIMILAR_TO {score: $similarity}]->(a2)
        RETURN a1.name, a2.name
        """
        result = self.execute_query(query, {
            "name1": attraction_name1,
            "name2": attraction_name2,
            "similarity": similarity
        })
        return len(result) > 0

    def get_attraction_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """根据名称查询景点"""
        query = """
        MATCH (a:Attraction {name: $name})
        OPTIONAL MATCH (a)-[:LOCATED_IN]->(c:City)
        OPTIONAL MATCH (a)-[:BELONGS_TO]->(cat:Category)
        RETURN a, COLLECT(DISTINCT c.name) as cities, COLLECT(DISTINCT cat.name) as categories
        """
        result = self.execute_query(query, {"name": name})
        if result:
            record = result[0]
            node = record["a"]
            return {
                "name": node.get("name"),
                "address": node.get("address"),
                "latitude": node.get("latitude"),
                "longitude": node.get("longitude"),
                "rating": node.get("rating"),
                "price": node.get("price"),
                "duration": node.get("duration"),
                "category": node.get("category"),
                "description": node.get("description"),
                "city": record["cities"][0] if record["cities"] else "",
                "categories": record["categories"]
            }
        return None

    def search_attractions_by_city(self, city: str) -> List[Dict[str, Any]]:
        """按城市查询景点"""
        query = """
        MATCH (a:Attraction)-[:LOCATED_IN]->(c:City {name: $city})
        OPTIONAL MATCH (a)-[:BELONGS_TO]->(cat:Category)
        RETURN a, COLLECT(DISTINCT cat.name) as categories
        ORDER BY a.rating DESC
        """
        result = self.execute_query(query, {"city": city})
        return self._format_search_results(result)

    def search_attractions_by_category(self, category: str) -> List[Dict[str, Any]]:
        """按类别查询景点"""
        query = """
        MATCH (a:Attraction)-[:BELONGS_TO]->(cat:Category {name: $category})
        OPTIONAL MATCH (a)-[:LOCATED_IN]->(c:City)
        RETURN a, COLLECT(DISTINCT c.name) as cities
        ORDER BY a.rating DESC
        """
        result = self.execute_query(query, {"category": category})
        return self._format_search_results(result)

    def get_nearby_attractions(self, attraction_name: str, max_distance: float = 5.0) -> List[Dict[str, Any]]:
        """获取邻近景点"""
        query = """
        MATCH (a:Attraction {name: $name})-[r:NEARBY]->(nearby:Attraction)
        WHERE r.distance <= $max_distance
        OPTIONAL MATCH (nearby)-[:LOCATED_IN]->(c:City)
        RETURN nearby, r.distance as distance
        ORDER BY r.distance ASC
        """
        result = self.execute_query(query, {"name": attraction_name, "max_distance": max_distance})
        return self._format_search_results(result, include_distance=True)

    def get_similar_attractions(self, attraction_name: str, limit: int = 5) -> List[Dict[str, Any]]:
        """获取相似景点"""
        query = """
        MATCH (a:Attraction {name: $name})-[r:SIMILAR_TO]->(similar:Attraction)
        RETURN similar, r.score as similarity
        ORDER BY r.score DESC
        LIMIT $limit
        """
        result = self.execute_query(query, {"name": attraction_name, "limit": limit})
        return self._format_search_results(result, include_distance=True)

    def get_recommendations_by_category(self, city: str, category: str, limit: int = 5) -> List[Dict[str, Any]]:
        """按城市和类别推荐景点"""
        query = """
        MATCH (a:Attraction)-[:LOCATED_IN]->(c:City {name: $city})
        MATCH (a)-[:BELONGS_TO]->(cat:Category {name: $category})
        RETURN a
        ORDER BY a.rating DESC
        LIMIT $limit
        """
        result = self.execute_query(query, {"city": city, "category": category, "limit": limit})
        return self._format_search_results(result)

    def get_graph_stats(self) -> Dict[str, Any]:
        """获取图统计信息"""
        query = """
        CALL db.labels() YIELD label
        WITH label
        CALL { WITH label
            MATCH (n) WHERE labels(n) = [label]
            RETURN count(n) as count
        }
        RETURN label, count
        """
        result = self.execute_query(query)
        stats = {"labels": {}}
        for record in result:
            stats["labels"][record["label"]] = record["count"]

        rel_query = """
        CALL db.relationshipTypes() YIELD relationshipType
        WITH relationshipType
        CALL { WITH relationshipType
            MATCH ()-[r]-() WHERE type(r) = relationshipType
            RETURN count(r) as count
        }
        RETURN relationshipType, count
        """
        rel_result = self.execute_query(rel_query)
        stats["relationships"] = {}
        for record in rel_result:
            stats["relationships"][record["relationshipType"]] = record["count"]

        return stats

    def clear_database(self) -> bool:
        """清空数据库"""
        query = """
        MATCH (n)
        DETACH DELETE n
        """
        try:
            self.execute_query(query)
            print("✅ 数据库已清空")
            return True
        except Exception as e:
            print(f"❌ 清空数据库失败: {e}")
            return False

    def _format_search_results(self, results: List[Record], include_distance: bool = False) -> List[Dict[str, Any]]:
        """格式化搜索结果"""
        formatted = []
        for record in results:
            node = record.get("a") or record.get("nearby") or record.get("similar")
            if node:
                item = {
                    "name": node.get("name"),
                    "address": node.get("address"),
                    "latitude": node.get("latitude"),
                    "longitude": node.get("longitude"),
                    "rating": node.get("rating"),
                    "price": node.get("price"),
                    "duration": node.get("duration"),
                    "category": node.get("category"),
                    "description": node.get("description"),
                    "city": node.get("city", "")
                }
                if include_distance:
                    item["distance"] = record.get("distance")
                    item["similarity"] = record.get("similarity")
                formatted.append(item)
        return formatted