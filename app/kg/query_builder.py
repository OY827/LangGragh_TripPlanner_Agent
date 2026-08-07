"""知识图谱查询构建器"""

from typing import Dict, List, Any, Optional


class GraphQueryBuilder:
    """Cypher查询构建器"""

    @staticmethod
    def build_search_query(
        keywords: Optional[List[str]] = None,
        city: Optional[str] = None,
        category: Optional[str] = None,
        rating_min: Optional[float] = None,
        price_max: Optional[int] = None,
        limit: int = 10
    ) -> str:
        """构建景点搜索查询"""
        clauses = ["MATCH (a:Attraction)"]
        where_conditions = []
        params = {}
        param_counter = 1

        if keywords:
            keyword_conditions = [f"a.name CONTAINS $keyword{i}" for i in range(len(keywords))]
            where_conditions.append(f"({' OR '.join(keyword_conditions)})")
            for i, kw in enumerate(keywords):
                params[f"keyword{i}"] = kw

        if city:
            clauses.append("OPTIONAL MATCH (a)-[:LOCATED_IN]->(c:City)")
            where_conditions.append(f"c.name = $city")
            params["city"] = city

        if category:
            clauses.append("OPTIONAL MATCH (a)-[:BELONGS_TO]->(cat:Category)")
            where_conditions.append(f"cat.name = $category")
            params["category"] = category

        if rating_min:
            where_conditions.append(f"a.rating >= $rating_min")
            params["rating_min"] = rating_min

        if price_max:
            where_conditions.append(f"a.price <= $price_max")
            params["price_max"] = price_max

        if where_conditions:
            clauses.append(f"WHERE {' AND '.join(where_conditions)}")

        clauses.append("RETURN a ORDER BY a.rating DESC LIMIT $limit")
        params["limit"] = limit

        query = " ".join(clauses)
        return query, params

    @staticmethod
    def build_recommendation_query(
        city: str,
        duration_days: int = 1,
        preferences: Optional[List[str]] = None,
        budget: Optional[int] = None
    ) -> str:
        """构建推荐查询"""
        clauses = ["MATCH (a:Attraction)-[:LOCATED_IN]->(c:City {name: $city})"]
        where_conditions = []
        params = {"city": city}

        if preferences:
            clauses.append("OPTIONAL MATCH (a)-[:BELONGS_TO]->(cat:Category)")
            pref_conditions = [f"cat.name = $pref{i}" for i in range(len(preferences))]
            where_conditions.append(f"({' OR '.join(pref_conditions)})")
            for i, pref in enumerate(preferences):
                params[f"pref{i}"] = pref

        if budget:
            where_conditions.append(f"a.price <= $budget")
            params["budget"] = budget

        if where_conditions:
            clauses.append(f"WHERE {' AND '.join(where_conditions)}")

        clauses.append("RETURN a ORDER BY a.rating DESC LIMIT $limit")
        params["limit"] = duration_days * 4

        query = " ".join(clauses)
        return query, params

    @staticmethod
    def build_nearby_query(
        attraction_name: str,
        max_distance: float = 5.0,
        limit: int = 5
    ) -> str:
        """构建邻近查询"""
        query = """
        MATCH (a:Attraction {name: $name})-[r:NEARBY]->(nearby:Attraction)
        WHERE r.distance <= $max_distance
        RETURN nearby, r.distance as distance
        ORDER BY r.distance ASC
        LIMIT $limit
        """
        params = {
            "name": attraction_name,
            "max_distance": max_distance,
            "limit": limit
        }
        return query, params

    @staticmethod
    def build_route_query(
        start_attraction: str,
        end_attraction: str,
        max_hops: int = 3
    ) -> str:
        """构建路径查询"""
        query = f"""
        MATCH path = (start:Attraction {{name: $start}})-[*1..{max_hops}]->(end:Attraction {{name: $end}})
        RETURN path, length(path) as hops
        ORDER BY hops ASC
        LIMIT 5
        """
        params = {
            "start": start_attraction,
            "end": end_attraction
        }
        return query, params

    @staticmethod
    def build_category_stats_query(city: Optional[str] = None) -> str:
        """构建类别统计查询"""
        if city:
            query = """
            MATCH (a:Attraction)-[:LOCATED_IN]->(c:City {name: $city})
            MATCH (a)-[:BELONGS_TO]->(cat:Category)
            RETURN cat.name as category, COUNT(a) as count, AVG(a.rating) as avg_rating
            ORDER BY count DESC
            """
            params = {"city": city}
        else:
            query = """
            MATCH (a:Attraction)-[:BELONGS_TO]->(cat:Category)
            RETURN cat.name as category, COUNT(a) as count, AVG(a.rating) as avg_rating
            ORDER BY count DESC
            """
            params = {}

        return query, params

    @staticmethod
    def build_city_stats_query() -> str:
        """构建城市统计查询"""
        query = """
        MATCH (a:Attraction)-[:LOCATED_IN]->(c:City)
        RETURN c.name as city, COUNT(a) as attraction_count, AVG(a.rating) as avg_rating
        ORDER BY attraction_count DESC
        """
        return query, {}

    @staticmethod
    def build_similarity_query(
        attraction_name: str,
        limit: int = 5
    ) -> str:
        """构建相似景点查询"""
        query = """
        MATCH (a:Attraction {name: $name})-[r:SIMILAR_TO]->(similar:Attraction)
        RETURN similar, r.score as similarity
        ORDER BY r.score DESC
        LIMIT $limit
        """
        params = {
            "name": attraction_name,
            "limit": limit
        }
        return query, params