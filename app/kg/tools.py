"""知识图谱工具类 - 封装为可被Agent调用"""

from typing import List, Dict, Any, Optional

try:
    from langchain_core.tools import BaseTool
    from langchain_core.pydantic_v1 import BaseModel, Field
except ImportError:
    from pydantic import BaseModel, Field
    
    class BaseTool:
        name: str = ""
        description: str = ""
        args_schema = None
        
        def __init__(self):
            pass
        
        def _run(self, **kwargs):
            raise NotImplementedError("Subclasses must implement _run")
import os
import importlib.util

try:
    from .graph_store import Neo4jGraphStore
    from .query_builder import GraphQueryBuilder
except ImportError:
    try:
        from app.kg.graph_store import Neo4jGraphStore
        from app.kg.query_builder import GraphQueryBuilder
    except ImportError:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        graph_store_path = os.path.join(current_dir, "graph_store.py")
        spec = importlib.util.spec_from_file_location("graph_store", graph_store_path)
        graph_store_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(graph_store_module)
        Neo4jGraphStore = graph_store_module.Neo4jGraphStore
        
        query_builder_path = os.path.join(current_dir, "query_builder.py")
        spec = importlib.util.spec_from_file_location("query_builder", query_builder_path)
        query_builder_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(query_builder_module)
        GraphQueryBuilder = query_builder_module.GraphQueryBuilder


class GraphSearchToolInput(BaseModel):
    """图搜索工具输入"""
    keywords: Optional[List[str]] = Field(default=None, description="搜索关键词列表")
    city: Optional[str] = Field(default=None, description="城市名称")
    category: Optional[str] = Field(default=None, description="景点类别")
    rating_min: Optional[float] = Field(default=None, description="最低评分")
    price_max: Optional[int] = Field(default=None, description="最高票价(元)")
    limit: int = Field(default=10, description="返回数量限制")


class GraphSearchTool(BaseTool):
    """知识图谱搜索工具"""
    
    name: str = "graph_search"
    description: str = "搜索景点知识图谱，支持按关键词、城市、类别、评分、价格筛选"
    args_schema = GraphSearchToolInput
    
    def __init__(self):
        super().__init__()
        self.graph_store = Neo4jGraphStore()
    
    def _run(
        self,
        keywords: Optional[List[str]] = None,
        city: Optional[str] = None,
        category: Optional[str] = None,
        rating_min: Optional[float] = None,
        price_max: Optional[int] = None,
        limit: int = 10
    ) -> str:
        """执行图搜索"""
        if not self.graph_store.driver:
            return "❌ Neo4j连接不可用"
        
        try:
            query, params = GraphQueryBuilder.build_search_query(
                keywords=keywords,
                city=city,
                category=category,
                rating_min=rating_min,
                price_max=price_max,
                limit=limit
            )
            
            results = self.graph_store.execute_query(query, params)
            
            if not results:
                return "未找到符合条件的景点"
            
            lines = ["找到以下景点:"]
            for i, record in enumerate(results, 1):
                node = record["a"]
                lines.append(
                    f"{i}. {node.get('name')} | {node.get('city', '')} | "
                    f"评分: {node.get('rating', 0)} | 票价: {node.get('price', 0)}元"
                )
            
            return "\n".join(lines)
        
        except Exception as e:
            return f"搜索失败: {str(e)}"


class GraphRecommendToolInput(BaseModel):
    """图推荐工具输入"""
    city: str = Field(..., description="城市名称")
    duration_days: int = Field(default=1, description="游玩天数")
    preferences: Optional[List[str]] = Field(default=None, description="偏好类别列表")
    budget: Optional[int] = Field(default=None, description="预算(元)")


class GraphRecommendTool(BaseTool):
    """知识图谱推荐工具"""
    
    name: str = "graph_recommend"
    description: str = "基于知识图谱推荐景点，考虑城市、游玩天数、偏好类别和预算"
    args_schema = GraphRecommendToolInput
    
    def __init__(self):
        super().__init__()
        self.graph_store = Neo4jGraphStore()
    
    def _run(
        self,
        city: str,
        duration_days: int = 1,
        preferences: Optional[List[str]] = None,
        budget: Optional[int] = None
    ) -> str:
        """执行图推荐"""
        if not self.graph_store.driver:
            return "❌ Neo4j连接不可用"
        
        try:
            query, params = GraphQueryBuilder.build_recommendation_query(
                city=city,
                duration_days=duration_days,
                preferences=preferences,
                budget=budget
            )
            
            results = self.graph_store.execute_query(query, params)
            
            if not results:
                return f"未找到{city}的推荐景点"
            
            lines = [f"{city}{duration_days}日游推荐景点:"]
            total_price = 0
            total_duration = 0
            
            for i, record in enumerate(results, 1):
                node = record["a"]
                price = node.get("price", 0)
                duration = node.get("duration", 60)
                total_price += price
                total_duration += duration
                
                lines.append(
                    f"{i}. {node.get('name')} | {node.get('category', '')} | "
                    f"评分: {node.get('rating', 0)} | 票价: {price}元 | 时长: {duration}分钟"
                )
            
            lines.append(f"\n预计总票价: {total_price}元")
            lines.append(f"预计总时长: {total_duration}分钟 ({total_duration//60}小时{total_duration%60}分钟)")
            
            return "\n".join(lines)
        
        except Exception as e:
            return f"推荐失败: {str(e)}"


class GraphNearbyToolInput(BaseModel):
    """邻近搜索工具输入"""
    attraction_name: str = Field(..., description="景点名称")
    max_distance: float = Field(default=5.0, description="最大距离(公里)")
    limit: int = Field(default=5, description="返回数量限制")


class GraphNearbyTool(BaseTool):
    """邻近景点搜索工具"""
    
    name: str = "graph_nearby"
    description: str = "查找景点附近的其他景点"
    args_schema = GraphNearbyToolInput
    
    def __init__(self):
        super().__init__()
        self.graph_store = Neo4jGraphStore()
    
    def _run(
        self,
        attraction_name: str,
        max_distance: float = 5.0,
        limit: int = 5
    ) -> str:
        """执行邻近搜索"""
        if not self.graph_store.driver:
            return "❌ Neo4j连接不可用"
        
        try:
            results = self.graph_store.get_nearby_attractions(
                attraction_name=attraction_name,
                max_distance=max_distance
            )
            
            if not results:
                return f"{attraction_name}附近没有找到其他景点"
            
            lines = [f"{attraction_name}附近{max_distance}公里内的景点:"]
            for i, item in enumerate(results[:limit], 1):
                lines.append(
                    f"{i}. {item.get('name')} | 距离: {item.get('distance', 0):.2f}km | "
                    f"评分: {item.get('rating', 0)} | 票价: {item.get('price', 0)}元"
                )
            
            return "\n".join(lines)
        
        except Exception as e:
            return f"邻近搜索失败: {str(e)}"


class GraphSimilarToolInput(BaseModel):
    """相似景点工具输入"""
    attraction_name: str = Field(..., description="景点名称")
    limit: int = Field(default=5, description="返回数量限制")


class GraphSimilarTool(BaseTool):
    """相似景点推荐工具"""
    
    name: str = "graph_similar"
    description: str = "查找与指定景点相似的其他景点"
    args_schema = GraphSimilarToolInput
    
    def __init__(self):
        super().__init__()
        self.graph_store = Neo4jGraphStore()
    
    def _run(
        self,
        attraction_name: str,
        limit: int = 5
    ) -> str:
        """执行相似搜索"""
        if not self.graph_store.driver:
            return "❌ Neo4j连接不可用"
        
        try:
            results = self.graph_store.get_similar_attractions(
                attraction_name=attraction_name,
                limit=limit
            )
            
            if not results:
                return f"没有找到与{attraction_name}相似的景点"
            
            lines = [f"与{attraction_name}相似的景点:"]
            for i, item in enumerate(results, 1):
                lines.append(
                    f"{i}. {item.get('name')} | 相似度: {item.get('similarity', 0):.2f} | "
                    f"{item.get('city', '')} | 评分: {item.get('rating', 0)}"
                )
            
            return "\n".join(lines)
        
        except Exception as e:
            return f"相似搜索失败: {str(e)}"