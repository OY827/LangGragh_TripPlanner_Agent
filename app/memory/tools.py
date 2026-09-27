"""
旅行记忆查询工具
供 Agent 查询用户历史旅行数据
"""

import json
from typing import Any, Optional, Type
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from app.memory.memory_manager import TravelMemoryManager


class TravelMemoryQueryInput(BaseModel):
    """旅行记忆查询输入参数"""
    user_id: str = Field(description="用户ID")
    query_type: str = Field(
        description="查询类型，可选值: stats(统计), date_range(日期范围), budget_range(预算范围), "
                    "attraction(景点), duration(天数), search(组合搜索), cities(城市列表), "
                    "semantic(语义检索长期记忆), preferences(用户偏好)"
    )
    start_date: Optional[str] = Field(default=None, description="开始日期 (YYYY-MM-DD)")
    end_date: Optional[str] = Field(default=None, description="结束日期 (YYYY-MM-DD)")
    min_budget: Optional[float] = Field(default=None, description="最低预算")
    max_budget: Optional[float] = Field(default=None, description="最高预算")
    attraction: Optional[str] = Field(default=None, description="景点名称")
    min_days: Optional[int] = Field(default=None, description="最短天数")
    max_days: Optional[int] = Field(default=None, description="最长天数")
    city: Optional[str] = Field(default=None, description="城市名称")
    limit: Optional[int] = Field(default=10, description="返回结果数量限制")


class TravelMemoryQueryTool(BaseTool):
    """
    旅行记忆查询工具
    
    功能说明:
    - 查询用户历史旅行计划
    - 获取旅行统计数据
    - 按日期、预算、景点等条件筛选
    
    使用场景:
    - "我去年去了哪些地方？"
    - "帮我找一下预算在1000-2000元之间的行程"
    - "我都去过哪些景点？"
    - "我平均每次旅行花多少钱？"
    """
    name: str = "travel_memory_query"
    description: str = (
        "查询用户历史旅行数据的工具。支持多种查询模式：\n"
        "1. stats: 获取用户旅行统计（总行程数、平均预算、热门景点等）\n"
        "2. date_range: 按日期范围查询（如：2026年的所有行程）\n"
        "3. budget_range: 按预算范围查询（如：预算1000-2000元的行程）\n"
        "4. attraction: 按景点查询（如：包含外滩的所有行程）\n"
        "5. duration: 按天数查询（如：3-5天的行程）\n"
        "6. search: 组合条件搜索\n"
        "7. cities: 获取所有去过的城市列表\n"
        "8. semantic: 用自然语言检索长期记忆（偏好/去过的景点/历史行程）\n"
        "9. preferences: 获取用户偏好列表\n"
    )
    args_schema: Type[BaseModel] = TravelMemoryQueryInput
    memory_manager: TravelMemoryManager = None
    
    def __init__(self, memory_manager: TravelMemoryManager = None, **kwargs):
        super().__init__(**kwargs)
        self.memory_manager = memory_manager or TravelMemoryManager()
    
    def _run(
        self,
        user_id: str,
        query_type: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        min_budget: Optional[float] = None,
        max_budget: Optional[float] = None,
        attraction: Optional[str] = None,
        min_days: Optional[int] = None,
        max_days: Optional[int] = None,
        city: Optional[str] = None,
        limit: Optional[int] = 10
    ) -> str:
        """
        执行旅行记忆查询
        
        Args:
            user_id: 用户ID
            query_type: 查询类型
            其他参数: 根据查询类型可选的筛选条件
            
        Returns:
            JSON格式的查询结果
        """
        try:
            if query_type == "stats":
                return self._query_stats(user_id)
            elif query_type == "date_range":
                return self._query_by_date_range(user_id, start_date, end_date, limit)
            elif query_type == "budget_range":
                return self._query_by_budget_range(user_id, min_budget, max_budget, limit)
            elif query_type == "attraction":
                return self._query_by_attraction(user_id, attraction, limit)
            elif query_type == "duration":
                return self._query_by_duration(user_id, min_days, max_days, limit)
            elif query_type == "search":
                return self._query_search(user_id, start_date, end_date, min_budget, max_budget, attraction, min_days, max_days, city, limit)
            elif query_type == "cities":
                return self._query_cities(user_id)
            elif query_type == "semantic":
                return self._query_semantic(user_id, attraction or city or "", limit)
            elif query_type == "preferences":
                return self._query_preferences(user_id, city)
            else:
                return json.dumps({
                    "success": False,
                    "error": f"不支持的查询类型: {query_type}",
                    "supported_types": [
                        "stats", "date_range", "budget_range", "attraction",
                        "duration", "search", "cities", "semantic", "preferences",
                    ]
                }, ensure_ascii=False, indent=2)
        except Exception as e:
            return json.dumps({
                "success": False,
                "error": str(e)
            }, ensure_ascii=False, indent=2)
    
    def _query_stats(self, user_id: str) -> str:
        """查询用户统计数据"""
        stats = self.memory_manager.get_user_statistics(user_id)
        return json.dumps({
            "success": True,
            "query_type": "stats",
            "data": stats
        }, ensure_ascii=False, indent=2)
    
    def _query_by_date_range(self, user_id: str, start_date: str, end_date: str, limit: int) -> str:
        """按日期范围查询"""
        if not start_date or not end_date:
            return json.dumps({
                "success": False,
                "error": "需要提供 start_date 和 end_date 参数，格式: YYYY-MM-DD"
            }, ensure_ascii=False, indent=2)
        
        plans = self.memory_manager.get_plans_by_date_range(user_id, start_date, end_date)
        return self._format_plans_result(plans, "date_range", limit, {
            "start_date": start_date,
            "end_date": end_date
        })
    
    def _query_by_budget_range(self, user_id: str, min_budget: float, max_budget: float, limit: int) -> str:
        """按预算范围查询"""
        min_budget = min_budget or 0
        max_budget = max_budget or float('inf')
        
        plans = self.memory_manager.get_plans_by_budget_range(user_id, min_budget, max_budget)
        return self._format_plans_result(plans, "budget_range", limit, {
            "min_budget": min_budget if min_budget != float('inf') else None,
            "max_budget": max_budget if max_budget != float('inf') else None
        })
    
    def _query_by_attraction(self, user_id: str, attraction: str, limit: int) -> str:
        """按景点查询"""
        if not attraction:
            return json.dumps({
                "success": False,
                "error": "需要提供 attraction 参数"
            }, ensure_ascii=False, indent=2)
        
        plans = self.memory_manager.get_plans_by_attraction(user_id, attraction)
        return self._format_plans_result(plans, "attraction", limit, {
            "attraction": attraction
        })
    
    def _query_by_duration(self, user_id: str, min_days: int, max_days: int, limit: int) -> str:
        """按天数查询"""
        min_days = min_days or 0
        max_days = max_days or float('inf')
        
        plans = self.memory_manager.get_plans_by_duration(user_id, min_days, max_days)
        return self._format_plans_result(plans, "duration", limit, {
            "min_days": min_days if min_days != float('inf') else None,
            "max_days": max_days if max_days != float('inf') else None
        })
    
    def _query_search(self, user_id: str, start_date: str, end_date: str, 
                      min_budget: float, max_budget: float, attraction: str,
                      min_days: int, max_days: int, city: str, limit: int) -> str:
        """组合条件搜索"""
        filters = {}
        if start_date and end_date:
            filters['start_date'] = start_date
            filters['end_date'] = end_date
        if min_budget is not None:
            filters['min_budget'] = min_budget
        if max_budget is not None:
            filters['max_budget'] = max_budget
        if attraction:
            filters['attraction'] = attraction
        if min_days is not None:
            filters['min_days'] = min_days
        if max_days is not None:
            filters['max_days'] = max_days
        if city:
            filters['city'] = city
        
        plans = self.memory_manager.search_plans(user_id, **filters)
        return self._format_plans_result(plans, "search", limit, filters)
    
    def _query_cities(self, user_id: str) -> str:
        """查询所有去过的城市"""
        plans = self.memory_manager.get_user_plans(user_id)
        cities = sorted(set(p.city for p in plans))
        city_counts = {}
        for city in cities:
            city_counts[city] = len([p for p in plans if p.city == city])
        
        return json.dumps({
            "success": True,
            "query_type": "cities",
            "data": {
                "total_cities": len(cities),
                "cities": [{"name": c, "visit_count": city_counts[c]} for c in cities],
                "all_plans_count": len(plans)
            }
        }, ensure_ascii=False, indent=2)

    def _query_semantic(self, user_id: str, query: str, limit: int) -> str:
        if not query:
            return json.dumps({
                "success": False,
                "error": "semantic 查询需要 attraction 或 city 作为自然语言 query",
            }, ensure_ascii=False, indent=2)
        hits = self.memory_manager.search_semantic(user_id, query, k=limit or 5)
        return json.dumps({
            "success": True,
            "query_type": "semantic",
            "data": hits,
        }, ensure_ascii=False, indent=2)

    def _query_preferences(self, user_id: str, city: Optional[str]) -> str:
        prefs = self.memory_manager.get_preferences(user_id, city or "")
        return json.dumps({
            "success": True,
            "query_type": "preferences",
            "data": {"preferences": prefs},
        }, ensure_ascii=False, indent=2)
    
    def _format_plans_result(self, plans, query_type: str, limit: int, extra: dict) -> str:
        """格式化计划列表结果"""
        plans_data = []
        for plan in plans[:limit]:
            plan_dict = {
                "plan_id": plan.plan_id,
                "city": plan.city,
                "duration_days": plan.duration_days,
                "created_at": plan.created_at,
                "attractions": plan.attractions,
            }
            # 添加预算信息摘要
            if plan.budget:
                if isinstance(plan.budget, dict):
                    plan_dict["budget_summary"] = {
                        "total": plan.budget.get("total", 0),
                        "has_detail": True
                    }
                else:
                    plan_dict["budget_summary"] = {
                        "total": plan.budget,
                        "has_detail": False
                    }
            plans_data.append(plan_dict)
        
        return json.dumps({
            "success": True,
            "query_type": query_type,
            "total_results": len(plans),
            "returned_results": len(plans_data),
            "filters": extra,
            "plans": plans_data
        }, ensure_ascii=False, indent=2)


class DiffRecommendationInput(BaseModel):
    """差异推荐输入参数"""
    user_id: str = Field(description="用户ID")
    city: str = Field(description="目标城市")


class DiffRecommendationTool(BaseTool):
    """
    差异推荐工具 - 对比用户历史记录，推荐新景点
    """
    name: str = "diff_recommend"
    description: str = (
        "根据用户历史旅行记录，推荐与过去不同的新景点。"
        "如果用户之前去过某个城市，会推荐未游览过的景点。"
    )
    args_schema: Type[BaseModel] = DiffRecommendationInput
    memory_manager: TravelMemoryManager = None
    
    def __init__(self, memory_manager: TravelMemoryManager = None, **kwargs):
        super().__init__(**kwargs)
        self.memory_manager = memory_manager or TravelMemoryManager()
    
    def _run(self, user_id: str, city: str) -> str:
        """执行差异推荐"""
        try:
            comparison = self.memory_manager.compare_plans(user_id, city)
            return json.dumps({
                "success": True,
                "data": comparison
            }, ensure_ascii=False, indent=2)
        except Exception as e:
            return json.dumps({
                "success": False,
                "error": str(e)
            }, ensure_ascii=False, indent=2)


class PlanVersionQueryInput(BaseModel):
    session_id: str = Field(description="对话会话ID")
    query_type: str = Field(
        description="查询类型: history(版本列表), current(当前计划), search(关键词检索版本), compare(对比两个版本)"
    )
    query: Optional[str] = Field(default=None, description="search 时的关键词")
    old_version_id: Optional[str] = Field(default=None, description="compare 时的旧版本ID")
    new_version_id: Optional[str] = Field(default=None, description="compare 时的新版本ID")


class PlanVersionQueryTool(BaseTool):
    """查询 Redis 中当前会话的旅行计划版本，支持列表、检索、对比。"""
    name: str = "plan_version_query"
    description: str = (
        "查询当前对话会话里的旅行计划版本。"
        "用于「上一版有哪些景点」「对比这次修改」等，不查询用户跨会话长期记忆。"
    )
    args_schema: Type[BaseModel] = PlanVersionQueryInput
    version_manager: Any = None

    def __init__(self, version_manager=None, **kwargs):
        super().__init__(**kwargs)
        if version_manager is not None:
            self.version_manager = version_manager
        else:
            from app.agents.version_manager import VersionManager
            self.version_manager = VersionManager(use_redis=True)

    def _run(
        self,
        session_id: str,
        query_type: str,
        query: Optional[str] = None,
        old_version_id: Optional[str] = None,
        new_version_id: Optional[str] = None,
    ) -> str:
        vm = self.version_manager
        try:
            if query_type == "history":
                return json.dumps({
                    "success": True,
                    "data": vm.format_versions_for_llm(session_id),
                }, ensure_ascii=False)
            if query_type == "current":
                version = vm.get_current_version(session_id)
                return json.dumps({
                    "success": True,
                    "data": version.plan_data if version else None,
                    "version_id": version.version_id if version else None,
                }, ensure_ascii=False)
            if query_type == "search":
                versions = vm.search_versions(session_id, query or "")
                return json.dumps({
                    "success": True,
                    "data": [
                        {
                            "version_id": v.version_id,
                            "version_number": v.version_number,
                            "changes": v.changes,
                            "attractions": (v.plan_data or {}).get("attractions", []),
                        }
                        for v in versions
                    ],
                }, ensure_ascii=False)
            if query_type == "compare":
                if not old_version_id or not new_version_id:
                    return json.dumps({"success": False, "error": "compare 需要 old_version_id 和 new_version_id"}, ensure_ascii=False)
                diff = vm.compare_versions(old_version_id, new_version_id)
                return json.dumps({
                    "success": True,
                    "data": diff.to_dict() if diff else None,
                }, ensure_ascii=False)
            return json.dumps({"success": False, "error": f"不支持的 query_type: {query_type}"}, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"success": False, "error": str(e)}, ensure_ascii=False)


def get_memory_tools(memory_manager: TravelMemoryManager = None, version_manager=None) -> list:
    """
    获取所有记忆相关工具
    
    Args:
        memory_manager: 共享的 TravelMemoryManager 实例
        version_manager: 可选 VersionManager，用于查询 Redis 中的计划版本
        
    Returns:
        工具列表
    """
    mm = memory_manager or TravelMemoryManager()
    tools = [
        TravelMemoryQueryTool(memory_manager=mm),
        DiffRecommendationTool(memory_manager=mm),
    ]
    if version_manager is not None:
        tools.append(PlanVersionQueryTool(version_manager=version_manager))
    return tools
