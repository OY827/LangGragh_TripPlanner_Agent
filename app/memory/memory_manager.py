"""旅行记忆管理器 — 长期记忆默认本地 Chroma，测试可用文件后端。"""

import json
import uuid
import os
from datetime import datetime
from typing import List, Dict, Optional, Any, Union
from dataclasses import dataclass, asdict

try:
    import redis
    from redis import Redis
except ImportError:
    redis = None

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


@dataclass
class TravelPlan:
    plan_id: str
    user_id: str
    city: str
    duration_days: int
    created_at: str
    attractions: List[str]
    daily_itinerary: Dict[str, List[str]]
    preferences: List[str] = None
    budget: Optional[Union[int, Dict[str, Any]]] = None
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TravelPlan":
        return cls(
            plan_id=data.get("plan_id", ""),
            user_id=data.get("user_id", ""),
            city=data.get("city", ""),
            duration_days=data.get("duration_days", 1),
            created_at=data.get("created_at", ""),
            attractions=data.get("attractions", []),
            daily_itinerary=data.get("daily_itinerary", {}),
            preferences=data.get("preferences", []),
            budget=data.get("budget"),
            notes=data.get("notes")
        )


class TravelMemoryManager:
    def __init__(
        self,
        redis_host: str = None,
        redis_port: int = None,
        redis_password: str = None,
        use_redis: bool = None,
        backend: str = None,
        embeddings=None,
        persist_directory: str = None,
    ):
        """
        backend:
          - chroma: 用户长期记忆（默认）
          - file: 本地 JSON，给单测 / 无 embedding 环境
          - redis: 旧后端，仅显式指定时使用
        use_redis=False 仍映射到 file，兼容现有测试。
        """
        self.client = None
        self._chroma = None
        if backend is None:
            if use_redis is False:
                backend = "file"
            else:
                backend = os.getenv("USER_MEMORY_BACKEND", "chroma")
        self.backend = (backend or "chroma").lower()

        if self.backend == "chroma":
            try:
                from app.memory.chroma_memory import ChromaUserMemory
                self._chroma = ChromaUserMemory(
                    persist_directory=persist_directory,
                    embeddings=embeddings,
                )
                self.use_redis = False
                return
            except Exception as e:
                print(f"Chroma 长期记忆初始化失败，回退到本地文件: {e}")
                self.backend = "file"

        if self.backend == "redis" and redis:
            if load_dotenv:
                load_dotenv()
            host = redis_host or os.getenv("REDIS_HOST", "localhost")
            port = redis_port or int(os.getenv("REDIS_PORT", 6379))
            password = redis_password or os.getenv("REDIS_PASSWORD", "")
            try:
                self.client = Redis(
                    host=host,
                    port=port,
                    password=password or None,
                    decode_responses=True,
                    socket_timeout=10,
                )
                self.client.ping()
                self.use_redis = True
                return
            except Exception as e:
                print(f"Redis连接失败，回退到本地文件存储: {e}")
                self.backend = "file"

        self.use_redis = False
        self.backend = "file"
        self.storage_path = "app/data/travel_plans"
        os.makedirs(self.storage_path, exist_ok=True)

    def _get_user_key(self, user_id: str) -> str:
        return f"travel:user:{user_id}:plans"

    def _get_plan_key(self, user_id: str, plan_id: str) -> str:
        return f"travel:user:{user_id}:plan:{plan_id}"

    def _get_city_key(self, user_id: str, city: str) -> str:
        return f"travel:user:{user_id}:city:{city}"

    def _get_user_folder(self, user_id: str) -> str:
        user_folder = os.path.join(self.storage_path, user_id)
        os.makedirs(user_folder, exist_ok=True)
        return user_folder

    def _get_plan_file(self, user_id: str, plan_id: str) -> str:
        return os.path.join(self._get_user_folder(user_id), f"{plan_id}.json")

    def save_plan(self, user_id: str, city: str, duration_days: int, 
                  attractions: List[str], daily_itinerary: Dict[str, List[str]],
                  preferences: List[str] = None, budget: Optional[Union[int, Dict[str, Any]]] = None,
                  notes: Optional[str] = None) -> str:
        plan_id = str(uuid.uuid4())[:8]
        plan = TravelPlan(
            plan_id=plan_id,
            user_id=user_id,
            city=city,
            duration_days=duration_days,
            created_at=datetime.now().isoformat(),
            attractions=attractions,
            daily_itinerary=daily_itinerary,
            preferences=preferences or [],
            budget=budget,
            notes=notes
        )

        if self._chroma:
            return self._chroma.save_plan(
                user_id=user_id,
                city=city,
                duration_days=duration_days,
                attractions=attractions,
                daily_itinerary=daily_itinerary,
                preferences=preferences or [],
                budget=budget,
                notes=notes,
                plan_id=plan_id,
                created_at=plan.created_at,
            )

        if self.use_redis:
            plan_data = json.dumps(plan.to_dict(), ensure_ascii=False)
            
            self.client.set(self._get_plan_key(user_id, plan_id), plan_data)
            
            self.client.rpush(self._get_user_key(user_id), plan_id)
            
            self.client.rpush(self._get_city_key(user_id, city), plan_id)
            
            for attraction in attractions:
                attraction_key = f"travel:user:{user_id}:city:{city}:attraction:{attraction}"
                self.client.sadd(attraction_key, plan_id)
        else:
            file_path = self._get_plan_file(user_id, plan_id)
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(plan.to_dict(), f, ensure_ascii=False, indent=2)

        return plan_id

    def get_plan(self, user_id: str, plan_id: str) -> Optional[TravelPlan]:
        if self._chroma:
            data = self._chroma.get_plan(user_id, plan_id)
            return TravelPlan.from_dict(data) if data else None
        if self.use_redis:
            plan_data = self.client.get(self._get_plan_key(user_id, plan_id))
            if plan_data:
                data = json.loads(plan_data)
                return TravelPlan.from_dict(data)
            return None
        else:
            file_path = self._get_plan_file(user_id, plan_id)
            if not os.path.exists(file_path):
                return None

            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return TravelPlan.from_dict(data)

    def get_user_plans(self, user_id: str) -> List[TravelPlan]:
        if self._chroma:
            return [TravelPlan.from_dict(item) for item in self._chroma.get_user_plans(user_id)]
        if self.use_redis:
            plan_ids = self.client.lrange(self._get_user_key(user_id), 0, -1)
            plans = []
            for plan_id in plan_ids:
                plan = self.get_plan(user_id, plan_id)
                if plan:
                    plans.append(plan)
            plans.sort(key=lambda p: p.created_at, reverse=True)
            return plans
        else:
            user_folder = self._get_user_folder(user_id)
            plans = []

            for filename in os.listdir(user_folder):
                if filename.endswith(".json"):
                    plan_id = filename.replace(".json", "")
                    plan = self.get_plan(user_id, plan_id)
                    if plan:
                        plans.append(plan)

            plans.sort(key=lambda p: p.created_at, reverse=True)
            return plans

    def get_user_plans_by_city(self, user_id: str, city: str) -> List[TravelPlan]:
        if self._chroma:
            return [TravelPlan.from_dict(item) for item in self._chroma.get_user_plans_by_city(user_id, city)]
        if self.use_redis:
            plan_ids = self.client.lrange(self._get_city_key(user_id, city), 0, -1)
            plans = []
            for plan_id in plan_ids:
                plan = self.get_plan(user_id, plan_id)
                if plan:
                    plans.append(plan)
            plans.sort(key=lambda p: p.created_at, reverse=True)
            return plans
        else:
            all_plans = self.get_user_plans(user_id)
            return [p for p in all_plans if p.city == city]

    def has_visited_city(self, user_id: str, city: str) -> bool:
        plans = self.get_user_plans_by_city(user_id, city)
        return len(plans) > 0

    def get_visited_attractions(self, user_id: str, city: str) -> List[str]:
        if self._chroma:
            return self._chroma.get_visited_attractions(user_id, city)
        visited = set()
        plans = self.get_user_plans_by_city(user_id, city)
        
        for plan in plans:
            visited.update(plan.attractions)
        
        return sorted(list(visited))

    def get_preferences(self, user_id: str, city: str = "") -> List[str]:
        if self._chroma:
            return self._chroma.get_preferences(user_id, city)
        prefs = []
        seen = set()
        plans = self.get_user_plans_by_city(user_id, city) if city else self.get_user_plans(user_id)
        for plan in plans:
            for item in plan.preferences or []:
                if item and item not in seen:
                    seen.add(item)
                    prefs.append(item)
        return prefs

    def add_preference(self, user_id: str, preference: str, city: str = "") -> str:
        if self._chroma:
            return self._chroma.add_preference(user_id, preference, city)
        return ""

    def search_semantic(self, user_id: str, query: str, k: int = 5, memory_type: str = None) -> List[Dict[str, Any]]:
        if self._chroma:
            return self._chroma.search(user_id, query, k=k, memory_type=memory_type)
        return []

    def compare_plans(self, user_id: str, city: str) -> Dict[str, Any]:
        if self._chroma:
            return self._chroma.compare_plans(user_id, city)
        plans = self.get_user_plans_by_city(user_id, city)
        
        if len(plans) == 0:
            return {
                "has_history": False,
                "plan_count": 0,
                "visited_attractions": [],
                "visited_count": 0,
                "recommended_new_attractions": []
            }

        visited = self.get_visited_attractions(user_id, city)
        
        return {
            "has_history": True,
            "plan_count": len(plans),
            "latest_plan_date": plans[0].created_at if plans else None,
            "visited_attractions": visited,
            "visited_count": len(visited)
        }

    def delete_plan(self, user_id: str, plan_id: str) -> bool:
        if self._chroma:
            return self._chroma.delete_plan(user_id, plan_id)
        if self.use_redis:
            plan = self.get_plan(user_id, plan_id)
            if not plan:
                return False
            
            self.client.delete(self._get_plan_key(user_id, plan_id))
            
            self.client.lrem(self._get_user_key(user_id), 0, plan_id)
            
            if plan.city:
                self.client.lrem(self._get_city_key(user_id, plan.city), 0, plan_id)
            
            return True
        else:
            file_path = self._get_plan_file(user_id, plan_id)
            if os.path.exists(file_path):
                os.remove(file_path)
                return True
            return False

    def clear_user_plans(self, user_id: str) -> int:
        if self._chroma:
            return self._chroma.clear_user_plans(user_id)
        if self.use_redis:
            plan_ids = self.client.lrange(self._get_user_key(user_id), 0, -1)
            count = 0
            for plan_id in plan_ids:
                self.delete_plan(user_id, plan_id)
                count += 1
            return count
        else:
            user_folder = self._get_user_folder(user_id)
            count = 0
            for filename in os.listdir(user_folder):
                if filename.endswith(".json"):
                    os.remove(os.path.join(user_folder, filename))
                    count += 1
            return count

    def get_storage_type(self) -> str:
        if self._chroma:
            return "chroma"
        return "redis" if self.use_redis else "file"

    def _parse_datetime(self, date_str: str) -> Optional[datetime]:
        """
        解析日期时间字符串，支持多种格式
        
        Args:
            date_str: 日期时间字符串
            
        Returns:
            datetime 对象或 None
        """
        if not date_str:
            return None
        
        formats = [
            "%Y-%m-%dT%H:%M:%S.%f",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M:%S.%f",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d",
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        
        # 尝试使用 fromisoformat（Python 3.7+）
        try:
            return datetime.fromisoformat(date_str)
        except (ValueError, TypeError):
            pass
        
        return None

    def get_plans_by_date_range(self, user_id: str, start_date: str, end_date: str) -> List[TravelPlan]:
        """
        按日期范围筛选计划
        
        Args:
            user_id: 用户ID
            start_date: 开始日期 (格式: YYYY-MM-DD)
            end_date: 结束日期 (格式: YYYY-MM-DD)
            
        Returns:
            在指定日期范围内的计划列表
        """
        try:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
            # 设置结束日期为当天最后一秒
            end_dt = end_dt.replace(hour=23, minute=59, second=59)
        except ValueError:
            return []
        
        all_plans = self.get_user_plans(user_id)
        filtered_plans = []
        
        for plan in all_plans:
            plan_date = self._parse_datetime(plan.created_at)
            if plan_date and start_dt <= plan_date <= end_dt:
                filtered_plans.append(plan)
        
        return filtered_plans

    def get_plans_by_budget_range(self, user_id: str, min_budget: float = 0, max_budget: float = float('inf')) -> List[TravelPlan]:
        """
        按预算范围筛选计划
        
        Args:
            user_id: 用户ID
            min_budget: 最低预算
            max_budget: 最高预算
            
        Returns:
            在指定预算范围内的计划列表
        """
        all_plans = self.get_user_plans(user_id)
        filtered_plans = []
        
        for plan in all_plans:
            budget_total = self._extract_budget_total(plan.budget)
            if budget_total is not None and min_budget <= budget_total <= max_budget:
                filtered_plans.append(plan)
        
        return filtered_plans

    def get_plans_by_attraction(self, user_id: str, attraction: str) -> List[TravelPlan]:
        """
        按景点筛选计划
        
        Args:
            user_id: 用户ID
            attraction: 景点名称（支持模糊匹配）
            
        Returns:
            包含指定景点的计划列表
        """
        all_plans = self.get_user_plans(user_id)
        filtered_plans = []
        
        attraction_lower = attraction.lower()
        for plan in all_plans:
            for a in plan.attractions:
                if attraction_lower in a.lower():
                    filtered_plans.append(plan)
                    break
        
        return filtered_plans

    def get_plans_by_duration(self, user_id: str, min_days: int = 0, max_days: int = float('inf')) -> List[TravelPlan]:
        """
        按行程天数筛选计划
        
        Args:
            user_id: 用户ID
            min_days: 最短天数
            max_days: 最长天数
            
        Returns:
            在指定天数范围内的计划列表
        """
        all_plans = self.get_user_plans(user_id)
        return [p for p in all_plans if min_days <= p.duration_days <= max_days]

    def search_plans(self, user_id: str, **filters) -> List[TravelPlan]:
        """
        组合条件搜索计划
        
        支持的筛选条件:
        - start_date, end_date: 日期范围
        - min_budget, max_budget: 预算范围
        - attraction: 景点名称
        - min_days, max_days: 行程天数
        - city: 城市名称
        
        Args:
            user_id: 用户ID
            **filters: 筛选条件
            
        Returns:
            符合所有条件的计划列表
        """
        all_plans = self.get_user_plans(user_id)
        
        # 按城市筛选
        if 'city' in filters:
            city = filters['city']
            city_lower = city.lower()
            all_plans = [p for p in all_plans if city_lower in p.city.lower()]
        
        # 按日期范围筛选
        if 'start_date' in filters and 'end_date' in filters:
            start_date = filters['start_date']
            end_date = filters['end_date']
            try:
                start_dt = datetime.strptime(start_date, "%Y-%m-%d")
                end_dt = datetime.strptime(end_date, "%Y-%m-%d")
                end_dt = end_dt.replace(hour=23, minute=59, second=59)
                filtered = []
                for plan in all_plans:
                    plan_date = self._parse_datetime(plan.created_at)
                    if plan_date and start_dt <= plan_date <= end_dt:
                        filtered.append(plan)
                all_plans = filtered
            except ValueError:
                pass
        
        # 按预算范围筛选
        min_budget = filters.get('min_budget', 0)
        max_budget = filters.get('max_budget', float('inf'))
        filtered = []
        for plan in all_plans:
            budget_total = self._extract_budget_total(plan.budget)
            if budget_total is not None and min_budget <= budget_total <= max_budget:
                filtered.append(plan)
        all_plans = filtered
        
        # 按景点筛选
        if 'attraction' in filters:
            attraction = filters['attraction']
            attraction_lower = attraction.lower()
            filtered = []
            for plan in all_plans:
                for a in plan.attractions:
                    if attraction_lower in a.lower():
                        filtered.append(plan)
                        break
            all_plans = filtered
        
        # 按天数筛选
        min_days = filters.get('min_days', 0)
        max_days = filters.get('max_days', float('inf'))
        all_plans = [p for p in all_plans if min_days <= p.duration_days <= max_days]
        
        return all_plans

    def get_user_statistics(self, user_id: str) -> Dict[str, Any]:
        """
        获取用户旅行统计数据
        
        Args:
            user_id: 用户ID
            
        Returns:
            统计数据字典
        """
        plans = self.get_user_plans(user_id)
        
        if not plans:
            return {
                "total_trips": 0,
                "total_cities": 0,
                "total_days": 0,
                "total_budget": 0,
                "avg_budget": 0,
                "cities_visited": [],
                "popular_attractions": [],
                "budget_distribution": {},
                "duration_distribution": {}
            }
        
        # 基础统计
        total_trips = len(plans)
        cities = set(p.city for p in plans)
        total_days = sum(p.duration_days for p in plans)
        
        # 预算统计
        budgets = []
        for plan in plans:
            budget_total = self._extract_budget_total(plan.budget)
            if budget_total is not None:
                budgets.append(budget_total)
        
        total_budget = sum(budgets) if budgets else 0
        avg_budget = total_budget / len(budgets) if budgets else 0
        
        # 景点统计
        attraction_counter = {}
        for plan in plans:
            for attraction in plan.attractions:
                if attraction not in attraction_counter:
                    attraction_counter[attraction] = 0
                attraction_counter[attraction] += 1
        
        popular_attractions = sorted(attraction_counter.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # 预算分布
        budget_ranges = {
            "low (<500元)": 0,
            "medium (500-1500元)": 0,
            "high (1500-3000元)": 0,
            "very_high (>3000元)": 0
        }
        for budget in budgets:
            if budget < 500:
                budget_ranges["low (<500元)"] += 1
            elif budget < 1500:
                budget_ranges["medium (500-1500元)"] += 1
            elif budget < 3000:
                budget_ranges["high (1500-3000元)"] += 1
            else:
                budget_ranges["very_high (>3000元)"] += 1
        
        # 天数分布
        duration_dist = {}
        for plan in plans:
            days = plan.duration_days
            if days <= 2:
                cat = "短途 (1-2天)"
            elif days <= 4:
                cat = "中途 (3-4天)"
            elif days <= 7:
                cat = "长途 (5-7天)"
            else:
                cat = "超长途 (>7天)"
            duration_dist[cat] = duration_dist.get(cat, 0) + 1
        
        # 年份分布
        year_dist = {}
        for plan in plans:
            plan_date = self._parse_datetime(plan.created_at)
            if plan_date:
                year = plan_date.year
                year_dist[year] = year_dist.get(year, 0) + 1
        
        return {
            "total_trips": total_trips,
            "total_cities": len(cities),
            "total_days": total_days,
            "total_budget": total_budget,
            "avg_budget": round(avg_budget, 2),
            "min_budget": min(budgets) if budgets else 0,
            "max_budget": max(budgets) if budgets else 0,
            "cities_visited": sorted(list(cities)),
            "popular_attractions": popular_attractions,
            "budget_distribution": budget_ranges,
            "duration_distribution": duration_dist,
            "year_distribution": year_dist,
            "has_budget_data": len(budgets) > 0
        }

    def _extract_budget_total(self, budget: Optional[Union[int, Dict[str, Any]]]) -> Optional[float]:
        """
        从预算字段提取总金额
        
        Args:
            budget: 预算数据（可以是int或Dict）
            
        Returns:
            预算总金额或None
        """
        if budget is None:
            return None
        
        if isinstance(budget, (int, float)):
            return float(budget)
        
        if isinstance(budget, dict):
            total = budget.get('total', budget.get('total_budget', 0))
            if total:
                return float(total)
            
            # 如果没有total字段，尝试求和
            att = budget.get('total_attractions', 0)
            hot = budget.get('total_hotels', 0)
            meal = budget.get('total_meals', 0)
            trans = budget.get('total_transportation', 0)
            
            calculated_total = att + hot + meal + trans
            if calculated_total > 0:
                return float(calculated_total)
        
        return None
