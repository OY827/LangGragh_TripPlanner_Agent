"""行程预算。"""

import re
from typing import Any, Dict, List, Optional

class BudgetMixin:
    """预算汇总。"""

    def _calculate_budget(
        self,
        parsed_plan: Dict[str, Any],
        state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        计算旅行预算

        数据来源：
        - 景点门票：从 KB 文档的 metadata.price 提取
        - 酒店费用：从酒店搜索结果估算
        - 餐饮：按 60 元/人/餐 × 2 餐 × 天数
        - 交通：按景点间距离估算
        """
        days = parsed_plan.get("days", [])
        travel_days = state.get("travel_days", len(days) if days else 1)
        kb_docs = state.get("kb_docs", [])
        hotel_response = state.get("hotel_response", "")
        amap_results = state.get("amap_results", "")

        # 1. 景点门票费用
        attraction_prices = self._extract_attraction_prices(
            kb_docs,
            days,
            state.get("attraction_candidates") or [],
        )
        total_attractions = sum(attraction_prices.values())

        # 2. 酒店费用（每晚 300 元估算）
        hotel_count = max(1, travel_days - 1)
        hotel_price_per_night = self._estimate_hotel_price(hotel_response)
        total_hotels = hotel_price_per_night * hotel_count

        # 3. 餐饮费用（午饭和晚饭，每餐 60 元）
        meal_cost_per_day = 120
        total_meals = meal_cost_per_day * travel_days

        # 4. 交通费用
        total_transportation = self._estimate_transport_cost(days, amap_results)

        # 5. 汇总
        total = total_attractions + total_hotels + total_meals + total_transportation

        return {
            "per_day_breakdown": {
                f"day_{i+1}": {
                    "attractions": [
                        {"name": a.get("name", ""), "ticket_price": attraction_prices.get(a.get("name", ""), 0)}
                        for a in day.get("attractions", [])
                        if isinstance(a, dict)
                    ]
                }
                for i, day in enumerate(days)
                if isinstance(day, dict)
            },
            "attraction_tickets": attraction_prices,
            "total_attractions": total_attractions,
            "hotel_details": {
                "nights": hotel_count,
                "price_per_night": hotel_price_per_night,
            },
            "total_hotels": total_hotels,
            "meal_details": {
                "meals_per_day": 2,
                "price_per_meal": 60,
                "days": travel_days,
            },
            "total_meals": total_meals,
            "total_transportation": total_transportation,
            "total": total,
            "currency": "CNY",
            "people_count": 1,
        }

    def _extract_attraction_prices(
        self,
        kb_docs: List[Dict],
        days: List[Dict],
        candidates: Optional[List[Dict]] = None,
    ) -> Dict[str, int]:
        """门票预算：优先用候选里的飞猪实价或知识库参考价。"""
        price_map = {}
        for doc in kb_docs:
            meta = doc.get("metadata", {})
            title = meta.get("title", "")
            price_str = meta.get("price", "0")
            try:
                price = int(re.sub(r'[^\d]', '', str(price_str)) or 0)
            except (ValueError, TypeError):
                price = 0
            if title:
                price_map[title] = price
        for item in candidates or []:
            name = str(item.get("name") or "").strip()
            price = item.get("ticket_price")
            if not name or price in (None, ""):
                continue
            try:
                price_map[name] = int(price)
            except (TypeError, ValueError):
                continue

        attraction_prices = {}
        for day in days:
            if not isinstance(day, dict):
                continue
            for attr in day.get("attractions", []):
                if isinstance(attr, dict):
                    name = attr.get("name", "")
                    if name:
                        attraction_prices[name] = price_map.get(name, 0)
                        if attraction_prices[name] == 0:
                            for title, price in price_map.items():
                                if name in title or title in name:
                                    attraction_prices[name] = price
                                    break

        return attraction_prices

    def _estimate_transport_cost(
        self,
        days: List[Dict],
        amap_results: str
    ) -> int:
        """估算交通费用"""
        base_cost_per_day = 30
        total = 0

        for day in days:
            if not isinstance(day, dict):
                continue
            attractions = day.get("attractions", [])
            route_segments = day.get("route_segments", [])

            day_cost = base_cost_per_day
            if route_segments and isinstance(route_segments, list):
                for seg in route_segments:
                    if isinstance(seg, dict):
                        distance = seg.get("distance", 0)
                        if distance:
                            day_cost += max(5, int(distance * 0.005))

            if len(attractions) > 2:
                day_cost += 20

            total += day_cost

        return total

