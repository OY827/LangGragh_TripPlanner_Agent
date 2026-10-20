"""酒店节点。"""

import re
from typing import Any, Dict, List

class HotelMixin:
    """酒店查询。"""

    def search_hotels(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """按城市、档次和入住日期直接搜索酒店，有报价的按价格从低到高排列。"""
        try:
            city = state.get("city", "")
            accommodation = state.get("accommodation") or "经济型"
            start_date = state.get("start_date") or ""
            travel_days = int(state.get("travel_days") or 1)
            stay_nights = max(1, travel_days - 1)
            hotel_budget = state.get("hotel_budget") or ""

            from app.services.hotel_mcp_service import get_hotel_mcp_service

            service = get_hotel_mcp_service()
            if service.is_configured():
                hotels = service.search_hotels(
                    city=city,
                    keywords=accommodation,
                    accommodation=accommodation,
                    check_in_date=start_date or None,
                    stay_nights=stay_nights,
                )
                hotels = service.filter_by_budget(service.sort_by_price(hotels), hotel_budget)
                options = [service.as_option(item) for item in hotels]
                if hotels or hotel_budget:
                    return {
                        "hotel_response": service.format_for_agent(
                            hotels, city=city, budget=hotel_budget
                        ),
                        "hotel_candidates": options,
                    }

            if hotel_budget:
                return {
                    "hotel_response": "未取得酒店报价，无法按每晚预算筛选",
                    "hotel_candidates": [],
                }

            results = self.amap_service.search_poi(
                keywords=f"{accommodation}酒店",
                city=city,
            )
            return {
                "hotel_response": self._format_hotel_results(results),
                "hotel_candidates": self._poi_candidates(results),
            }
        except Exception as e:
            return {"hotel_response": f"酒店搜索失败: {str(e)}", "hotel_candidates": []}
    
    def reflect_hotels(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """反思酒店搜索结果"""
        hotel_response = state.get("hotel_response", "")
        iteration = state.get("iteration_count", 0)
        
        has_hotels = "酒店" in hotel_response and len(hotel_response) > 100
        
        if has_hotels:
            return {"reflection_pass": True, "reflection_feedback": "酒店结果充足", "iteration_count": iteration}
        else:
            return {"reflection_pass": False, "reflection_feedback": "酒店结果不足，需要重新搜索", "iteration_count": iteration + 1}
    
    # ==================== 主图节点 ====================
    
    def build_hotel_query(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """构建酒店查询"""
        return {
            "current_phase": "hotels"
        }
    
    def _format_hotel_results(self, results) -> str:
        """格式化酒店结果（POIInfo 对象列表）"""
        if not results:
            return "未找到符合条件的酒店"
        
        lines = []
        for i, r in enumerate(results[:10], 1):
            name = r.name if hasattr(r, 'name') else r.get('name', '未知')
            lines.append(f"{i}. {name}")
            
            address = r.address if hasattr(r, 'address') else r.get('address')
            if address:
                lines.append(f"   地址: {address}")
            
            tel = r.tel if hasattr(r, 'tel') else r.get('tel')
            if tel:
                lines.append(f"   电话: {tel}")
            
            poi_type = r.type if hasattr(r, 'type') else r.get('type')
            if poi_type:
                lines.append(f"   类型: {poi_type}")
            
            lines.append("")
        
        return "\n".join(lines)

    def _poi_candidates(self, results) -> List[Dict[str, Any]]:
        """高德没有报价时，仍把名称和地址留给用户选择。"""
        options = []
        for item in (results or [])[:10]:
            name = item.name if hasattr(item, "name") else item.get("name", "")
            if not name:
                continue
            address = item.address if hasattr(item, "address") else item.get("address")
            poi_type = item.type if hasattr(item, "type") else item.get("type")
            options.append({
                "name": name,
                "address": address or "",
                "price_range": "",
                "rating": "",
                "distance": "",
                "type": poi_type or "",
                "estimated_cost": 0,
            })
        return options
    
    def _estimate_hotel_price(self, hotel_response: str) -> int:
        """从酒店搜索结果估算平均房价"""
        if not hotel_response:
            return 300

        prices = []
        for match in re.finditer(r'(\d+)\s*元', hotel_response):
            try:
                price = int(match.group(1))
                if 50 <= price <= 3000:
                    prices.append(price)
            except ValueError:
                pass

        if prices:
            return max(200, sum(prices) // len(prices))
        return 300

