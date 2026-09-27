"""
RollingGo / 道旅酒店 MCP 服务

连 https://mcp.rollinggo.cn/mcp，调用 searchHotels 拿带价格的酒店。
与高德 POI 无关；高德仍只负责地图坐标和路线。
"""

from __future__ import annotations

import os
import re
from datetime import date, timedelta
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from app.tools.mcp_http_client import StreamableHttpMCPClient

load_dotenv()

_STAR_HINTS = [
    (["五星", "5星", "豪华", "奢华", "高档"], [5.0]),
    (["四星", "4星", "舒适", "商务"], [4.0]),
    (["三星", "3星", "经济", "快捷", "民宿"], [3.0, 2.0]),
]


class HotelMCPService:
    """RollingGo 酒店搜索封装（单例）。"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self.url = os.getenv("HOTEL_MCP_URL", "https://mcp.rollinggo.cn/mcp").strip()
        raw_key = os.getenv("HOTEL_API_KEY", "").strip()
        if raw_key.lower().startswith("bearer "):
            raw_key = raw_key[7:].strip()
        self.api_key = raw_key
        self._client: Optional[StreamableHttpMCPClient] = None
        if self.api_key:
            self._client = StreamableHttpMCPClient(
                url=self.url,
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
        self._initialized = True
        print(
            f"酒店 MCP 初始化完成 - 已配置: {self.is_configured()}, "
            f"endpoint: {self.url}"
        )

    def is_configured(self) -> bool:
        return bool(self._client)

    def search_hotels(
        self,
        city: str,
        keywords: str = "",
        accommodation: str = "",
        check_in_date: Optional[str] = None,
        stay_nights: int = 1,
        size: int = 8,
    ) -> List[Dict[str, Any]]:
        if not self._client:
            return []
        if not city:
            raise ValueError("缺少城市")

        stay_nights = max(1, int(stay_nights or 1))
        check_in_date = self._normalize_date(check_in_date)
        query_hint = " ".join(part for part in [city, accommodation, keywords] if part).strip()
        origin_query = query_hint if "酒店" in query_hint else f"{query_hint}酒店".strip()

        arguments: Dict[str, Any] = {
            "originQuery": origin_query or f"{city}酒店",
            "place": city,
            "placeType": "城市",
            "countryCode": "CN",
            "size": max(1, min(int(size or 8), 20)),
            "checkInParam": {
                "checkInDate": check_in_date,
                "stayNights": stay_nights,
            },
        }
        stars = self._star_ratings(f"{accommodation} {keywords}")
        if stars:
            arguments["filterOptions"] = {"starRatings": stars}

        result = self._client.call_tool("searchHotels", arguments)
        hotels = self._extract_hotels(result)
        if hotels:
            return hotels

        # 星级过滤过严时，去掉过滤再搜一次
        if "filterOptions" in arguments:
            arguments.pop("filterOptions", None)
            result = self._client.call_tool("searchHotels", arguments)
            hotels = self._extract_hotels(result)
        return hotels

    def search_and_format(
        self,
        city: str,
        keywords: str = "",
        accommodation: str = "",
        check_in_date: Optional[str] = None,
        stay_nights: int = 1,
        size: int = 8,
    ) -> str:
        hotels = self.search_hotels(
            city=city,
            keywords=keywords,
            accommodation=accommodation,
            check_in_date=check_in_date,
            stay_nights=stay_nights,
            size=size,
        )
        return self.format_for_agent(hotels, city=city)

    def format_for_agent(self, hotels: List[Dict[str, Any]], city: str = "") -> str:
        if not hotels:
            return f"未找到{city}符合条件的酒店" if city else "未找到符合条件的酒店"

        lines = [f"共找到 {len(hotels)} 家酒店（RollingGo 实时报价）：", ""]
        for i, hotel in enumerate(hotels, 1):
            name = self._pick(hotel, "name", "hotelName", "hotel_name", "title") or "未知酒店"
            lines.append(f"{i}. {name}")

            address = self._pick(hotel, "address", "hotelAddress", "addr", "location")
            if address:
                lines.append(f"   地址: {address}")

            star = self._pick(hotel, "star", "starRating", "stars", "hotelStar")
            if star:
                lines.append(f"   星级: {star}")

            rating = self._pick(hotel, "rating", "score", "commentScore", "reviewScore")
            if rating:
                lines.append(f"   评分: {rating}")

            price = self._format_price(hotel)
            if price:
                lines.append(f"   价格: {price}")

            distance = self._pick(hotel, "distance", "distanceDesc", "distanceText")
            if distance:
                lines.append(f"   距离: {distance}")

            hotel_type = self._pick(hotel, "type", "category", "hotelType")
            if hotel_type:
                lines.append(f"   类型: {hotel_type}")
            lines.append("")
        return "\n".join(lines).strip()

    @staticmethod
    def _normalize_date(check_in_date: Optional[str]) -> str:
        if check_in_date:
            text = str(check_in_date).strip()[:10]
            if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
                return text
        return (date.today() + timedelta(days=7)).isoformat()

    @staticmethod
    def _star_ratings(text: str) -> List[float]:
        blob = text or ""
        for hints, stars in _STAR_HINTS:
            if any(h in blob for h in hints):
                return stars
        return []

    @staticmethod
    def _extract_hotels(payload: Any) -> List[Dict[str, Any]]:
        if payload is None:
            return []
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        if not isinstance(payload, dict):
            return []

        for key in (
            "hotelInformationList",
            "hotels",
            "hotelList",
            "list",
            "items",
            "records",
            "data",
            "result",
        ):
            value = payload.get(key)
            if isinstance(value, list) and value and isinstance(value[0], dict):
                return [item for item in value if isinstance(item, dict)]
            if isinstance(value, dict):
                nested = HotelMCPService._extract_hotels(value)
                if nested:
                    return nested

        if HotelMCPService._pick(payload, "name", "hotelName"):
            return [payload]
        return []

    @staticmethod
    def _pick(data: Dict[str, Any], *keys: str) -> Any:
        for key in keys:
            value = data.get(key)
            if value not in (None, "", [], {}):
                return value
        return None

    @classmethod
    def _format_price(cls, hotel: Dict[str, Any]) -> str:
        price = cls._pick(
            hotel,
            "price",
            "displayPrice",
            "pricePerNight",
            "minPrice",
            "lowestPrice",
            "nightlyPrice",
            "salePrice",
            "amount",
        )
        if isinstance(price, dict):
            price = cls._pick(
                price,
                "lowestPrice",
                "amount",
                "value",
                "min",
                "price",
                "display",
                "message",
            )
        if price in (None, ""):
            return ""
        text = str(price).strip()
        if re.search(r"元", text):
            return text if "晚" in text else f"{text}/晚"
        digits = re.search(r"\d+(\.\d+)?", text.replace(",", ""))
        if digits:
            return f"{digits.group(0)}元/晚"
        return text


def get_hotel_mcp_service() -> HotelMCPService:
    return HotelMCPService()
