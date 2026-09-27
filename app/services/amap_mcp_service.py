"""
高德地图 MCP 服务（Streamable HTTP）

默认连官方托管：https://mcp.amap.com/mcp?key={AMAP_MAPS_API_KEY}
本地 sugarforever/amap-mcp-server 可用 AMAP_MCP_URL 覆盖，例如 http://localhost:8001/mcp
（不要占用本仓库 FastAPI 的 8000）。

不再启动 python -m amap_mcp_server stdio。
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from app.tools.mcp_http_client import StreamableHttpMCPClient

load_dotenv()

_COORD_RE = re.compile(r"^\s*-?\d+(\.\d+)?\s*,\s*-?\d+(\.\d+)?\s*$")

OFFICIAL_DIRECTION_TOOLS = {
    "driving": "maps_direction_driving",
    "walking": "maps_direction_walking",
    "transit": "maps_direction_transit_integrated",
    "bicycling": "maps_direction_bicycling",
}

ADDRESS_DIRECTION_TOOLS = {
    "driving": "maps_direction_driving_by_address",
    "walking": "maps_direction_walking_by_address",
    "transit": "maps_direction_transit_integrated_by_address",
    "bicycling": "maps_direction_bicycling_by_address",
}


def is_amap_mcp_enabled() -> bool:
    """USE_MCP 或 ENABLE_MCP_TOOLS 任一为 true 即走 MCP。"""
    for name in ("USE_MCP", "ENABLE_MCP_TOOLS"):
        if os.getenv(name, "false").strip().lower() == "true":
            return True
    return False


def _masked_url(url: str) -> str:
    if not url:
        return ""
    if "key=" in url.lower():
        base = url.split("?", 1)[0]
        return f"{base}?key=***"
    return url


def _is_coordinate(value: str) -> bool:
    return bool(value and _COORD_RE.match(value.strip()))


def _clean_args(arguments: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    cleaned: Dict[str, Any] = {}
    for key, value in (arguments or {}).items():
        if value is None:
            continue
        if isinstance(value, str) and value.strip() == "":
            continue
        cleaned[key] = value
    return cleaned


def resolve_amap_mcp_url() -> str:
    """显式 AMAP_MCP_URL 优先；否则拼官方托管地址。"""
    explicit = os.getenv("AMAP_MCP_URL", "").strip()
    api_key = os.getenv("AMAP_MAPS_API_KEY", "").strip()
    if explicit:
        if "mcp.amap.com" in explicit.lower() and "key=" not in explicit.lower() and api_key:
            sep = "&" if "?" in explicit else "?"
            return f"{explicit}{sep}key={api_key}"
        return explicit
    if not api_key:
        return ""
    return f"https://mcp.amap.com/mcp?key={api_key}"


class AmapMCPService:
    """高德 Streamable HTTP MCP 客户端封装（单例）。"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self.url = resolve_amap_mcp_url()
        self._client: Optional[StreamableHttpMCPClient] = None
        self._tool_names: Optional[List[str]] = None
        if self.url:
            self._client = StreamableHttpMCPClient(url=self.url, timeout=40.0)
        self._initialized = True
        print(
            f"高德 MCP 初始化完成 - 已配置: {self.is_configured()}, "
            f"endpoint: {_masked_url(self.url)}"
        )

    def is_configured(self) -> bool:
        return bool(self._client)

    def list_tool_names(self) -> List[str]:
        if self._tool_names is not None:
            return self._tool_names
        if not self._client:
            self._tool_names = []
            return self._tool_names
        try:
            tools = self._client.list_tools()
            self._tool_names = [item["name"] for item in tools if item.get("name")]
        except Exception as exc:
            print(f"高德 MCP tools/list 失败: {exc}")
            self._tool_names = []
        return self._tool_names

    def call_tool(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> Optional[Any]:
        if not self._client:
            return None
        args = _clean_args(arguments)
        try:
            return self._client.call_tool(name, args)
        except Exception as exc:
            print(f"高德 MCP 调用失败 {name}: {exc}")
            return None

    def text_search(
        self,
        keywords: str,
        city: str = "",
        citylimit: bool = True,
    ) -> Optional[Dict[str, Any]]:
        result = self.call_tool(
            "maps_text_search",
            {
                "keywords": keywords,
                "city": city,
                "citylimit": str(bool(citylimit)).lower(),
            },
        )
        return result if isinstance(result, dict) else None

    def weather(self, city: str) -> Optional[Dict[str, Any]]:
        result = self.call_tool("maps_weather", {"city": city})
        return result if isinstance(result, dict) else None

    def geo(self, address: str, city: str = "") -> Optional[Dict[str, Any]]:
        result = self.call_tool("maps_geo", {"address": address, "city": city})
        return result if isinstance(result, dict) else None

    def regeocode(self, location: str) -> Optional[Dict[str, Any]]:
        result = self.call_tool("maps_regeocode", {"location": location})
        return result if isinstance(result, dict) else None

    def search_detail(self, poi_id: str) -> Optional[Dict[str, Any]]:
        result = self.call_tool("maps_search_detail", {"id": poi_id})
        return result if isinstance(result, dict) else None

    def direction(
        self,
        origin: str,
        destination: str,
        mode: str = "driving",
        city: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """官方工具吃坐标；sugarforever 另有 *_by_address。先探测再调。"""
        mode = (mode or "driving").lower()
        names = set(self.list_tool_names())
        address_tool = ADDRESS_DIRECTION_TOOLS.get(mode)
        official_tool = OFFICIAL_DIRECTION_TOOLS.get(mode, OFFICIAL_DIRECTION_TOOLS["driving"])
        hosted_official = "mcp.amap.com" in (self.url or "")

        if address_tool and (address_tool in names or (not names and not hosted_official)):
            args: Dict[str, Any] = {
                "origin_address": origin,
                "destination_address": destination,
            }
            if city:
                args["origin_city"] = city
                args["destination_city"] = city
            elif mode == "transit":
                args["origin_city"] = "全国"
                args["destination_city"] = "全国"
            result = self.call_tool(address_tool, args)
            if self._looks_like_route(result):
                return result if isinstance(result, dict) else None

        origin_loc = origin.strip() if _is_coordinate(origin) else self._geo_location(origin, city)
        dest_loc = destination.strip() if _is_coordinate(destination) else self._geo_location(destination, city)
        if not origin_loc or not dest_loc:
            return None

        args = {"origin": origin_loc, "destination": dest_loc}
        if mode == "transit" and city:
            args["city"] = city
            args["cityd"] = city
        result = self.call_tool(official_tool, args)
        return result if isinstance(result, dict) else None

    def extract_pois(self, result: Any) -> List[Dict[str, Any]]:
        if isinstance(result, list):
            return [item for item in result if isinstance(item, dict)]
        if not isinstance(result, dict):
            return []
        for key in ("pois", "return", "data"):
            value = result.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
            if isinstance(value, dict) and isinstance(value.get("pois"), list):
                return [item for item in value["pois"] if isinstance(item, dict)]
        return []

    def extract_geocode(self, result: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(result, dict):
            return None
        geos = result.get("geocodes") or result.get("return") or result.get("results") or []
        if isinstance(geos, list) and geos:
            first = geos[0]
            return first if isinstance(first, dict) else None
        if result.get("location"):
            return result
        return None

    def _geo_location(self, address: str, city: Optional[str] = None) -> Optional[str]:
        result = self.geo(address, city or "")
        item = self.extract_geocode(result)
        if not item:
            return None
        location = item.get("location")
        return str(location).strip() if location else None

    @staticmethod
    def _looks_like_route(result: Any) -> bool:
        if not isinstance(result, dict) or result.get("error"):
            return False
        if str(result.get("status")) == "0":
            return False
        route = result.get("route") if isinstance(result.get("route"), dict) else result
        if not isinstance(route, dict):
            return False
        return bool(route.get("paths") or route.get("transits") or route.get("distance"))


def get_amap_mcp_service() -> AmapMCPService:
    return AmapMCPService()
