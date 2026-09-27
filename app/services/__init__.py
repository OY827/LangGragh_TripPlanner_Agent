"""
服务层 - 提供各类外部服务封装

包含：
- AmapService: 高德地图服务
- AmapMCPService: 高德 Streamable HTTP MCP
- HotelMCPService: RollingGo 酒店 MCP
- get_amap_service / get_amap_mcp_service / get_hotel_mcp_service
- 数据模型: POIInfo, AmapWeatherInfo, AmapRouteInfo
- LLM服务: get_llm, get_openai_llm, get_aliyun_llm
"""

from .amap_service import (
    AmapService,
    get_amap_service,
)
from .amap_mcp_service import (
    AmapMCPService,
    get_amap_mcp_service,
    is_amap_mcp_enabled,
)
from .hotel_mcp_service import (
    HotelMCPService,
    get_hotel_mcp_service,
)

from .llm_service import (
    get_llm,
    get_openai_llm,
    get_aliyun_llm
)

from app.models import POIInfo, AmapWeatherInfo, AmapRouteInfo

__all__ = [
    # 高德地图服务
    "AmapService",
    "get_amap_service",
    "AmapMCPService",
    "get_amap_mcp_service",
    "is_amap_mcp_enabled",
    "HotelMCPService",
    "get_hotel_mcp_service",
    "POIInfo",
    "AmapWeatherInfo",
    "AmapRouteInfo",
    # LLM服务
    "get_llm",
    "get_openai_llm",
    "get_aliyun_llm"
]
