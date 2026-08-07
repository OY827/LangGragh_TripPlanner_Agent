"""
服务层 - 提供各类外部服务封装

包含：
- AmapService: 高德地图服务
- get_amap_service: 获取高德地图服务实例（单例模式）
- 数据模型: POIInfo, AmapWeatherInfo, AmapRouteInfo
- LLM服务: get_llm, get_openai_llm, get_aliyun_llm
"""

from .amap_service import (
    AmapService,
    get_amap_service,
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
    "POIInfo",
    "AmapWeatherInfo",
    "AmapRouteInfo",
    # LLM服务
    "get_llm",
    "get_openai_llm",
    "get_aliyun_llm"
]
