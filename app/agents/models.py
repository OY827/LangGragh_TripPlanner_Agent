"""
数据模型定义 - 从app.models统一导入
此文件保留用于向后兼容，新代码应直接使用app.models
"""

# 从统一模型层导入所有模型
from app.models import (
    # 基础模型
    Location,
    
    # POI与景点模型
    POIInfo,
    Attraction,
    
    # 餐饮模型
    Meal,
    
    # 酒店模型
    Hotel,
    
    # 天气模型
    WeatherInfo,
    AmapWeatherInfo,
    
    # 路线模型
    RouteInfo,
    AmapRouteInfo,
    
    # 预算模型
    Budget,
    
    # 行程模型
    DayPlan,
    TripPlan,
    
    # 请求模型
    TripRequest,
    POISearchRequest,
    RouteRequest,
    
    # 响应模型
    TripPlanResponse,
    POISearchResponse,
    WeatherResponse,
    RouteResponse,
    ErrorResponse,
)

__all__ = [
    "Location",
    "POIInfo",
    "Attraction",
    "Meal",
    "Hotel",
    "WeatherInfo",
    "AmapWeatherInfo",
    "RouteInfo",
    "AmapRouteInfo",
    "Budget",
    "DayPlan",
    "TripPlan",
    "TripRequest",
    "POISearchRequest",
    "RouteRequest",
    "TripPlanResponse",
    "POISearchResponse",
    "WeatherResponse",
    "RouteResponse",
    "ErrorResponse",
]
