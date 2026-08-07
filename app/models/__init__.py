"""
Models层 - 统一数据模型定义

从schemas模块导入所有模型定义
"""

from app.models.schemas import (
    Location,
    POIInfo,
    Attraction,
    Meal,
    Hotel,
    WeatherInfo,
    AmapWeatherInfo,
    RouteInfo,
    AmapRouteInfo,
    RouteSegment,
    TimeSlot,
    Budget,
    DayPlan,
    TripPlan,
    TripRequest,
    POISearchRequest,
    RouteRequest,
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
    "RouteSegment",
    "TimeSlot",
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
