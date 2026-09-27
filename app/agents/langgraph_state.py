"""
LangGraph 全局状态 Schema 定义

定义所有 Agent 共享的全局状态字段，用于在图节点间传递数据。
状态分为：
- TripState: 主旅行规划图的状态
- AttractionState: 景点子图的状态
- WeatherState: 天气子图的状态
- HotelState: 酒店子图的状态
"""

from typing import TypedDict, List, Dict, Any, Optional, Literal
from operator import add


class AttractionState(TypedDict, total=False):
    """景点子图状态"""
    query: str
    city: str
    preferences: List[str]
    visited_attractions: List[str]
    
    # 检索状态
    kb_docs: List[Dict[str, Any]]
    amap_results: str
    merged_results: str
    
    # 反思状态
    reflection_feedback: str
    reflection_pass: bool
    
    # 最终输出
    attraction_response: str
    iteration_count: int


class WeatherState(TypedDict, total=False):
    """天气子图状态"""
    city: str
    start_date: str
    end_date: str
    
    weather_response: str
    iteration_count: int


class HotelState(TypedDict, total=False):
    """酒店子图状态"""
    city: str
    accommodation: str
    start_date: str
    travel_days: int
    
    hotel_response: str
    iteration_count: int


class TripState(TypedDict, total=False):
    """主旅行规划图状态

    包含所有子图需要的字段，用于子图嵌套时状态共享。
    """
    # 用户请求
    user_query: str
    city: str
    start_date: str
    end_date: str
    travel_days: int
    transportation: str
    accommodation: str
    preferences: List[str]
    user_id: str
    free_text_input: str

    # 景点子图字段（与 AttractionState 共享）
    query: str
    kb_docs: List[Dict[str, Any]]
    amap_results: str
    merged_results: str
    reflection_pass: bool

    # 子图结果
    attraction_response: str
    weather_response: str
    hotel_response: str

    # 天气子图字段（与 WeatherState 共享）
    # start_date, end_date, weather_response, city 已定义

    # 酒店子图字段（与 HotelState 共享）
    # city, accommodation, hotel_response 已定义

    # 路线
    route_info: str

    # 规划结果
    trip_plan: str
    trip_plan_parsed: Optional[Dict[str, Any]]

    # 预算
    budget: Optional[Dict[str, Any]]
    budget_total: int

    # ========== 对话Agent字段 ==========
    # 会话管理
    session_id: str              # 对话会话ID
    current_version_id: str      # 当前版本ID

    # 对话交互
    dialog_input: str            # 用户修改输入
    dialog_reply: str            # 对话Agent回复
    dialog_changes: List[str]    # 变更描述列表
    dialog_active: bool          # 是否继续对话循环

    # 全局控制
    current_phase: Literal[
        "init", "memory", "attractions", "weather",
        "hotels", "route", "plan", "reflect", "save_version",
        "dialog", "final", "parallel_fetch"
    ]
    reflection_feedback: str
    iteration_count: int
    max_iterations: int
    messages: List[Dict[str, Any]]
    errors: List[str]
