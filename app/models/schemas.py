"""
统一数据模型定义
整合schemas.py和agents/models.py的模型，并添加services层需要的模型
"""

from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, field_validator


# ============ 基础模型 ============

class Location(BaseModel):
    """地理位置"""
    longitude: float = Field(..., description="经度")
    latitude: float = Field(..., description="纬度")
    
    @classmethod
    def from_string(cls, location_str: str) -> Optional["Location"]:
        """从字符串创建Location，格式如 '116.397128,39.916527'"""
        if not location_str:
            return None
        try:
            parts = location_str.split(",")
            if len(parts) == 2:
                return cls(longitude=float(parts[0]), latitude=float(parts[1]))
        except (ValueError, TypeError):
            pass
        return None


# ============ 景点与POI相关模型 ============

class POIInfo(BaseModel):
    """POI信息（来自高德地图API）"""
    id: str = Field(..., description="POI ID")
    name: str = Field(..., description="名称")
    type: Optional[str] = Field(default=None, description="类型")
    address: Optional[str] = Field(default=None, description="地址")
    location: Optional[Location] = Field(default=None, description="经纬度坐标")
    tel: Optional[str] = Field(default=None, description="电话")
    city: Optional[str] = Field(default=None, description="城市")
    typecode: Optional[str] = Field(default=None, description="类型编码")
    business_area: Optional[str] = Field(default=None, description="商圈")
    alias: Optional[str] = Field(default=None, description="别名")
    rating: Optional[str] = Field(default=None, description="评分")
    cost: Optional[str] = Field(default=None, description="消费等级")
    tag: Optional[str] = Field(default=None, description="标签")
    photos: Optional[str] = Field(default=None, description="图片")
    
    @classmethod
    def from_amap_response(cls, poi_data: Dict[str, Any]) -> "POIInfo":
        """从高德地图API响应创建POIInfo"""
        location = None
        if poi_data.get("location"):
            location = Location.from_string(poi_data["location"])
        
        # 处理tel字段，可能是字符串或列表
        tel = poi_data.get("tel")
        if isinstance(tel, list):
            tel = tel[0] if tel else None
        elif not isinstance(tel, str):
            tel = None
        
        # 处理alias字段，可能是字符串或列表
        alias = poi_data.get("alias")
        if isinstance(alias, list):
            alias = alias[0] if alias else None
        elif not isinstance(alias, str):
            alias = None
        
        # 处理business_area字段，可能是字符串或列表
        business_area = poi_data.get("business_area")
        if isinstance(business_area, list):
            business_area = business_area[0] if business_area else None
        elif not isinstance(business_area, str):
            business_area = None
        
        # 处理rating字段
        rating = poi_data.get("rating")
        if rating and not isinstance(rating, str):
            rating = str(rating)
        
        # 处理cost字段，可能是字符串或列表
        cost = poi_data.get("cost")
        if isinstance(cost, list):
            cost = cost[0] if cost else None
        elif not isinstance(cost, str):
            cost = None
        
        # 处理tag字段，可能是字符串或列表
        tag = poi_data.get("tag")
        if isinstance(tag, list):
            tag = tag[0] if tag else None
        elif not isinstance(tag, str):
            tag = None
        
        return cls(
            id=poi_data.get("id", ""),
            name=poi_data.get("name", ""),
            type=poi_data.get("type"),
            address=poi_data.get("address"),
            location=location,
            tel=tel,
            city=poi_data.get("cityname") or poi_data.get("city"),
            typecode=poi_data.get("typecode"),
            business_area=business_area,
            alias=alias,
            rating=rating,
            cost=cost,
            tag=tag,
            photos=poi_data.get("photos")
        )


class Attraction(BaseModel):
    """景点信息"""
    name: str = Field(..., description="景点名称")
    address: str = Field(..., description="地址")
    location: Location = Field(..., description="经纬度坐标")
    visit_duration: int = Field(default=120, description="建议游览时间(分钟)")
    description: Optional[str] = Field(default="", description="景点描述")
    category: Optional[str] = Field(default="景点", description="景点类别")
    rating: Optional[float] = Field(default=None, description="评分")
    photos: Optional[List[str]] = Field(default_factory=list, description="景点图片URL列表")
    poi_id: Optional[str] = Field(default="", description="POI ID")
    image_url: Optional[str] = Field(default=None, description="图片URL")
    ticket_price: int = Field(default=0, description="门票价格(元)")
    
    @classmethod
    def from_poi(cls, poi: POIInfo, category: str = "景点", visit_duration: int = 120) -> "Attraction":
        """从POIInfo创建Attraction"""
        return cls(
            name=poi.name,
            address=poi.address or "",
            location=poi.location or Location(longitude=0, latitude=0),
            visit_duration=visit_duration,
            description=f"{poi.type or ''} {poi.business_area or ''}".strip(),
            category=category,
            poi_id=poi.id
        )


# ============ 餐饮相关模型 ============

class Meal(BaseModel):
    """餐饮信息"""
    type: str = Field(..., description="餐饮类型: breakfast/lunch/dinner/snack")
    name: str = Field(..., description="餐饮名称")
    address: Optional[str] = Field(default=None, description="地址")
    location: Optional[Location] = Field(default=None, description="经纬度坐标")
    description: Optional[str] = Field(default=None, description="描述")
    estimated_cost: int = Field(default=0, description="预估费用(元)")
    
    @classmethod
    def from_poi(cls, poi: POIInfo, meal_type: str = "lunch") -> "Meal":
        """从POIInfo创建Meal"""
        return cls(
            type=meal_type,
            name=poi.name,
            address=poi.address,
            location=poi.location,
            description=poi.type
        )


# ============ 酒店相关模型 ============

class Hotel(BaseModel):
    """酒店信息"""
    name: str = Field(..., description="酒店名称")
    address: str = Field(default="", description="酒店地址")
    location: Optional[Location] = Field(default=None, description="酒店位置")
    price_range: str = Field(default="", description="价格范围")
    rating: str = Field(default="", description="评分")
    distance: str = Field(default="", description="距离景点距离")
    type: str = Field(default="", description="酒店类型")
    estimated_cost: int = Field(default=0, description="预估费用(元/晚)")
    
    @classmethod
    def from_poi(cls, poi: POIInfo, hotel_type: str = "经济型") -> "Hotel":
        """从POIInfo创建Hotel"""
        return cls(
            name=poi.name,
            address=poi.address or "",
            location=poi.location,
            type=hotel_type,
            description=poi.type
        )


# ============ 天气相关模型 ============

class WeatherInfo(BaseModel):
    """天气信息"""
    date: str = Field(..., description="日期 YYYY-MM-DD")
    day_weather: str = Field(default="", description="白天天气")
    night_weather: str = Field(default="", description="夜间天气")
    day_temp: Union[int, str] = Field(default=0, description="白天温度")
    night_temp: Union[int, str] = Field(default=0, description="夜间温度")
    wind_direction: str = Field(default="", description="风向")
    wind_power: str = Field(default="", description="风力")
    
    @field_validator('day_temp', 'night_temp', mode='before')
    @classmethod
    def parse_temperature(cls, v):
        """解析温度,移除°C等单位"""
        if isinstance(v, str):
            v = v.replace('°C', '').replace('℃', '').replace('°', '').strip()
            try:
                return int(v)
            except ValueError:
                return 0
        return v


class AmapWeatherInfo(BaseModel):
    """高德地图天气信息（原始API响应）"""
    city: str = Field(..., description="城市名称")
    weather: str = Field(default="", description="天气状况")
    temperature: Optional[str] = Field(default=None, description="温度")
    wind_direction: Optional[str] = Field(default=None, description="风向")
    wind_power: Optional[str] = Field(default=None, description="风力")
    humidity: Optional[str] = Field(default=None, description="湿度")
    report_time: Optional[str] = Field(default=None, description="更新时间")
    
    def to_weather_info(self, date: str) -> WeatherInfo:
        """转换为WeatherInfo"""
        temp = self.temperature or "0"
        try:
            day_temp = int(temp.replace('°C', '').replace('℃', '').replace('°', '').strip())
        except ValueError:
            day_temp = 0
        
        return WeatherInfo(
            date=date,
            day_weather=self.weather,
            night_weather=self.weather,
            day_temp=day_temp,
            night_temp=day_temp - 5,
            wind_direction=self.wind_direction or "",
            wind_power=self.wind_power or ""
        )


# ============ 路线相关模型 ============

class RouteInfo(BaseModel):
    """路线信息"""
    distance: float = Field(..., description="距离(米)")
    duration: int = Field(..., description="时间(秒)")
    route_type: str = Field(default="", description="路线类型")
    description: str = Field(default="", description="路线描述")


class AmapRouteInfo(BaseModel):
    """高德地图路线信息（原始API响应）"""
    origin: str = Field(..., description="起点")
    destination: str = Field(..., description="终点")
    distance: Optional[str] = Field(default=None, description="距离(米)")
    duration: Optional[str] = Field(default=None, description="时长(秒)")
    paths: Optional[List[Dict[str, Any]]] = Field(default=None, description="路径列表")
    transits: Optional[List[Dict[str, Any]]] = Field(default=None, description="公交路线列表")
    
    def to_route_info(self, route_type: str = "driving") -> RouteInfo:
        """转换为RouteInfo"""
        try:
            distance = float(self.distance) if self.distance else 0
        except ValueError:
            distance = 0
        
        try:
            duration = int(self.duration) if self.duration else 0
        except ValueError:
            duration = 0
        
        return RouteInfo(
            distance=distance,
            duration=duration,
            route_type=route_type,
            description=f"从{self.origin}到{self.destination}"
        )


class RouteSegment(BaseModel):
    """路线段信息"""
    start_point: str = Field(..., description="起点名称")
    end_point: str = Field(..., description="终点名称")
    start_location: Optional[Location] = Field(default=None, description="起点坐标")
    end_location: Optional[Location] = Field(default=None, description="终点坐标")
    transport_mode: str = Field(default="步行", description="交通方式：步行/地铁/公交/打车/驾车")
    distance: float = Field(default=0, description="距离(米)")
    duration: int = Field(default=0, description="时长(分钟)")
    description: str = Field(default="", description="路线描述")


class TimeSlot(BaseModel):
    """时间安排"""
    start_time: str = Field(..., description="开始时间，格式如 09:00")
    end_time: str = Field(..., description="结束时间，格式如 11:00")
    activity_type: str = Field(..., description="活动类型：景点游览/餐饮/交通/休息")
    activity_name: str = Field(..., description="活动名称")
    description: str = Field(default="", description="活动描述")
    location: Optional[Location] = Field(default=None, description="活动地点坐标")


# ============ 预算相关模型 ============

class Budget(BaseModel):
    """预算信息"""
    total_attractions: int = Field(default=0, description="景点门票总费用")
    total_hotels: int = Field(default=0, description="酒店总费用")
    total_meals: int = Field(default=0, description="餐饮总费用")
    total_transportation: int = Field(default=0, description="交通总费用")
    total: int = Field(default=0, description="总费用")


# ============ 行程相关模型 ============

class DayPlan(BaseModel):
    """单日行程"""
    date: str = Field(..., description="日期 YYYY-MM-DD")
    day_index: int = Field(..., description="第几天(从0开始)")
    description: str = Field(..., description="当日行程描述")
    transportation: str = Field(default="", description="交通方式")
    accommodation: str = Field(default="", description="住宿")
    hotel: Optional[Hotel] = Field(default=None, description="推荐酒店")
    attractions: List[Attraction] = Field(default=[], description="景点列表")
    meals: List[Meal] = Field(default=[], description="餐饮列表")
    time_slots: List[TimeSlot] = Field(default=[], description="详细时间安排")
    route_segments: List[RouteSegment] = Field(default=[], description="路线段列表")


class TripPlan(BaseModel):
    """旅行计划"""
    city: str = Field(..., description="目的地城市")
    start_date: str = Field(..., description="开始日期")
    end_date: str = Field(..., description="结束日期")
    days: List[DayPlan] = Field(default=[], description="每日行程")
    weather_info: List[WeatherInfo] = Field(default=[], description="天气信息")
    overall_suggestions: str = Field(default="", description="总体建议")
    budget: Optional[Budget] = Field(default=None, description="预算信息")

    def normalize_day_indices(self) -> "TripPlan":
        """将 day_index 统一为从 0 开始，与 days 数组顺序一致"""
        for i, day in enumerate(self.days):
            day.day_index = i
        return self


def normalize_trip_plan_day_indices(plan: Optional[TripPlan]) -> Optional[TripPlan]:
    """规范化 TripPlan 的 day_index（LLM 常输出 1-based，需转为 0-based）"""
    if plan is None:
        return None
    return plan.normalize_day_indices()


# ============ 请求相关模型 ============

class TripRequest(BaseModel):
    """旅行规划请求"""
    city: str = Field(..., description="目的地城市", example="北京")
    start_date: str = Field(..., description="开始日期 YYYY-MM-DD", example="2025-06-01")
    end_date: str = Field(..., description="结束日期 YYYY-MM-DD", example="2025-06-03")
    travel_days: int = Field(..., description="旅行天数", ge=1, le=30, example=3)
    transportation: str = Field(default="公共交通", description="交通方式")
    accommodation: str = Field(default="经济型酒店", description="住宿偏好")
    preferences: List[str] = Field(default_factory=list, description="旅行偏好标签")
    free_text_input: Optional[str] = Field(default="", description="额外要求")
    user_id: Optional[str] = Field(default="default_user", description="用户ID，用于旅行记忆")
    
    class Config:
        json_schema_extra = {
            "example": {
                "city": "北京",
                "start_date": "2025-06-01",
                "end_date": "2025-06-03",
                "travel_days": 3,
                "transportation": "公共交通",
                "accommodation": "经济型酒店",
                "preferences": ["历史文化", "美食"],
                "free_text_input": "希望多安排一些博物馆",
                "user_id": "user_123"
            }
        }


class POISearchRequest(BaseModel):
    """POI搜索请求"""
    keywords: str = Field(..., description="搜索关键词", example="故宫")
    city: str = Field(..., description="城市", example="北京")
    citylimit: bool = Field(default=True, description="是否限制在城市范围内")


class RouteRequest(BaseModel):
    """路线规划请求"""
    origin_address: str = Field(..., description="起点地址")
    destination_address: str = Field(..., description="终点地址")
    origin_city: Optional[str] = Field(default=None, description="起点城市")
    destination_city: Optional[str] = Field(default=None, description="终点城市")
    route_type: str = Field(default="walking", description="路线类型: walking/driving/transit")


# ============ 响应相关模型 ============

class TripPlanResponse(BaseModel):
    """旅行计划响应"""
    success: bool = Field(..., description="是否成功")
    message: str = Field(default="", description="消息")
    data: Optional[TripPlan] = Field(default=None, description="旅行计划数据")


class DialogSessionRequest(BaseModel):
    """创建对话会话请求（计划生成后进入修改对话）"""
    user_id: str = Field(..., description="用户ID")
    trip_plan: TripPlan = Field(..., description="已生成的旅行计划")
    transportation: str = Field(default="公共交通", description="交通方式")
    accommodation: str = Field(default="经济型酒店", description="住宿偏好")
    travel_days: Optional[int] = Field(default=None, description="旅行天数")


class DialogSessionResponse(BaseModel):
    """创建对话会话响应"""
    success: bool = Field(..., description="是否成功")
    message: str = Field(default="", description="消息")
    session_id: Optional[str] = Field(default=None, description="对话会话ID")
    welcome_message: Optional[str] = Field(default=None, description="欢迎/引导消息")


class DialogChatRequest(BaseModel):
    """对话修改请求"""
    session_id: str = Field(..., description="对话会话ID")
    message: str = Field(..., description="用户修改指令")


class DialogChatResponse(BaseModel):
    """对话修改响应"""
    success: bool = Field(..., description="是否成功")
    message: str = Field(default="", description="消息")
    reply: Optional[str] = Field(default=None, description="助手回复")
    trip_plan: Optional[TripPlan] = Field(default=None, description="更新后的旅行计划（用于前端展示）")
    changes: List[str] = Field(default_factory=list, description="变更描述")
    new_version_id: Optional[str] = Field(default=None, description="新版本ID")


class POISearchResponse(BaseModel):
    """POI搜索响应"""
    success: bool = Field(..., description="是否成功")
    message: str = Field(default="", description="消息")
    data: List[POIInfo] = Field(default=[], description="POI列表")


class WeatherResponse(BaseModel):
    """天气查询响应"""
    success: bool = Field(..., description="是否成功")
    message: str = Field(default="", description="消息")
    data: List[WeatherInfo] = Field(default=[], description="天气信息")


class RouteResponse(BaseModel):
    """路线规划响应"""
    success: bool = Field(..., description="是否成功")
    message: str = Field(default="", description="消息")
    data: Optional[RouteInfo] = Field(default=None, description="路线信息")


class ErrorResponse(BaseModel):
    """错误响应"""
    success: bool = Field(default=False, description="是否成功")
    message: str = Field(..., description="错误消息")
    error_code: Optional[str] = Field(default=None, description="错误代码")
