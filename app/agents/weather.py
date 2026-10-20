"""天气节点。"""

from typing import Any, Dict

class WeatherMixin:
    """天气查询。"""

    def fetch_weather(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """查询天气"""
        try:
            city = state.get("city", "")
            weather_info = self.amap_service.get_weather(city)
            
            if weather_info:
                weather_text = f"""天气信息：
城市: {weather_info.city}
天气: {weather_info.weather}
温度: {weather_info.temperature or '未知'}
风向: {weather_info.wind_direction or '未知'}
风力: {weather_info.wind_power or '未知'}
湿度: {weather_info.humidity or '未知'}"""
            else:
                weather_text = f"天气信息：城市 {city} 查询失败"
            
            return {"weather_response": weather_text}
        except Exception as e:
            return {"weather_response": f"天气查询失败: {str(e)}"}
    
    # ==================== 酒店子图节点 ====================
    
    def build_weather_query(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """构建天气查询"""
        return {
            "current_phase": "weather"
        }
    
