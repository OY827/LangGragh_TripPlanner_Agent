from typing import List, Dict, Optional, Any
import os
from dotenv import load_dotenv
import requests

# 导入统一的数据模型
from app.models import POIInfo, AmapWeatherInfo, AmapRouteInfo, Location

# 导入MCP工具
from app.tools import MCPToolManager, settings

# 加载环境变量
load_dotenv()


def get_amap_api_key() -> str:
    """获取高德地图API密钥"""
    return os.getenv("AMAP_MAPS_API_KEY", "")


class AmapService:
    """高德地图服务封装类"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AmapService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        """初始化服务"""
        self.api_key = get_amap_api_key()
        self._use_real_api = bool(self.api_key)
        self._use_mcp = os.getenv("USE_MCP", "false").lower() == "true"
        self._mcp_tool = None
        
        # 仅当启用 MCP 时尝试导入
        if self._use_mcp:
            try:
                from app.tools import MCPToolManager, settings
                manager = MCPToolManager()
                if settings.amap_api_key:
                    manager.register_tool(
                        name="amap",
                        description="高德地图服务",
                        server_command=["python", "-m", "amap_mcp_server"],
                        env={"AMAP_MAPS_API_KEY": settings.amap_api_key},
                        auto_expand=True
                    )
                    self._mcp_tool = manager.get_tool("amap")
            except Exception as e:
                print(f"MCP工具初始化失败，将使用直接API调用: {str(e)}")
                self._use_mcp = False
        
        self._initialized = True
        print(f"高德地图服务初始化完成 - 使用真实API: {self._use_real_api}, 使用MCP: {self._use_mcp}")
    
    def _call_mcp_tool(self, tool_name: str, **kwargs) -> Any:
        """调用MCP工具"""
        if self._mcp_tool:
            try:
                return self._mcp_tool.invoke(name=tool_name, arguments=kwargs)
            except Exception as e:
                # 只在第一次失败时打印警告
                if not getattr(self, '_mcp_warned', False):
                    print(f"⚠️ MCP工具调用失败，将使用直接API调用")
                    self._mcp_warned = True
        return None
    
    def search_poi(self, keywords: str, city: str, citylimit: bool = True) -> List[POIInfo]:
        """
        搜索POI（兴趣点）
        
        Args:
            keywords: 搜索关键词
            city: 城市名称
            citylimit: 是否限制在城市范围内
        
        Returns:
            POI信息列表
        """
        if self._use_mcp:
            result = self._call_mcp_tool(
                "maps_text_search",
                keywords=keywords,
                city=city,
                citylimit=str(citylimit).lower()
            )
            if result and "error" not in result:
                pois = result.get("pois", [])
                return [POIInfo.from_amap_response(poi) for poi in pois]
        
        # 降级到直接API调用或模拟数据
        if self._use_real_api:
            return self._search_poi_real(keywords, city, citylimit)
        else:
            return self._search_poi_mock(keywords, city)
    
    def _search_poi_real(self, keywords: str, city: str, citylimit: bool) -> List[POIInfo]:
        """真实API搜索POI"""
        try:
            url = "https://restapi.amap.com/v3/place/text"
            params = {
                "key": self.api_key,
                "keywords": keywords,
                "city": city,
                "citylimit": "true" if citylimit else "false"
            }
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if data.get("status") != "1":
                print(f"POI搜索失败: {data.get('info')}")
                return self._search_poi_mock(keywords, city)
            
            pois = []
            for poi in data.get("pois", []):
                pois.append(POIInfo.from_amap_response(poi))
            return pois
        except Exception as e:
            print(f"POI搜索API调用失败: {str(e)}")
            return self._search_poi_mock(keywords, city)
    
    def _search_poi_mock(self, keywords: str, city: str) -> List[POIInfo]:
        """模拟POI搜索数据"""
        mock_data = {
            "景点": [
                {"id": "1", "name": f"{city}故宫博物院", "address": "北京市东城区景山前街4号", "typecode": "110100", "city": city},
                {"id": "2", "name": f"{city}颐和园", "address": "北京市海淀区新建宫门路19号", "typecode": "110100", "city": city},
                {"id": "3", "name": f"{city}天坛公园", "address": "北京市东城区天坛东里甲1号", "typecode": "110100", "city": city}
            ],
            "餐厅": [
                {"id": "1", "name": "全聚德", "address": "北京市前门西大街14号", "typecode": "050100", "city": city},
                {"id": "2", "name": "四季民福", "address": "北京市东城区王府井大街150号", "typecode": "050100", "city": city}
            ],
            "酒店": [
                {"id": "1", "name": "北京国贸大酒店", "address": "北京市朝阳区建国门外大街1号", "typecode": "080100", "city": city},
                {"id": "2", "name": "北京华尔道夫酒店", "address": "北京市东城区金鱼胡同5-15号", "typecode": "080100", "city": city}
            ]
        }
        
        # 根据关键词返回模拟数据
        for key, items in mock_data.items():
            if key in keywords or keywords in key:
                return [POIInfo.from_amap_response(item) for item in items]
        
        # 默认返回景点数据
        return [POIInfo.from_amap_response(item) for item in mock_data["景点"]]
    
    def get_poi_detail(self, poi_id: str) -> Optional[POIInfo]:
        """
        获取POI详细信息
        
        Args:
            poi_id: POI ID
        
        Returns:
            POI详细信息
        """
        if self._use_mcp:
            result = self._call_mcp_tool("maps_search_detail", id=poi_id)
            if result and "error" not in result:
                return POIInfo.from_amap_response(result)
        
        if self._use_real_api:
            return self._get_poi_detail_real(poi_id)
        else:
            return self._get_poi_detail_mock(poi_id)
    
    def _get_poi_detail_real(self, poi_id: str) -> Optional[POIInfo]:
        """真实API获取POI详情"""
        try:
            url = "https://restapi.amap.com/v3/place/detail"
            params = {
                "key": self.api_key,
                "id": poi_id
            }
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if data.get("status") != "1":
                print(f"POI详情查询失败: {data.get('info')}")
                return None
            
            pois = data.get("pois", [])
            if pois:
                return POIInfo.from_amap_response(pois[0])
            
            return None
        except Exception as e:
            print(f"POI详情API调用失败: {str(e)}")
            return None
    
    def _get_poi_detail_mock(self, poi_id: str) -> Optional[POIInfo]:
        """模拟POI详情数据"""
        mock_data = {
            "1": {"id": "1", "name": "故宫博物院", "address": "北京市东城区景山前街4号", "type": "风景名胜", "location": "116.397026,39.918058", "city": "北京"},
            "2": {"id": "2", "name": "颐和园", "address": "北京市海淀区新建宫门路19号", "type": "风景名胜", "location": "116.2730,39.9995", "city": "北京"},
            "3": {"id": "3", "name": "天坛公园", "address": "北京市东城区天坛东里甲1号", "type": "风景名胜", "location": "116.4108,39.8837", "city": "北京"}
        }
        
        if poi_id in mock_data:
            return POIInfo.from_amap_response(mock_data[poi_id])
        
        return None
    
    def get_weather(self, city: str) -> Optional[AmapWeatherInfo]:
        """
        获取天气信息
        
        Args:
            city: 城市名称或城市编码
        
        Returns:
            天气信息对象
        """
        if self._use_mcp:
            result = self._call_mcp_tool("maps_weather", city=city)
            if result and "error" not in result:
                # MCP 返回格式: {"city": "...", "forecasts": [...]}
                forecasts = result.get("forecasts", [])
                if forecasts:
                    # 取今天的预报数据
                    today = forecasts[0]
                    return AmapWeatherInfo(
                        city=result.get("city", city),
                        weather=today.get("dayweather", ""),
                        temperature=f"{today.get('daytemp', '')}°C",
                        wind_direction=today.get("daywind"),
                        wind_power=today.get("daypower"),
                        humidity=None,  # 预报数据没有湿度
                        report_time=today.get("date")
                    )
        
        if self._use_real_api:
            return self._get_weather_real(city)
        else:
            return self._get_weather_mock(city)
    
    def _get_weather_real(self, city: str) -> Optional[AmapWeatherInfo]:
        """真实API获取天气"""
        try:
            url = "https://restapi.amap.com/v3/weather/weatherInfo"
            params = {
                "key": self.api_key,
                "city": city,
                "extensions": "base"
            }
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if data.get("status") != "1":
                print(f"天气查询失败: {data.get('info')}")
                return self._get_weather_mock(city)
            
            lives = data.get("lives", [])
            if lives:
                live = lives[0]
                return AmapWeatherInfo(
                    city=live.get("city", city),
                    weather=live.get("weather", ""),
                    temperature=live.get("temperature", ""),
                    wind_direction=live.get("winddirection"),
                    wind_power=live.get("windpower"),
                    humidity=live.get("humidity"),
                    report_time=live.get("reporttime")
                )
            return self._get_weather_mock(city)
        except Exception as e:
            print(f"天气查询API调用失败: {str(e)}")
            return self._get_weather_mock(city)
    
    def _get_weather_mock(self, city: str) -> AmapWeatherInfo:
        """模拟天气数据"""
        return AmapWeatherInfo(
            city=city,
            weather="晴",
            temperature="25°C",
            wind_direction="南风",
            wind_power="2级",
            humidity="45%",
            report_time="2024-01-15 10:00:00"
        )
    
    def geocode(self, address: str, city: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        地理编码（地址转坐标）
        
        Args:
            address: 地址描述
            city: 城市名称（可选）
        
        Returns:
            包含坐标信息的字典
        """
        if self._use_mcp:
            result = self._call_mcp_tool("maps_geo", address=address, city=city or "")
            if result and "error" not in result:
                # MCP 返回格式: {"return": [{"location": "...", ...}]}
                if "return" in result and len(result["return"]) > 0:
                    return result["return"][0]
                return result
        
        if self._use_real_api:
            return self._geocode_real(address, city)
        else:
            return self._geocode_mock(address, city)
    
    def _geocode_real(self, address: str, city: Optional[str]) -> Optional[Dict[str, Any]]:
        """真实API地理编码"""
        try:
            url = "https://restapi.amap.com/v3/geocode/geo"
            params = {
                "key": self.api_key,
                "address": address
            }
            if city:
                params["city"] = city
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if data.get("status") != "1":
                print(f"地理编码失败: {data.get('info')}")
                return self._geocode_mock(address, city)
            
            geocodes = data.get("geocodes", [])
            if geocodes:
                return geocodes[0]
            return self._geocode_mock(address, city)
        except Exception as e:
            print(f"地理编码API调用失败: {str(e)}")
            return self._geocode_mock(address, city)
    
    def _geocode_mock(self, address: str, city: Optional[str]) -> Dict[str, Any]:
        """模拟地理编码数据"""
        return {
            "formatted_address": f"{city or '北京'}{address}",
            "province": city or "北京",
            "city": city or "北京",
            "district": "朝阳区",
            "location": "116.407413,39.904211",
            "level": "门牌号"
        }
    
    def regeocode(self, location: str) -> Optional[Dict[str, Any]]:
        """
        逆地理编码（坐标转地址）
        
        Args:
            location: 坐标，格式为"经度,纬度"
        
        Returns:
            包含地址信息的字典
        """
        if self._use_mcp:
            result = self._call_mcp_tool("maps_regeocode", location=location)
            if result and "error" not in result:
                # MCP 返回格式: {"province": "...", "city": "...", "district": "..."}
                return result
        
        if self._use_real_api:
            return self._regeocode_real(location)
        else:
            return self._regeocode_mock(location)
    
    def _regeocode_real(self, location: str) -> Optional[Dict[str, Any]]:
        """真实API逆地理编码"""
        try:
            url = "https://restapi.amap.com/v3/geocode/regeo"
            params = {
                "key": self.api_key,
                "location": location
            }
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if data.get("status") != "1":
                print(f"逆地理编码失败: {data.get('info')}")
                return self._regeocode_mock(location)
            
            return data.get("regeocode", {}).get("addressComponent", {})
        except Exception as e:
            print(f"逆地理编码API调用失败: {str(e)}")
            return self._regeocode_mock(location)
    
    def _regeocode_mock(self, location: str) -> Dict[str, Any]:
        """模拟逆地理编码数据"""
        return {
            "province": "北京市",
            "city": "北京市",
            "district": "朝阳区",
            "streetNumber": "阜通东大街6号"
        }
    
    def route_direction(self, origin: str, destination: str, mode: str = "driving", city: str = None) -> Optional[AmapRouteInfo]:
        """
        路线规划

        Args:
            origin: 起点（地址或坐标）
            destination: 终点（地址或坐标）
            mode: 出行方式（driving/walking/transit）
            city: 城市名称（用于公交/地铁路线规划）

        Returns:
            路线信息对象
        """
        if self._use_mcp:
            tool_name = {
                "driving": "maps_direction_driving_by_address",
                "walking": "maps_direction_walking_by_address",
                "transit": "maps_direction_transit_integrated_by_address"
            }.get(mode, "maps_direction_driving_by_address")

            mcp_kwargs = {
                "origin_address": origin,
                "destination_address": destination,
            }
            # 公交/地铁模式需要城市参数
            if mode == "transit" and city:
                mcp_kwargs["origin_city"] = city
                mcp_kwargs["destination_city"] = city
            elif mode == "transit":
                mcp_kwargs["origin_city"] = "全国"
                mcp_kwargs["destination_city"] = "全国"

            result = self._call_mcp_tool(tool_name, **mcp_kwargs)
            if result and "error" not in result:
                route_data = result.get("route", {})
                return AmapRouteInfo(
                    origin=route_data.get("origin", origin),
                    destination=route_data.get("destination", destination),
                    distance=route_data.get("distance"),
                    duration=route_data.get("duration"),
                    paths=route_data.get("paths"),
                    transits=route_data.get("transits")
                )

        if self._use_real_api:
            return self._route_direction_real(origin, destination, mode, city)
        else:
            return self._route_direction_mock(origin, destination, mode)
    
    def _route_direction_real(self, origin: str, destination: str, mode: str, city: str = None) -> Optional[AmapRouteInfo]:
        """真实API路线规划"""
        try:
            # 检查是否为坐标格式（经纬度用逗号分隔）
            def is_coordinate(s: str) -> bool:
                parts = s.split(",")
                if len(parts) != 2:
                    return False
                try:
                    float(parts[0])
                    float(parts[1])
                    return True
                except ValueError:
                    return False
            
            # 如果输入是地址，先进行地理编码转换为坐标
            origin_coord = origin
            dest_coord = destination
            
            if not is_coordinate(origin):
                geo_result = self._geocode_real(origin, None)
                if geo_result and geo_result.get("location"):
                    origin_coord = geo_result["location"]
                    print(f"地址 '{origin}' 转换为坐标: {origin_coord}")
                else:
                    print(f"无法将地址 '{origin}' 转换为坐标，使用模拟数据")
                    return self._route_direction_mock(origin, destination, mode)
            
            if not is_coordinate(destination):
                geo_result = self._geocode_real(destination, None)
                if geo_result and geo_result.get("location"):
                    dest_coord = geo_result["location"]
                    print(f"地址 '{destination}' 转换为坐标: {dest_coord}")
                else:
                    print(f"无法将地址 '{destination}' 转换为坐标，使用模拟数据")
                    return self._route_direction_mock(origin, destination, mode)
            
            url_map = {
                "driving": "https://restapi.amap.com/v3/direction/driving",
                "walking": "https://restapi.amap.com/v3/direction/walking",
                "transit": "https://restapi.amap.com/v3/direction/transit/integrated"
            }
            
            url = url_map.get(mode, url_map["driving"])
            params = {
                "key": self.api_key,
                "origin": origin_coord,
                "destination": dest_coord
            }
            
            if mode == "transit":
                params["city"] = city or "全国"
                params["cityd"] = city or "全国"
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if data.get("status") != "1":
                print(f"路线规划失败: {data.get('info')}")
                return self._route_direction_mock(origin, destination, mode)
            
            route = data.get("route", {})
            
            # 提取距离和时长（在paths数组中），duration 保持为秒数字符串，便于下游解析
            distance = None
            duration = None
            paths = route.get("paths", [])
            if paths and isinstance(paths, list) and len(paths) > 0:
                first_path = paths[0]
                distance = first_path.get("distance")
                duration = first_path.get("duration")
            
            # transit 模式距离/时长在 transits 中
            transits = route.get("transits")
            if (not distance or not duration) and transits and isinstance(transits, list) and transits:
                first_transit = transits[0]
                distance = distance or first_transit.get("distance")
                duration = duration or first_transit.get("duration")
            
            return AmapRouteInfo(
                origin=origin,
                destination=destination,
                distance=str(distance) if distance is not None else None,
                duration=str(duration) if duration is not None else None,
                paths=paths,
                transits=transits
            )
        except Exception as e:
            print(f"路线规划API调用失败: {str(e)}")
            return self._route_direction_mock(origin, destination, mode)
    
    def _route_direction_mock(self, origin: str, destination: str, mode: str) -> AmapRouteInfo:
        """模拟路线规划数据"""
        mode_info = {
            "driving": {"distance": "15000", "duration": "25分钟"},
            "walking": {"distance": "5000", "duration": "1小时"},
            "transit": {"distance": "8000", "duration": "45分钟"}
        }
        info = mode_info.get(mode, mode_info["driving"])
        
        return AmapRouteInfo(
            origin=origin,
            destination=destination,
            distance=info["distance"],
            duration=info["duration"],
            paths=None,
            transits=None
        )


# 全局服务实例
_amap_service = None


def get_amap_service() -> AmapService:
    """获取高德地图服务单例实例"""
    global _amap_service
    if _amap_service is None:
        _amap_service = AmapService()
    return _amap_service


# 测试代码
if __name__ == "__main__":
    print("=== 测试高德地图服务 ===")
    
    # 获取服务实例
    service = get_amap_service()
    
    print(f"\n服务配置: 使用真实API={service._use_real_api}, 使用MCP={service._use_mcp}")
    
    # 测试POI搜索
    print("\n1. 测试POI搜索:")
    pois = service.search_poi("景点", "深圳")
    for poi in pois:
        print(f"  - {poi.name}: {poi.address}")
    
    # 测试天气查询
    print("\n2. 测试天气查询:")
    weather = service.get_weather("深圳")
    if weather:
        print(f"  深圳天气: {weather.weather}, 温度: {weather.temperature}")
    
    # 测试地理编码
    print("\n3. 测试地理编码:")
    geo = service.geocode("深圳坪山区比亚迪二期高管宿舍")
    if geo:
        print(f"  坐标: {geo.get('location')}")
    
    # 测试逆地理编码
    print("\n4. 测试逆地理编码:")
    regeo = service.regeocode(geo.get('location'))
    if regeo:
        print(f"  地址: {regeo.get('province')}{regeo.get('city')}{regeo.get('district')}")
    
    # 测试路线规划
    print("\n5. 测试路线规划:")
    route = service.route_direction("深圳锦龙地铁站", "深圳坪山区比亚迪二期高管宿舍", "driving")
    if route:
        print(f"  距离: {route.distance}米, 时长: {route.duration}")
    
    print("\n=== 测试完成 ===")
