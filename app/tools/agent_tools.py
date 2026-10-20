"""
Agent工具定义
包含用于Agent执行的各类工具封装
"""

from langchain_core.tools import BaseTool
from typing import Optional, Dict, Any


class CalculatorTool(BaseTool):
    name: str = "calculator"
    description: str = "用于执行数学计算，输入应为数学表达式字符串"

    def _run(self, expression: str) -> str:
        try:
            result = eval(expression, {"__builtins__": {}}, {})
            return str(result)
        except Exception as e:
            return f"错误: {str(e)}"

    async def _arun(self, expression: str) -> str:
        return self._run(expression)


class CodeInterpreterTool(BaseTool):
    """
    代码解释器工具
    允许Agent动态生成并执行Python代码，用于处理复杂的计算、数据转换、分析等任务。
    
    安全限制：
    - 仅允许使用安全模块（math, json, itertools, datetime, re, statistics, collections）
    - 禁用文件操作、系统调用、子进程等危险功能
    - 设置执行超时（默认10秒）
    
    使用场景：
    - 复杂预算计算（多维度费用拆解）
    - 数据格式转换
    - 路线优化问题
    - 统计分析
    """
    name: str = "code_interpreter"
    description: str = (
        "执行Python代码的工具。当需要进行复杂计算、数据转换、统计分析时使用此工具。"
        "输入必须是完整的Python代码字符串。支持的模块：math, json, itertools, datetime, re, statistics, collections。"
        "代码中应使用print()输出结果。"
        "示例场景：\n"
        "1. 预算计算：计算景点门票、酒店、餐饮、交通等各项费用\n"
        "2. 数据转换：将API返回的数据转换为所需格式\n"
        "3. 统计分析：计算平均值、汇总、分布等\n"
        "4. 优化问题：简单的排序、筛选、组合计算\n"
    )
    
    def _get_safe_modules(self) -> dict:
        """获取允许使用的安全模块"""
        return {
            'math': __import__('math'),
            'json': __import__('json'),
            'itertools': __import__('itertools'),
            'datetime': __import__('datetime'),
            're': __import__('re'),
            'statistics': __import__('statistics'),
            'collections': __import__('collections'),
        }
    
    def _safe_import(self, name: str, *args, **kwargs):
        """安全的import函数，只允许导入白名单中的模块"""
        safe_modules = self._get_safe_modules()
        if name in safe_modules:
            return safe_modules[name]
        # 尝试导入子模块（如 datetime.datetime）
        base_name = name.split('.')[0]
        if base_name in safe_modules:
            return __import__(name, *args, **kwargs)
        raise ImportError(f"安全限制：不允许导入 '{name}' 模块。允许的模块: {list(safe_modules.keys())}")
    
    def _get_safe_builtins(self) -> dict:
        """获取允许使用的内置函数（有限白名单）"""
        return {
            '__import__': self._safe_import,
            'abs': abs,
            'all': all,
            'any': any,
            'bin': bin,
            'bool': bool,
            'chr': chr,
            'dict': dict,
            'divmod': divmod,
            'enumerate': enumerate,
            'filter': filter,
            'float': float,
            'format': format,
            'frozenset': frozenset,
            'hash': hash,
            'hex': hex,
            'int': int,
            'isinstance': isinstance,
            'issubclass': issubclass,
            'iter': iter,
            'len': len,
            'list': list,
            'map': map,
            'max': max,
            'min': min,
            'next': next,
            'oct': oct,
            'ord': ord,
            'print': print,
            'range': range,
            'repr': repr,
            'reversed': reversed,
            'round': round,
            'set': set,
            'slice': slice,
            'sorted': sorted,
            'str': str,
            'sum': sum,
            'tuple': tuple,
            'type': type,
            'zip': zip,
            'True': True,
            'False': False,
            'None': None,
        }
    
    def _run(self, code: str) -> str:
        """
        执行Python代码并返回输出结果
        
        Args:
            code: 要执行的Python代码字符串
            
        Returns:
            代码执行的输出结果或错误信息
        """
        import io
        import sys
        import threading
        
        output_capture = io.StringIO()
        execution_result = {'output': '', 'error': None, 'completed': False}
        
        def execute_code():
            try:
                # 重定向stdout
                old_stdout = sys.stdout
                sys.stdout = output_capture
                
                # 构建受限的执行环境
                safe_globals = {
                    '__builtins__': self._get_safe_builtins(),
                }
                safe_globals.update(self._get_safe_modules())
                
                # 执行代码
                exec(code, safe_globals, {})
                
                # 恢复stdout
                sys.stdout = old_stdout
                
                execution_result['output'] = output_capture.getvalue()
                execution_result['completed'] = True
                
            except Exception as e:
                sys.stdout = old_stdout
                execution_result['error'] = str(e)
                execution_result['completed'] = True
        
        # 创建执行线程并设置超时
        thread = threading.Thread(target=execute_code)
        thread.daemon = True
        thread.start()
        thread.join(timeout=10)  # 10秒超时
        
        if not execution_result['completed']:
            return "执行超时：代码运行时间超过10秒限制"
        
        if execution_result['error']:
            return f"代码执行错误：{execution_result['error']}"
        
        output = execution_result['output']
        if not output:
            return "代码执行成功，但没有输出。请使用print()输出结果。"
        
        return output.strip()

    async def _arun(self, code: str) -> str:
        return self._run(code)


class SearchTool(BaseTool):
    name: str = "web_search"
    description: str = "用于在网上搜索信息，输入应为搜索查询词"

    def _run(self, query: str) -> str:
        return f"搜索结果 '{query}': [这是模拟搜索结果。在生产环境中，您需要调用真实的搜索API。]"

    async def _arun(self, query: str) -> str:
        return self._run(query)


class DateTimeTool(BaseTool):
    name: str = "datetime_now"
    description: str = "获取当前日期和时间，不需要输入参数"

    def _run(self, input: str = "") -> str:
        from datetime import datetime
        return datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")

    async def _arun(self, input: str = "") -> str:
        return self._run(input)


class WeatherTool(BaseTool):
    name: str = "mock_weather"
    description: str = "获取特定位置的天气信息，输入应为城市名称（模拟，生产请用 amap_weather）"

    def _run(self, location: str) -> str:
        return f"{location}天气: [这是模拟天气结果。在生产环境中，您需要调用真实的天气API。]"

    async def _arun(self, location: str) -> str:
        return self._run(location)


# ============ 高德地图MCP工具 ============

from pydantic import BaseModel, Field

class AttractionSearchInput(BaseModel):
    city: str = Field(..., description="城市名称")
    keywords: str = Field(default="景点", description="景点关键词")

class AmapAttractionTool(BaseTool):
    """
    高德地图景点搜索工具
    使用MCP调用高德地图POI搜索API获取景点信息
    """
    name: str = "amap_attraction_search"
    description: str = "用高德搜索指定城市的景点（名称、地址、坐标）。无评分无票价。"
    args_schema: type = AttractionSearchInput

    def _run(self, city: str, keywords: str = "景点") -> str:
        try:
            from app.services.amap_service import get_amap_service

            if not city:
                return "错误：缺少城市参数"

            pois = get_amap_service().search_poi(keywords=keywords or "景点", city=city)
            if not pois:
                return f"未找到{city}的{keywords}相关景点"

            result_list = []
            for poi in pois[:10]:
                loc = ""
                if getattr(poi, "location", None):
                    loc = f"{poi.location.longitude},{poi.location.latitude}"
                info = f"- {poi.name}\n  地址: {poi.address or ''}\n  类型: {poi.type or ''}"
                if loc:
                    info += f"\n  坐标: {loc}"
                result_list.append(info)
            return "\n\n".join(result_list)

        except Exception as e:
            return f"景点搜索出错：{str(e)}"

    async def _arun(self, city: str, keywords: str = "景点") -> str:
        return self._run(city, keywords)


class WeatherSearchInput(BaseModel):
    city: str = Field(..., description="城市名称")

class AmapWeatherTool(BaseTool):
    """
    高德地图天气查询工具
    使用MCP调用高德地图天气API获取天气信息
    """
    name: str = "amap_weather"
    description: str = "查询指定城市的天气（高德，含多日预报）"
    args_schema: type = WeatherSearchInput

    def _run(self, city: str) -> str:
        try:
            from app.services.amap_mcp_service import get_amap_mcp_service, is_amap_mcp_enabled
            from app.services.amap_service import get_amap_service

            if not city:
                return "错误：缺少城市参数"

            if is_amap_mcp_enabled():
                mcp = get_amap_mcp_service()
                if mcp.is_configured():
                    result = mcp.weather(city)
                    forecasts = (result or {}).get("forecasts") or []
                    if forecasts:
                        weather_info = f"{city} 天气预报：\n"
                        week_names = ["一", "二", "三", "四", "五", "六", "日"]
                        for forecast in forecasts[:3]:
                            try:
                                week = week_names[int(forecast.get("week", 1)) - 1]
                            except (TypeError, ValueError, IndexError):
                                week = str(forecast.get("week", ""))
                            weather_info += f"\n{forecast.get('date')} (周{week}):\n"
                            weather_info += f"  白天: {forecast.get('dayweather', '')} {forecast.get('daytemp', '')}°C\n"
                            weather_info += f"  夜间: {forecast.get('nightweather', '')} {forecast.get('nighttemp', '')}°C\n"
                        return weather_info

            weather = get_amap_service().get_weather(city)
            if not weather:
                return "天气查询失败：未获取到预报信息"
            return (
                f"{weather.city or city} 天气：\n"
                f"  天气: {weather.weather}\n"
                f"  温度: {weather.temperature or ''}\n"
                f"  风向: {weather.wind_direction or ''}\n"
                f"  风力: {weather.wind_power or ''}"
            )

        except Exception as e:
            return f"天气查询出错：{str(e)}"

    async def _arun(self, city: str) -> str:
        return self._run(city)


class RoutePlanInput(BaseModel):
    origin: str = Field(..., description="起点地址或坐标，格式如'北京市东城区天安门'或'116.397128,39.916527'")
    destination: str = Field(..., description="终点地址或坐标，格式如'北京市西城区颐和园'或'116.2730,39.9995'")
    mode: str = Field(default="walking", description="交通方式：driving(驾车)/walking(步行)/transit(公共交通)")
    city: str = Field(default="", description="城市名称")

class AmapRouteTool(BaseTool):
    """
    高德地图路线规划工具
    使用MCP调用高德地图路线规划API，获取两点之间的路线信息
    """
    name: str = "amap_route"
    description: str = "规划两点之间的路线，返回距离、时长和交通方式。支持 walking/driving/transit"
    args_schema: type = RoutePlanInput

    def _run(self, origin: str, destination: str, mode: str = "walking", city: str = "") -> str:
        try:
            from app.services.amap_service import get_amap_service

            if not origin or not destination:
                return "错误：缺少起点或终点参数"

            info = get_amap_service().route_direction(
                origin=origin,
                destination=destination,
                mode=mode or "walking",
                city=city or None,
            )
            if not info:
                return f"未找到从{origin}到{destination}的路线"

            if mode == "transit":
                transits = info.transits or []
                if transits:
                    return self._format_transit_result(origin, destination, transits[0])
                return f"未找到从{origin}到{destination}的公共交通路线"

            paths = info.paths or []
            path = paths[0] if paths else {
                "distance": info.distance or "0",
                "duration": info.duration or "0",
            }
            return self._format_driving_walking_result(origin, destination, path, mode)

        except Exception as e:
            return f"路线规划出错：{str(e)}"

    def _format_driving_walking_result(self, origin: str, destination: str, path: dict, mode: str) -> str:
        """格式化驾车/步行路线结果"""
        distance = path.get("distance", "0")
        duration = path.get("duration", "0")
        
        try:
            distance_m = int(distance)
            if distance_m >= 1000:
                distance_str = f"{distance_m / 1000:.1f}公里"
            else:
                distance_str = f"{distance_m}米"
        except:
            distance_str = f"{distance}米"
        
        try:
            duration_s = int(duration)
            hours = duration_s // 3600
            minutes = (duration_s % 3600) // 60
            if hours > 0:
                duration_str = f"{hours}小时{minutes}分钟"
            else:
                duration_str = f"{minutes}分钟"
        except:
            duration_str = duration
        
        mode_name = {
            "driving": "驾车",
            "walking": "步行"
        }.get(mode, mode)
        
        result = f"路线规划结果：\n"
        result += f"  起点：{origin}\n"
        result += f"  终点：{destination}\n"
        result += f"  交通方式：{mode_name}\n"
        result += f"  距离：{distance_str}\n"
        result += f"  预计时长：{duration_str}\n"
        
        return result

    def _format_transit_result(self, origin: str, destination: str, transit: dict) -> str:
        """格式化公共交通路线结果"""
        duration = transit.get("duration", "0")
        walking_distance = transit.get("walking_distance", "0")
        cost = transit.get("cost", "0")
        
        try:
            duration_s = int(duration)
            hours = duration_s // 3600
            minutes = (duration_s % 3600) // 60
            if hours > 0:
                duration_str = f"{hours}小时{minutes}分钟"
            else:
                duration_str = f"{minutes}分钟"
        except:
            duration_str = duration
        
        try:
            walking_m = int(walking_distance)
            if walking_m >= 1000:
                walking_str = f"{walking_m / 1000:.1f}公里"
            else:
                walking_str = f"{walking_m}米"
        except:
            walking_str = f"{walking_distance}米"
        
        result = f"路线规划结果：\n"
        result += f"  起点：{origin}\n"
        result += f"  终点：{destination}\n"
        result += f"  交通方式：公共交通\n"
        result += f"  预计时长：{duration_str}\n"
        result += f"  步行距离：{walking_str}\n"
        result += f"  预计费用：{cost}元\n"
        
        segments = transit.get("segments", [])
        if segments:
            result += "\n  换乘路线：\n"
            for i, segment in enumerate(segments):
                bus_line = segment.get("bus", {}).get("name", "")
                start_stop = segment.get("bus", {}).get("departure_stop", {}).get("name", "")
                end_stop = segment.get("bus", {}).get("arrival_stop", {}).get("name", "")
                stops_count = segment.get("bus", {}).get("stops", 0)
                
                if bus_line:
                    result += f"    {i+1}. [{bus_line}] {start_stop} -> {end_stop}（{stops_count}站）\n"
                else:
                    walk_desc = segment.get("walking", {}).get("instruction", {}).get("desc", "")
                    result += f"    {i+1}. 步行：{walk_desc}\n"
        
        return result

    async def _arun(self, origin: str, destination: str, mode: str = "walking", city: str = "") -> str:
        return self._run(origin, destination, mode, city)


class HotelSearchInput(BaseModel):
    city: str = Field(..., description="城市名称")
    keywords: str = Field(default="酒店", description="酒店类型关键词，如经济型酒店、豪华酒店")
    check_in_date: str = Field(default="", description="入住日期，格式 YYYY-MM-DD，可空")
    stay_nights: int = Field(default=1, description="住宿晚数，默认1")


def _coerce_stay_nights(stay_nights) -> int:
    try:
        nights = int(stay_nights or 1)
    except (TypeError, ValueError):
        nights = 1
    return max(1, nights)


class RollingGoHotelTool(BaseTool):
    """
    RollingGo（道旅）酒店 MCP 工具。
    酒店推荐 Agent 的主工具：实时星级和每晚报价。
    """
    name: str = "hotel_search"
    description: str = (
        "用 RollingGo 酒店 MCP 搜索带价格、星级的酒店（主工具）。"
        "必填 city；建议传入 keywords、check_in_date（YYYY-MM-DD）、stay_nights。"
    )
    args_schema: type = HotelSearchInput

    def _run(
        self,
        city: str,
        keywords: str = "酒店",
        check_in_date: str = "",
        stay_nights: int = 1,
    ) -> str:
        if not city:
            return "错误：缺少城市参数"
        try:
            from app.services.hotel_mcp_service import get_hotel_mcp_service

            service = get_hotel_mcp_service()
            if not service.is_configured():
                return "酒店搜索失败：未配置 HOTEL_API_KEY / HOTEL_MCP_URL"
            return service.search_and_format(
                city=city,
                keywords=keywords or "",
                accommodation=keywords or "",
                check_in_date=check_in_date or None,
                stay_nights=_coerce_stay_nights(stay_nights),
            )
        except Exception as e:
            return f"RollingGo 酒店搜索出错：{str(e)}"

    async def _arun(
        self,
        city: str,
        keywords: str = "酒店",
        check_in_date: str = "",
        stay_nights: int = 1,
    ) -> str:
        return self._run(city, keywords, check_in_date, stay_nights)


class AmapHotelTool(BaseTool):
    """高德 POI 酒店位置兜底（通常无报价）。"""
    name: str = "amap_hotel_search"
    description: str = "仅在 hotel_search 无结果时使用。返回高德酒店名称和地址，通常没有价格。"
    args_schema: type = HotelSearchInput

    def _run(
        self,
        city: str,
        keywords: str = "酒店",
        check_in_date: str = "",
        stay_nights: int = 1,
    ) -> str:
        if not city:
            return "错误：缺少城市参数"
        try:
            from app.services.amap_service import get_amap_service

            service = get_amap_service()
            pois = service.search_poi(keywords=keywords or "酒店", city=city) or []
            if not pois:
                return f"未找到{city}的{keywords}相关酒店"

            result_list = []
            for poi in pois[:10]:
                name = poi.name if hasattr(poi, "name") else poi.get("name", "")
                address = poi.address if hasattr(poi, "address") else poi.get("address", "")
                poi_type = poi.type if hasattr(poi, "type") else poi.get("type", "")
                info = f"- {name}\n  地址: {address}"
                if poi_type:
                    info += f"\n  类型: {poi_type}"
                result_list.append(info)
            return "\n\n".join(result_list) + "\n\n（高德 POI 兜底，可能无价格）"
        except Exception as e:
            return f"高德酒店搜索出错：{str(e)}"

    async def _arun(
        self,
        city: str,
        keywords: str = "酒店",
        check_in_date: str = "",
        stay_nights: int = 1,
    ) -> str:
        return self._run(city, keywords, check_in_date, stay_nights)


def get_basic_tools() -> list:
    return [
        CalculatorTool(),
        CodeInterpreterTool(),
        DateTimeTool(),
    ]


def get_all_tools() -> list:
    tools = [
        CalculatorTool(),
        CodeInterpreterTool(),
        SearchTool(),
        DateTimeTool(),
        WeatherTool(),
        AmapAttractionTool(),
        FlyaiTicketSearchTool(),
        AmapWeatherTool(),
        RollingGoHotelTool(),
        AmapHotelTool(),
        AmapRouteTool(),
    ]
    
    try:
        from app.memory.tools import get_memory_tools
        tools.extend(get_memory_tools())
    except ImportError:
        pass
    
    return tools


def get_amap_tools() -> list:
    """获取高德地图相关工具"""
    return [
        AmapAttractionTool(),
        AmapWeatherTool(),
        AmapHotelTool(),
        AmapRouteTool(),
    ]


def get_route_tools() -> list:
    """获取路线规划相关工具"""
    return [
        AmapRouteTool(),
    ]


class SmartAttractionSearchInput(BaseModel):
    city: str = Field(..., description="城市名称")
    keywords: str = Field(default="", description="搜索关键词，如历史文化、自然风光等")


class FlyaiTicketSearchInput(BaseModel):
    scenic_name: str = Field(..., description="景点名称，如外滩、中山陵")
    city: str = Field(..., description="城市中文全称，如上海")


_retriever_instance = None

def _get_retriever():
    """获取全局共享的检索器实例，避免重复加载Embedding模型"""
    global _retriever_instance
    if _retriever_instance is None:
        try:
            from app.rag.retriever import KnowledgeRetriever
            _retriever_instance = KnowledgeRetriever()
            print("✅ 检索器实例初始化完成")
        except Exception as e:
            print(f"⚠️ 检索器初始化失败: {e}")
    return _retriever_instance


def _format_location_value(location) -> str:
    """将知识库/高德 location 统一为 lon,lat 字符串"""
    if not location:
        return ""
    if isinstance(location, dict):
        lon = location.get("longitude")
        lat = location.get("latitude")
        if lon is not None and lat is not None:
            return f"{lon},{lat}"
        return ""
    return str(location).strip()


def _append_attraction_fields(info: str, metadata: dict, poi: dict = None) -> str:
    """追加地址/坐标/评分/票价/游玩时长等字段（知识库优先，高德补充）"""
    metadata = metadata or {}
    poi = poi or {}

    address = metadata.get("address") or poi.get("address")
    if address:
        info += f"\n   地址: {address}"

    loc = _format_location_value(metadata.get("location")) or str(poi.get("location") or "").strip()
    if loc:
        info += f"\n   坐标: {loc}"

    if metadata.get("rating"):
        info += f"\n   评分: {metadata['rating']}"

    if metadata.get("price") is not None and str(metadata.get("price")).strip() != "":
        price = str(metadata["price"]).strip()
        if price == "免费" or "元" in price:
            info += f"\n   票价: {price}"
        else:
            try:
                price_num = float(price)
                info += f"\n   票价: {int(price_num)}元" if price_num == int(price_num) else f"\n   票价: {price}元"
            except Exception:
                info += f"\n   票价: {price}元"

    if metadata.get("visit_duration"):
        info += f"\n   建议游览: {metadata['visit_duration']}分钟"

    if metadata.get("open_time"):
        info += f"\n   开放时间: {metadata['open_time']}"

    if metadata.get("category"):
        info += f"\n   分类: {metadata['category']}"
    elif poi.get("type"):
        info += f"\n   类型: {poi['type']}"

    if metadata.get("description"):
        desc = str(metadata["description"])
        info += f"\n   简介: {desc[:150]}..." if len(desc) > 150 else f"\n   简介: {desc}"

    return info


# 全局共享 HybridRAG 实例（避免重复加载 CrossEncoder 模型）
_hybrid_rag_instance = None


def _get_hybrid_rag():
    """获取全局共享的 HybridRAG 实例（含 BGE Rerank 精排）"""
    global _hybrid_rag_instance
    if _hybrid_rag_instance is None:
        try:
            from app.rag.hybrid_rag import HybridRAG, HybridRAGConfig

            config = HybridRAGConfig(
                enable_rerank=True,
                rerank_model="BAAI/bge-reranker-base",
                retrieval_top_k=30,
                rerank_top_k=12,
                top_k=12,
                max_retrieval_rounds=1,
                enable_query_rewrite=False,
                enable_retrieval_validation=False,
                enable_answer_validation=False,
                use_mmr=False,
            )
            _hybrid_rag_instance = HybridRAG(config=config)
            print("✅ HybridRAG(含BGE Rerank) 实例初始化完成")
        except Exception as e:
            print(f"⚠️ HybridRAG 初始化失败: {e}，降级为普通检索器")
            _hybrid_rag_instance = None
    return _hybrid_rag_instance


class SmartAttractionTool(BaseTool):
    """高德拉名单，知识库补简介，飞猪补门票。"""

    name: str = "smart_attraction_search"
    description: str = (
        "搜索景点：高德返回名称、地址和坐标，按关键词筛选并保留足够数量，"
        "知识库补简介和游览时长，飞猪补门票价格。"
    )
    args_schema: type = SmartAttractionSearchInput

    def _run(self, city: str, keywords: str = "") -> str:
        try:
            from app.services.amap_service import get_amap_service
            from app.services.attraction_recommend import build_recommendation

            service = get_amap_service()
            if service is None:
                return f"无法获取{city}的景点信息"
            preferences = [keywords] if keywords else []
            payload = build_recommendation(
                search_poi=service.search_poi,
                city=city,
                query=keywords or f"{city}景点",
                preferences=preferences,
                travel_days=3,
            )
            text = payload.get("text") or ""
            if not text:
                return f"无法获取{city}的景点信息"
            return text
        except Exception as e:
            return f"智能景点搜索出错：{str(e)}"

    def _search_hybrid_rag(self, city: str, keywords: str):
        """HybridRAG + BGE Rerank 主检索"""
        hybrid_rag = _get_hybrid_rag()
        if hybrid_rag is None:
            # 降级普通检索
            retriever = _get_retriever()
            if retriever is None:
                return [], {"rerank_enabled": False}
            prefs = [keywords] if keywords else []
            results = retriever.retrieve_attractions(
                city=city,
                preferences=prefs,
                travel_days=3,
                search_type="hybrid",
            )
            return results or [], {"rerank_enabled": False}

        query = f"{city} {keywords}".strip() if keywords else f"{city} 景点推荐"
        preferences = [keywords] if keywords else []
        results, meta = hybrid_rag.retrieve_with_validation(
            query=query,
            city=city,
            preferences=preferences,
        )
        return results or [], meta or {}

    def _search_amap(self, city: str, keywords: str) -> list:
        """高德搜索景点：Streamable HTTP MCP，失败则 REST。"""
        return self._search_amap_rest(city, keywords)

    def _search_amap_rest(self, city: str, keywords: str) -> list:
        try:
            from app.services.amap_service import get_amap_service

            service = get_amap_service()
            if not service:
                return []
            results = service.search_poi(keywords=keywords or "景点", city=city) or []
            pois = []
            for r in results[:12]:
                if hasattr(r, "name"):
                    loc = ""
                    if getattr(r, "location", None):
                        loc = f"{r.location.longitude},{r.location.latitude}" if hasattr(r.location, "longitude") else str(r.location)
                    pois.append({
                        "name": r.name,
                        "address": getattr(r, "address", "") or "",
                        "location": loc,
                        "type": getattr(r, "type", "") or "",
                    })
                elif isinstance(r, dict):
                    pois.append(r)
            return pois
        except Exception:
            return []

    def _index_amap_pois(self, amap_results: list) -> dict:
        indexed = {}
        for poi in amap_results or []:
            name = (poi.get("name") or "").strip()
            if name:
                indexed[name] = poi
        return indexed

    def _match_amap_poi(self, name: str, amap_by_name: dict) -> dict:
        if name in amap_by_name:
            return amap_by_name[name]
        for amap_name, poi in amap_by_name.items():
            if name in amap_name or amap_name in name:
                return poi
        return {}

    def _name_matched(self, name: str, used_names: set) -> bool:
        if name in used_names:
            return True
        for used in used_names:
            if name in used or used in name:
                return True
        return False

    async def _arun(self, city: str, keywords: str = "") -> str:
        return self._run(city, keywords)


class RAGAttractionSearchInput(BaseModel):
    city: str = Field(..., description="城市名称")
    keywords: str = Field(default="", description="搜索关键词，如历史文化、自然风光等")


class RAGAttractionTool(BaseTool):
    """
    RAG景点推荐工具
    使用向量数据库检索知识库中的景点信息进行推荐
    采用 Hybrid RAG 策略：向量检索 + BM25 + CrossEncoder Rerank 精排
    """
    name: str = "rag_attraction_search"
    description: str = (
        "仅从知识库检索景点（评分、票价、开放时间、建议游览时长、简介）。"
        "向量检索+BM25，BGE Rerank 精排。不含实时高德坐标。"
    )
    args_schema: type = RAGAttractionSearchInput

    def _run(self, city: str, keywords: str = "") -> str:
        try:
            hybrid_rag = _get_hybrid_rag()

            if hybrid_rag is not None:
                query = f"{city} {keywords}".strip() if keywords else f"{city} 景点推荐"
                preferences = [keywords] if keywords else []
                results, meta = hybrid_rag.retrieve_with_validation(
                    query=query,
                    city=city,
                    preferences=preferences,
                )
            else:
                from app.rag.retriever import KnowledgeRetriever

                retriever = KnowledgeRetriever()
                preferences = [keywords] if keywords else []
                results = retriever.retrieve_attractions(
                    city=city,
                    preferences=preferences,
                    travel_days=3,
                    search_type="mmr",
                )
                meta = {"rerank_enabled": False}

            if results:
                result_list = []
                rerank_enabled = meta.get("rerank_enabled", False)
                for idx, result in enumerate(results[:10], 1):
                    metadata = result.get("metadata", {}) or {}
                    info = f"{idx}. {metadata.get('title', '')}"
                    if rerank_enabled and result.get("rerank_score") is not None:
                        info += f" (相关性: {float(result['rerank_score']):.2f})"
                    info = _append_attraction_fields(info, metadata)
                    result_list.append(info)

                footer = ""
                if rerank_enabled:
                    rerank_info = meta.get("rerank_info", {})
                    mode = rerank_info.get("mode", "N/A")
                    footer = f"\n\n[检索模式: Hybrid+Rerank({mode}), 精排后返回{len(results)}篇]"
                return "\n\n".join(result_list) + footer
            else:
                return f"知识库中未找到{city}的相关景点"

        except Exception as e:
            return f"RAG景点推荐出错：{str(e)}"

    async def _arun(self, city: str, keywords: str = "") -> str:
        return self._run(city, keywords)


class FlyaiTicketSearchTool(BaseTool):
    """飞猪 search-poi：按城市和景点名查门票。"""

    name: str = "flyai_ticket_search"
    description: str = (
        "用飞猪查询景点门票。必填 city 和 scenic_name。"
        "返回票名、价格和详情链接。没有数字价格时以详情页为准。"
    )
    args_schema: type = FlyaiTicketSearchInput

    def _run(self, scenic_name: str, city: str) -> str:
        if not (scenic_name or "").strip():
            return "错误：缺少景点名称"
        if not (city or "").strip():
            return "错误：缺少城市"
        try:
            from app.services.flyai_ticket_service import get_flyai_ticket_service

            service = get_flyai_ticket_service()
            if not service.is_configured():
                return "飞猪景点搜索失败：未配置 FLYAI_API_KEY"
            return service.search_and_format(city=city, scenic_name=scenic_name)
        except Exception as e:
            return f"飞猪景点搜索出错：{str(e)}"

    async def _arun(self, scenic_name: str, city: str) -> str:
        return self._run(scenic_name, city)


def get_attraction_tools() -> list:
    """获取景点搜索相关工具"""
    return [
        AmapAttractionTool(),
        SmartAttractionTool(),
        FlyaiTicketSearchTool(),
    ]


def get_weather_tools() -> list:
    """获取天气查询相关工具"""
    return [
        AmapWeatherTool(),
    ]


def get_hotel_tools() -> list:
    """酒店推荐 Agent 的工具：RollingGo 主搜，高德仅作位置兜底。"""
    return [
        RollingGoHotelTool(),
        AmapHotelTool(),
    ]
