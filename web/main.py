"""
旅行规划Web应用 - FastAPI
提供旅行计划生成、POI搜索、天气查询等API接口
"""

import sys
import os
import asyncio
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from app.models.schemas import (
    TripRequest, TripPlan, TripPlanResponse,
    POISearchRequest, POISearchResponse, POIInfo,
    WeatherResponse, WeatherInfo,
    RouteRequest, RouteResponse, RouteInfo,
    ErrorResponse,
    DialogSessionRequest, DialogSessionResponse,
    DialogChatRequest, DialogChatResponse,
    normalize_trip_plan_day_indices,
)
from app.agents.langgraph_builder import get_langgraph_agent
from app.agents.dialog_agent import DialogAgent
from app.agents.version_manager import VersionManager
from app.services.amap_service import AmapService
from app.memory.memory_manager import TravelMemoryManager
from app.monitoring import get_metrics, get_alert_manager, setup_default_alerts, setup_langsmith, langsmith_health_check


app = FastAPI(
    title="旅行规划API",
    description="智能旅行规划服务。出游计划走 LangGraph 主图（景点/天气/酒店并行子图）。",
    version="1.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


trip_planner = None
amap_service = None
memory_manager = None
dialog_agent = None
version_manager = None


def build_plan_data(trip_plan: TripPlan, request: DialogSessionRequest) -> dict:
    """将 TripPlan 转为 DialogAgent 使用的 plan_data 结构"""
    plan_dict = trip_plan.model_dump()
    attractions = []
    for day in plan_dict.get("days", []):
        for attr in day.get("attractions", []):
            name = attr.get("name", "")
            if name:
                attractions.append(name)

    travel_days = request.travel_days or len(plan_dict.get("days", [])) or 1
    budget = plan_dict.get("budget") or {}

    return {
        "city": plan_dict.get("city", ""),
        "duration_days": travel_days,
        "attractions": attractions,
        "budget": budget,
        "daily_itinerary": plan_dict,
        "transportation": request.transportation,
        "accommodation": request.accommodation,
        "start_date": plan_dict.get("start_date", ""),
        "end_date": plan_dict.get("end_date", ""),
    }


def extract_trip_plan(plan_data: dict) -> Optional[TripPlan]:
    """从 plan_data 中提取可展示的 TripPlan"""
    if not plan_data:
        return None

    daily = plan_data.get("daily_itinerary")
    if isinstance(daily, dict) and daily.get("days"):
        try:
            # 补齐缺失 location，避免校验失败导致右侧预览回退到旧结构
            patched = _ensure_attraction_locations(daily)
            return normalize_trip_plan_day_indices(TripPlan(**patched))
        except Exception as e:
            print(f"⚠️ extract_trip_plan 解析 daily_itinerary 失败: {e}")

    city = plan_data.get("city", "")
    if not city:
        return None

    return TripPlan(
        city=city,
        start_date=plan_data.get("start_date", ""),
        end_date=plan_data.get("end_date", ""),
        days=[],
        overall_suggestions=f"当前计划包含景点：{', '.join(plan_data.get('attractions', []))}",
        budget=plan_data.get("budget"),
    )


def _ensure_attraction_locations(daily: dict) -> dict:
    """保证每日景点含 location 字段，便于 TripPlan 校验通过"""
    import copy
    data = copy.deepcopy(daily)
    for day in data.get("days", []) or []:
        attrs = day.get("attractions", []) or []
        fixed = []
        for a in attrs:
            if not isinstance(a, dict):
                continue
            if not a.get("location"):
                a["location"] = {"longitude": 116.4, "latitude": 39.9}
            elif isinstance(a["location"], str):
                parts = str(a["location"]).split(",")
                try:
                    a["location"] = {
                        "longitude": float(parts[0]),
                        "latitude": float(parts[1]) if len(parts) > 1 else 39.9,
                    }
                except Exception:
                    a["location"] = {"longitude": 116.4, "latitude": 39.9}
            a.setdefault("address", a.get("address") or "")
            a.setdefault("visit_duration", a.get("visit_duration") or 120)
            a.setdefault("ticket_price", a.get("ticket_price") or 0)
            fixed.append(a)
        day["attractions"] = fixed
    return data


def graph_result_to_trip_plan(result: dict, request: TripRequest) -> Optional[TripPlan]:
    """把 LangGraph 的 trip_plan_parsed 收成前端用的 TripPlan。"""
    parsed = (result or {}).get("trip_plan_parsed")
    if not isinstance(parsed, dict):
        return None

    data = dict(parsed)
    data.setdefault("city", result.get("city") or request.city)
    data.setdefault("start_date", request.start_date)
    data.setdefault("end_date", request.end_date)
    data.setdefault("days", [])
    data.setdefault("weather_info", [])
    data.setdefault("overall_suggestions", data.get("overall_suggestions") or "")

    for i, day in enumerate(data.get("days") or []):
        if not isinstance(day, dict):
            continue
        day.setdefault("date", request.start_date)
        day.setdefault("day_index", i)
        day.setdefault("description", day.get("description") or f"第{i + 1}天")
        day.setdefault("attractions", [])
        day.setdefault("meals", [])

    patched = _ensure_attraction_locations(data)
    try:
        return normalize_trip_plan_day_indices(TripPlan(**patched))
    except Exception as e:
        print(f"⚠️ graph_result_to_trip_plan 校验失败: {e}")
        return None


def get_trip_planner():
    global trip_planner
    if trip_planner is None:
        try:
            trip_planner = get_langgraph_agent(llm_provider="aliyun")
            print("✅ LangGraph TripPlanner 初始化成功")
        except Exception as e:
            print(f"⚠️ LangGraph TripPlanner 初始化失败: {e}")
    return trip_planner


def get_amap_service():
    global amap_service
    if amap_service is None:
        try:
            amap_service = AmapService()
            print("✅ AmapService初始化成功")
        except Exception as e:
            print(f"⚠️ AmapService初始化失败: {e}")
    return amap_service


def get_memory_manager():
    global memory_manager
    if memory_manager is None:
        try:
            memory_manager = TravelMemoryManager()
            print("✅ MemoryManager初始化成功")
        except Exception as e:
            print(f"⚠️ MemoryManager初始化失败: {e}")
    return memory_manager


def get_version_manager():
    global version_manager
    if version_manager is None:
        try:
            version_manager = VersionManager(use_redis=True)
            print("✅ VersionManager初始化成功")
        except Exception as e:
            print(f"⚠️ VersionManager初始化失败: {e}")
    return version_manager


def get_dialog_agent():
    global dialog_agent
    if dialog_agent is None:
        try:
            dialog_agent = DialogAgent(
                llm_provider="aliyun",
                use_redis=True,
                enable_rag=True,
                enable_amap=True,
                enable_memory=True,
            )
            print("✅ DialogAgent初始化成功")
        except Exception as e:
            print(f"⚠️ DialogAgent初始化失败: {e}")
    return dialog_agent


@app.on_event("startup")
async def startup_event():
    print("✅ 旅行规划API服务启动中...")

    # LangSmith 链路追踪（需配置 LANGSMITH_TRACING + LANGSMITH_API_KEY）
    ls_cfg = setup_langsmith()
    if ls_cfg.enabled:
        print(f"✅ LangSmith 已启用 project={ls_cfg.project}")
    else:
        print("ℹ️  LangSmith 未启用（可选：在 .env 配置 LANGSMITH_TRACING=true 与 LANGSMITH_API_KEY）")

    setup_default_alerts()
    alert_manager = get_alert_manager()
    alert_manager.start_monitoring(lambda: get_metrics().get_summary())
    print("✅ 监控告警系统已启动")


@app.get("/", tags=["首页"])
async def root():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/health", tags=["健康检查"])
async def health_check():
    return {
        "status": "healthy",
        "langsmith": langsmith_health_check(),
    }


@app.get("/api/monitoring/langsmith", tags=["监控"])
async def langsmith_status():
    """LangSmith 追踪状态与最近 runs"""
    from app.monitoring import get_langsmith_config, get_recent_runs

    cfg = get_langsmith_config()
    return {
        "success": True,
        "config": cfg.to_dict(),
        "health": langsmith_health_check(),
        "recent_runs": get_recent_runs(limit=5) if cfg.enabled else [],
    }


@app.post("/api/trip/plan", response_model=TripPlanResponse, tags=["旅行计划"])
async def generate_trip_plan(request: TripRequest, background_tasks: BackgroundTasks = None):
    """
    生成旅行计划（LangGraph 主图：景点 / 天气 / 酒店并行）
    """
    try:
        planner = get_trip_planner()
        if not planner:
            raise HTTPException(status_code=500, detail="旅行规划服务初始化失败")

        result = await planner.aplan_trip(
            city=request.city,
            travel_days=request.travel_days,
            start_date=request.start_date,
            end_date=request.end_date,
            transportation=request.transportation,
            accommodation=request.accommodation,
            preferences=request.preferences or [],
            user_id=request.user_id or "",
            free_text_input=request.free_text_input or "",
        )
        plan = graph_result_to_trip_plan(result, request)

        if plan:
            return TripPlanResponse(
                success=True,
                message="旅行计划已生成，图已暂停，可以继续修改",
                data=plan,
                thread_id=result.get("thread_id") or None,
                session_id=result.get("session_id") or None,
                welcome_message=result.get("dialog_reply") or None,
            )
        return TripPlanResponse(
            success=False,
            message=result.get("trip_plan") or "旅行计划生成失败（图结果无法解析为 TripPlan）",
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/poi/search", response_model=POISearchResponse, tags=["POI搜索"])
async def search_poi(request: POISearchRequest):
    """
    搜索POI（景点/酒店等）
    
    参数:
    - keywords: 搜索关键词
    - city: 城市
    - citylimit: 是否限制在城市范围内
    """
    try:
        service = get_amap_service()
        if not service:
            raise HTTPException(status_code=500, detail="高德地图服务初始化失败")
        
        result = service.search_poi(
            keywords=request.keywords,
            city=request.city,
            citylimit=str(request.citylimit).lower()
        )
        
        if result and "pois" in result:
            pois = [POIInfo.from_amap_response(poi) for poi in result["pois"]]
            return POISearchResponse(
                success=True,
                message="搜索成功",
                data=pois[:10]
            )
        else:
            return POISearchResponse(
                success=False,
                message="未找到相关结果"
            )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/weather/{city}", response_model=WeatherResponse, tags=["天气查询"])
async def get_weather(city: str):
    """
    查询天气信息
    
    参数:
    - city: 城市名称
    """
    try:
        service = get_amap_service()
        if not service:
            raise HTTPException(status_code=500, detail="高德地图服务初始化失败")
        
        result = service.get_weather(city)
        
        if result:
            if hasattr(result, 'to_weather_info'):
                weather_info = result.to_weather_info(date=result.report_time or "")
                return WeatherResponse(
                    success=True,
                    message="天气查询成功",
                    data=[weather_info]
                )
            elif isinstance(result, dict) and "forecasts" in result:
                forecasts = result.get("forecasts", [])
                weather_list = []
                for forecast in forecasts[:5]:
                    weather_info = WeatherInfo(
                        date=forecast.get("date", ""),
                        day_weather=forecast.get("dayweather", ""),
                        night_weather=forecast.get("nightweather", ""),
                        day_temp=forecast.get("daytemp", 0),
                        night_temp=forecast.get("nighttemp", 0),
                        wind_direction=forecast.get("daywind", ""),
                        wind_power=forecast.get("daypower", "")
                    )
                    weather_list.append(weather_info)
                return WeatherResponse(
                    success=True,
                    message="天气查询成功",
                    data=weather_list
                )
        
        return WeatherResponse(
            success=False,
            message="天气查询失败"
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/route", response_model=RouteResponse, tags=["路线规划"])
async def get_route(request: RouteRequest):
    """
    规划路线
    
    参数:
    - origin_address: 起点地址
    - destination_address: 终点地址
    - origin_city: 起点城市（可选）
    - destination_city: 终点城市（可选）
    - route_type: 路线类型 walking/driving/transit
    """
    try:
        service = get_amap_service()
        if not service:
            raise HTTPException(status_code=500, detail="高德地图服务初始化失败")
        
        result = None
        city = request.origin_city or request.destination_city
        route = service.route_direction(
            origin=request.origin_address,
            destination=request.destination_address,
            mode=request.route_type or "driving",
            city=city,
        )
        if route:
            # 距离/时长：优先 paths，其次顶层字段
            distance = 0.0
            duration_sec = 0
            if route.paths and isinstance(route.paths, list) and route.paths:
                path = route.paths[0]
                try:
                    distance = float(path.get("distance") or route.distance or 0)
                except (TypeError, ValueError):
                    distance = 0.0
                try:
                    duration_sec = int(float(path.get("duration") or 0))
                except (TypeError, ValueError):
                    duration_sec = 0
            else:
                try:
                    distance = float(route.distance or 0)
                except (TypeError, ValueError):
                    distance = 0.0
                # duration 可能是秒数字符串或「25分钟」
                raw = str(route.duration or "")
                import re
                if "分钟" in raw or "小时" in raw:
                    hours = re.search(r"(\d+)\s*小时", raw)
                    mins = re.search(r"(\d+)\s*分钟", raw)
                    duration_sec = ((int(hours.group(1)) if hours else 0) * 60 + (int(mins.group(1)) if mins else 0)) * 60
                else:
                    try:
                        duration_sec = int(float(re.search(r"(\d+)", raw).group(1))) if re.search(r"(\d+)", raw) else 0
                    except Exception:
                        duration_sec = 0

            result = {
                "distance": distance,
                "duration": duration_sec,
            }
        
        if result:
            route_info = RouteInfo(
                distance=float(result.get("distance", "0")),
                duration=int(result.get("duration", "0")),
                route_type=request.route_type,
                description=f"从{request.origin_address}到{request.destination_address}"
            )
            return RouteResponse(
                success=True,
                message="路线规划成功",
                data=route_info
            )
        else:
            return RouteResponse(
                success=False,
                message="路线规划失败"
            )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/memory/{user_id}/{city}", tags=["旅行记忆"])
async def get_travel_memory(user_id: str, city: str):
    """
    获取用户旅行记忆
    
    参数:
    - user_id: 用户ID
    - city: 城市名称
    """
    try:
        manager = get_memory_manager()
        if not manager:
            raise HTTPException(status_code=500, detail="旅行记忆服务初始化失败")
        
        comparison = manager.compare_plans(user_id, city)
        
        return {
            "success": True,
            "message": "旅行记忆查询成功",
            "data": comparison
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/dialog/session", response_model=DialogSessionResponse, tags=["对话修改"])
async def create_dialog_session(request: DialogSessionRequest):
    """
    计划生成后创建对话会话，用于后续多轮修改。

    前端在展示旅行计划后调用此接口，获取 session_id 并进入聊天面板。
    """
    try:
        vm = get_version_manager()
        if not vm:
            raise HTTPException(status_code=500, detail="版本管理服务初始化失败")

        session = vm.create_session(user_id=request.user_id, plan_id="")
        plan_data = build_plan_data(request.trip_plan, request)
        initial_version = vm.create_initial_version(
            session_id=session.session_id,
            plan_data=plan_data,
        )

        budget = plan_data.get("budget", {})
        budget_total = budget.get("total", 0) if isinstance(budget, dict) else 0
        city = plan_data.get("city", "")

        welcome_message = f"""✅ 旅行计划已就绪，可以开始修改！

📍 城市：{city}
💰 预算总计：{budget_total}元

您可以输入修改指令，例如：
• "添加迪士尼"
• "降低酒店预算"
• "延长到5天"
• "对比一下变化"
• "回退到之前的版本"

或输入"结束"完成修改。"""

        return DialogSessionResponse(
            success=True,
            message="对话会话创建成功",
            session_id=session.session_id,
            welcome_message=welcome_message,
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/dialog/chat", response_model=DialogChatResponse, tags=["对话修改"])
async def dialog_chat(request: DialogChatRequest):
    """把修改指令送回暂停中的主图。"""
    try:
        planner = get_trip_planner()
        if not planner:
            raise HTTPException(status_code=500, detail="旅行规划服务初始化失败")

        result = await planner.aresume_dialog(request.thread_id, request.message.strip())
        if result.get("error"):
            raise HTTPException(status_code=404, detail=result["error"])

        parsed = result.get("trip_plan_parsed") if isinstance(result.get("trip_plan_parsed"), dict) else {}
        stub = TripRequest(
            city=result.get("city") or parsed.get("city") or "未指定",
            start_date=parsed.get("start_date") or "1970-01-01",
            end_date=parsed.get("end_date") or parsed.get("start_date") or "1970-01-01",
            travel_days=int(result.get("travel_days") or 1),
        )
        trip_plan = graph_result_to_trip_plan(result, stub)

        return DialogChatResponse(
            success=True,
            message="对话处理成功" if result.get("awaiting_input") else "对话已结束",
            reply=result.get("dialog_reply") or "",
            trip_plan=trip_plan,
            changes=result.get("dialog_changes") or [],
            new_version_id=result.get("current_version_id") or None,
        )
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/dialog/session/{session_id}", tags=["对话修改"])
async def get_dialog_session(session_id: str):
    """获取对话会话当前计划状态"""
    try:
        vm = get_version_manager()
        if not vm:
            raise HTTPException(status_code=500, detail="版本管理服务初始化失败")

        session = vm.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="会话不存在")

        current_version = vm.get_current_version(session_id)
        trip_plan = extract_trip_plan(current_version.plan_data) if current_version else None
        history = vm.get_conversation_history(session_id, limit=20)

        return {
            "success": True,
            "message": "会话查询成功",
            "data": {
                "session_id": session_id,
                "trip_plan": trip_plan.model_dump() if trip_plan else None,
                "current_version_id": session.current_version_id,
                "conversation_history": history,
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/memory/user/{user_id}", tags=["旅行记忆"])
async def get_user_memories(user_id: str):
    """
    获取用户所有旅行记忆
    
    参数:
    - user_id: 用户ID
    """
    try:
        manager = get_memory_manager()
        if not manager:
            raise HTTPException(status_code=500, detail="旅行记忆服务初始化失败")
        
        plans = manager.get_user_plans(user_id)
        
        return {
            "success": True,
            "message": "用户旅行记忆查询成功",
            "data": plans
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
