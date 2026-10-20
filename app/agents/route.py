"""按确定顺序计算相邻地点的距离。"""

import re
from typing import Any, Dict, List, Optional

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from app.agents.schedule import ScheduleMixin

class RouteDistanceArgs(BaseModel):
    city: str = Field(..., description="城市")
    transportation: str = Field(default="公共交通", description="公共交通、驾车或步行")
    stops: List[str] = Field(..., description="某一天已经排好的地点顺序，包含出发酒店和返回酒店")


class RouteDistanceTool(BaseTool):
    """按已经排好的当天顺序计算相邻地点的距离和时长。"""

    name: str = "calculate_route_distance"
    description: str = (
        "在某一天的游览顺序确定之后调用。"
        "按给定顺序计算相邻地点的距离、时长和方式。不要在顺序确定之前调用。"
    )
    args_schema: type = RouteDistanceArgs
    nodes: Any = None
    catalog: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    last_segments: List[Dict[str, Any]] = Field(default_factory=list)

    def _run(self, city: str, transportation: str = "公共交通", stops: Optional[List[str]] = None) -> str:
        places = []
        for name in stops or []:
            found = self.catalog.get(name)
            places.append(dict(found) if found else {"name": name})
        if len(places) < 2 or self.nodes is None:
            self.last_segments = []
            return "地点不足两个，无法计算距离。"
        segments = self.nodes._generate_route_segments_from_attractions(
            places, city=city, transportation=transportation or "公共交通"
        )
        self.last_segments = list(segments or [])
        if not self.last_segments:
            return "没有算出相邻路段。"
        lines = []
        for seg in self.last_segments:
            distance = seg.get("distance") or 0
            distance_text = f"{distance / 1000:.1f}公里" if distance >= 1000 else f"{distance}米"
            lines.append(
                f"{seg.get('start_point')} -> {seg.get('end_point')}: "
                f"{seg.get('transport_mode')}，约{seg.get('duration')}分钟，{distance_text}"
            )
        return "\n".join(lines)

    async def _arun(self, city: str, transportation: str = "公共交通", stops: Optional[List[str]] = None) -> str:
        return self._run(city, transportation, stops)


class RouteMixin:
    """高德路段。"""

    def _extract_route_info(
        self,
        attraction_response: str,
        city: str = "",
        transportation: str = "公共交通",
        travel_days: int = 1,
    ) -> str:
        """从景点响应提取列表，并调用高德真实路径规划生成路线摘要"""
        try:
            attractions = []
            current = {}

            for line in attraction_response.split("\n"):
                line = line.strip()
                if line.startswith(tuple(f"{i}." for i in range(1, 16))):
                    if current:
                        attractions.append(current)
                    current = {"name": line.split(".", 1)[1].strip() if "." in line else line}
                elif line.startswith("坐标:"):
                    current["location"] = line[3:].strip()
                elif line.startswith("地址:"):
                    current["address"] = line[3:].strip()

            if current:
                attractions.append(current)

            max_n = max(8, (travel_days or 1) * 5)
            attractions = attractions[:max_n]
            if len(attractions) < 2:
                return ""

            # 生成相邻段真实路线
            segments = self._generate_route_segments_from_attractions(
                attractions, city=city, transportation=transportation
            )

            lines = [
                "以下为高德真实路径规划结果，规划行程时请优先采用这些距离与时长：",
                "",
                "景点列表:",
            ]
            for i, attr in enumerate(attractions, 1):
                loc = attr.get("location") or attr.get("address") or "未知"
                lines.append(f"  {i}. {attr.get('name', '')}: {loc}")

            lines.append("")
            lines.append("相邻景点高德路线:")
            for seg in segments:
                dist = seg.get("distance") or 0
                dist_txt = f"{dist/1000:.1f}公里" if dist >= 1000 else f"{dist}米"
                detail = seg.get("route_detail") or seg.get("description") or ""
                lines.append(
                    f"  {seg.get('start_point')} -> {seg.get('end_point')}: "
                    f"{seg.get('transport_mode')}，约{seg.get('duration')}分钟，{dist_txt}"
                    f"{(' · ' + detail) if detail else ''}（高德真实路线）"
                )
            return "\n".join(lines)
        except Exception as e:
            if self.logger:
                self.logger.log_info("ROUTE", msg=f"[路线规划] 提取失败: {e}")
            return ""
    
    def _route_between(
        self,
        origin: Optional[Dict[str, Any]],
        dest: Optional[Dict[str, Any]],
        city: str,
        transportation: str,
    ) -> Optional[Dict[str, Any]]:
        if not origin or not dest:
            return None
        cache = getattr(self, "_route_cache", None)
        if cache is None:
            cache = {}
            self._route_cache = cache
        key = (origin.get("name"), dest.get("name"), city, transportation)
        if key in cache:
            return cache[key]
        try:
            segs = self._generate_route_segments_from_attractions(
                [origin, dest], city, transportation
            )
        except Exception:
            segs = []
        seg = segs[0] if segs else None
        cache[key] = seg
        return seg

    def _travel_block(self, seg: Optional[Dict[str, Any]]) -> tuple:
        """时间轴上的交通占用。真实路程加 10 分钟；没有路线时按步行 5 分钟。"""
        if not isinstance(seg, dict):
            text = f"未取得高德路线，按步行{self.UNKNOWN_TRAVEL_MIN}分钟估算"
            return self.UNKNOWN_TRAVEL_MIN, "步行", text
        detail = str(seg.get("route_detail") or "")
        mode = str(seg.get("transport_mode") or "交通")
        try:
            minutes = int(float(seg.get("duration") or 0))
        except (TypeError, ValueError):
            minutes = 0
        if minutes <= 0 or "未取得" in detail:
            length = self.UNKNOWN_TRAVEL_MIN
            text = f"未取得高德路线，按步行{length}分钟估算"
            return length, "步行", text
        return (
            minutes + self.TRAVEL_BUFFER_MIN,
            detail or mode,
            f"{mode} · 约{minutes}分钟，另留{self.TRAVEL_BUFFER_MIN}分钟缓冲",
        )

    @staticmethod
    def _transit_line_names(route_info) -> List[str]:
        """收集一次公交换乘里每一段线路名。"""
        names = []
        transits = getattr(route_info, "transits", None) or []
        if not isinstance(transits, list) or not transits or not isinstance(transits[0], dict):
            return names
        for seg in transits[0].get("segments") or []:
            if not isinstance(seg, dict):
                continue
            bus = seg.get("bus") or {}
            if not isinstance(bus, dict):
                continue
            for line in bus.get("buslines") or []:
                if not isinstance(line, dict):
                    continue
                name = str(line.get("name") or "").strip()
                if name and name not in names:
                    names.append(name)
        return names

    @staticmethod
    def _qualify_route_leg(
        distance: int,
        duration_min: int,
        line_names: List[str],
        transportation: str,
        amap_mode: str,
    ) -> Dict[str, Any]:
        """500 米以内按步行计时。没有路线时按步行 5 分钟。"""
        try:
            distance = int(distance or 0)
        except (TypeError, ValueError):
            distance = 0
        try:
            duration_min = int(duration_min or 0)
        except (TypeError, ValueError):
            duration_min = 0
        if 0 < distance < ScheduleMixin.WALK_MAX_METERS:
            minutes = max(5, int(round(distance / 70)))
            return {
                "distance": distance,
                "duration": minutes,
                "transport_mode": "步行",
                "route_detail": f"步行约{minutes}分钟，{distance}米",
            }
        if duration_min <= 0:
            return {
                "distance": distance,
                "duration": ScheduleMixin.UNKNOWN_TRAVEL_MIN,
                "transport_mode": "步行",
                "route_detail": f"未取得高德路线，按步行{ScheduleMixin.UNKNOWN_TRAVEL_MIN}分钟估算",
            }
        if line_names:
            detail = " → ".join(line_names) if len(line_names) > 1 else f"乘坐{line_names[0]}"
            mode = detail if len(line_names) > 1 else line_names[0]
        elif amap_mode == "driving":
            detail = f"驾车约{duration_min}分钟，{distance / 1000:.1f}公里"
            mode = "驾车"
        elif amap_mode == "walking":
            detail = f"步行约{duration_min}分钟，{distance}米"
            mode = "步行"
        else:
            detail = f"{transportation}约{duration_min}分钟"
            mode = transportation or "交通"
        return {
            "distance": distance,
            "duration": duration_min,
            "transport_mode": mode,
            "route_detail": detail,
        }
    
    def _generate_route_segments_from_attractions(self, attractions: List[Dict], city: str = "", transportation: str = "地铁") -> List[Dict]:
        """根据景点列表生成路线段（调用高德API获取真实路线信息）"""
        segments = []
        
        # 确定交通模式
        transport_mode_map = {
            "地铁": "transit",
            "公交": "transit",
            "公共交通": "transit",
            "打车": "driving",
            "驾车": "driving",
            "自驾": "driving",
            "步行": "walking",
            "骑行": "walking",
        }
        amap_mode = transport_mode_map.get(transportation, "transit")
        
        for i in range(len(attractions) - 1):
            a1 = attractions[i] if isinstance(attractions[i], dict) else {}
            a2 = attractions[i+1] if isinstance(attractions[i+1], dict) else {}
            
            name1 = a1.get("name", f"景点{i+1}")
            name2 = a2.get("name", f"景点{i+2}")
            
            segment = {
                "start_point": name1,
                "end_point": name2,
                "transport_mode": transportation,
                "distance": 0,
                "duration": 0,
                "description": f"从{name1}到{name2}",
                "route_detail": ""
            }
            
            # 调用高德路线规划API（优先坐标，其次城市+名称）
            try:
                loc1 = a1.get("location") or ""
                loc2 = a2.get("location") or ""
                if loc1 and "," in str(loc1):
                    origin = str(loc1).strip()
                else:
                    origin = a1.get("address") or (f"{city}{name1}" if city else name1)
                if loc2 and "," in str(loc2):
                    destination = str(loc2).strip()
                else:
                    destination = a2.get("address") or (f"{city}{name2}" if city else name2)

                cache = getattr(self, "_leg_cache", None)
                if cache is None:
                    cache = {}
                    self._leg_cache = cache
                cache_key = (origin, destination, amap_mode, city)
                if cache_key in cache:
                    segment.update(cache[cache_key])
                    segments.append(segment)
                    continue

                near = self._adjacent_walk(a1, a2)
                if near:
                    segment.update(near)
                    segment["description"] = (
                        f"从{name1}到{name2}，步行，约{segment['duration']}分钟"
                    )
                    cache[cache_key] = {
                        "distance": segment.get("distance"),
                        "duration": segment.get("duration"),
                        "transport_mode": segment.get("transport_mode"),
                        "route_detail": segment.get("route_detail") or "",
                    }
                    segments.append(segment)
                    continue

                route_info = self.amap_service.route_direction(
                    origin=origin,
                    destination=destination,
                    mode=amap_mode,
                    city=city
                )
                
                if route_info:
                    distance, duration_min = self._parse_amap_distance_duration(route_info)
                    line_names = self._transit_line_names(route_info) if amap_mode == "transit" else []
                    segment.update(self._qualify_route_leg(
                        distance, duration_min, line_names, transportation, amap_mode
                    ))
                else:
                    segment.update(self._qualify_route_leg(0, 0, [], transportation, amap_mode))
                segment["description"] = (
                    f"从{name1}到{name2}，{segment['transport_mode']}，约{segment['duration']}分钟"
                    + ("" if "未取得" in segment.get("route_detail", "") else "（高德路径规划）")
                )
                cache[cache_key] = {
                    "distance": segment.get("distance"),
                    "duration": segment.get("duration"),
                    "transport_mode": segment.get("transport_mode"),
                    "route_detail": segment.get("route_detail") or "",
                }
                    
            except Exception as e:
                if getattr(self, "logger", None):
                    self.logger.log_info("ROUTE", msg=f"[路线规划] {name1}→{name2} 失败: {e}")
                segment.update(self._qualify_route_leg(0, 0, [], transportation, amap_mode))
            
            segments.append(segment)
        
        return segments

    def _parse_amap_distance_duration(self, route_info) -> tuple:
        """从 AmapRouteInfo 解析距离(米)与时长(分钟)"""
        import re

        distance = 0
        duration_sec = 0

        if route_info.paths and isinstance(route_info.paths, list) and route_info.paths:
            path = route_info.paths[0]
            try:
                distance = int(float(path.get("distance") or 0))
            except (TypeError, ValueError):
                distance = 0
            try:
                duration_sec = int(float(path.get("duration") or 0))
            except (TypeError, ValueError):
                duration_sec = 0
        elif route_info.transits and isinstance(route_info.transits, list) and route_info.transits:
            transit = route_info.transits[0]
            try:
                distance = int(float(transit.get("distance") or 0))
            except (TypeError, ValueError):
                distance = 0
            try:
                duration_sec = int(float(transit.get("duration") or 0))
            except (TypeError, ValueError):
                duration_sec = 0

        if not distance and route_info.distance:
            try:
                raw_d = str(route_info.distance)
                distance = int(float(re.sub(r"[^\d.]", "", raw_d) or 0))
                if "公里" in raw_d or "km" in raw_d.lower():
                    distance = int(distance * 1000)
            except (TypeError, ValueError):
                pass

        if duration_sec:
            return distance, max(1, duration_sec // 60)

        raw = str(route_info.duration or "")
        hours = re.search(r"(\d+)\s*小时", raw)
        mins = re.search(r"(\d+)\s*分钟", raw)
        if hours or mins:
            total = (int(hours.group(1)) if hours else 0) * 60 + (int(mins.group(1)) if mins else 0)
            return distance, max(1, total)
        try:
            m = re.search(r"(\d+)", raw)
            val = int(m.group(1)) if m else 0
            return distance, max(1, val // 60) if val > 180 else max(1, val) if val else 0
        except Exception:
            return distance, 0
