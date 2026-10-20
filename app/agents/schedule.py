"""当天时间轴、开放时间、用餐和最近邻排序。"""

import json
import re
from typing import Any, Dict, List, Optional

class ScheduleMixin:
    """时间轴。"""

    # 当天时间轴：09:00 出发，排到 20:00 即算满，超过 20:00 可以，单段游览仍在 21:00 前结束。
    # 游览写入知识库里的完整分钟数；高德路程再加 10 分钟缓冲。
    # 查不到路线时按步行 5 分钟估算。500 米以内按步行计。
    # 3 小时及以上的景点排在上午，其余按最近邻。
    # 午饭 11:30–14:00、晚饭 17:00–19:30，各 60 分钟。不早于窗口开饭。下一站会盖住饭点时先用餐。不排早饭。
    DAY_START = 9
    DAY_END = 21
    DAY_START_MIN = 9 * 60
    DAY_END_MIN = 21 * 60
    FILL_UNTIL_MIN = 20 * 60
    TRAVEL_BUFFER_MIN = 10
    UNKNOWN_TRAVEL_MIN = 5
    WALK_MAX_METERS = 500
    LONG_VISIT_MIN = 180
    MEAL_MINUTES = 60
    LUNCH_START = 11 * 60 + 30
    LUNCH_END = 14 * 60
    DINNER_START = 17 * 60
    DINNER_END = 19 * 60 + 30
    OPEN_WAIT_MAX = 180
    GENERIC_STAYS = frozenset({
        "经济型", "舒适型", "豪华型", "高档型", "中档", "低档",
        "经济", "舒适", "豪华", "高档", "酒店", "住宿", "宾馆",
    })

    @staticmethod
    def _visit_minutes(value: Any, default: int = 120) -> int:
        try:
            minutes = int(float(value))
        except (TypeError, ValueError):
            return default
        return minutes if minutes > 0 else default

    @staticmethod
    def _format_hhmm(minutes: int) -> str:
        minutes = max(0, int(minutes))
        return f"{minutes // 60:02d}:{minutes % 60:02d}"

    @staticmethod
    def _parse_hhmm(value: Any) -> int:
        text = str(value or "")
        if ":" not in text:
            return -1
        hour, minute = text.split(":", 1)
        try:
            return int(hour) * 60 + int(minute)
        except ValueError:
            return -1

    def _generate_time_slots_from_attractions(self, attractions: List[Dict], route_segments: List[Dict] = None) -> List[Dict]:
        """按游览分钟和高德路程分钟排时间。一段游览要能在 21:00 前结束，才写入时间轴。"""
        time_slots = []
        current = self.DAY_START_MIN
        route_segs = route_segments or []

        for i, attr in enumerate(attractions):
            if not isinstance(attr, dict):
                continue
            if current >= self.DAY_END_MIN:
                break
            name = attr.get("name", "景点")
            duration = self._visit_minutes(attr.get("visit_duration"), default=120)
            if current + duration > self.DAY_END_MIN:
                break
            end = current + duration
            time_slots.append({
                "start_time": self._format_hhmm(current),
                "end_time": self._format_hhmm(end),
                "activity_type": "景点游览",
                "activity_name": name,
                "description": attr.get("description", f"游览{name}")
            })
            current = end

            if i < len(attractions) - 1:
                travel_duration = 20
                travel_desc = "前往下一个景点"
                transport_mode = "交通"
                unknown = False

                if i < len(route_segs):
                    seg = route_segs[i]
                    try:
                        parsed = int(float(seg.get("duration") or 0))
                    except (TypeError, ValueError):
                        parsed = 0
                    travel_desc = seg.get("route_detail") or travel_desc
                    transport_mode = seg.get("transport_mode") or transport_mode
                    if parsed > 0 and "未取得" not in str(travel_desc):
                        travel_duration = parsed
                    else:
                        unknown = True
                        travel_duration = parsed or self.UNKNOWN_TRAVEL_MIN

                if unknown:
                    buffered = travel_duration
                    description = (
                        travel_desc if "未取得" in str(travel_desc)
                        else f"未取得高德路线，按步行{buffered}分钟估算"
                    )
                    transport_mode = "步行"
                else:
                    buffered = travel_duration + self.TRAVEL_BUFFER_MIN
                    description = (
                        f"{transport_mode} · 约{travel_duration}分钟，"
                        f"另留{self.TRAVEL_BUFFER_MIN}分钟缓冲"
                    )

                next_item = attractions[i + 1]
                next_duration = self._visit_minutes(
                    next_item.get("visit_duration") if isinstance(next_item, dict) else None,
                    default=120,
                )
                if current + buffered + next_duration > self.DAY_END_MIN:
                    break
                travel_end = current + buffered
                time_slots.append({
                    "start_time": self._format_hhmm(current),
                    "end_time": self._format_hhmm(travel_end),
                    "activity_type": "交通",
                    "activity_name": travel_desc,
                    "description": description,
                })
                current = travel_end

        return time_slots

    @staticmethod
    def _day_end_hour(day: Dict[str, Any]) -> int:
        if not isinstance(day, dict):
            return ScheduleMixin.DAY_START
        slots = day.get("time_slots") or []
        if not slots:
            return ScheduleMixin.DAY_START
        end = str(slots[-1].get("end_time") or "")
        if ":" not in end:
            return ScheduleMixin.DAY_START
        try:
            return int(end.split(":", 1)[0])
        except ValueError:
            return ScheduleMixin.DAY_START

    @staticmethod
    def _location_text(data: Dict[str, Any]) -> str:
        if not isinstance(data, dict):
            return ""
        location = data.get("location")
        if isinstance(location, str) and "," in location:
            return location.strip()
        if isinstance(location, dict):
            lon = location.get("longitude")
            lat = location.get("latitude")
        else:
            lon = data.get("longitude")
            lat = data.get("latitude")
        if lon in (None, "") or lat in (None, ""):
            return ""
        return f"{lon},{lat}"

    def _coerce_attraction(self, item: Any) -> Optional[Dict[str, Any]]:
        if isinstance(item, str) and item.strip():
            return {
                "name": item.strip(),
                "address": "",
                "location": "",
                "visit_duration": 120,
            }
        if not isinstance(item, dict) or not item.get("name"):
            return None
        copied = dict(item)
        copied["location"] = self._location_text(copied)
        copied["visit_duration"] = self._visit_minutes(copied.get("visit_duration"), default=120)
        return copied

    def _lookup_kb_meta(self, name: str, city: str) -> Dict[str, Any]:
        store = getattr(getattr(self, "retriever", None), "vector_store", None)
        finder = getattr(store, "find_by_title", None)
        if finder is None or not name:
            return {}
        try:
            hits = finder(name, city=city, limit=1)
        except Exception:
            return {}
        if not hits:
            return {}
        meta = hits[0].get("metadata") or {}
        return meta if isinstance(meta, dict) else {}

    def _enrich_from_kb(self, attr: Dict[str, Any], city: str) -> Dict[str, Any]:
        """按景点标题从知识库补开放时间、坐标和游览分钟。"""
        cache = getattr(self, "_kb_cache", None)
        if cache is None:
            cache = {}
            self._kb_cache = cache
        key = (city or "", attr.get("name") or "")
        if key not in cache:
            cache[key] = self._lookup_kb_meta(attr.get("name") or "", city)
        meta = cache[key] or {}
        if not meta:
            return attr
        if not attr.get("open_time"):
            attr["open_time"] = meta.get("open_time") or ""
        if not attr.get("location"):
            attr["location"] = self._location_text(meta)
        if not attr.get("address"):
            attr["address"] = meta.get("address") or ""
        if meta.get("visit_duration"):
            attr["visit_duration"] = self._visit_minutes(
                meta.get("visit_duration"),
                default=attr.get("visit_duration") or 120,
            )
        return attr

    @staticmethod
    def _coords(item: Any) -> Optional[tuple]:
        if not isinstance(item, dict):
            return None
        text = item.get("location") or ""
        if not isinstance(text, str) or "," not in text:
            return None
        lon, lat = text.split(",", 1)
        try:
            return float(lon), float(lat)
        except ValueError:
            return None

    @staticmethod
    def _distance2(origin: tuple, dest: tuple) -> float:
        return (origin[0] - dest[0]) ** 2 + (origin[1] - dest[1]) ** 2

    @staticmethod
    def _meters_between(origin: tuple, dest: tuple) -> int:
        import math

        lon1, lat1 = origin
        lon2, lat2 = dest
        radius = 6371000
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)
        chord = (
            math.sin(dphi / 2) ** 2
            + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
        )
        return int(round(2 * radius * math.asin(math.sqrt(chord))))

    def _adjacent_walk(self, origin: Dict[str, Any], dest: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """直线距离不到 500 米时按步行，不再采用绕远的公交结果。"""
        start = self._coords(origin)
        end = self._coords(dest)
        if not start or not end:
            return None
        meters = self._meters_between(start, end)
        if meters >= self.WALK_MAX_METERS:
            return None
        minutes = self.UNKNOWN_TRAVEL_MIN if meters <= 0 else max(5, int(round(meters / 70)))
        return {
            "distance": meters,
            "duration": minutes,
            "transport_mode": "步行",
            "route_detail": f"步行约{minutes}分钟，{meters}米",
        }

    def _order_nearest(self, items: List[Dict[str, Any]], origin: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """有坐标的点按最近邻排，没有坐标的保持原顺序跟在后面。"""
        located = [item for item in items if self._coords(item)]
        rest = [item for item in items if not self._coords(item)]
        ordered = []
        cursor = self._coords(origin) if origin else None
        pool = list(located)
        while pool:
            if cursor is None:
                pick = pool.pop(0)
            else:
                pick = min(pool, key=lambda item: self._distance2(cursor, self._coords(item)))
                pool.remove(pick)
            ordered.append(pick)
            cursor = self._coords(pick)
        ordered.extend(rest)
        return ordered

    @classmethod
    def _is_generic_stay(cls, name: str) -> bool:
        text = (name or "").strip()
        return not text or text in cls.GENERIC_STAYS

    def _resolve_hotel_anchor(self, day: Dict[str, Any], city: str) -> Optional[Dict[str, Any]]:
        """能定位的具体酒店作为当天的出发点和返回点。"""
        raw = day.get("hotel")
        if isinstance(raw, str):
            raw = {"name": raw}
        if not isinstance(raw, dict):
            return None
        name = str(raw.get("name") or "").strip()
        if self._is_generic_stay(name):
            return None
        location = self._location_text(raw)
        if location:
            return {"name": name, "location": location, "address": raw.get("address") or ""}
        cache = getattr(self, "_hotel_cache", None)
        if cache is None:
            cache = {}
            self._hotel_cache = cache
        key = (city or "", name)
        if key in cache:
            return cache[key]
        anchor = None
        amap = getattr(self, "amap_service", None)
        if amap is not None and city and hasattr(amap, "search_poi"):
            try:
                pois = amap.search_poi(name, city, True) or []
            except Exception:
                pois = []
            if pois:
                poi = pois[0]
                loc = getattr(poi, "location", None)
                if loc is not None and getattr(loc, "longitude", None) is not None:
                    anchor = {
                        "name": getattr(poi, "name", None) or name,
                        "location": f"{loc.longitude},{loc.latitude}",
                        "address": getattr(poi, "address", "") or "",
                    }
        cache[key] = anchor
        return anchor

    @staticmethod
    def _open_window(text: Any) -> tuple:
        """取开放时间里和 09:00–21:00 重叠最长的一段。原文没有钟点时，这一项不设开关门约束。"""
        spans = re.findall(
            r"(\d{1,2}):(\d{2})\s*[-~～—至到]+\s*(\d{1,2}):(\d{2})",
            str(text or ""),
        )
        best = None
        best_overlap = 0
        for oh, om, ch, cm in spans:
            try:
                open_m = int(oh) * 60 + int(om)
                close_m = int(ch) * 60 + int(cm)
            except ValueError:
                continue
            if close_m <= open_m:
                continue
            overlap = min(close_m, ScheduleMixin.DAY_END_MIN) - max(open_m, ScheduleMixin.DAY_START_MIN)
            if overlap > best_overlap:
                best_overlap = overlap
                best = (open_m, close_m)
        if not best:
            return None, None
        return best

    @staticmethod
    def _overlap(start: int, end: int, win_start: int, win_end: int) -> int:
        return max(0, min(end, win_end) - max(start, win_start))

    def _preview_visit(
        self,
        attr: Dict[str, Any],
        prev: Optional[Dict[str, Any]],
        current: int,
        city: str,
        transportation: str,
        hotel: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """游览结束和返回酒店都要落在关门时间与 21:00 之前，这一段才放入当天。"""
        duration = self._visit_minutes(attr.get("visit_duration"), default=120)
        travel_slot = 0
        travel_name = ""
        travel_desc = ""
        segment = None
        if prev is not None:
            segment = self._route_between(prev, attr, city, transportation)
            travel_slot, travel_name, travel_desc = self._travel_block(segment)
        arrival = current + travel_slot
        if arrival >= self.DAY_END_MIN:
            return None
        open_m, close_m = self._open_window(attr.get("open_time"))
        too_early = False
        if open_m is not None and arrival < open_m:
            if open_m - arrival <= 30:
                arrival = open_m
            else:
                too_early = True
        close_limit = self.DAY_END_MIN if close_m is None else min(self.DAY_END_MIN, close_m)
        if too_early:
            fits_later = open_m is not None and open_m + duration <= close_limit
            if not fits_later:
                return None
            return {"attr": attr, "too_early": True}
        end = arrival + duration
        if end > close_limit:
            return None
        if hotel is not None:
            back = self._route_between(attr, hotel, city, transportation)
            back_slot, _, _ = self._travel_block(back)
            if end + back_slot > self.DAY_END_MIN:
                return None
        return {
            "attr": attr,
            "too_early": False,
            "start": arrival,
            "end": end,
            "depart": current,
            "travel_slot": travel_slot,
            "travel_name": travel_name,
            "travel_desc": travel_desc,
            "segment": segment,
        }

    def _preview_wait(
        self,
        pending: List[Dict[str, Any]],
        prev: Optional[Dict[str, Any]],
        current: int,
        city: str,
        transportation: str,
        hotel: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """开放时间晚于当前时刻、且间隔在 3 小时以内时，出发时间对准开门。"""
        best = None
        best_open = None
        for attr in pending:
            open_m, _close_m = self._open_window(attr.get("open_time"))
            if open_m is None or open_m <= current or open_m - current > self.OPEN_WAIT_MAX:
                continue
            travel_slot = 0
            if prev is not None:
                seg = self._route_between(prev, attr, city, transportation)
                travel_slot, _, _ = self._travel_block(seg)
            depart = open_m - travel_slot
            if depart < current:
                continue
            preview = self._preview_visit(attr, prev, depart, city, transportation, hotel)
            if not preview or preview.get("too_early"):
                continue
            if best_open is None or open_m < best_open:
                best = preview
                best_open = open_m
        return best

    def _commit_visit(self, preview: Dict[str, Any], slots: List[Dict], segments: List[Dict], kept: List[Dict]) -> None:
        attr = preview["attr"]
        depart = preview["depart"]
        if preview["travel_slot"] > 0:
            if preview.get("segment"):
                segments.append(preview["segment"])
            slots.append({
                "start_time": self._format_hhmm(depart),
                "end_time": self._format_hhmm(depart + preview["travel_slot"]),
                "activity_type": "交通",
                "activity_name": preview["travel_name"],
                "description": preview["travel_desc"],
            })
        slots.append({
            "start_time": self._format_hhmm(preview["start"]),
            "end_time": self._format_hhmm(preview["end"]),
            "activity_type": "景点游览",
            "activity_name": attr.get("name", "景点"),
            "description": attr.get("description") or f"游览{attr.get('name', '景点')}",
        })
        kept.append(attr)

    def _note_meals(self, start: int, end: int, had_lunch: bool, had_dinner: bool) -> tuple:
        if self._overlap(start, end, self.LUNCH_START, self.LUNCH_END) >= 45:
            had_lunch = True
        if self._overlap(start, end, self.DINNER_START, self.DINNER_END) >= 45:
            had_dinner = True
        return had_lunch, had_dinner

    def _meal_blocked(self, preview: Optional[Dict[str, Any]], had_lunch: bool, had_dinner: bool) -> Optional[str]:
        """短游览若盖住饭点，先吃饭再出发。3 小时及以上的景点不从中切开。"""
        if not preview or preview.get("too_early"):
            return None
        if preview["end"] - preview["start"] >= self.LONG_VISIT_MIN:
            return None
        if not had_lunch and self._overlap(preview["start"], preview["end"], self.LUNCH_START, self.LUNCH_END) >= 45:
            return "lunch"
        if not had_dinner and self._overlap(preview["start"], preview["end"], self.DINNER_START, self.DINNER_END) >= 45:
            return "dinner"
        return None

    def _align_meal_start(self, current: int, window_start: int, force: bool) -> int:
        """午饭不早于 11:30，晚饭不早于 17:00。已经进入窗口就按当前时刻开饭。"""
        del force
        return current if current >= window_start else window_start

    def _meal_named(self, meals: Optional[List[Dict[str, Any]]], meal_type: str) -> Dict[str, Any]:
        for meal in meals or []:
            if isinstance(meal, dict) and meal.get("type") == meal_type and meal.get("name"):
                return meal
        return {}

    def _append_meal_slot(
        self,
        slots: List[Dict],
        current: int,
        meal: Optional[Dict[str, Any]],
        fallback_name: str,
        fallback_desc: str,
    ) -> int:
        chosen = meal or {}
        slots.append({
            "start_time": self._format_hhmm(current),
            "end_time": self._format_hhmm(current + self.MEAL_MINUTES),
            "activity_type": "用餐",
            "activity_name": chosen.get("name") or fallback_name,
            "description": chosen.get("address") or fallback_desc,
        })
        return current + self.MEAL_MINUTES

    def _located_stop(self, meal: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not isinstance(meal, dict):
            return None
        name = str(meal.get("name") or "").strip()
        location = str(meal.get("location") or "").strip()
        if not name or "," not in location:
            return None
        return {"name": name, "location": location, "address": meal.get("address") or ""}

    def _commit_travel(
        self,
        slots: List[Dict],
        segments: List[Dict],
        prev: Dict[str, Any],
        dest: Dict[str, Any],
        current: int,
        city: str,
        transportation: str,
    ) -> int:
        seg = self._route_between(prev, dest, city, transportation)
        minutes, name, desc = self._travel_block(seg)
        if seg:
            segments.append(seg)
        slots.append({
            "start_time": self._format_hhmm(current),
            "end_time": self._format_hhmm(current + minutes),
            "activity_type": "交通",
            "activity_name": name,
            "description": desc,
        })
        return current + minutes

    def _insert_due_meals(
        self,
        slots: List[Dict],
        current: int,
        had_lunch: bool,
        had_dinner: bool,
        reserve: int = 0,
        lunch: Optional[Dict[str, Any]] = None,
        dinner: Optional[Dict[str, Any]] = None,
        prev: Optional[Dict[str, Any]] = None,
        segments: Optional[List[Dict]] = None,
        city: str = "",
        transportation: str = "公共交通",
        force: Optional[str] = None,
        catch_up: bool = False,
    ) -> tuple:
        """到了饭点，或下一站会把饭点盖住时，先用餐 60 分钟。有坐标的餐馆算进路程。"""
        segments = segments if segments is not None else []

        def place(meal: Optional[Dict[str, Any]], fallback_name: str, fallback_desc: str) -> bool:
            nonlocal current, prev
            point = self._located_stop(meal)
            travel = 0
            if point is not None and prev is not None:
                seg = self._route_between(prev, point, city, transportation)
                travel, _, _ = self._travel_block(seg)
            if current + travel + self.MEAL_MINUTES + reserve > self.DAY_END_MIN:
                return False
            if point is not None and prev is not None and travel:
                current = self._commit_travel(slots, segments, prev, point, current, city, transportation)
                prev = point
            current = self._append_meal_slot(slots, current, meal, fallback_name, fallback_desc)
            return True

        lunch_due = not had_lunch and (
            force == "lunch"
            or self.LUNCH_START <= current < self.LUNCH_END
        )
        dinner_due = not had_dinner and (
            force == "dinner"
            or self.DINNER_START <= current < self.DINNER_END
            or (catch_up and self.LUNCH_START <= current < self.DINNER_END)
        )
        if lunch_due:
            current = self._align_meal_start(current, self.LUNCH_START, force == "lunch")
            if place(lunch, "午餐", "午饭，约60分钟"):
                had_lunch = True
        if dinner_due:
            current = self._align_meal_start(current, self.DINNER_START, force == "dinner")
            if place(dinner, "晚餐", "晚饭，约60分钟"):
                had_dinner = True
        return current, had_lunch, had_dinner, prev

    def _first_fit(
        self,
        pending: List[Dict[str, Any]],
        prev: Optional[Dict[str, Any]],
        current: int,
        city: str,
        transportation: str,
        hotel: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        for attr in pending:
            preview = self._preview_visit(attr, prev, current, city, transportation, hotel)
            if preview and not preview.get("too_early"):
                return preview
        return None

    @staticmethod
    def _take(pending: List[Dict[str, Any]], attr: Dict[str, Any]) -> None:
        for index, item in enumerate(pending):
            if item is attr or item.get("name") == attr.get("name"):
                pending.pop(index)
                return

    def _place_one(
        self,
        pending: List[Dict[str, Any]],
        slots: List[Dict],
        segments: List[Dict],
        kept: List[Dict],
        current: int,
        prev: Optional[Dict[str, Any]],
        city: str,
        transportation: str,
        hotel: Optional[Dict[str, Any]],
        had_lunch: bool,
        had_dinner: bool,
        lunch: Optional[Dict[str, Any]] = None,
        dinner: Optional[Dict[str, Any]] = None,
    ) -> tuple:
        preview = self._first_fit(pending, prev, current, city, transportation, hotel)
        swallowed = self._meal_blocked(preview, had_lunch, had_dinner)
        if swallowed:
            current, had_lunch, had_dinner, prev = self._insert_due_meals(
                slots, current, had_lunch, had_dinner,
                lunch=lunch, dinner=dinner,
                prev=prev, segments=segments, city=city, transportation=transportation,
                force=swallowed,
            )
            preview = self._first_fit(pending, prev, current, city, transportation, hotel)
        meal_due = (
            (not had_lunch and self.LUNCH_START <= current < self.LUNCH_END)
            or (not had_dinner and self.DINNER_START <= current < self.DINNER_END)
        )
        if preview and not meal_due:
            self._commit_visit(preview, slots, segments, kept)
            self._take(pending, preview["attr"])
            return preview["end"], preview["attr"], had_lunch, had_dinner, True
        if meal_due:
            current, had_lunch, had_dinner, prev = self._insert_due_meals(
                slots, current, had_lunch, had_dinner,
                lunch=lunch, dinner=dinner,
                prev=prev, segments=segments, city=city, transportation=transportation,
            )
            preview = self._first_fit(pending, prev, current, city, transportation, hotel)
            if preview:
                self._commit_visit(preview, slots, segments, kept)
                self._take(pending, preview["attr"])
                return preview["end"], preview["attr"], had_lunch, had_dinner, True
        waited = self._preview_wait(pending, prev, current, city, transportation, hotel)
        blocked = self._meal_blocked(waited, had_lunch, had_dinner)
        if blocked:
            current, had_lunch, had_dinner, prev = self._insert_due_meals(
                slots, current, had_lunch, had_dinner,
                lunch=lunch, dinner=dinner,
                prev=prev, segments=segments, city=city, transportation=transportation,
                force=blocked,
            )
            waited = self._preview_wait(pending, prev, current, city, transportation, hotel)
        if waited and not self._meal_blocked(waited, had_lunch, had_dinner):
            self._commit_visit(waited, slots, segments, kept)
            self._take(pending, waited["attr"])
            return waited["end"], waited["attr"], had_lunch, had_dinner, True
        return current, prev, had_lunch, had_dinner, False

    def _drain(
        self,
        pending: List[Dict[str, Any]],
        slots: List[Dict],
        segments: List[Dict],
        kept: List[Dict],
        current: int,
        prev: Optional[Dict[str, Any]],
        city: str,
        transportation: str,
        hotel: Optional[Dict[str, Any]],
        had_lunch: bool,
        had_dinner: bool,
        lunch: Optional[Dict[str, Any]] = None,
        dinner: Optional[Dict[str, Any]] = None,
    ) -> tuple:
        pending = list(pending)
        while pending:
            current, prev, had_lunch, had_dinner, placed = self._place_one(
                pending, slots, segments, kept, current, prev, city, transportation,
                hotel, had_lunch, had_dinner, lunch, dinner,
            )
            if not placed:
                break
        return current, prev, had_lunch, had_dinner, pending

    def _too_early_pinned(self, pending: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        kept = []
        for attr in pending:
            if attr.get("filler"):
                continue
            open_m, close_m = self._open_window(attr.get("open_time"))
            if open_m is None:
                continue
            close_limit = self.DAY_END_MIN if close_m is None else min(self.DAY_END_MIN, close_m)
            duration = self._visit_minutes(attr.get("visit_duration"), default=120)
            if open_m + duration <= close_limit and open_m > self.DAY_START_MIN:
                kept.append(attr)
        return kept

    def _assemble_day(
        self,
        attractions: List[Dict[str, Any]],
        city: str,
        transportation: str,
        hotel: Optional[Dict[str, Any]],
        lunch: Optional[Dict[str, Any]] = None,
        dinner: Optional[Dict[str, Any]] = None,
    ) -> tuple:
        """3 小时及以上的景点从上午排起，其余景点和补点按最近邻接在后面。"""
        pinned = [item for item in attractions if not item.get("filler")]
        fillers = [item for item in attractions if item.get("filler")]
        long_items = [
            item for item in pinned
            if self._visit_minutes(item.get("visit_duration"), default=120) >= self.LONG_VISIT_MIN
        ]
        long_names = {item.get("name") for item in long_items}
        short_items = [item for item in pinned if item.get("name") not in long_names]
        long_items = self._order_nearest(long_items, hotel)
        slots: List[Dict] = []
        segments: List[Dict] = []
        kept: List[Dict] = []
        current = self.DAY_START_MIN
        prev = hotel
        had_lunch = False
        had_dinner = False
        current, prev, had_lunch, had_dinner, left = self._drain(
            long_items, slots, segments, kept, current, prev, city, transportation,
            hotel, had_lunch, had_dinner, lunch, dinner,
        )
        deferred = self._too_early_pinned(left)
        short_items = self._order_nearest(short_items, prev or hotel)
        current, prev, had_lunch, had_dinner, left = self._drain(
            short_items, slots, segments, kept, current, prev, city, transportation,
            hotel, had_lunch, had_dinner, lunch, dinner,
        )
        deferred.extend(self._too_early_pinned(left))
        pending_fillers = self._order_nearest(fillers, prev or hotel)
        pending_deferred = deferred
        while pending_fillers or pending_deferred:
            if pending_deferred:
                current, prev, had_lunch, had_dinner, placed = self._place_one(
                    pending_deferred, slots, segments, kept, current, prev, city, transportation,
                    hotel, had_lunch, had_dinner, lunch, dinner,
                )
                if placed:
                    continue
            if not pending_fillers:
                break
            one = [pending_fillers.pop(0)]
            current, prev, had_lunch, had_dinner, left = self._drain(
                one, slots, segments, kept, current, prev, city, transportation,
                hotel, had_lunch, had_dinner, lunch, dinner,
            )
            if left:
                break
        back_origin = prev
        if (
            not had_lunch
            and self.LUNCH_START <= current < self.LUNCH_END
            and self._located_stop(lunch)
        ):
            back_origin = self._located_stop(lunch)
        elif (
            not had_dinner
            and self.DINNER_START <= current < self.DINNER_END
            and self._located_stop(dinner)
        ):
            back_origin = self._located_stop(dinner)
        back_slot = 0
        if hotel and kept and back_origin is not None:
            back = self._route_between(back_origin, hotel, city, transportation)
            back_slot, _, _ = self._travel_block(back)
        current, had_lunch, had_dinner, prev = self._insert_due_meals(
            slots, current, had_lunch, had_dinner,
            reserve=back_slot, lunch=lunch, dinner=dinner,
            prev=prev, segments=segments, city=city, transportation=transportation,
            catch_up=True,
        )
        if hotel and kept and prev is not None:
            seg = self._route_between(prev, hotel, city, transportation)
            slot, name, desc = self._travel_block(seg)
            if current + slot <= self.DAY_END_MIN:
                if seg:
                    segments.append(seg)
                else:
                    segments.append({
                        "start_point": prev.get("name"),
                        "end_point": hotel.get("name"),
                        "transport_mode": "交通",
                        "distance": 0,
                        "duration": slot,
                        "description": desc,
                        "route_detail": name,
                    })
                slots.append({
                    "start_time": self._format_hhmm(current),
                    "end_time": self._format_hhmm(current + slot),
                    "activity_type": "交通",
                    "activity_name": name,
                    "description": desc,
                })
        still_early = [item for item in pending_deferred if item.get("name") not in {kept_item.get("name") for kept_item in kept}]
        return kept, slots, segments, self._too_early_pinned(still_early)

    def _schedule_day_clock(self, day: Dict[str, Any], city: str, transportation: str) -> None:
        attractions = []
        for item in day.get("attractions") or []:
            coerced = self._coerce_attraction(item)
            if coerced:
                attractions.append(self._enrich_from_kb(coerced, city))
        hotel = self._resolve_hotel_anchor(day, city)
        meals = day.get("meals")
        kept, slots, segments, deferred = self._assemble_day(
            attractions,
            city,
            transportation or "公共交通",
            hotel,
            self._meal_named(meals, "lunch"),
            self._meal_named(meals, "dinner"),
        )
        day["attractions"] = kept + deferred
        day["route_segments"] = segments
        day["time_slots"] = slots
        names = [item["name"] for item in kept]
        if names:
            day["description"] = f"第{int(day.get('day_index', 0)) + 1}天行程：{', '.join(names)}"

    def _slot_start_minutes(self, day: Dict[str, Any], name: str) -> Optional[int]:
        for slot in day.get("time_slots") or []:
            if slot.get("activity_type") == "景点游览" and slot.get("activity_name") == name:
                start = self._parse_hhmm(slot.get("start_time"))
                return start if start >= 0 else None
        return None

    @staticmethod
    def _day_end_minutes(day: Dict[str, Any]) -> int:
        if not isinstance(day, dict):
            return ScheduleMixin.DAY_START_MIN
        slots = day.get("time_slots") or []
        if not slots:
            return ScheduleMixin.DAY_START_MIN
        end = ScheduleMixin._parse_hhmm(slots[-1].get("end_time"))
        return end if end >= 0 else ScheduleMixin.DAY_START_MIN

    def _fill_day_until_close(
        self,
        day: Dict[str, Any],
        city: str,
        transportation: str,
        used: set,
        pool: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """先排已有景点。收尾早于 20:00 且距离 21:00 还空出至少 1 小时时，从候选池补景点。超过 20:00 就不再补。"""
        if not isinstance(day, dict):
            return
        self._schedule_day_clock(day, city, transportation or "公共交通")
        for slot in day.get("time_slots") or []:
            if slot.get("activity_type") == "景点游览" and slot.get("activity_name"):
                used.add(slot.get("activity_name"))
        if not city and pool is None:
            return
        rejected: set = set()
        attempts = max(6, len(pool or []))
        for _ in range(attempts):
            end = self._day_end_minutes(day)
            if end >= self.FILL_UNTIL_MIN:
                break
            existing = [
                slot.get("activity_name")
                for slot in day.get("time_slots") or []
                if slot.get("activity_type") == "景点游览" and slot.get("activity_name")
            ]
            if len(existing) >= 6:
                break
            remaining = self.DAY_END_MIN - end
            if existing and remaining < 60:
                break
            slotted = [
                slot.get("activity_name")
                for slot in day.get("time_slots") or []
                if slot.get("activity_type") == "景点游览" and slot.get("activity_name")
            ]
            last_name = slotted[-1] if slotted else (existing[-1] if existing else "")
            blocked = set(used) | rejected
            try:
                if pool is not None:
                    record = self._next_pool_filler(pool, blocked)
                else:
                    record = self._suggest_filler_attraction(
                        city, list(blocked), max(1, remaining // 60), last_name
                    )
            except Exception as e:
                if getattr(self, "logger", None):
                    self.logger.log_info("PLAN", msg=f"[补点] 失败: {e}")
                break
            if not isinstance(record, dict):
                break
            name = str(record.get("name") or "").strip()
            if not name or name in blocked:
                break
            record = dict(record)
            record["name"] = name
            record["filler"] = True
            record.setdefault("description", "补足当日空档")
            day.setdefault("attractions", []).append(record)
            used.add(name)
            self._schedule_day_clock(day, city, transportation or "公共交通")
            start = self._slot_start_minutes(day, name)
            if start is None or start >= self.DAY_END_MIN:
                used.discard(name)
                rejected.add(name)
                day["attractions"] = [
                    item for item in day.get("attractions") or []
                    if not (isinstance(item, dict) and item.get("name") == name)
                ]
                self._schedule_day_clock(day, city, transportation or "公共交通")
                continue
        self._drop_unslotted(day)
        self._schedule_day_clock(day, city, transportation or "公共交通")

    def _drop_unslotted(self, day: Dict[str, Any]) -> None:
        slotted = {
            slot.get("activity_name")
            for slot in day.get("time_slots") or []
            if slot.get("activity_type") == "景点游览"
        }
        day["attractions"] = [
            item for item in day.get("attractions") or []
            if isinstance(item, dict) and item.get("name") in slotted
        ]

    @staticmethod
    def _next_pool_filler(
        pool: List[Dict[str, Any]],
        used: set,
    ) -> Optional[Dict[str, Any]]:
        """从推荐候选里取一个还没排上的景点。"""
        for item in pool:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip()
            if name and name not in used:
                return dict(item)
        return None

    def _suggest_filler_attraction(
        self,
        city: str,
        existing: List[str],
        remaining_hours: int,
        last_name: str,
    ) -> Optional[Dict[str, Any]]:
        """搜索一个知识库里标题能对上的景点，用来填 21:00 前的空档。"""
        llm = getattr(self, "llm", None)
        if llm is None or remaining_hours < 1:
            return None
        if getattr(self, "_fill_agent", None) is None:
            from app.agents.react_agent import ReActAgent
            from app.services.structured_output import FillNameOutput
            from app.tools.agent_tools import SmartAttractionTool

            self._fill_agent = ReActAgent(
                name="行程补点",
                tools=[SmartAttractionTool()],
                llm=llm,
                system_prompt=(
                    "你负责把还没到 20:00 的一天补满。超过 20:00 就不要再补。\n"
                    "调用 smart_attraction_search，找一个不在已有列表里、离当天最后一个景点近、并能在 21:00 前结束的景点。\n"
                    "搜索结果太远，或游览结束会超过 21:00，就换一个更近的关键词再搜一次。不要编造标题。\n"
                    "name 必须与某条工具结果的标题逐字相同。"
                ),
                max_iterations=2,
                verbose=False,
                response_format=FillNameOutput,
            )
        query = (
            f"城市：{city}\n"
            f"已有景点：{', '.join(existing) or '无'}\n"
            f"当天最后一个景点：{last_name or '无'}\n"
            f"距离 20:00 还空着约 {remaining_hours} 小时。\n"
            "请调用 smart_attraction_search 补一个景点。"
        )
        try:
            text = self._fill_agent.run(query)
        except Exception as e:
            if getattr(self, "logger", None):
                self.logger.log_info("PLAN", msg=f"[补点] 检索失败: {e}")
            return None
        name = self._parse_fill_name(text)
        if not name or name in existing:
            return None
        return self._lookup_filler(name, city)

    @staticmethod
    def _parse_fill_name(text: str) -> str:
        if not text:
            return ""
        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            return ""
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return ""
        if not isinstance(data, dict):
            return ""
        return str(data.get("name") or "").strip()

    def _lookup_filler(self, name: str, city: str) -> Optional[Dict[str, Any]]:
        store = getattr(getattr(self, "retriever", None), "vector_store", None)
        finder = getattr(store, "find_by_title", None)
        if finder is None:
            return None
        try:
            hits = finder(name, city=city, limit=1)
        except Exception:
            return None
        if not hits:
            return None
        meta = hits[0].get("metadata") or {}
        price_str = meta.get("price", "0")
        try:
            price = int(re.sub(r"[^\d]", "", str(price_str)) or 0)
        except (TypeError, ValueError):
            price = 0
        return {
            "name": meta.get("title") or name,
            "visit_duration": self._visit_minutes(meta.get("visit_duration"), default=90),
            "address": meta.get("address") or "",
            "location": self._location_text(meta),
            "category": meta.get("category") or "景点",
            "ticket_price": price,
            "open_time": meta.get("open_time") or "",
            "filler": True,
            "description": "补足当日空档",
        }

