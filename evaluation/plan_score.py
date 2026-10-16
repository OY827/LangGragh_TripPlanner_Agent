"""可执行行程：天数、候选约束、坐标、时刻表。"""

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from evaluation.metrics import rate

DAY_END_MIN = 21 * 60
BEIJING_LON = 116.4
BEIJING_LAT = 39.9
BEIJING_EPS = 0.05


def score_plans(cases: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    rows = [score_plan(case) for case in cases]
    executable = sum(1 for row in rows if row["executable"])
    names = sum(row["name_count"] for row in rows)
    outside = sum(row["outside_candidate_count"] for row in rows)
    slots = sum(row["timed_slot_count"] for row in rows)
    clipped = sum(row["clipped_slot_count"] for row in rows)
    return {
        "completion_rate": rate(executable, len(rows)),
        "executable_count": executable,
        "case_count": len(rows),
        "outside_candidate_rate": rate(outside, names),
        "clipped_visit_rate": rate(clipped, slots),
        "cases": rows,
    }


def score_plan(case: Dict[str, Any]) -> Dict[str, Any]:
    plan = case.get("plan") if isinstance(case.get("plan"), dict) else {}
    travel_days = int(case.get("travel_days") or 0)
    city = str(case.get("city") or "")
    candidates = _name_set(case.get("candidate_attractions") or [])
    hotels = _name_set(case.get("candidate_hotels") or [])
    days = plan.get("days") if isinstance(plan.get("days"), list) else []
    names = _attraction_names(days)
    reasons: List[str] = []

    if len(days) != travel_days or travel_days <= 0:
        reasons.append("day_count")
    if not candidates:
        reasons.append("missing_candidates")
    outside = [name for name in names if name not in candidates]
    if outside or not names:
        reasons.append("outside_candidate")

    hotel_name = _hotel_name(plan, days)
    if hotels and hotel_name not in hotels:
        reasons.append("hotel_not_in_candidates")
    if hotel_name and not _hotel_geocoded(plan, days, city):
        reasons.append("hotel_not_geocoded")
    if _has_beijing_default(plan, days, city):
        reasons.append("beijing_default_coord")

    timed, clipped, untimed, past_close = _timeline_issues(days)
    if untimed:
        reasons.append("missing_timeline")
    if clipped:
        reasons.append("clipped_visit")
    if past_close:
        reasons.append("past_21_00")

    return {
        "id": case.get("id", ""),
        "executable": not reasons,
        "reasons": reasons,
        "name_count": len(names),
        "outside_candidate_count": len(outside),
        "timed_slot_count": timed,
        "clipped_slot_count": clipped,
    }


def _attraction_names(days: Sequence[Any]) -> List[str]:
    names: List[str] = []
    for day in days:
        if not isinstance(day, dict):
            continue
        for item in day.get("attractions") or []:
            if isinstance(item, dict):
                name = str(item.get("name") or "").strip()
            else:
                name = str(item).strip()
            if name:
                names.append(name)
    return names


def _hotel_name(plan: Dict[str, Any], days: Sequence[Any]) -> str:
    hotel = plan.get("hotel")
    if isinstance(hotel, dict) and str(hotel.get("name") or "").strip():
        return str(hotel["name"]).strip()
    accommodation = plan.get("accommodation")
    if isinstance(accommodation, str) and accommodation.strip():
        return accommodation.strip()
    for day in days:
        if isinstance(day, dict):
            day_hotel = day.get("hotel")
            if isinstance(day_hotel, dict) and str(day_hotel.get("name") or "").strip():
                return str(day_hotel["name"]).strip()
    return ""


def _hotel_geocoded(plan: Dict[str, Any], days: Sequence[Any], city: str) -> bool:
    points = []
    hotel = plan.get("hotel")
    if isinstance(hotel, dict):
        points.append(hotel.get("location"))
    for day in days:
        if isinstance(day, dict) and isinstance(day.get("hotel"), dict):
            points.append(day["hotel"].get("location"))
    parsed = [parse_lon_lat(point) for point in points]
    parsed = [point for point in parsed if point is not None]
    if not parsed:
        return False
    return all(not _bad_coord(lon, lat, city) for lon, lat in parsed)


def _has_beijing_default(plan: Dict[str, Any], days: Sequence[Any], city: str) -> bool:
    for lon, lat in _iter_coords(plan, days):
        if _is_beijing_default(lon, lat) and city != "北京":
            return True
    return False


def _iter_coords(plan: Dict[str, Any], days: Sequence[Any]) -> Iterable[Tuple[float, float]]:
    blobs: List[Any] = [plan.get("hotel")]
    for day in days:
        if not isinstance(day, dict):
            continue
        blobs.append(day.get("hotel"))
        blobs.extend(day.get("attractions") or [])
    for blob in blobs:
        if not isinstance(blob, dict):
            continue
        parsed = parse_lon_lat(blob.get("location"))
        if parsed is not None:
            yield parsed


def _timeline_issues(days: Sequence[Any]) -> Tuple[int, int, int, int]:
    timed = clipped = untimed = past_close = 0
    for day in days:
        if not isinstance(day, dict):
            continue
        for item in day.get("attractions") or []:
            if not isinstance(item, dict):
                untimed += 1
                continue
            duration = _minutes(item.get("visit_duration"))
            start = _parse_hhmm(item.get("start_time"))
            end = _parse_hhmm(item.get("end_time"))
            if duration <= 0 or start < 0 or end < 0:
                untimed += 1
                continue
            timed += 1
            if end - start < duration:
                clipped += 1
            if end > DAY_END_MIN:
                past_close += 1
    return timed, clipped, untimed, past_close


def parse_lon_lat(location: Any) -> Optional[Tuple[float, float]]:
    if isinstance(location, dict):
        lon = location.get("longitude", location.get("lng"))
        lat = location.get("latitude", location.get("lat"))
    elif isinstance(location, str) and "," in location:
        lon, lat = location.split(",", 1)
    else:
        return None
    try:
        return float(lon), float(lat)
    except (TypeError, ValueError):
        return None


def _bad_coord(lon: float, lat: float, city: str) -> bool:
    if abs(lon) < 1e-6 and abs(lat) < 1e-6:
        return True
    return _is_beijing_default(lon, lat) and city != "北京"


def _is_beijing_default(lon: float, lat: float) -> bool:
    return abs(lon - BEIJING_LON) <= BEIJING_EPS and abs(lat - BEIJING_LAT) <= BEIJING_EPS


def _parse_hhmm(value: Any) -> int:
    text = str(value or "")
    if ":" not in text:
        return -1
    hour, minute = text.split(":", 1)
    try:
        return int(hour) * 60 + int(minute)
    except ValueError:
        return -1


def _minutes(value: Any) -> int:
    try:
        minutes = int(float(value))
    except (TypeError, ValueError):
        return 0
    return minutes if minutes > 0 else 0


def _name_set(names: Iterable[Any]) -> set:
    return {str(name).strip() for name in names if str(name).strip()}
