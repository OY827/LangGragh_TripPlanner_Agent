"""高德景点候选、知识库介绍、飞猪门票。

名单只来自高德。知识库按景点名补介绍和游览时长，按城市补概述。
飞猪只补门票价格。筛选后若数量不够，按偏好排序从原名单补足。
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

_KB_ROOT = Path(__file__).resolve().parents[1] / "knowledge_base"
_INTRO_LIMIT = 420
_CITY_LIMIT = 700


def attraction_limits(travel_days: int) -> tuple:
    """候选池每天 4 到 6 个。规划再按游览时长取用，短景点多排几个才能到晚上。"""
    days = max(1, int(travel_days or 1))
    return days * 4, days * 6


def minimum_attraction_count(travel_days: int, available: int) -> int:
    """候选尽量给满每天 6 个；高德不够时全留。"""
    _low, high = attraction_limits(travel_days)
    return max(0, min(int(available or 0), high))


def expand_keywords(query: str, preferences: Optional[List[str]] = None) -> List[str]:
    keyword_map = {
        "历史文化": ["古迹", "博物馆", "古镇"],
        "自然风光": ["公园", "园林", "山水"],
        "主题乐园": ["乐园", "海洋馆", "动物园"],
        "艺术文化": ["美术馆", "博物馆", "展览"],
        "购物休闲": ["步行街", "商场"],
        "亲子": ["动物园", "海洋馆", "乐园"],
    }
    text = f"{query or ''} {' '.join(preferences or [])}"
    matched: List[str] = []
    for words in keyword_map.values():
        for word in words:
            if word in text and word not in matched:
                matched.append(word)
                break
    if not matched:
        if any(token in text for token in ("历史", "古迹", "文化", "博物馆")):
            matched = ["古迹", "博物馆", "古镇"]
        elif any(token in text for token in ("自然", "风景", "公园")):
            matched = ["公园", "园林", "山水"]
        else:
            matched = ["景点", "名胜", "热门"]
    if "景点" not in matched:
        matched.append("景点")
    return matched


def record_from_poi(poi: Any) -> Dict[str, Any]:
    if isinstance(poi, dict):
        name = str(poi.get("name") or "").strip()
        address = str(poi.get("address") or "").strip()
        location = poi.get("location") or ""
        if isinstance(location, dict):
            lon = location.get("longitude")
            lat = location.get("latitude")
            location = f"{lon},{lat}" if lon is not None and lat is not None else ""
        poi_type = str(poi.get("type") or poi.get("poi_type") or "").strip()
    else:
        name = str(getattr(poi, "name", "") or "").strip()
        address = str(getattr(poi, "address", "") or "").strip()
        location = ""
        loc = getattr(poi, "location", None)
        if loc is not None and hasattr(loc, "longitude") and hasattr(loc, "latitude"):
            location = f"{loc.longitude},{loc.latitude}"
        elif loc:
            location = str(loc)
        poi_type = str(getattr(poi, "type", "") or "").strip()
    return {
        "name": name,
        "address": address,
        "location": str(location or ""),
        "type": poi_type,
    }


def gather_amap_pois(
    search_poi: Callable[..., List[Any]],
    city: str,
    query: str = "",
    preferences: Optional[List[str]] = None,
    limit: int = 24,
) -> List[Dict[str, Any]]:
    seen = set()
    records: List[Dict[str, Any]] = []
    for keyword in expand_keywords(query, preferences)[:4]:
        try:
            results = search_poi(keywords=keyword, city=city) or []
        except TypeError:
            results = search_poi(keyword, city) or []
        except Exception:
            results = []
        for poi in results:
            record = record_from_poi(poi)
            name = record["name"]
            if not name or name in seen:
                continue
            seen.add(name)
            records.append(record)
            if len(records) >= limit:
                return records
    return records


def search_keyword(
    search_poi: Callable[..., List[Any]],
    city: str,
    keyword: str,
    limit: int = 10,
) -> List[Dict[str, Any]]:
    """按智能体给出的关键词查一次高德，不另做词扩展。"""
    keyword = (keyword or "景点").strip() or "景点"
    try:
        results = search_poi(keywords=keyword, city=city) or []
    except TypeError:
        try:
            results = search_poi(keyword, city) or []
        except Exception:
            results = []
    except Exception:
        results = []
    records: List[Dict[str, Any]] = []
    seen = set()
    for poi in results:
        record = record_from_poi(poi)
        name = record["name"]
        if not name or name in seen:
            continue
        seen.add(name)
        records.append(record)
        if len(records) >= limit:
            break
    return records


def names_from_agent_text(text: str) -> List[str]:
    raw = (text or "").strip()
    if not raw:
        return []
    data: Any = None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            try:
                data = json.loads(raw[start:end + 1])
            except json.JSONDecodeError:
                data = None
    if isinstance(data, dict):
        items = data.get("names") or []
    elif isinstance(data, list):
        items = data
    else:
        return []
    return [str(item).strip() for item in items if str(item).strip()]


def _norm(name: str) -> str:
    return re.sub(r"\s+", "", name or "")


def rank_by_preference(
    pois: List[Dict[str, Any]],
    preferences: Optional[List[str]] = None,
    query: str = "",
) -> List[Dict[str, Any]]:
    words = [item for item in expand_keywords(query, preferences) if item != "景点"]
    pref_text = " ".join(preferences or [])

    def score(poi: Dict[str, Any]) -> int:
        blob = f"{poi.get('name', '')} {poi.get('type', '')}"
        hits = sum(1 for word in words if word and word in blob)
        if pref_text and any(part and part in blob for part in pref_text.split()):
            hits += 1
        return hits

    ranked = sorted(enumerate(pois), key=lambda pair: (-score(pair[1]), pair[0]))
    return [poi for _, poi in ranked]


def select_names(
    pois: List[Dict[str, Any]],
    chosen_names: Optional[List[str]],
    travel_days: int,
) -> List[Dict[str, Any]]:
    """按模型点名保留，数量不够时用原顺序补足。"""
    by_name = {poi.get("name"): poi for poi in pois if poi.get("name")}
    selected: List[Dict[str, Any]] = []
    used = set()
    for name in chosen_names or []:
        poi = by_name.get(str(name).strip())
        if poi is None or poi["name"] in used:
            continue
        selected.append(poi)
        used.add(poi["name"])
    _low, high = attraction_limits(travel_days)
    selected = selected[:high]
    used = {poi["name"] for poi in selected}
    target = minimum_attraction_count(travel_days, len(pois))
    if len(selected) < target:
        for poi in pois:
            if poi.get("name") in used:
                continue
            selected.append(poi)
            used.add(poi["name"])
            if len(selected) >= target:
                break
    return selected


_DINNER_CUISINES = (
    "火锅", "烧烤", "海鲜", "本帮菜", "小吃", "西餐", "日料",
    "川菜", "粤菜", "素食", "面", "烤鸭", "小笼",
)


def dining_keywords(preferences: Optional[List[str]] = None, free_text: str = "") -> tuple:
    """按餐饮偏好给出午饭、晚饭的搜索词。智能体每次只用其中一个。"""
    text = f"{' '.join(preferences or [])} {free_text or ''}"
    lunch: List[str] = []
    dinner: List[str] = []
    for word in _DINNER_CUISINES:
        if word in text:
            if word not in lunch:
                lunch.append(word)
            if word not in dinner:
                dinner.append(word)
    if "美食" in text:
        for word in ("特色菜", "小吃"):
            if word not in lunch:
                lunch.append(word)
            if word not in dinner:
                dinner.append(word)
    if "历史文化" in text:
        lunch.append("老字号")
        dinner.append("老字号")
    if "亲子" in text:
        lunch.append("家庭餐厅")
        dinner.append("家庭餐厅")
    if "购物" in text:
        lunch.append("商场餐厅")
        dinner.append("商场餐厅")
    if "自然风光" in text:
        lunch.append("农家菜")
        dinner.append("农家菜")
    if "休闲" in text:
        if "餐厅" not in lunch:
            lunch.append("餐厅")
        if "餐厅" not in dinner:
            dinner.append("餐厅")
    if not lunch:
        lunch = ["午餐", "简餐"]
    if not dinner:
        dinner = ["特色菜", "餐厅"]
    return lunch[:2], dinner[:2]


def tag_restaurant(record: Dict[str, Any], meal: str) -> Dict[str, Any]:
    item = dict(record)
    item["meal"] = "dinner" if str(meal).strip().lower() == "dinner" else "lunch"
    item["description"] = item.get("type") or ""
    item["estimated_cost"] = 0
    item["duration_minutes"] = 60
    return item


def select_restaurants(
    pois: List[Dict[str, Any]],
    chosen_names: Optional[List[str]],
    meal: str,
    travel_days: int,
) -> List[Dict[str, Any]]:
    """按餐次保留工具返回过的餐馆，不够时从同一次搜索结果里补。"""
    tagged = [poi for poi in pois if poi.get("meal") == meal and poi.get("name")]
    by_name = {poi["name"]: poi for poi in tagged}
    selected: List[Dict[str, Any]] = []
    used = set()
    for name in chosen_names or []:
        poi = by_name.get(str(name).strip())
        if poi is None or poi["name"] in used:
            continue
        selected.append(poi)
        used.add(poi["name"])
    target = min(len(tagged), max(1, int(travel_days or 1)))
    if len(selected) < target:
        for poi in tagged:
            if poi["name"] in used:
                continue
            selected.append(poi)
            used.add(poi["name"])
            if len(selected) >= target:
                break
    return selected


def format_restaurants(items: List[Dict[str, Any]]) -> str:
    groups = (("lunch", "午饭"), ("dinner", "晚饭"))
    lines: List[str] = []
    for meal, title in groups:
        group = [item for item in items if item.get("meal") == meal]
        if not group:
            continue
        lines.append(f"## {title}")
        for index, item in enumerate(group, 1):
            lines.append(f"{index}. {item.get('name') or '未知'}")
            if item.get("address"):
                lines.append(f"   地址: {item['address']}")
            if item.get("location"):
                lines.append(f"   坐标: {item['location']}")
            if item.get("type"):
                lines.append(f"   类型: {item['type']}")
            lines.append("   用餐: 60分钟")
            lines.append("   来源: 高德")
            lines.append("")
    return "\n".join(lines).strip()


def picks_from_agent_text(text: str) -> Dict[str, List[str]]:
    raw = (text or "").strip()
    data: Any = None
    if raw:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            start = raw.find("{")
            end = raw.rfind("}")
            if start >= 0 and end > start:
                try:
                    data = json.loads(raw[start:end + 1])
                except json.JSONDecodeError:
                    data = None
    if not isinstance(data, dict):
        return {"names": names_from_agent_text(text), "lunch": [], "dinner": []}

    def _list(key: str) -> List[str]:
        value = data.get(key) or []
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            return []
        return [str(item).strip() for item in value if str(item).strip()]

    return {
        "names": _list("names"),
        "lunch": _list("lunch"),
        "dinner": _list("dinner"),
    }


def extract_intro(content: str, limit: int = _INTRO_LIMIT) -> str:
    text = content or ""
    marker = "景区介绍："
    index = text.find(marker)
    if index < 0:
        return ""
    rest = text[index + len(marker):]
    for stop in ("\n优待政策", "\n服务设施"):
        cut = rest.find(stop)
        if cut >= 0:
            rest = rest[:cut]
            break
    cleaned = re.sub(r"\s+", " ", rest).strip()
    if len(cleaned) > limit:
        return cleaned[:limit] + "..."
    return cleaned


@lru_cache(maxsize=32)
def _city_index(city: str) -> Dict[str, Dict[str, Any]]:
    path = _KB_ROOT / "attractions" / f"{city}.json"
    if not path.is_file():
        return {}
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    index: Dict[str, Dict[str, Any]] = {}
    for row in rows if isinstance(rows, list) else []:
        meta = row.get("metadata") if isinstance(row, dict) else None
        if not isinstance(meta, dict):
            continue
        title = _norm(str(meta.get("title") or ""))
        if title:
            index[title] = row
    return index


def lookup_profile(city: str, name: str) -> Dict[str, Any]:
    index = _city_index(city)
    key = _norm(name)
    row = index.get(key)
    if row is None and len(key) >= 2:
        ranked = []
        for title, item in index.items():
            if key in title or title in key:
                ranked.append((abs(len(title) - len(key)), item))
        if ranked:
            ranked.sort(key=lambda pair: pair[0])
            row = ranked[0][1]
    if not isinstance(row, dict):
        return {}
    meta = row.get("metadata") or {}
    duration = meta.get("visit_duration")
    try:
        duration = int(duration) if duration else None
    except (TypeError, ValueError):
        duration = None
    try:
        reference_price = int(float(meta.get("price")))
    except (TypeError, ValueError):
        reference_price = None
    return {
        "description": extract_intro(row.get("content") or ""),
        "visit_duration": duration,
        "category": meta.get("category") or "",
        "open_time": meta.get("open_time") or "",
        "reference_price": reference_price,
    }


def load_city_description(city: str, limit: int = _CITY_LIMIT) -> str:
    path = _KB_ROOT / "guides" / f"{city}.json"
    if not path.is_file():
        return ""
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    overview = ""
    fallback = ""
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict):
            continue
        meta = row.get("metadata") or {}
        content = re.sub(r"\s+", " ", str(row.get("content") or "")).strip()
        if not fallback and content:
            fallback = content
        section = str(meta.get("section") or meta.get("title") or "")
        if section == "概述" or "概述" in content[:40]:
            overview = content
            break
    text = overview or fallback
    if len(text) > limit:
        return text[:limit] + "..."
    return text


def _flyai_quote(city: str, name: str, depart_date: str = "") -> Dict[str, Any]:
    del depart_date
    from app.services.flyai_ticket_service import get_flyai_ticket_service

    service = get_flyai_ticket_service()
    if not service.is_configured():
        return {"price": None, "jump_url": ""}
    try:
        return service.quote_for(city, name)
    except Exception:
        return {"price": None, "jump_url": ""}


def _as_quote(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        price = value.get("price")
        try:
            price = int(price) if price not in (None, "") else None
        except (TypeError, ValueError):
            price = None
        return {"price": price, "jump_url": str(value.get("jump_url") or "")}
    if value in (None, ""):
        return {"price": None, "jump_url": ""}
    try:
        return {"price": int(value), "jump_url": ""}
    except (TypeError, ValueError):
        return {"price": None, "jump_url": ""}


def enrich_attractions(
    pois: List[Dict[str, Any]],
    city: str,
    depart_date: str = "",
    ticket_lookup: Optional[Callable[..., Any]] = None,
) -> List[Dict[str, Any]]:
    lookup = ticket_lookup if ticket_lookup is not None else _flyai_quote
    enriched: List[Dict[str, Any]] = []
    quotes: Dict[str, Dict[str, Any]] = {}
    names = [poi.get("name") or "" for poi in pois]
    use_ticket = ticket_lookup is not None or _flyai_ready()
    if use_ticket and names:
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {
                pool.submit(lookup, city, name, depart_date): name
                for name in names
                if name
            }
            for future in as_completed(futures):
                name = futures[future]
                try:
                    quotes[name] = _as_quote(future.result())
                except Exception:
                    quotes[name] = {"price": None, "jump_url": ""}
    for poi in pois:
        profile = lookup_profile(city, poi.get("name") or "")
        quote = quotes.get(poi.get("name") or "", {"price": None, "jump_url": ""})
        flyai_price = quote.get("price")
        reference_price = profile.get("reference_price")
        item = dict(poi)
        item["description"] = profile.get("description") or ""
        item["visit_duration"] = profile.get("visit_duration")
        item["category"] = profile.get("category") or poi.get("type") or ""
        item["open_time"] = profile.get("open_time") or ""
        item["jump_url"] = quote.get("jump_url") or ""
        item["reference_price"] = reference_price
        if flyai_price is not None:
            item["ticket_price"] = flyai_price
            item["ticket_source"] = "flyai"
            item["needs_user_check"] = False
        else:
            item["ticket_price"] = reference_price
            item["ticket_source"] = "knowledge_base" if reference_price is not None else "unknown"
            item["needs_user_check"] = True
        item["source"] = "高德"
        enriched.append(item)
    return enriched


def _flyai_ready() -> bool:
    try:
        from app.services.flyai_ticket_service import get_flyai_ticket_service

        return get_flyai_ticket_service().is_configured()
    except Exception:
        return False


def format_candidates(items: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for index, item in enumerate(items, 1):
        lines.append(f"{index}. {item.get('name') or '未知'}")
        if item.get("address"):
            lines.append(f"   地址: {item['address']}")
        if item.get("location"):
            lines.append(f"   坐标: {item['location']}")
        if item.get("type"):
            lines.append(f"   类型: {item['type']}")
        if item.get("category"):
            lines.append(f"   分类: {item['category']}")
        if item.get("description"):
            lines.append(f"   简介: {item['description']}")
        duration = item.get("visit_duration")
        if duration:
            lines.append(f"   建议游览: {duration}分钟")
        if item.get("open_time"):
            lines.append(f"   开放时间: {item['open_time']}")
        if "ticket_price" in item:
            price = item.get("ticket_price")
            source = item.get("ticket_source")
            if price not in (None, "") and source == "knowledge_base":
                lines.append(f"   票价: {price}元（知识库参考，飞猪未返回实价）")
            elif price not in (None, ""):
                lines.append(f"   票价: {price}元")
            else:
                lines.append("   票价: 未查到")
        if item.get("jump_url"):
            lines.append(f"   详情: {item['jump_url']}")
        lines.append("   来源: 高德")
        lines.append("")
    return "\n".join(lines).strip()


def price_checks_for_plan(plan: Dict[str, Any], candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """计划里飞猪没有实价的景点，附上知识库参考价和详情链接。"""
    names: List[str] = []
    for day in plan.get("days") or []:
        if not isinstance(day, dict):
            continue
        for attraction in day.get("attractions") or []:
            if isinstance(attraction, dict):
                name = str(attraction.get("name") or "").strip()
            else:
                name = str(attraction or "").strip()
            if name and name not in names:
                names.append(name)
    checks: List[Dict[str, Any]] = []
    for name in names:
        item = _match_candidate(candidates, name)
        if not item or not item.get("needs_user_check"):
            continue
        checks.append({
            "name": item.get("name") or name,
            "reference_price": item.get("reference_price"),
            "jump_url": item.get("jump_url") or "",
        })
    return checks


def _match_candidate(candidates: List[Dict[str, Any]], name: str) -> Optional[Dict[str, Any]]:
    target = _norm(name)
    exact = None
    closest = None
    closest_gap = None
    for item in candidates:
        title = _norm(str(item.get("name") or ""))
        if not title:
            continue
        if title == target:
            exact = item
            break
        if len(target) >= 2 and (target in title or title in target):
            gap = abs(len(title) - len(target))
            if closest_gap is None or gap < closest_gap:
                closest = item
                closest_gap = gap
    return exact or closest


def build_recommendation(
    search_poi: Callable[..., List[Any]],
    city: str,
    query: str = "",
    preferences: Optional[List[str]] = None,
    travel_days: int = 1,
    depart_date: str = "",
    chosen_names: Optional[List[str]] = None,
    visited: Optional[List[str]] = None,
    ticket_lookup: Optional[Callable[[str, str, str], Optional[int]]] = None,
) -> Dict[str, Any]:
    pois = gather_amap_pois(search_poi, city, query, preferences)
    blocked = set(visited or [])
    pois = [poi for poi in pois if poi.get("name") not in blocked]
    ranked = rank_by_preference(pois, preferences, query)
    selected = select_names(ranked, chosen_names, travel_days)
    enriched = enrich_attractions(selected, city, depart_date, ticket_lookup=ticket_lookup)
    return {
        "amap_pois": pois,
        "attraction_candidates": enriched,
        "city_description": load_city_description(city),
        "text": format_candidates(enriched),
    }
