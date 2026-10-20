"""把单次 LLM 调用收成 Pydantic 对象。

优先走 ChatModel.with_structured_output（function calling）。
提供方没有返回结构化结果时，再从原文里取出 JSON 做校验。
"""

from __future__ import annotations

import json
import re
from typing import Any, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _as_int(value: Any) -> int:
    if isinstance(value, bool) or value is None:
        return 0
    if isinstance(value, (int, float)):
        return int(value)
    digits = re.sub(r"[^\d]", "", str(value))
    return int(digits) if digits else 0


def _as_float(value: Any) -> float:
    if isinstance(value, bool) or value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    match = re.search(r"\d+(?:\.\d+)?", str(value))
    return float(match.group(0)) if match else 0.0


class IntentDetails(BaseModel):
    model_config = ConfigDict(extra="ignore")

    action: str = ""
    item: str = ""
    remove: str = ""
    old_item: str = ""
    new_item: str = ""


class IntentOutput(BaseModel):
    """对话修改的意图。"""

    model_config = ConfigDict(extra="ignore")

    change_type: str = "no_change"
    scope: str = "incremental"
    description: str = ""
    target_field: str = ""
    details: IntentDetails = Field(default_factory=IntentDetails)
    confidence: float = 0.8

    @field_validator("details", mode="before")
    @classmethod
    def _details(cls, value: Any) -> Any:
        if value is None or value == "":
            return {}
        return value

    @field_validator("confidence", mode="before")
    @classmethod
    def _confidence(cls, value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.8


class QueryVariants(BaseModel):
    model_config = ConfigDict(extra="ignore")

    queries: List[str] = Field(default_factory=list)

    @field_validator("queries", mode="before")
    @classmethod
    def _queries(cls, value: Any) -> List[str]:
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            return []
        return [str(item).strip() for item in value if str(item).strip()]


class AnswerCheck(BaseModel):
    model_config = ConfigDict(extra="ignore")

    grounded: bool = True
    completeness: float = 0.5
    alignment_score: float = 0.5
    issues: List[str] = Field(default_factory=list)

    @field_validator("grounded", mode="before")
    @classmethod
    def _grounded(cls, value: Any) -> bool:
        if isinstance(value, str):
            return value.strip().lower() not in {"false", "0", "no", "否"}
        return bool(value)

    @field_validator("completeness", "alignment_score", mode="before")
    @classmethod
    def _score(cls, value: Any) -> float:
        return _as_float(value)

    @field_validator("issues", mode="before")
    @classmethod
    def _issues(cls, value: Any) -> List[str]:
        if value is None or value == "":
            return []
        if isinstance(value, str):
            return [value]
        if isinstance(value, list):
            return [str(item) for item in value if str(item).strip()]
        return []


class GeoPoint(BaseModel):
    model_config = ConfigDict(extra="ignore")

    longitude: float = 0
    latitude: float = 0

    @field_validator("longitude", "latitude", mode="before")
    @classmethod
    def _coord(cls, value: Any) -> float:
        return _as_float(value)


class AttractionOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str = ""
    address: str = ""
    location: Optional[GeoPoint] = None
    visit_duration: int = 120
    description: str = Field(default="", max_length=40)
    category: str = ""
    ticket_price: int = 0

    @field_validator("description", mode="before")
    @classmethod
    def _clip_description(cls, value: Any) -> str:
        return str(value or "")[:40]

    @field_validator("location", mode="before")
    @classmethod
    def _location(cls, value: Any) -> Any:
        if isinstance(value, dict) and ("longitude" in value or "latitude" in value):
            lon = value.get("longitude")
            lat = value.get("latitude")
            if lon in (None, "", 0, 0.0) or lat in (None, "", 0, 0.0):
                return None
            return value
        return None

    @field_validator("visit_duration", "ticket_price", mode="before")
    @classmethod
    def _minutes(cls, value: Any) -> int:
        return _as_int(value)


class MealOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type: str = ""
    name: str = ""
    address: str = ""
    location: Optional[GeoPoint] = None
    description: str = Field(default="", max_length=40)
    estimated_cost: int = 0
    duration_minutes: int = 60

    @field_validator("description", mode="before")
    @classmethod
    def _clip_description(cls, value: Any) -> str:
        return str(value or "")[:40]

    @field_validator("location", mode="before")
    @classmethod
    def _location(cls, value: Any) -> Any:
        if isinstance(value, str) and "," in value:
            lon, lat = value.split(",", 1)
            return {"longitude": lon, "latitude": lat}
        if isinstance(value, dict) and ("longitude" in value or "latitude" in value):
            lon = value.get("longitude")
            lat = value.get("latitude")
            if lon in (None, "", 0, 0.0) or lat in (None, "", 0, 0.0):
                return None
            return value
        return None

    @field_validator("duration_minutes", mode="before")
    @classmethod
    def _duration(cls, value: Any) -> int:
        minutes = _as_int(value)
        return minutes if minutes > 0 else 60

    @field_validator("estimated_cost", mode="before")
    @classmethod
    def _cost(cls, value: Any) -> int:
        return _as_int(value)


class HotelOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str = ""
    address: str = ""
    location: Optional[GeoPoint] = None
    price_range: str = ""
    rating: str = ""
    distance: str = ""
    type: str = ""
    estimated_cost: int = 0

    @field_validator("location", mode="before")
    @classmethod
    def _location(cls, value: Any) -> Any:
        if isinstance(value, dict) and ("longitude" in value or "latitude" in value):
            lon = value.get("longitude")
            lat = value.get("latitude")
            if lon in (None, "", 0, 0.0) or lat in (None, "", 0, 0.0):
                return None
            return value
        return None

    @field_validator("estimated_cost", mode="before")
    @classmethod
    def _cost(cls, value: Any) -> int:
        return _as_int(value)


class WeatherOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    date: str = ""
    day_weather: str = ""
    night_weather: str = ""
    day_temp: int = 0
    night_temp: int = 0
    wind_direction: str = ""
    wind_power: str = ""

    @field_validator("day_temp", "night_temp", mode="before")
    @classmethod
    def _temp(cls, value: Any) -> int:
        return _as_int(value)


class BudgetOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    total_attractions: float = 0
    total_hotels: float = 0
    total_meals: float = 0
    total_transportation: float = 0
    total: float = 0

    @field_validator(
        "total_attractions",
        "total_hotels",
        "total_meals",
        "total_transportation",
        "total",
        mode="before",
    )
    @classmethod
    def _money(cls, value: Any) -> float:
        return _as_float(value)


class DayOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    date: str = ""
    day_index: int = 0
    description: str = Field(default="", max_length=40)
    transportation: str = ""
    accommodation: str = ""
    hotel: Optional[HotelOutput] = None
    attractions: List[AttractionOutput] = Field(default_factory=list)
    meals: List[MealOutput] = Field(default_factory=list)

    @field_validator("description", mode="before")
    @classmethod
    def _clip_description(cls, value: Any) -> str:
        return str(value or "")[:40]

    @field_validator("day_index", mode="before")
    @classmethod
    def _index(cls, value: Any) -> int:
        return _as_int(value)

    @field_validator("attractions", mode="before")
    @classmethod
    def _attractions(cls, value: Any) -> List[Any]:
        if not isinstance(value, list):
            return []
        items = []
        for item in value:
            if isinstance(item, str):
                items.append({"name": item})
            elif isinstance(item, dict):
                items.append(item)
        return items

    @field_validator("hotel", mode="before")
    @classmethod
    def _hotel(cls, value: Any) -> Any:
        if isinstance(value, str):
            return {"name": value} if value else None
        return value


class AttractionPickOutput(BaseModel):
    """从高德名单里点名。names 是景点，lunch / dinner 是餐馆原名。"""

    model_config = ConfigDict(extra="ignore")

    names: List[str] = Field(default_factory=list)
    lunch: List[str] = Field(default_factory=list)
    dinner: List[str] = Field(default_factory=list)

    @field_validator("names", "lunch", "dinner", mode="before")
    @classmethod
    def _names(cls, value: Any) -> List[str]:
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            return []
        return [str(item).strip() for item in value if str(item).strip()]


class TripPlanOutput(BaseModel):
    """主图规划节点的行程 JSON。"""

    model_config = ConfigDict(extra="ignore")

    city: str = ""
    start_date: str = ""
    end_date: str = ""
    introduction: str = Field(default="", max_length=360)
    days: List[DayOutput] = Field(default_factory=list)
    weather_info: List[WeatherOutput] = Field(default_factory=list)
    overall_suggestions: str = Field(default="", max_length=160)
    budget: Optional[BudgetOutput] = None

    @field_validator("introduction", "overall_suggestions", mode="before")
    @classmethod
    def _clip_prose(cls, value: Any, info) -> str:
        limit = 360 if info.field_name == "introduction" else 160
        return str(value or "")[:limit]

    @field_validator("budget", mode="before")
    @classmethod
    def _budget(cls, value: Any) -> Any:
        if isinstance(value, (int, float)):
            return {"total": value}
        return value


class DialogPlanUpdate(BaseModel):
    """对话里全量重生成时交回的计划。未填写的字段保持默认，合并时丢掉。"""

    model_config = ConfigDict(extra="ignore")

    city: str = ""
    duration_days: int = 0
    attractions: List[str] = Field(default_factory=list)
    transportation: str = ""
    accommodation: str = ""
    budget: Optional[BudgetOutput] = None
    daily_itinerary: Optional[TripPlanOutput] = None

    @field_validator("duration_days", mode="before")
    @classmethod
    def _days(cls, value: Any) -> int:
        return _as_int(value)

    @field_validator("attractions", mode="before")
    @classmethod
    def _names(cls, value: Any) -> List[str]:
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            return []
        names = []
        for item in value:
            if isinstance(item, str) and item.strip():
                names.append(item.strip())
            elif isinstance(item, dict) and item.get("name"):
                names.append(str(item["name"]).strip())
        return names

    @field_validator("budget", mode="before")
    @classmethod
    def _budget(cls, value: Any) -> Any:
        if isinstance(value, (int, float)):
            return {"total": value}
        return value


class SearchResultOutput(BaseModel):
    """对话检索图的最终结果。"""

    model_config = ConfigDict(extra="ignore")

    attractions: List[str] = Field(default_factory=list)
    hotel: str = ""
    hotel_price: Optional[float] = None
    summary: str = ""

    @field_validator("attractions", mode="before")
    @classmethod
    def _names(cls, value: Any) -> List[str]:
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            return []
        return [str(item).strip() for item in value if str(item).strip()]

    @field_validator("hotel_price", mode="before")
    @classmethod
    def _price(cls, value: Any) -> Optional[float]:
        if value in (None, ""):
            return None
        return _as_float(value)


class FillNameOutput(BaseModel):
    """行程补点图的最终结果。"""

    model_config = ConfigDict(extra="ignore")

    name: str = ""

    @field_validator("name", mode="before")
    @classmethod
    def _name(cls, value: Any) -> str:
        return str(value or "").strip()


class PlanReflectOutput(BaseModel):
    """根据已排时间表判断这一轮是否达标。"""

    model_config = ConfigDict(extra="ignore")

    passed: bool = False
    note: str = ""

    @field_validator("passed", mode="before")
    @classmethod
    def _passed(cls, value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value != 0
        return str(value or "").strip().lower() in {"1", "true", "yes", "y", "是", "达标", "通过"}

    @field_validator("note", mode="before")
    @classmethod
    def _note(cls, value: Any) -> str:
        return str(value or "").strip()[:200]


def message_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("text"):
                parts.append(str(block["text"]))
        return "\n".join(parts)
    return str(content)


def extract_json_dict(text: str) -> Optional[dict]:
    """从模型原文里取出一个 JSON 对象。"""
    if not text:
        return None
    raw = text.strip()
    candidates = [raw]
    for marker in ("```json", "```"):
        start = raw.find(marker)
        if start == -1:
            continue
        body = raw[start + len(marker):]
        end = body.find("```")
        if end != -1:
            candidates.append(body[:end].strip())
    first = raw.find("{")
    last = raw.rfind("}")
    if first != -1 and last > first:
        candidates.append(raw[first:last + 1])

    for item in candidates:
        try:
            data = json.loads(item)
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
        if isinstance(data, dict):
            return data
    return None


def as_dict(result: Any, *, exclude_defaults: bool = False) -> Optional[dict]:
    if isinstance(result, BaseModel):
        return result.model_dump(exclude_defaults=exclude_defaults)
    if isinstance(result, dict):
        return result
    return None


def _bind_structured(llm: Any, schema: type[BaseModel]):
    try:
        return llm.with_structured_output(
            schema,
            method="function_calling",
            include_raw=True,
        )
    except TypeError:
        return llm.with_structured_output(schema, include_raw=True)


def _coerce(schema: type[BaseModel], value: Any) -> Optional[Union[BaseModel, dict]]:
    if isinstance(value, schema):
        return value
    if isinstance(value, BaseModel):
        value = value.model_dump()
    if isinstance(value, dict):
        try:
            return schema.model_validate(value)
        except Exception:
            return value
    return None


def invoke_structured(llm: Any, schema: type[BaseModel], prompt: Any) -> Optional[Union[BaseModel, dict]]:
    """调用模型并收成 schema。失败时返回原文里的 JSON 对象，仍失败则返回 None。"""
    parsed: Any = None
    raw_text = ""
    try:
        payload = _bind_structured(llm, schema).invoke(prompt)
    except Exception:
        payload = None

    if isinstance(payload, dict) and ("parsed" in payload or "raw" in payload):
        parsed = payload.get("parsed")
        raw = payload.get("raw")
        if raw is not None:
            raw_text = message_text(getattr(raw, "content", raw))
    else:
        parsed = payload

    coerced = _coerce(schema, parsed)
    if isinstance(coerced, schema):
        return coerced

    if not raw_text:
        try:
            response = llm.invoke(prompt)
            raw_text = message_text(getattr(response, "content", response))
        except Exception:
            raw_text = ""

    data = extract_json_dict(raw_text)
    if data is None:
        return coerced if isinstance(coerced, dict) else None
    try:
        return schema.model_validate(data)
    except Exception:
        return data
