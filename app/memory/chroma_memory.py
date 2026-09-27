"""
用户长期记忆：本地 Chroma。

存用户偏好、去过的景点、历史行程摘要，供语义检索和精确过滤。
与攻略知识库（travel_knowledge / travel_guides）分集合，互不混写。
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from app.rag.embeddings import build_embeddings


MEMORY_TYPE_TRIP = "trip"
MEMORY_TYPE_VISITED = "visited_attraction"
MEMORY_TYPE_PREFERENCE = "preference"


def _project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _as_float_list(values: Any) -> List[float]:
    return [float(x) for x in values]


def _json_dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False) if value is not None else ""


def _json_load(value: Any, default: Any):
    if value in (None, ""):
        return default
    if isinstance(value, (list, dict)):
        return value
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return default


def _stable_id(*parts: Any) -> str:
    raw = "|".join(str(p) for p in parts)
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def _where(user_id: str, extra: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    conditions = [{"user_id": {"$eq": user_id}}]
    for key, value in (extra or {}).items():
        if value is not None and value != "":
            conditions.append({key: {"$eq": value}})
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


class ChromaUserMemory:
    """按 user_id 持久化长期记忆，默认集合 user_memory。"""

    def __init__(
        self,
        persist_directory: Optional[str] = None,
        collection_name: Optional[str] = None,
        embeddings: Any = None,
    ):
        default_path = os.path.join(_project_root(), "app", "data", "chroma")
        self.persist_directory = persist_directory or os.getenv("CHROMA_PATH", default_path)
        os.makedirs(self.persist_directory, exist_ok=True)
        self.collection_name = (
            collection_name
            or os.getenv("CHROMA_USER_COLLECTION", "user_memory")
        )

        if embeddings is not None:
            self.embeddings = embeddings
        else:
            backend = os.getenv("CHROMA_EMBEDDING_BACKEND") or os.getenv("EMBEDDING_BACKEND", "dashscope")
            self.embeddings, _, _ = build_embeddings(backend)

        import chromadb

        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def _embed_one(self, text: str) -> List[float]:
        return _as_float_list(self.embeddings.embed_query(text))

    def _embed_many(self, texts: List[str]) -> List[List[float]]:
        if hasattr(self.embeddings, "embed_documents"):
            return [_as_float_list(item) for item in self.embeddings.embed_documents(texts)]
        return [self._embed_one(text) for text in texts]

    def _upsert(self, ids: List[str], documents: List[str], metadatas: List[Dict[str, Any]]) -> None:
        if not ids:
            return
        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=self._embed_many(documents),
        )

    def _get(self, user_id: str, extra: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
        try:
            raw = self.collection.get(
                where=_where(user_id, extra),
                include=["documents", "metadatas"],
            )
        except Exception:
            return []
        ids = raw.get("ids") or []
        documents = raw.get("documents") or []
        metadatas = raw.get("metadatas") or []
        items = []
        for doc_id, content, metadata in zip(ids, documents, metadatas):
            items.append({
                "id": doc_id,
                "content": content or "",
                "metadata": dict(metadata or {}),
            })
        return items

    def save_plan(
        self,
        user_id: str,
        city: str,
        duration_days: int,
        attractions: List[str],
        daily_itinerary: Dict[str, List[str]],
        preferences: Optional[List[str]] = None,
        budget: Optional[Union[int, Dict[str, Any]]] = None,
        notes: Optional[str] = None,
        plan_id: Optional[str] = None,
        created_at: Optional[str] = None,
    ) -> str:
        plan_id = plan_id or str(uuid.uuid4())[:8]
        created_at = created_at or datetime.now().isoformat()
        preferences = list(dict.fromkeys(
            [(item or "").strip() for item in (preferences or []) if (item or "").strip()]
        ))
        attractions = list(dict.fromkeys([name for name in (attractions or []) if name]))

        content = (
            f"用户{user_id}曾在{city}旅行{duration_days}天。"
            f"去过景点：{'、'.join(attractions) or '无'}。"
            f"偏好：{'、'.join(preferences) or '无'}。"
            f"{notes or ''}"
        ).strip()

        trip_meta = {
            "user_id": user_id,
            "city": city or "",
            "memory_type": MEMORY_TYPE_TRIP,
            "plan_id": plan_id,
            "duration_days": int(duration_days or 1),
            "created_at": created_at,
            "attractions_json": _json_dump(attractions),
            "daily_itinerary_json": _json_dump(daily_itinerary or {}),
            "preferences_json": _json_dump(preferences),
            "budget_json": _json_dump(budget),
            "notes": notes or "",
            "title": f"{city}行程",
        }
        ids = [_stable_id("trip", user_id, plan_id)]
        documents = [content]
        metadatas = [trip_meta]

        for attraction in attractions:
            ids.append(_stable_id("visited", user_id, city, attraction))
            documents.append(f"用户{user_id}去过{city}的{attraction}")
            metadatas.append({
                "user_id": user_id,
                "city": city or "",
                "memory_type": MEMORY_TYPE_VISITED,
                "plan_id": plan_id,
                "attraction": attraction,
                "created_at": created_at,
                "title": attraction,
            })

        for pref in preferences:
            pref = (pref or "").strip()
            if not pref:
                continue
            ids.append(_stable_id("pref", user_id, pref))
            documents.append(f"用户{user_id}的旅行偏好：{pref}。相关城市：{city}")
            metadatas.append({
                "user_id": user_id,
                "city": city or "",
                "memory_type": MEMORY_TYPE_PREFERENCE,
                "preference": pref,
                "plan_id": plan_id,
                "created_at": created_at,
                "title": pref,
            })

        self._upsert(ids, documents, metadatas)
        return plan_id

    def add_preference(self, user_id: str, preference: str, city: str = "") -> str:
        preference = (preference or "").strip()
        if not preference:
            return ""
        created_at = datetime.now().isoformat()
        doc_id = _stable_id("pref", user_id, preference)
        self._upsert(
            [doc_id],
            [f"用户{user_id}的旅行偏好：{preference}。相关城市：{city or '不限'}"],
            [{
                "user_id": user_id,
                "city": city or "",
                "memory_type": MEMORY_TYPE_PREFERENCE,
                "preference": preference,
                "plan_id": "",
                "created_at": created_at,
                "title": preference,
            }],
        )
        return doc_id

    def add_visited_attraction(self, user_id: str, city: str, attraction: str) -> str:
        attraction = (attraction or "").strip()
        if not attraction:
            return ""
        created_at = datetime.now().isoformat()
        doc_id = _stable_id("visited", user_id, city, attraction)
        self._upsert(
            [doc_id],
            [f"用户{user_id}去过{city}的{attraction}"],
            [{
                "user_id": user_id,
                "city": city or "",
                "memory_type": MEMORY_TYPE_VISITED,
                "plan_id": "",
                "attraction": attraction,
                "created_at": created_at,
                "title": attraction,
            }],
        )
        return doc_id

    def _trip_from_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        meta = item.get("metadata") or {}
        return {
            "plan_id": meta.get("plan_id", ""),
            "user_id": meta.get("user_id", ""),
            "city": meta.get("city", ""),
            "duration_days": int(meta.get("duration_days") or 1),
            "created_at": meta.get("created_at", ""),
            "attractions": _json_load(meta.get("attractions_json"), []),
            "daily_itinerary": _json_load(meta.get("daily_itinerary_json"), {}),
            "preferences": _json_load(meta.get("preferences_json"), []),
            "budget": _json_load(meta.get("budget_json"), None),
            "notes": meta.get("notes") or "",
        }

    def get_plan(self, user_id: str, plan_id: str) -> Optional[Dict[str, Any]]:
        items = self._get(user_id, {"memory_type": MEMORY_TYPE_TRIP, "plan_id": plan_id})
        if not items:
            return None
        return self._trip_from_item(items[0])

    def get_user_plans(self, user_id: str) -> List[Dict[str, Any]]:
        plans = [self._trip_from_item(item) for item in self._get(user_id, {"memory_type": MEMORY_TYPE_TRIP})]
        plans.sort(key=lambda p: p.get("created_at") or "", reverse=True)
        return plans

    def get_user_plans_by_city(self, user_id: str, city: str) -> List[Dict[str, Any]]:
        return [plan for plan in self.get_user_plans(user_id) if plan.get("city") == city]

    def get_visited_attractions(self, user_id: str, city: str) -> List[str]:
        items = self._get(user_id, {"memory_type": MEMORY_TYPE_VISITED, "city": city})
        visited = {item["metadata"].get("attraction") for item in items if item.get("metadata", {}).get("attraction")}
        if not visited:
            for plan in self.get_user_plans_by_city(user_id, city):
                visited.update(plan.get("attractions") or [])
        return sorted(visited)

    def get_preferences(self, user_id: str, city: str = "") -> List[str]:
        extra = {"memory_type": MEMORY_TYPE_PREFERENCE}
        if city:
            extra["city"] = city
        items = self._get(user_id, extra)
        prefs = []
        seen = set()
        for item in items:
            value = (item.get("metadata") or {}).get("preference") or ""
            if value and value not in seen:
                seen.add(value)
                prefs.append(value)
        return prefs

    def compare_plans(self, user_id: str, city: str) -> Dict[str, Any]:
        plans = self.get_user_plans_by_city(user_id, city)
        if not plans:
            return {
                "has_history": False,
                "plan_count": 0,
                "visited_attractions": [],
                "visited_count": 0,
                "recommended_new_attractions": [],
                "preferences": self.get_preferences(user_id, city),
            }
        visited = self.get_visited_attractions(user_id, city)
        return {
            "has_history": True,
            "plan_count": len(plans),
            "latest_plan_date": plans[0].get("created_at"),
            "visited_attractions": visited,
            "visited_count": len(visited),
            "preferences": self.get_preferences(user_id, city),
            "recommended_new_attractions": [],
        }

    def search(
        self,
        user_id: str,
        query: str,
        k: int = 5,
        memory_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        extra = {"memory_type": memory_type} if memory_type else None
        where = _where(user_id, extra)
        try:
            raw = self.collection.query(
                query_embeddings=[self._embed_one(query)],
                n_results=max(1, k),
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        except Exception:
            return []

        documents = (raw.get("documents") or [[]])[0]
        metadatas = (raw.get("metadatas") or [[]])[0]
        distances = (raw.get("distances") or [[]])[0]
        ids = (raw.get("ids") or [[]])[0]
        results = []
        for doc_id, content, metadata, distance in zip(ids, documents, metadatas, distances):
            score = 1.0 - float(distance) if distance is not None else 0.0
            results.append({
                "id": doc_id,
                "content": content or "",
                "metadata": dict(metadata or {}),
                "score": max(0.0, score),
            })
        return results

    def delete_plan(self, user_id: str, plan_id: str) -> bool:
        items = self._get(user_id, {"plan_id": plan_id})
        ids = [item["id"] for item in items]
        if not ids:
            return False
        self.collection.delete(ids=ids)
        return True

    def clear_user_plans(self, user_id: str) -> int:
        items = self._get(user_id)
        ids = [item["id"] for item in items]
        if ids:
            self.collection.delete(ids=ids)
        return len(ids)

    def get_storage_type(self) -> str:
        return "chroma"
