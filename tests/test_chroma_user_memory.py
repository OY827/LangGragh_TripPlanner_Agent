"""本地 Chroma 用户长期记忆（FakeEmbeddings，不打真实 API）。"""

import tempfile
import uuid

from langchain_core.embeddings.fake import FakeEmbeddings

from app.memory.chroma_memory import ChromaUserMemory
from app.memory.memory_manager import TravelMemoryManager


def _memory(tmp_dir: str) -> ChromaUserMemory:
    return ChromaUserMemory(
        persist_directory=tmp_dir,
        collection_name=f"user_memory_{uuid.uuid4().hex[:8]}",
        embeddings=FakeEmbeddings(size=16),
    )


def test_chroma_saves_preferences_and_visited_attractions():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        store = _memory(tmp)
        user_id = "user_chroma_1"
        plan_id = store.save_plan(
            user_id=user_id,
            city="上海",
            duration_days=2,
            attractions=["外滩", "豫园"],
            daily_itinerary={"day0": ["外滩"], "day1": ["豫园"]},
            preferences=["少走路", "夜景"],
            budget={"total": 1200},
        )
        assert plan_id
        assert store.get_visited_attractions(user_id, "上海") == ["外滩", "豫园"]
        prefs = store.get_preferences(user_id)
        assert "少走路" in prefs
        assert "夜景" in prefs

        store.add_preference(user_id, "怕热", city="上海")
        assert "怕热" in store.get_preferences(user_id, "上海")

        comparison = store.compare_plans(user_id, "上海")
        assert comparison["has_history"] is True
        assert comparison["visited_count"] == 2

        hits = store.search(user_id, "夜景和少走路", k=3)
        assert hits
        types = {item["metadata"].get("memory_type") for item in hits}
        assert types & {"preference", "trip", "visited_attraction"}


def test_travel_memory_manager_file_backend_still_works():
    manager = TravelMemoryManager(use_redis=False)
    assert manager.get_storage_type() == "file"
    user_id = f"file_user_{uuid.uuid4().hex[:6]}"
    try:
        plan_id = manager.save_plan(
            user_id=user_id,
            city="杭州",
            duration_days=1,
            attractions=["西湖"],
            daily_itinerary={"day0": ["西湖"]},
            preferences=["自然风光"],
            budget=800,
        )
        assert manager.get_plan(user_id, plan_id).city == "杭州"
        assert "西湖" in manager.get_visited_attractions(user_id, "杭州")
    finally:
        manager.clear_user_plans(user_id)


def test_travel_memory_manager_chroma_backend():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        manager = TravelMemoryManager(
            backend="chroma",
            embeddings=FakeEmbeddings(size=16),
            persist_directory=tmp,
        )
        assert manager.get_storage_type() == "chroma"
        user_id = "mgr_chroma_1"
        manager.save_plan(
            user_id=user_id,
            city="上海",
            duration_days=1,
            attractions=["东方明珠"],
            daily_itinerary={"day0": ["东方明珠"]},
            preferences=["地标"],
        )
        assert manager.get_visited_attractions(user_id, "上海") == ["东方明珠"]
        assert "地标" in manager.get_preferences(user_id)


if __name__ == "__main__":
    test_chroma_saves_preferences_and_visited_attractions()
    test_travel_memory_manager_file_backend_still_works()
    test_travel_memory_manager_chroma_backend()
    print("ok")
