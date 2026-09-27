"""VersionManager 版本查询（文件后端，不依赖 Redis）。"""

import os
import tempfile

from app.agents.dialog_models import ChangeIntent, PlanDiff
from app.agents.version_manager import VersionManager


def test_format_and_search_versions():
    with tempfile.TemporaryDirectory() as tmp:
        vm = VersionManager(use_redis=False)
        vm.storage_path = os.path.join(tmp, "dialog_sessions")
        os.makedirs(vm.storage_path, exist_ok=True)

        session = vm.create_session(user_id="u1", plan_id="p1")
        v1 = vm.create_initial_version(session.session_id, {
            "city": "上海",
            "duration_days": 2,
            "attractions": ["外滩"],
        })
        intent = ChangeIntent(change_type="modify_attractions", description="加豫园")
        diff = PlanDiff(
            old_version_id=v1.version_id,
            added_attractions=["豫园"],
            change_descriptions=["新增景点: 豫园"],
            summary="新增景点: 豫园",
        )
        v2 = vm.create_modified_version(
            session.session_id,
            {"city": "上海", "duration_days": 2, "attractions": ["外滩", "豫园"]},
            intent,
            diff,
            changes=["新增景点: 豫园"],
        )

        text = vm.format_versions_for_llm(session.session_id)
        assert "外滩" in text
        assert v2.version_id in text

        found = vm.search_versions(session.session_id, "豫园")
        assert any(item.version_id == v2.version_id for item in found)

        rolled = vm.rollback_to_version(session.session_id, v1.version_id)
        assert rolled is not None
        current = vm.get_current_version(session.session_id)
        assert current.plan_data.get("attractions") == ["外滩"]


if __name__ == "__main__":
    test_format_and_search_versions()
    print("ok")
