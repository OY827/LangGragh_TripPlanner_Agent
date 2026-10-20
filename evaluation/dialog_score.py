"""修改对话：意图、工具强制调用、名称差、版本号。"""

from typing import Any, Dict, List, Sequence

from evaluation.metrics import rate

TOOL_REQUIRED = {"add_item", "modify_attractions", "modify_hotel"}
TOOL_BY_TYPE = {
    "add_item": "amap_attraction_search",
    "modify_attractions": "amap_attraction_search",
    "modify_hotel": "hotel_search",
}


def score_dialogs(cases: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    rows = [score_dialog(case) for case in cases]
    required = [row for row in rows if row["requires_tool"]]
    covered = sum(1 for row in required if row["tool_covered"])
    violations = sum(1 for row in required if not row["tool_covered"])
    intent_ok = sum(1 for row in rows if row["intent_correct"])
    version_ok = sum(1 for row in rows if row["version_incremented"])
    edit_ok = sum(1 for row in rows if row["edit_applied"])
    return {
        "intent_accuracy": rate(intent_ok, len(rows)),
        "tool_coverage": rate(covered, len(required)),
        "tool_required_count": len(required),
        "tool_covered_count": covered,
        "violation_count": violations,
        "version_increment_rate": rate(version_ok, len(rows)),
        "edit_applied_rate": rate(edit_ok, len(rows)),
        "case_count": len(rows),
        "cases": rows,
    }


def score_dialog(case: Dict[str, Any]) -> Dict[str, Any]:
    expected = str(case.get("expected_change_type") or "")
    predicted = str(case.get("change_type") or "")
    requires_tool = expected in TOOL_REQUIRED
    tool_calls = [str(name) for name in (case.get("tool_calls") or [])]
    expected_tool = str(case.get("expected_tool") or TOOL_BY_TYPE.get(expected, ""))
    search_names = {str(name).strip() for name in (case.get("search_names") or []) if str(name).strip()}
    search_hotel = str(case.get("search_hotel") or "").strip()
    added = [str(name).strip() for name in (case.get("added") or []) if str(name).strip()]
    removed = [str(name).strip() for name in (case.get("removed") or []) if str(name).strip()]
    after_names = {str(name).strip() for name in (case.get("after_names") or [])}
    added_hotel = str(case.get("added_hotel") or "").strip()

    tool_called = bool(expected_tool) and expected_tool in tool_calls
    names_from_search = all(name in search_names for name in added) if added else True
    hotel_from_search = (added_hotel == search_hotel) if added_hotel else True
    tool_covered = (not requires_tool) or (tool_called and names_from_search and hotel_from_search)

    before_version = case.get("before_version")
    after_version = case.get("after_version")
    version_incremented = (
        isinstance(before_version, int)
        and isinstance(after_version, int)
        and after_version == before_version + 1
    )
    edit_applied = all(name not in after_names for name in removed) and all(
        name in after_names for name in added
    )
    if added_hotel:
        edit_applied = edit_applied and added_hotel == str(case.get("after_hotel") or "").strip()

    return {
        "id": case.get("id", ""),
        "intent_correct": expected == predicted and bool(expected),
        "requires_tool": requires_tool,
        "tool_covered": tool_covered,
        "version_incremented": version_incremented,
        "edit_applied": edit_applied,
        "reasons": _reasons(
            expected == predicted and bool(expected),
            requires_tool,
            tool_called,
            names_from_search and hotel_from_search,
            version_incremented,
            edit_applied,
        ),
    }


def _reasons(
    intent_ok: bool,
    requires_tool: bool,
    tool_called: bool,
    grounded_in_search: bool,
    version_ok: bool,
    edit_ok: bool,
) -> List[str]:
    reasons: List[str] = []
    if not intent_ok:
        reasons.append("intent")
    if requires_tool and not tool_called:
        reasons.append("tool_not_called")
    if requires_tool and not grounded_in_search:
        reasons.append("name_not_in_tool_result")
    if not version_ok:
        reasons.append("version")
    if not edit_ok:
        reasons.append("edit_not_applied")
    return reasons
