"""量化评测口径：样例上的完成率、工具覆盖、耗时和检索指标。"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.dialog_score import score_dialog, score_dialogs
from evaluation.latency_score import summarize_latency
from evaluation.metrics import ndcg_at_k, precision_at_k, recall_at_k
from evaluation.plan_score import score_plan, score_plans
from evaluation.rag_score import score_retrievals
from evaluation.report import build_report
from evaluation.run import run_offline

CASES = Path(__file__).resolve().parents[1] / "evaluation" / "cases"


def _load(name):
    with (CASES / name).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def test_executable_plan_passes_and_known_failures_are_named():
    rows = {row["id"]: row for row in score_plans(_load("plan_cases.json"))["cases"]}
    assert rows["shanghai_2d_ok"]["executable"] is True
    assert rows["shanghai_2d_ok"]["reasons"] == []
    assert "outside_candidate" in rows["outside_candidate"]["reasons"]
    assert "clipped_visit" in rows["clipped_visit"]["reasons"]
    assert "beijing_default_coord" in rows["beijing_default_coord"]["reasons"]
    assert "day_count" in rows["day_count_short"]["reasons"]
    summary = score_plans(_load("plan_cases.json"))
    assert summary["executable_count"] == 1
    assert summary["case_count"] == 5
    assert summary["completion_rate"] == 0.2


def test_zero_coordinate_is_not_geocoded():
    case = _load("plan_cases.json")[0]
    case = json.loads(json.dumps(case))
    case["plan"]["hotel"]["location"] = "0,0"
    scored = score_plan(case)
    assert "hotel_not_geocoded" in scored["reasons"]
    assert scored["executable"] is False


def test_tool_coverage_counts_only_search_required_edits():
    summary = score_dialogs(_load("dialog_cases.json"))
    assert summary["tool_required_count"] == 5
    assert summary["tool_covered_count"] == 2
    assert summary["violation_count"] == 3
    assert summary["tool_coverage"] == 0.4
    by_id = {row["id"]: row for row in summary["cases"]}
    assert by_id["replace_with_tool"]["tool_covered"] is True
    assert by_id["budget_no_tool"]["requires_tool"] is False
    assert by_id["budget_no_tool"]["intent_correct"] is True
    assert "tool_not_called" in by_id["replace_without_tool"]["reasons"]
    assert "name_not_in_tool_result" in by_id["name_not_in_search"]["reasons"]
    assert score_dialog(by_id and _load("dialog_cases.json")[5])["intent_correct"] is False


def test_latency_reduction_is_the_mean_of_per_request_ratios():
    summary = summarize_latency(_load("latency_cases.json"))
    assert summary["mean_tool_s"] == (1.0 + 1.4 + 1.0) / 3
    assert summary["mean_e2e_s"] == 10
    assert summary["wait_reduction"] == (0.4 + 0.2) / 2


def test_ranking_metrics_match_binary_relevance():
    titles = ["外滩", "南京路步行街", "豫园", "上海博物馆", "上海迪士尼度假区"]
    relevant = ["外滩", "豫园", "上海博物馆", "城隍庙", "龙华寺"]
    assert precision_at_k(titles, relevant, 5) == 0.6
    assert recall_at_k(titles, relevant, 5) == 0.6
    assert ndcg_at_k(titles, relevant, 5) > ndcg_at_k(list(reversed(titles)), relevant, 5)
    summary = score_retrievals(_load("rag_cases.json"), k=5, pool_k=30)
    assert abs(summary["precision"] - 0.8) < 1e-9
    assert summary["grounded_rate"] == 0.5
    assert abs(summary["recall_at_pool"] - 0.9) < 1e-9


def test_offline_report_labels_itself_as_fixture():
    report = run_offline()
    built = build_report(
        {"completion_rate": 0.2, "executable_count": 1, "case_count": 5,
         "outside_candidate_rate": 0, "clipped_visit_rate": 0, "cases": []},
        {**report["dialog"], "cases": []},
        report["latency"],
        report["rag"] | {"cases": []},
    )
    assert report["source"] == "offline_fixture"
    assert "样例自检" in report["sentences"][0]
    assert "20.0%" in report["sentences"][0]
    assert "40.0%" in report["sentences"][1]
    assert built["source"] == "offline_fixture"


if __name__ == "__main__":
    test_executable_plan_passes_and_known_failures_are_named()
    test_zero_coordinate_is_not_geocoded()
    test_tool_coverage_counts_only_search_required_edits()
    test_latency_reduction_is_the_mean_of_per_request_ratios()
    test_ranking_metrics_match_binary_relevance()
    test_offline_report_labels_itself_as_fixture()
    print("ok")
