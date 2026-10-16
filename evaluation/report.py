"""把四类打分收成一份报告。样例自检和线上实测用 source 区分。"""

from typing import Any, Dict, List, Optional


def build_report(
    plan: Dict[str, Any],
    dialog: Dict[str, Any],
    latency: Dict[str, Any],
    rag: Dict[str, Any],
    source: str = "offline_fixture",
) -> Dict[str, Any]:
    report = {
        "source": source,
        "plan": {key: value for key, value in plan.items() if key != "cases"},
        "dialog": {key: value for key, value in dialog.items() if key != "cases"},
        "latency": latency,
        "rag": {key: value for key, value in rag.items() if key != "cases"},
        "plan_cases": plan.get("cases") or [],
        "dialog_cases": dialog.get("cases") or [],
    }
    report["sentences"] = render_sentences(report)
    return report


def render_sentences(report: Dict[str, Any]) -> List[str]:
    plan = report["plan"]
    dialog = report["dialog"]
    latency = report["latency"]
    rag = report["rag"]
    label = "样例自检" if report.get("source") == "offline_fixture" else "本次实测"
    grounded = rag.get("grounded_rate")
    grounded_text = "未测" if grounded is None else _pct(grounded)
    return [
        (
            f"{label}：多日行程可执行完成率 {_pct(plan['completion_rate'])}"
            f"（{plan['executable_count']}/{plan['case_count']}），"
            f"候选外景点占比 {_pct(plan['outside_candidate_rate'])}，"
            f"游览时长被截短的时段占比 {_pct(plan['clipped_visit_rate'])}。"
        ),
        (
            f"{label}：改景点、改酒店、新增景点的工具强制调用覆盖率 "
            f"{_pct(dialog['tool_coverage'])}"
            f"（{dialog['tool_covered_count']}/{dialog['tool_required_count']}），"
            f"未检索就写入计划 {dialog['violation_count']} 次。"
            f"意图分类准确率 {_pct(dialog['intent_accuracy'])}，"
            f"版本号加一的比例 {_pct(dialog['version_increment_rate'])}。"
        ),
        (
            f"{label}：工具调用平均耗时 {latency['mean_tool_s']:.2f} 秒"
            f"（{latency['tool_sample_count']} 次），"
            f"端到端等待平均 {latency['mean_e2e_s']:.2f} 秒，"
            f"四路并行相对串行的等待降幅 {_pct(latency['wait_reduction'])}。"
        ),
        (
            f"{label}：检索 Precision@{rag['k']} {_num(rag['precision'])}，"
            f"Recall@{rag['k']} {_num(rag['recall'])}，"
            f"NDCG@{rag['k']} {_num(rag['ndcg'])}，"
            f"MRR {_num(rag['mrr'])}，"
            f"类别命中率 {_num(rag['category_hit_rate'])}，"
            f"Recall@{rag['pool_k']} {_num(rag['recall_at_pool'])}，"
            f"答案可溯源比例 {grounded_text}。"
        ),
    ]


def format_report(report: Dict[str, Any]) -> str:
    lines = [
        "量化评测报告",
        f"来源: {report.get('source')}",
        "",
    ]
    lines.extend(report.get("sentences") or [])
    if report.get("source") == "offline_fixture":
        lines.extend([
            "",
            "上面的比例来自 evaluation/cases 里的固定样例，用来核对口径。",
            "线上数字需要把真实计划和检索结果交到同一套打分函数后再生成报告。",
        ])
    return "\n".join(lines)


def _pct(value: Optional[float]) -> str:
    if value is None:
        return "未测"
    return f"{value * 100:.1f}%"


def _num(value: Optional[float]) -> str:
    if value is None:
        return "未测"
    return f"{value:.3f}"
