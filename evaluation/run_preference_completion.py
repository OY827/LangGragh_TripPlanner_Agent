"""按城市和偏好跑真实规划，汇总任务完成率。

每条任务都会调用模型、高德和酒店接口。默认 6 条，大约十几分钟。

    .\\.venv\\Scripts\\python.exe evaluation/run_preference_completion.py
    .\\.venv\\Scripts\\python.exe evaluation/run_preference_completion.py --cities 上海,杭州 --preferences 历史文化,自然风光 --days 1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass

os.environ.setdefault("VECTOR_BACKEND", "chroma")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

from evaluation.preference_completion import TASKS, case_from_result, format_matrix_report, summarize_matrix


def _pairs(cities: list[str], preferences: list[str]) -> list[tuple[str, list[str]]]:
    if not cities and not preferences:
        return [(city, list(prefs)) for city, prefs in TASKS]
    chosen_cities = cities or []
    seen_cities = []
    for city, _prefs in TASKS:
        if city not in seen_cities:
            seen_cities.append(city)
    chosen_cities = chosen_cities or seen_cities
    if preferences:
        chosen_prefs = [item.split("|") for item in preferences]
    else:
        chosen_prefs = []
        for _city, prefs in TASKS:
            group = list(prefs)
            if group not in chosen_prefs:
                chosen_prefs.append(group)
    return [(city, prefs) for city in chosen_cities for prefs in chosen_prefs]


def main() -> int:
    parser = argparse.ArgumentParser(description="按城市和偏好统计真实任务完成率")
    parser.add_argument("--cities", default="", help="逗号分隔，例如 上海,北京")
    parser.add_argument("--preferences", default="", help="逗号分隔。一组里的多个偏好用 | ，例如 夜景|步行街,历史文化")
    parser.add_argument("--days", type=int, default=1)
    args = parser.parse_args()
    cities = [item.strip() for item in args.cities.split(",") if item.strip()]
    preferences = [item.strip() for item in args.preferences.split(",") if item.strip()]
    pairs = _pairs(cities, preferences)
    start = (datetime.now() + timedelta(days=14)).strftime("%Y-%m-%d")
    end = (datetime.now() + timedelta(days=13 + args.days)).strftime("%Y-%m-%d")

    from app.orchestrator import get_langgraph_agent

    print(f"任务 {len(pairs)} 条，每条 {args.days} 天，{start} 起")
    agent = get_langgraph_agent(llm_provider="aliyun")
    cases = []
    for index, (city, prefs) in enumerate(pairs, 1):
        label = "+".join(prefs)
        print(f"\n[{index}/{len(pairs)}] {city} / {label}")
        result = agent.plan_trip(
            city=city,
            travel_days=args.days,
            start_date=start,
            end_date=end,
            transportation="公共交通",
            accommodation="经济型",
            preferences=prefs,
            user_id="preference_completion",
            free_text_input=f"偏好：{label}",
            thread_id=f"pref-{city}-{label}-{index}",
        )
        case = case_from_result(result, city=city, preferences=prefs, travel_days=args.days)
        cases.append(case)
        parsed = result.get("trip_plan_parsed") or {}
        days = parsed.get("days") if isinstance(parsed, dict) else None
        print(f"  天数={len(days) if isinstance(days, list) else 0}  预算={result.get('budget_total')}")

    summary = summarize_matrix(cases)
    text = format_matrix_report(summary)
    print("\n" + text)
    out = ROOT / "evaluation" / "results"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "preference_completion.json"
    path.write_text(json.dumps({"cases": cases, "summary_text": text}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已写入 {path}")
    return 0 if summary["overall"]["full_chain"]["rate"] == 1 else 1


if __name__ == "__main__":
    raise SystemExit(main())
