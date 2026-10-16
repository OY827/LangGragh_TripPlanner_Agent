"""离线样例自检。加上 --rag-live 才调用混合检索。"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.dialog_score import score_dialogs
from evaluation.latency_score import summarize_latency
from evaluation.plan_score import score_plans
from evaluation.rag_score import case_from_docs, score_retrievals
from evaluation.report import build_report, format_report

CASES = Path(__file__).resolve().parent / "cases"


def load_cases(name: str):
    path = CASES / name
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def run_offline():
    return build_report(
        score_plans(load_cases("plan_cases.json")),
        score_dialogs(load_cases("dialog_cases.json")),
        summarize_latency(load_cases("latency_cases.json")),
        score_retrievals(load_cases("rag_cases.json"), k=5, pool_k=30),
        source="offline_fixture",
    )


def run_rag_live(k: int = 5, pool_k: int = 30):
    """用上海对齐标准答案打一次当前混合检索。失败时返回错误说明，不改口径。"""
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    gt_path = ROOT / "evaluation" / "gt_shanghai_kb_aligned.json"
    with gt_path.open("r", encoding="utf-8") as handle:
        dataset = json.load(handle)

    from app.rag.hybrid_rag import HybridRAG, HybridRAGConfig

    rag = HybridRAG(config=HybridRAGConfig(
        enable_rerank=True,
        rerank_model="BAAI/bge-reranker-base",
        retrieval_top_k=pool_k,
        rerank_top_k=k,
        top_k=k,
        max_retrieval_rounds=1,
        enable_query_rewrite=False,
        enable_retrieval_validation=False,
        use_mmr=False,
        enable_intent_boost=True,
        intent_boost_weight=0.12,
    ))
    cases = []
    for item in dataset:
        docs, meta = rag.retrieve_with_validation(
            query=item["query"],
            city=item.get("city") or "上海",
            preferences=list(item.get("intent_categories") or []),
        )
        validation = meta.get("validation") if isinstance(meta, dict) else None
        grounded = None
        if isinstance(validation, dict) and "grounded" in validation:
            grounded = validation.get("grounded")
        cases.append(case_from_docs(
            query=item["query"],
            relevant_docs=item.get("relevant_docs") or [],
            docs=docs,
            intent_categories=item.get("intent_categories") or [],
            grounded=grounded,
            pool_docs=docs,
        ))
    return score_retrievals(cases, k=k, pool_k=pool_k)


def main():
    parser = argparse.ArgumentParser(description="量化评测")
    parser.add_argument("--json", action="store_true", help="输出 JSON")
    parser.add_argument("--rag-live", action="store_true", help="对上海标准答案跑混合检索")
    args = parser.parse_args()
    report = run_offline()
    if args.rag_live:
        report["rag_live"] = run_rag_live()
        report["rag_live"].pop("cases", None)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(format_report(report))
        if args.rag_live:
            live = report["rag_live"]
            print()
            print(
                "上海标准答案实测："
                f"Precision@{live['k']} {live['precision']:.3f}，"
                f"Recall@{live['k']} {live['recall']:.3f}，"
                f"NDCG@{live['k']} {live['ndcg']:.3f}，"
                f"MRR {live['mrr']:.3f}。"
            )


if __name__ == "__main__":
    main()
