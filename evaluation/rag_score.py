"""固定标准答案上的 Precision、Recall、NDCG、MRR、类别命中。"""

from typing import Any, Dict, Iterable, List, Sequence

from evaluation.metrics import (
    category_hit_rate,
    mean,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


def score_retrievals(cases: Sequence[Dict[str, Any]], k: int = 5, pool_k: int = 30) -> Dict[str, Any]:
    rows = [score_retrieval(case, k=k, pool_k=pool_k) for case in cases]
    grounded_rows = [row for row in rows if row["grounded"] is not None]
    grounded_hit = sum(1 for row in grounded_rows if row["grounded"])
    return {
        "k": k,
        "pool_k": pool_k,
        "query_count": len(rows),
        "precision": mean([row["precision"] for row in rows]),
        "recall": mean([row["recall"] for row in rows]),
        "ndcg": mean([row["ndcg"] for row in rows]),
        "mrr": mean([row["mrr"] for row in rows]),
        "category_hit_rate": mean([row["category_hit_rate"] for row in rows]),
        "recall_at_pool": mean([row["recall_at_pool"] for row in rows]),
        "grounded_rate": (grounded_hit / len(grounded_rows)) if grounded_rows else None,
        "cases": rows,
    }


def score_retrieval(case: Dict[str, Any], k: int = 5, pool_k: int = 30) -> Dict[str, Any]:
    relevant = list(case.get("relevant_docs") or [])
    titles = [str(title) for title in (case.get("retrieved_titles") or [])]
    pool = [str(title) for title in (case.get("retrieved_pool") or titles)]
    categories = [str(text) for text in (case.get("categories") or [])]
    intents = list(case.get("intent_categories") or [])
    grounded = case.get("grounded")
    return {
        "id": case.get("id") or case.get("query") or "",
        "precision": precision_at_k(titles, relevant, k),
        "recall": recall_at_k(titles, relevant, k),
        "ndcg": ndcg_at_k(titles, relevant, k),
        "mrr": mrr(titles, relevant),
        "category_hit_rate": category_hit_rate(categories, intents, k),
        "recall_at_pool": recall_at_k(pool, relevant, pool_k),
        "grounded": bool(grounded) if grounded is not None else None,
    }


def titles_of(docs: Sequence[Dict[str, Any]]) -> List[str]:
    titles = []
    for doc in docs:
        metadata = doc.get("metadata") if isinstance(doc, dict) else None
        title = ""
        if isinstance(metadata, dict):
            title = str(metadata.get("title") or "")
        if title:
            titles.append(title)
    return titles


def category_texts(docs: Sequence[Dict[str, Any]]) -> List[str]:
    texts = []
    for doc in docs:
        metadata = doc.get("metadata") if isinstance(doc, dict) else {}
        if not isinstance(metadata, dict):
            metadata = {}
        texts.append(
            " ".join(
                str(metadata.get(key) or "")
                for key in ("category", "tags", "title")
            )
        )
    return texts


def case_from_docs(
    query: str,
    relevant_docs: Iterable[str],
    docs: Sequence[Dict[str, Any]],
    intent_categories: Iterable[str],
    grounded: Any = None,
    pool_docs: Sequence[Dict[str, Any]] = (),
) -> Dict[str, Any]:
    """把一次检索结果收成 score_retrieval 能直接打分的记录。"""
    return {
        "query": query,
        "relevant_docs": list(relevant_docs),
        "retrieved_titles": titles_of(docs),
        "retrieved_pool": titles_of(pool_docs) if pool_docs else titles_of(docs),
        "categories": category_texts(docs),
        "intent_categories": list(intent_categories),
        "grounded": grounded,
    }
