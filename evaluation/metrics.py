"""与线上脚本一致的排序指标，以及比率、耗时降幅。"""

import math
from typing import Iterable, Sequence


def precision_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """前 K 条里相关文档数 / K。不足 K 条时分母仍是 K。"""
    if k <= 0:
        return 0.0
    relevant_set = set(relevant)
    hit = sum(1 for title in list(retrieved)[:k] if title in relevant_set)
    return hit / k


def recall_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """前 K 条里相关文档数 / 全部相关文档数。"""
    relevant_set = set(relevant)
    if not relevant_set or k <= 0:
        return 0.0
    hit = sum(1 for title in list(retrieved)[:k] if title in relevant_set)
    return hit / len(relevant_set)


def ndcg_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """二元相关的 NDCG@K。相关为 1，其余为 0。"""
    if k <= 0:
        return 0.0
    relevant_set = set(relevant)
    dcg = 0.0
    for index, title in enumerate(list(retrieved)[:k]):
        gain = 1.0 if title in relevant_set else 0.0
        dcg += (2 ** gain - 1) / math.log2(index + 2)
    ideal = min(len(relevant_set), k)
    idcg = sum((2 ** 1 - 1) / math.log2(index + 2) for index in range(ideal))
    return dcg / idcg if idcg else 0.0


def mrr(retrieved: Sequence[str], relevant: Iterable[str]) -> float:
    """第一条相关文档排名的倒数。没有命中则为 0。"""
    relevant_set = set(relevant)
    for rank, title in enumerate(retrieved, 1):
        if title in relevant_set:
            return 1.0 / rank
    return 0.0


def category_hit_rate(categories: Sequence[str], intent_categories: Iterable[str], k: int) -> float:
    """前 K 条的类别或标题文本里，包含任一意图类别的比例。"""
    if k <= 0:
        return 0.0
    top = list(categories)[:k]
    if not top:
        return 0.0
    intents = [item for item in intent_categories if item]
    if not intents:
        return 0.0
    hit = sum(1 for text in top if any(intent in text for intent in intents))
    return hit / k


def rate(passed: int, total: int) -> float:
    return passed / total if total else 0.0


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def latency_reduction(serial_s: float, parallel_s: float) -> float:
    """(串行 - 并行) / 串行。串行为 0 时返回 0。"""
    if serial_s <= 0:
        return 0.0
    return (serial_s - parallel_s) / serial_s
