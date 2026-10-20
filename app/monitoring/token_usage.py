"""从模型响应里读取 token，并按本次规划汇总。"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Dict, Iterator, List, Optional

from langchain_core.callbacks import BaseCallbackHandler


def empty_usage() -> Dict[str, Any]:
    return {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "calls": 0,
        "calls_without_usage": 0,
    }


_current: ContextVar[Optional[Dict[str, Any]]] = ContextVar("llm_token_ledger", default=None)
_lock = threading.Lock()
_inflight: List[Dict[str, Any]] = []
_seen_runs: List[str] = []


def usage_from_result(response: Any) -> Dict[str, int]:
    """读取 LangChain LLMResult 或单条消息上的用量。没有用量时返回 0。"""
    prompt, completion = _pair_from_mapping(_llm_output(response))
    model_prompt, model_completion = _from_generations(response)
    if prompt or completion:
        return {"input_tokens": prompt, "output_tokens": completion}
    return {"input_tokens": model_prompt, "output_tokens": model_completion}


def _llm_output(response: Any) -> Dict[str, Any]:
    output = getattr(response, "llm_output", None) or {}
    if not isinstance(output, dict):
        return {}
    token_usage = output.get("token_usage") or output.get("usage") or {}
    return token_usage if isinstance(token_usage, dict) else {}


def _from_generations(response: Any) -> tuple[int, int]:
    generations = getattr(response, "generations", None) or []
    for group in generations:
        for gen in group or []:
            message = getattr(gen, "message", None)
            if message is None:
                continue
            prompt, completion = _pair_from_mapping(getattr(message, "usage_metadata", None))
            if prompt or completion:
                return prompt, completion
            meta = getattr(message, "response_metadata", None) or {}
            if isinstance(meta, dict):
                token_usage = meta.get("token_usage") or meta.get("usage") or {}
                prompt, completion = _pair_from_mapping(token_usage)
                if prompt or completion:
                    return prompt, completion
    return 0, 0


def _pair_from_mapping(raw: Any) -> tuple[int, int]:
    if not isinstance(raw, dict):
        return 0, 0
    prompt = raw.get("prompt_tokens", raw.get("input_tokens", 0))
    completion = raw.get("completion_tokens", raw.get("output_tokens", 0))
    try:
        return int(prompt or 0), int(completion or 0)
    except (TypeError, ValueError):
        return 0, 0


def _targets() -> List[Dict[str, Any]]:
    current = _current.get()
    if current is not None:
        return [current]
    with _lock:
        return list(_inflight)


def record_token_usage(prompt_tokens: int, completion_tokens: int, *, model: str = "") -> None:
    prompt_tokens = max(0, int(prompt_tokens or 0))
    completion_tokens = max(0, int(completion_tokens or 0))
    for ledger in _targets():
        ledger["calls"] += 1
        if prompt_tokens or completion_tokens:
            ledger["input_tokens"] += prompt_tokens
            ledger["output_tokens"] += completion_tokens
            ledger["total_tokens"] = ledger["input_tokens"] + ledger["output_tokens"]
        else:
            ledger["calls_without_usage"] += 1
    if prompt_tokens or completion_tokens:
        _publish(prompt_tokens, completion_tokens, model)


def _publish(prompt_tokens: int, completion_tokens: int, model: str) -> None:
    from app.monitoring.logger import get_logger
    from app.monitoring.metrics import get_metrics

    get_metrics().record_llm_usage(prompt_tokens, completion_tokens)
    get_logger().log_llm_call(model or "unknown", prompt_tokens, completion_tokens, 0)


def note_llm_result(response: Any, run_id: Any = None) -> None:
    if run_id is not None:
        key = str(run_id)
        with _lock:
            if key in _seen_runs:
                return
            _seen_runs.append(key)
            if len(_seen_runs) > 400:
                del _seen_runs[:200]
    usage = usage_from_result(response)
    record_token_usage(usage["input_tokens"], usage["output_tokens"])


class TokenUsageHandler(BaseCallbackHandler):
    """挂在 ChatOpenAI 上，每次调用结束记一笔用量。"""

    def on_llm_end(self, response: Any, *, run_id: Any = None, **kwargs: Any) -> None:
        note_llm_result(response, run_id=run_id)

    def on_chat_model_end(self, response: Any, *, run_id: Any = None, **kwargs: Any) -> None:
        note_llm_result(response, run_id=run_id)


def token_usage_handler() -> TokenUsageHandler:
    return TokenUsageHandler()


@contextmanager
def track_llm_usage() -> Iterator[Dict[str, Any]]:
    """包住一次规划或一次对话恢复，得到这一次的 token 合计。"""
    ledger = empty_usage()
    token = _current.set(ledger)
    with _lock:
        _inflight.append(ledger)
    try:
        yield ledger
    finally:
        _current.reset(token)
        with _lock:
            if ledger in _inflight:
                _inflight.remove(ledger)
