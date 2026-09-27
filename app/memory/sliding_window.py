"""
对话上下文窗口：LangChain 官方 trim_messages。

压缩策略：
1. strategy="last" 滑动窗口，只保留最近 N 条消息
2. 可选再按近似 token 数裁一次，防止单条过长撑爆窗口
"""

from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List, Optional

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    trim_messages,
)


def _count_messages(messages: Any) -> int:
    """按条数计数，配合 trim_messages 做成「最近 N 条」滑动窗口。"""
    if isinstance(messages, list):
        return len(messages)
    return 1


def records_to_messages(records: Iterable[Dict[str, Any]]) -> List[BaseMessage]:
    messages: List[BaseMessage] = []
    for record in records or []:
        role = (record.get("role") or "").lower()
        content = record.get("content") or ""
        if role in ("user", "human"):
            messages.append(HumanMessage(content=content))
        elif role in ("assistant", "ai"):
            messages.append(AIMessage(content=content))
        elif role == "system":
            messages.append(SystemMessage(content=content))
    return messages


def messages_to_records(messages: Iterable[BaseMessage]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for message in messages or []:
        content = message.content if isinstance(message.content, str) else str(message.content or "")
        if isinstance(message, HumanMessage):
            records.append({"role": "user", "content": content})
        elif isinstance(message, AIMessage):
            records.append({"role": "assistant", "content": content})
        elif isinstance(message, SystemMessage):
            records.append({"role": "system", "content": content})
    return records


class SlidingWindowChatMemory:
    """
    LangChain 官方上下文窗口。

    不自己维护存储：调用方把完整历史（例如 Redis 会话里的 conversation_log）
    传进来，这里只负责压缩成可送进 LLM 的窗口。
    """

    def __init__(
        self,
        window_size: int = 6,
        max_tokens: Optional[int] = None,
    ):
        self.window_size = max(1, int(window_size))
        if max_tokens is None:
            raw = os.getenv("DIALOG_CONTEXT_MAX_TOKENS", "").strip()
            max_tokens = int(raw) if raw.isdigit() else None
        self.max_tokens = max_tokens

    def compress_messages(
        self,
        messages: List[BaseMessage],
        system_prompt: Optional[str] = None,
    ) -> List[BaseMessage]:
        packed: List[BaseMessage] = list(messages or [])
        if system_prompt:
            packed = [SystemMessage(content=system_prompt), *packed]

        if not packed:
            return []

        windowed = trim_messages(
            packed,
            max_tokens=self.window_size + (1 if system_prompt else 0),
            token_counter=_count_messages,
            strategy="last",
            start_on="human",
            include_system=True,
            allow_partial=False,
        )

        if self.max_tokens:
            windowed = trim_messages(
                windowed,
                max_tokens=self.max_tokens,
                token_counter="approximate",
                strategy="last",
                start_on="human",
                include_system=True,
                allow_partial=False,
            )
        return windowed

    def compress_records(
        self,
        records: Iterable[Dict[str, Any]],
        system_prompt: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        messages = records_to_messages(records)
        windowed = self.compress_messages(messages, system_prompt=system_prompt)
        if system_prompt:
            windowed = [m for m in windowed if not isinstance(m, SystemMessage)]
        return messages_to_records(windowed)

    def format_for_prompt(self, records: Iterable[Dict[str, Any]]) -> str:
        lines = []
        for item in self.compress_records(records):
            role = "用户" if item.get("role") == "user" else "助手"
            content = (item.get("content") or "").strip()
            if content:
                lines.append(f"  {role}: {content}")
        return "\n".join(lines)
