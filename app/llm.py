"""LLM服务封装（向后兼容）"""

from .services.llm_service import (
    get_llm,
    get_openai_llm,
    get_aliyun_llm
)

# 保持向后兼容的导出
__all__ = [
    "get_llm",
    "get_openai_llm",
    "get_aliyun_llm"
]
