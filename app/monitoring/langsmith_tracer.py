"""
LangSmith 链路追踪模块

职责：
1. 从环境变量初始化 LangSmith / LangChain tracing
2. 为 LangGraph / Agent 调用构建带 tags/metadata 的 runnable config
3. 提供 @traceable 装饰器封装，便于自定义业务跨度上报
4. 查询最近 runs（可选，用于健康检查）

环境变量（任选一套，推荐 LANGSMITH_*）：
    LANGSMITH_TRACING=true
    LANGSMITH_API_KEY=lsv2_...
    LANGSMITH_PROJECT=travel-planner
    LANGSMITH_ENDPOINT=https://api.smith.langchain.com   # 可选

兼容旧变量：
    LANGCHAIN_TRACING_V2 / LANGCHAIN_API_KEY / LANGCHAIN_PROJECT / LANGCHAIN_ENDPOINT
"""

from __future__ import annotations

import os
import functools
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
from uuid import uuid4

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


@dataclass
class LangSmithConfig:
    """LangSmith 配置快照"""
    enabled: bool = False
    api_key: str = ""
    project: str = "travel-planner"
    endpoint: str = "https://api.smith.langchain.com"
    sample_rate: float = 1.0
    tags: List[str] = field(default_factory=lambda: ["travel-planner"])
    source: str = "env"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "project": self.project,
            "endpoint": self.endpoint,
            "sample_rate": self.sample_rate,
            "tags": list(self.tags),
            "api_key_set": bool(self.api_key),
            "source": self.source,
        }


_tracer_initialized = False
_config: Optional[LangSmithConfig] = None


def _env_truthy(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def load_langsmith_config() -> LangSmithConfig:
    """从环境变量加载配置（不修改进程环境）"""
    api_key = (
        os.getenv("LANGSMITH_API_KEY")
        or os.getenv("LANGCHAIN_API_KEY")
        or ""
    ).strip()
    project = (
        os.getenv("LANGSMITH_PROJECT")
        or os.getenv("LANGCHAIN_PROJECT")
        or "travel-planner"
    ).strip()
    endpoint = (
        os.getenv("LANGSMITH_ENDPOINT")
        or os.getenv("LANGCHAIN_ENDPOINT")
        or "https://api.smith.langchain.com"
    ).strip()

    tracing_flag = _env_truthy("LANGSMITH_TRACING") or _env_truthy("LANGCHAIN_TRACING_V2")
    # 有 key 且显式开启 tracing 才启用；仅有 key 不自动开启，避免误上报
    enabled = tracing_flag and bool(api_key)

    try:
        sample_rate = float(os.getenv("LANGSMITH_SAMPLE_RATE", "1.0"))
    except ValueError:
        sample_rate = 1.0
    sample_rate = max(0.0, min(1.0, sample_rate))

    extra_tags = [
        t.strip()
        for t in os.getenv("LANGSMITH_TAGS", "travel-planner").split(",")
        if t.strip()
    ]

    return LangSmithConfig(
        enabled=enabled,
        api_key=api_key,
        project=project,
        endpoint=endpoint,
        sample_rate=sample_rate,
        tags=extra_tags or ["travel-planner"],
        source="env",
    )


def setup_langsmith(
    enabled: Optional[bool] = None,
    project: Optional[str] = None,
    api_key: Optional[str] = None,
    endpoint: Optional[str] = None,
    force: bool = False,
) -> LangSmithConfig:
    """
    初始化 LangSmith 追踪。

    Args:
        enabled: 强制开关；None 则跟随环境变量
        project: 项目名覆盖
        api_key: API Key 覆盖
        endpoint: Endpoint 覆盖
        force: 是否强制重新初始化

    Returns:
        当前生效的 LangSmithConfig
    """
    global _tracer_initialized, _config

    if _tracer_initialized and not force:
        return _config or LangSmithConfig()

    cfg = load_langsmith_config()

    if api_key is not None:
        cfg.api_key = api_key.strip()
    if project is not None:
        cfg.project = project.strip() or cfg.project
    if endpoint is not None:
        cfg.endpoint = endpoint.strip() or cfg.endpoint
    if enabled is not None:
        cfg.enabled = bool(enabled) and bool(cfg.api_key)
    else:
        cfg.enabled = cfg.enabled and bool(cfg.api_key)

    if cfg.enabled:
        # 写入两套环境变量，兼容 langchain / langsmith SDK
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGSMITH_API_KEY"] = cfg.api_key
        os.environ["LANGCHAIN_API_KEY"] = cfg.api_key
        os.environ["LANGSMITH_PROJECT"] = cfg.project
        os.environ["LANGCHAIN_PROJECT"] = cfg.project
        os.environ["LANGSMITH_ENDPOINT"] = cfg.endpoint
        os.environ["LANGCHAIN_ENDPOINT"] = cfg.endpoint

        print(f"✅ LangSmith 追踪已启用 | project={cfg.project} | endpoint={cfg.endpoint}")
    else:
        # 显式关闭，避免进程内残留开启状态
        os.environ["LANGSMITH_TRACING"] = "false"
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        reason = "未设置 API Key" if not cfg.api_key else "未开启 LANGSMITH_TRACING"
        print(f"ℹ️  LangSmith 追踪未启用（{reason}）")

    _config = cfg
    _tracer_initialized = True
    return cfg


def get_langsmith_config() -> LangSmithConfig:
    """获取当前配置（未初始化则先加载）"""
    global _config
    if _config is None:
        return setup_langsmith()
    return _config


def is_tracing_enabled() -> bool:
    return get_langsmith_config().enabled


def build_trace_config(
    *,
    run_name: str = "trip_planner",
    thread_id: Optional[str] = None,
    tags: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    configurable: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    构建 LangGraph / LangChain runnable config。

    即使 tracing 关闭也会返回可用的 config（含 thread_id），便于 checkpointer。
    """
    cfg = get_langsmith_config()
    run_id = str(uuid4())

    merged_tags = list(dict.fromkeys([*(cfg.tags or []), *(tags or [])]))
    merged_metadata = {
        "service": "travel-planner",
        "run_id": run_id,
        **(metadata or {}),
    }
    if thread_id:
        merged_metadata["thread_id"] = thread_id

    config: Dict[str, Any] = {
        "run_name": run_name,
        "tags": merged_tags,
        "metadata": merged_metadata,
        "configurable": {
            **(configurable or {}),
        },
    }
    if thread_id:
        config["configurable"]["thread_id"] = thread_id

    return config


def merge_run_config(
    base: Optional[Dict[str, Any]],
    *,
    run_name: str,
    thread_id: Optional[str] = None,
    tags: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """将 tracing 字段合并进已有 config（如已有 checkpointer thread_id）"""
    trace = build_trace_config(
        run_name=run_name,
        thread_id=thread_id,
        tags=tags,
        metadata=metadata,
        configurable=(base or {}).get("configurable"),
    )
    if not base:
        return trace

    merged = dict(base)
    merged["run_name"] = run_name
    merged["tags"] = list(dict.fromkeys([
        *(base.get("tags") or []),
        *(trace.get("tags") or []),
    ]))
    merged["metadata"] = {
        **(base.get("metadata") or {}),
        **(trace.get("metadata") or {}),
    }
    merged["configurable"] = {
        **(base.get("configurable") or {}),
        **(trace.get("configurable") or {}),
    }
    return merged


def traced(
    name: Optional[str] = None,
    *,
    run_type: str = "chain",
    tags: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Callable:
    """
    业务函数追踪装饰器。

    tracing 未启用或 langsmith 不可用时，退化为原函数（零开销旁路）。
    """

    def decorator(func: Callable) -> Callable:
        run_name = name or func.__name__

        try:
            from langsmith import traceable as ls_traceable
        except ImportError:
            return func

        wrapped = ls_traceable(
            name=run_name,
            run_type=run_type,
            tags=tags or [],
            metadata=metadata or {},
        )(func)

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            if not is_tracing_enabled():
                return func(*args, **kwargs)
            return wrapped(*args, **kwargs)

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            if not is_tracing_enabled():
                return await func(*args, **kwargs)
            return await wrapped(*args, **kwargs)

        if asyncio_iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


def asyncio_iscoroutinefunction(func: Callable) -> bool:
    import asyncio
    return asyncio.iscoroutinefunction(func)


def get_recent_runs(limit: int = 5) -> List[Dict[str, Any]]:
    """拉取当前项目最近 runs（用于健康检查；失败返回空列表）"""
    cfg = get_langsmith_config()
    if not cfg.enabled:
        return []

    try:
        from langsmith import Client
        client = Client(api_key=cfg.api_key, api_url=cfg.endpoint)
        runs = list(client.list_runs(project_name=cfg.project, limit=limit))
        result = []
        for run in runs:
            result.append({
                "id": str(getattr(run, "id", "")),
                "name": getattr(run, "name", ""),
                "status": getattr(run, "status", ""),
                "start_time": str(getattr(run, "start_time", "")),
                "end_time": str(getattr(run, "end_time", "")),
                "error": getattr(run, "error", None),
                "tags": list(getattr(run, "tags", []) or []),
            })
        return result
    except Exception as e:
        print(f"⚠️  拉取 LangSmith runs 失败: {e}")
        return []


def health_check() -> Dict[str, Any]:
    """LangSmith 模块健康检查"""
    cfg = get_langsmith_config()
    status = {
        "enabled": cfg.enabled,
        "project": cfg.project,
        "endpoint": cfg.endpoint,
        "ok": True,
        "message": "tracing disabled" if not cfg.enabled else "ok",
    }
    if not cfg.enabled:
        return status

    try:
        from langsmith import Client
        client = Client(api_key=cfg.api_key, api_url=cfg.endpoint)
        # 轻量探测：读取项目信息（无则创建由平台侧处理）
        _ = list(client.list_projects(limit=1))
        status["message"] = "connected"
    except Exception as e:
        status["ok"] = False
        status["message"] = str(e)
    return status
