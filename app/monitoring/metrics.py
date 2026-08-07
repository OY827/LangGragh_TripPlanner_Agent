"""
性能监控模块 - Performance Metrics
收集和统计Agent系统的各项性能指标
"""

import time
import threading
from typing import Dict, List, Optional
from datetime import datetime


class PerformanceMetrics:
    """性能指标收集器"""

    def __init__(self):
        self._metrics = {
            "total_latency": [],
            "llm_latency": [],
            "tool_latency": [],
            "rag_latency": [],
            "success_count": 0,
            "error_count": 0,
            "tokens_used": {"input": 0, "output": 0},
            "api_calls": {"llm": 0, "tool": 0, "rag": 0},
            "concurrent_requests": 0,
            "requests_per_minute": [],
        }
        self._lock = threading.Lock()
        self._last_second = 0
        self._requests_in_current_minute = 0
        self._start_time = datetime.now()

    def record_latency(self, metric_type: str, value: float):
        """记录延迟指标"""
        with self._lock:
            if metric_type in self._metrics and isinstance(self._metrics[metric_type], list):
                self._metrics[metric_type].append(value)
                if len(self._metrics[metric_type]) > 1000:
                    self._metrics[metric_type] = self._metrics[metric_type][-500:]

    def record_llm_usage(self, input_tokens: int, output_tokens: int):
        """记录LLM使用量"""
        with self._lock:
            self._metrics["tokens_used"]["input"] += input_tokens
            self._metrics["tokens_used"]["output"] += output_tokens
            self._metrics["api_calls"]["llm"] += 1

    def record_api_call(self, api_type: str):
        """记录API调用次数"""
        with self._lock:
            if api_type in self._metrics["api_calls"]:
                self._metrics["api_calls"][api_type] += 1

    def increment_success(self):
        """增加成功计数"""
        with self._lock:
            self._metrics["success_count"] += 1
            self._record_request_rate()

    def increment_error(self):
        """增加错误计数"""
        with self._lock:
            self._metrics["error_count"] += 1
            self._record_request_rate()

    def increment_concurrent_requests(self):
        """增加并发请求数"""
        with self._lock:
            self._metrics["concurrent_requests"] += 1

    def decrement_concurrent_requests(self):
        """减少并发请求数"""
        with self._lock:
            self._metrics["concurrent_requests"] = max(0, self._metrics["concurrent_requests"] - 1)

    def _record_request_rate(self):
        """记录请求速率"""
        current_second = int(time.time())
        current_minute = current_second // 60

        if current_minute != self._last_second:
            if self._last_second != 0:
                self._metrics["requests_per_minute"].append(self._requests_in_current_minute)
                if len(self._metrics["requests_per_minute"]) > 60:
                    self._metrics["requests_per_minute"] = self._metrics["requests_per_minute"][-30:]
            self._last_second = current_minute
            self._requests_in_current_minute = 1
        else:
            self._requests_in_current_minute += 1

    def get_summary(self) -> dict:
        """获取指标汇总"""
        with self._lock:
            total_latencies = self._metrics["total_latency"]
            llm_latencies = self._metrics["llm_latency"]
            tool_latencies = self._metrics["tool_latency"]
            rag_latencies = self._metrics["rag_latency"]

            total_requests = self._metrics["success_count"] + self._metrics["error_count"]

            return {
                "uptime": (datetime.now() - self._start_time).total_seconds(),
                "total_requests": total_requests,
                "success_rate": self._metrics["success_count"] / total_requests if total_requests > 0 else 0,
                "error_rate": self._metrics["error_count"] / total_requests if total_requests > 0 else 0,
                "avg_total_latency": sum(total_latencies) / len(total_latencies) if total_latencies else 0,
                "p95_total_latency": sorted(total_latencies)[int(len(total_latencies) * 0.95)] if len(total_latencies) >= 20 else 0,
                "avg_llm_latency": sum(llm_latencies) / len(llm_latencies) if llm_latencies else 0,
                "avg_tool_latency": sum(tool_latencies) / len(tool_latencies) if tool_latencies else 0,
                "avg_rag_latency": sum(rag_latencies) / len(rag_latencies) if rag_latencies else 0,
                "total_tokens": self._metrics["tokens_used"]["input"] + self._metrics["tokens_used"]["output"],
                "input_tokens": self._metrics["tokens_used"]["input"],
                "output_tokens": self._metrics["tokens_used"]["output"],
                "api_calls": {
                    "llm": self._metrics["api_calls"]["llm"],
                    "tool": self._metrics["api_calls"]["tool"],
                    "rag": self._metrics["api_calls"]["rag"],
                },
                "concurrent_requests": self._metrics["concurrent_requests"],
                "requests_per_minute": {
                    "current": self._requests_in_current_minute,
                    "average": sum(self._metrics["requests_per_minute"]) / len(self._metrics["requests_per_minute"]) if self._metrics["requests_per_minute"] else 0,
                },
            }

    def reset(self):
        """重置所有指标"""
        with self._lock:
            self._metrics = {
                "total_latency": [],
                "llm_latency": [],
                "tool_latency": [],
                "rag_latency": [],
                "success_count": 0,
                "error_count": 0,
                "tokens_used": {"input": 0, "output": 0},
                "api_calls": {"llm": 0, "tool": 0, "rag": 0},
                "concurrent_requests": 0,
                "requests_per_minute": [],
            }
            self._start_time = datetime.now()


_metrics_instance = None


def get_metrics() -> PerformanceMetrics:
    """获取全局性能指标实例（单例模式）"""
    global _metrics_instance
    if _metrics_instance is None:
        _metrics_instance = PerformanceMetrics()
    return _metrics_instance


class Timer:
    """计时器上下文管理器"""

    def __init__(self, metric_type: str = None):
        self.metric_type = metric_type
        self.start_time = None
        self.metrics = get_metrics()

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            elapsed = time.time() - self.start_time
            if self.metric_type:
                self.metrics.record_latency(self.metric_type, elapsed)
        return False

    def elapsed(self) -> float:
        """获取已流逝时间"""
        if self.start_time is not None:
            return time.time() - self.start_time
        return 0