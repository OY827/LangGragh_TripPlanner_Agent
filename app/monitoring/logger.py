"""
结构化日志模块 - Structured Logger
提供统一的结构化日志记录功能
"""

import logging
import json
import os
from datetime import datetime
from typing import Optional, Any
from logging.handlers import RotatingFileHandler


class AgentLogger:
    """Agent结构化日志记录器"""

    def __init__(self, name: str = "agent", log_level: str = "INFO"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, log_level.upper()))
        self.logger.propagate = False

        if not self.logger.handlers:
            self._setup_handlers()

    def _setup_handlers(self):
        """设置日志处理器"""
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
        os.makedirs(log_dir, exist_ok=True)

        file_handler = RotatingFileHandler(
            os.path.join(log_dir, "agent.log"),
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8"
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def _format_message(self, event_type: str, **kwargs) -> str:
        """格式化日志消息"""
        log_data = {
            "event_type": event_type,
            "timestamp": datetime.now().isoformat(),
            **kwargs
        }
        return json.dumps(log_data, ensure_ascii=False)

    def log_request(self, user_id: str, query: str, request_id: str = ""):
        """记录用户请求"""
        message = self._format_message(
            "REQUEST",
            user_id=user_id,
            query=query[:200],
            request_id=request_id
        )
        self.logger.info(message)

    def log_agent_step(self, agent_name: str, step: str, action: str, input_data: Any, output_data: Any):
        """记录Agent执行步骤"""
        message = self._format_message(
            "AGENT_STEP",
            agent_name=agent_name,
            step=step,
            action=action,
            input=str(input_data)[:200],
            output=str(output_data)[:200]
        )
        self.logger.info(message)

    def log_tool_call(self, tool_name: str, input_data: dict, output_data: Any, latency: float, success: bool):
        """记录工具调用"""
        message = self._format_message(
            "TOOL_CALL",
            tool_name=tool_name,
            input=str(input_data)[:200],
            output=str(output_data)[:200],
            latency=latency,
            success=success
        )
        level = logging.INFO if success else logging.WARNING
        self.logger.log(level, message)

    def log_llm_call(self, model: str, prompt_tokens: int, completion_tokens: int, latency: float):
        """记录LLM调用"""
        message = self._format_message(
            "LLM_CALL",
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency=latency
        )
        self.logger.info(message)

    def log_rag_call(self, query: str, num_results: int, latency: float, search_type: str = "similarity"):
        """记录RAG调用"""
        message = self._format_message(
            "RAG_CALL",
            query=query[:100],
            num_results=num_results,
            latency=latency,
            search_type=search_type
        )
        self.logger.info(message)

    def log_error(self, error_type: str, message: str, traceback: str = ""):
        """记录错误"""
        message = self._format_message(
            "ERROR",
            error_type=error_type,
            message=message[:200],
            traceback=traceback[:300]
        )
        self.logger.error(message)

    def log_performance(self, metrics: dict):
        """记录性能指标"""
        message = self._format_message(
            "PERFORMANCE",
            **metrics
        )
        self.logger.debug(message)

    def log_info(self, event_type: str, **kwargs):
        """通用信息日志"""
        message = self._format_message(event_type, **kwargs)
        self.logger.info(message)

    def log_warning(self, event_type: str, **kwargs):
        """警告日志"""
        message = self._format_message(event_type, **kwargs)
        self.logger.warning(message)


_logger_instance = None


def get_logger(name: str = "agent") -> AgentLogger:
    """获取全局日志实例（单例模式）"""
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = AgentLogger(name)
    return _logger_instance