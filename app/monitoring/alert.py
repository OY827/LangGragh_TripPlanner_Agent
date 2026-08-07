"""
告警系统模块 - Alert System
监控系统指标，当超过阈值时触发告警
"""

import threading
import time
from typing import Callable, Dict, List, Optional
from datetime import datetime
from .logger import get_logger


class AlertRule:
    """告警规则"""

    def __init__(
        self,
        name: str,
        condition: Callable[[Dict], bool],
        threshold: float,
        severity: str = "warning",
        description: str = "",
        cooldown_minutes: int = 5
    ):
        self.name = name
        self.condition = condition
        self.threshold = threshold
        self.severity = severity
        self.description = description
        self.cooldown_minutes = cooldown_minutes
        self.last_triggered = 0
        self.trigger_count = 0

    def should_trigger(self, metrics: Dict) -> bool:
        """判断是否应该触发告警"""
        now = time.time()
        if now - self.last_triggered < self.cooldown_minutes * 60:
            return False

        if self.condition(metrics):
            self.last_triggered = now
            self.trigger_count += 1
            return True
        return False


class AlertManager:
    """告警管理器"""

    _OPERATORS = {
        ">": lambda a, b: a > b,
        "<": lambda a, b: a < b,
        ">=": lambda a, b: a >= b,
        "<=": lambda a, b: a <= b,
        "==": lambda a, b: a == b,
        "!=": lambda a, b: a != b,
    }

    def __init__(self, check_interval_seconds: int = 30):
        self.rules: List[AlertRule] = []
        self.alerts: List[Dict] = []
        self._lock = threading.Lock()
        self._logger = get_logger("alert")
        self._check_interval = check_interval_seconds
        self._running = False
        self._thread = None

    def add_rule(self, rule: AlertRule):
        """添加告警规则"""
        with self._lock:
            self.rules.append(rule)
            self._logger.log_info("ALERT_RULE_ADDED", rule_name=rule.name, severity=rule.severity)

    def add_rules_from_config(self, config: List[Dict]):
        """从配置添加告警规则"""
        for rule_config in config:
            condition = self._create_condition(
                field=rule_config["field"],
                operator=rule_config["operator"],
                threshold=rule_config["threshold"]
            )
            rule = AlertRule(
                name=rule_config["name"],
                condition=condition,
                threshold=rule_config["threshold"],
                severity=rule_config.get("severity", "warning"),
                description=rule_config.get("description", ""),
                cooldown_minutes=rule_config.get("cooldown_minutes", 5)
            )
            self.add_rule(rule)

    def _create_condition(self, field: str, operator: str, threshold: float) -> Callable[[Dict], bool]:
        """创建条件函数（安全方式）"""
        op_func = self._OPERATORS.get(operator)
        
        def condition(metrics: Dict) -> bool:
            if op_func is None:
                return False
            value = metrics.get(field)
            if value is None:
                return False
            try:
                return op_func(float(value), threshold)
            except (ValueError, TypeError):
                return False
        
        return condition

    def check_alerts(self, metrics: Dict) -> List[Dict]:
        """检查所有告警规则"""
        triggered_alerts = []
        with self._lock:
            for rule in self.rules:
                if rule.should_trigger(metrics):
                    alert = {
                        "name": rule.name,
                        "severity": rule.severity,
                        "description": rule.description,
                        "threshold": rule.threshold,
                        "trigger_count": rule.trigger_count,
                        "timestamp": datetime.now().isoformat()
                    }
                    triggered_alerts.append(alert)
                    self.alerts.append(alert)
                    if len(self.alerts) > 100:
                        self.alerts = self.alerts[-50:]
                    self._notify(alert)

        return triggered_alerts

    def _notify(self, alert: Dict):
        """发送告警通知"""
        severity_color = {
            "critical": "\033[91m",
            "warning": "\033[93m",
            "info": "\033[94m"
        }
        reset_color = "\033[0m"

        message = (
            f"{severity_color.get(alert['severity'], '')}"
            f"[{alert['severity'].upper()}] {alert['name']}"
            f"{reset_color}\n"
            f"  描述: {alert['description']}\n"
            f"  阈值: {alert['threshold']}\n"
            f"  触发次数: {alert['trigger_count']}\n"
            f"  时间: {alert['timestamp']}"
        )

        print(message)
        self._logger.log_warning("ALERT_TRIGGERED", **alert)

    def start_monitoring(self, metrics_provider: Callable[[], Dict]):
        """启动监控线程"""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, args=(metrics_provider,))
        self._thread.daemon = True
        self._thread.start()

    def _monitor_loop(self, metrics_provider: Callable[[], Dict]):
        """监控循环"""
        while self._running:
            try:
                metrics = metrics_provider()
                self.check_alerts(metrics)
            except Exception as e:
                self._logger.log_error("ALERT_ERROR", message=str(e))
            time.sleep(self._check_interval)

    def stop_monitoring(self):
        """停止监控"""
        self._running = False
        if self._thread:
            self._thread.join()

    def get_alerts(self, limit: int = 20) -> List[Dict]:
        """获取最近的告警"""
        with self._lock:
            return self.alerts[-limit:]

    def get_rules_summary(self) -> List[Dict]:
        """获取规则汇总"""
        with self._lock:
            return [{
                "name": rule.name,
                "severity": rule.severity,
                "trigger_count": rule.trigger_count,
                "description": rule.description
            } for rule in self.rules]


_alert_manager_instance = None


def get_alert_manager() -> AlertManager:
    """获取全局告警管理器实例（单例模式）"""
    global _alert_manager_instance
    if _alert_manager_instance is None:
        _alert_manager_instance = AlertManager()
    return _alert_manager_instance


def setup_default_alerts():
    """设置默认告警规则"""
    manager = get_alert_manager()
    rules = [
        {
            "name": "错误率过高",
            "field": "error_rate",
            "operator": ">",
            "threshold": 0.1,
            "severity": "critical",
            "description": "系统错误率超过10%",
            "cooldown_minutes": 5
        },
        {
            "name": "平均响应时间过高",
            "field": "avg_total_latency",
            "operator": ">",
            "threshold": 10,
            "severity": "warning",
            "description": "平均响应时间超过10秒",
            "cooldown_minutes": 3
        },
        {
            "name": "LLM延迟过高",
            "field": "avg_llm_latency",
            "operator": ">",
            "threshold": 5,
            "severity": "warning",
            "description": "LLM调用平均延迟超过5秒",
            "cooldown_minutes": 3
        },
        {
            "name": "并发请求过多",
            "field": "concurrent_requests",
            "operator": ">",
            "threshold": 10,
            "severity": "warning",
            "description": "并发请求数超过10",
            "cooldown_minutes": 2
        },
        {
            "name": "RAG延迟过高",
            "field": "avg_rag_latency",
            "operator": ">",
            "threshold": 3,
            "severity": "info",
            "description": "RAG检索平均延迟超过3秒",
            "cooldown_minutes": 3
        }
    ]
    manager.add_rules_from_config(rules)