"""
监控模块测试脚本
验证性能监控、结构化日志和告警系统是否正常工作
"""

import time
import sys
sys.path.append("d:/Projects/LangChain")

from app.monitoring import get_metrics, get_logger, get_alert_manager, setup_default_alerts, Timer


def test_performance_metrics():
    """测试性能监控"""
    print("=== 测试性能监控 ===")
    
    metrics = get_metrics()
    
    metrics.record_latency("total_latency", 1.23)
    metrics.record_latency("total_latency", 2.34)
    metrics.record_latency("llm_latency", 0.56)
    metrics.record_latency("tool_latency", 0.12)
    metrics.record_latency("rag_latency", 0.34)
    
    metrics.record_llm_usage(1000, 500)
    metrics.record_llm_usage(1500, 800)
    
    metrics.record_api_call("llm")
    metrics.record_api_call("tool")
    metrics.record_api_call("rag")
    
    metrics.increment_success()
    metrics.increment_success()
    metrics.increment_error()
    
    summary = metrics.get_summary()
    print("性能指标汇总:")
    print(f"  总请求数: {summary['total_requests']}")
    print(f"  成功率: {summary['success_rate']:.2%}")
    print(f"  错误率: {summary['error_rate']:.2%}")
    print(f"  平均总延迟: {summary['avg_total_latency']:.2f}s")
    print(f"  平均LLM延迟: {summary['avg_llm_latency']:.2f}s")
    print(f"  平均工具延迟: {summary['avg_tool_latency']:.2f}s")
    print(f"  平均RAG延迟: {summary['avg_rag_latency']:.2f}s")
    print(f"  总Token数: {summary['total_tokens']}")
    print(f"  API调用次数: {summary['api_calls']}")
    
    assert summary['total_requests'] == 3
    assert summary['success_rate'] == 2/3
    assert summary['avg_total_latency'] > 0
    print("✅ 性能监控测试通过\n")


def test_timer():
    """测试计时器"""
    print("=== 测试计时器 ===")
    
    with Timer("test_latency") as timer:
        time.sleep(0.1)
    
    assert timer.elapsed() >= 0.1
    print(f"  计时结果: {timer.elapsed():.3f}s")
    print("✅ 计时器测试通过\n")


def test_structured_logger():
    """测试结构化日志"""
    print("=== 测试结构化日志 ===")
    
    logger = get_logger("test")
    
    logger.log_request("user123", "测试请求内容")
    logger.log_agent_step("TestAgent", "step1", "think", "输入", "输出")
    logger.log_tool_call("SearchTool", {"query": "测试"}, "工具结果", 0.5, True)
    logger.log_llm_call("qwen-turbo", 1000, 500, 1.2)
    logger.log_rag_call("测试查询", 5, 0.3, "hybrid")
    
    print("✅ 结构化日志测试通过（查看控制台输出）\n")


def test_alert_system():
    """测试告警系统"""
    print("=== 测试告警系统 ===")
    
    setup_default_alerts()
    alert_manager = get_alert_manager()
    
    test_metrics = {
        "error_rate": 0.2,
        "avg_total_latency": 15,
        "avg_llm_latency": 6,
        "concurrent_requests": 15,
        "avg_rag_latency": 4
    }
    
    alerts = alert_manager.check_alerts(test_metrics)
    print(f"  触发告警数量: {len(alerts)}")
    
    for alert in alerts:
        print(f"    - [{alert['severity'].upper()}] {alert['name']}")
    
    assert len(alerts) > 0
    print("✅ 告警系统测试通过\n")


def test_metrics_singleton():
    """测试单例模式"""
    print("=== 测试单例模式 ===")
    
    metrics1 = get_metrics()
    metrics2 = get_metrics()
    
    assert metrics1 is metrics2
    print("✅ 单例模式测试通过\n")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("监控模块测试")
    print("="*60 + "\n")
    
    test_performance_metrics()
    test_timer()
    test_structured_logger()
    test_alert_system()
    test_metrics_singleton()
    
    print("="*60)
    print("所有测试通过！")
    print("="*60)