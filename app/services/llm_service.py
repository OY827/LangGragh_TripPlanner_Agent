"""LLM服务封装"""

import os
from typing import Any, List, Optional

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

# 加载环境变量
load_dotenv()


def get_openai_llm(temperature: float = 0.7, model: str = "gpt-4") -> ChatOpenAI:
    """
    获取 OpenAI LLM 实例
    
    Args:
        temperature: 温度参数，控制输出的随机性
        model: 模型名称
    
    Returns:
        ChatOpenAI 实例
    
    Raises:
        ValueError: 当 OPENAI_API_KEY 未配置时
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set in environment variables")

    from app.monitoring.token_usage import token_usage_handler

    return ChatOpenAI(
        api_key=api_key,
        model=model,
        temperature=temperature,
        callbacks=[token_usage_handler()],
    )


def get_aliyun_llm(temperature: float = 0.7, model: str = None) -> ChatOpenAI:
    """
    获取阿里云 DashScope LLM 实例（使用 OpenAI 兼容模式）
    
    Args:
        temperature: 温度参数，控制输出的随机性
        model: 模型名称，默认从环境变量读取
    
    Returns:
        ChatOpenAI 实例（配置为调用阿里云 DashScope）
    
    Raises:
        ValueError: 当 DASHSCOPE_API_KEY 未配置时
    """
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError("DASHSCOPE_API_KEY is not set in environment variables")

    if model is None:
        model = os.getenv("DASHSCOPE_MODEL", "qwen-turbo")

    from app.monitoring.token_usage import token_usage_handler

    # 使用 OpenAI 兼容模式调用阿里云 DashScope。
    # qwen3 默认会把思考过程算进 completion，规划一次能写到两万 token。
    extra_body = {"enable_thinking": False} if "qwen3" in model.lower() else None
    return ChatOpenAI(
        api_key=api_key,
        model=model,
        temperature=temperature,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        timeout=180,
        max_tokens=4096,
        extra_body=extra_body,
        callbacks=[token_usage_handler()],
    )


# 全局LLM实例缓存
_llm_instances = {}


def get_llm(provider: str = "openai", temperature: float = 0.7, model: str = None) -> BaseChatModel:
    """
    获取 LLM 实例（单例模式）
    
    Args:
        provider: 服务商，支持 "openai" 或 "aliyun"
        temperature: 温度参数，控制输出的随机性
        model: 模型名称
    
    Returns:
        BaseChatModel 实例
    """
    # 构建缓存键
    cache_key = f"{provider}_{temperature}_{model}"

    # 如果缓存中存在，直接返回
    if cache_key in _llm_instances:
        return _llm_instances[cache_key]

    # 创建新实例
    if provider.lower() == "aliyun":
        if model is None:
            model = os.getenv("DASHSCOPE_MODEL", "qwen-turbo")
        llm = get_aliyun_llm(temperature, model)
    else:
        if model is None:
            model = "gpt-4"
        llm = get_openai_llm(temperature, model)

    # 缓存实例
    _llm_instances[cache_key] = llm

    print(f"✅ LLM服务初始化成功 - 服务商: {provider}, 模型: {model}")

    return llm


# 测试代码
if __name__ == "__main__":
    print("=== 测试LLM服务 ===")

    # 测试阿里云LLM
    try:
        llm = get_llm(provider="aliyun", temperature=0.5)
        response = llm.invoke("你好，请介绍一下自己")
        print(f"\n阿里云LLM响应: {response.content[:100]}...")
    except Exception as e:
        print(f"\n阿里云LLM测试失败: {str(e)}")

    # 测试OpenAI LLM
    try:
        llm = get_llm(provider="openai", temperature=0.5)
        response = llm.invoke("Hello, introduce yourself")
        print(f"\nOpenAI LLM响应: {response.content[:100]}...")
    except Exception as e:
        print(f"\nOpenAI LLM测试失败: {str(e)}")

    print("\n=== 测试完成 ===")
