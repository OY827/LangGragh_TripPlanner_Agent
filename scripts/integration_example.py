"""
RAG 工具与 Agent 集成示例

展示如何将 RAG 工具注册到现有的 ReAct Agent 中
"""

from typing import List
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from app.rag import get_rag_tools


def create_rag_enhanced_agent(
    llm_api_key: str = None,
    llm_base_url: str = None
) -> AgentExecutor:
    """
    创建一个增强RAG能力的Agent

    Args:
        llm_api_key: LLM API密钥
        llm_base_url: LLM API地址

    Returns:
        AgentExecutor: 可执行的Agent
    """

    # 1. 获取RAG工具
    rag_tools = get_rag_tools()

    # 2. 初始化LLM
    import os
    model_name = os.getenv("DASHSCOPE_MODEL", "qwen-turbo")
    if llm_api_key and llm_base_url:
        llm = ChatOpenAI(
            model=model_name,
            api_key=llm_api_key,
            base_url=llm_base_url,
            temperature=0.7
        )
    else:
        # 使用通义千问
        from langchain_community.chat_models import Tongyi
        llm = Tongyi(model_name=model_name, temperature=0.7)

    # 3. 创建Prompt
    prompt = PromptTemplate.from_template("""你是一个专业的旅行规划助手，擅长根据用户需求推荐景点和规划行程。

你可以使用以下工具：
{tools}

工具名称和功能：
{tool_names}

请用中文回答用户的问题。

用户问题：{input}

{agent_scratchpad}
""")

    # 4. 创建Agent
    agent = create_react_agent(
        llm=llm,
        tools=rag_tools,
        prompt=prompt
    )

    # 5. 创建执行器
    agent_executor = AgentExecutor(
        agent=agent,
        tools=rag_tools,
        verbose=True,
        max_iterations=10,
        handle_parsing_errors=True
    )

    return agent_executor


def create_hybrid_agent():
    """
    创建混合Agent（原有工具 + RAG工具）

    将RAG工具与原有的高德地图工具结合使用
    """

    # 获取原有工具
    from app.tools import get_all_tools
    original_tools = get_all_tools()

    # 获取RAG工具
    rag_tools = get_rag_tools()

    # 合并工具列表
    all_tools = original_tools + rag_tools

    # 初始化LLM
    import os
    model_name = os.getenv("DASHSCOPE_MODEL", "qwen-turbo")
    from langchain_community.chat_models import Tongyi
    llm = Tongyi(model_name=model_name, temperature=0.7)

    # 创建Prompt（包含更多工具描述）
    prompt = PromptTemplate.from_template("""你是一个智能旅行规划助手，结合知识库和实时API为用户提供服务。

可用工具：
{tools}

工具功能说明：
- rag_通用查询：从知识库检索景点、城市相关信息
- 景点推荐：根据城市和偏好推荐景点
- 景点详情：查询特定景点的详细信息
- 旅行攻略：获取旅行攻略和路线建议
- AmapWeather：高德地图天气查询
- POISearch：高德地图POI搜索
- HotelSearch：酒店搜索
- 等等...

请用中文回答用户问题。

用户：{input}

{agent_scratchpad}
""")

    # 创建Agent
    agent = create_react_agent(
        llm=llm,
        tools=all_tools,
        prompt=prompt
    )

    agent_executor = AgentExecutor(
        agent=agent,
        tools=all_tools,
        verbose=True,
        max_iterations=15
    )

    return agent_executor


# ==================== 使用示例 ====================

if __name__ == "__main__":

    print("=" * 60)
    print("🤖 RAG Agent 集成示例")
    print("=" * 60)

    # 方式1: 仅RAG增强的Agent
    print("\n📌 方式1: 仅RAG增强的Agent")
    print("-" * 40)

    try:
        agent = create_rag_enhanced_agent()

        # 测试问答
        result = agent.invoke({
            "input": "帮我介绍一下北京有哪些值得去的景点？"
        })
        print("\n🤖 Agent回答:")
        print(result["output"])
    except Exception as e:
        print(f"❌ Agent执行失败: {e}")

    # 方式2: 混合Agent
    print("\n\n📌 方式2: 混合Agent（原有工具 + RAG工具）")
    print("-" * 40)

    try:
        hybrid_agent = create_hybrid_agent()

        # 测试复杂问题
        result = hybrid_agent.invoke({
            "input": "我想去北京玩3天，喜欢历史文化，请问有哪些推荐？"
        })
        print("\n🤖 Agent回答:")
        print(result["output"])
    except Exception as e:
        print(f"❌ Hybrid Agent执行失败: {e}")
