from typing import Optional, Dict, Any, List
from langchain_core.messages import HumanMessage, SystemMessage
from app.llm import get_llm


class SimpleAgent:
    """
    纯推理 Agent - 不调用工具，仅基于大模型进行逻辑推理
    
    适用于：
    - 行程规划
    - 内容总结
    - 数据分析
    - 报告生成
    - 其他不需要外部工具的任务
    
    使用示例：
        agent = SimpleAgent(
            name="行程规划专家",
            system_prompt="你是一位专业的旅行规划师..."
        )
        result = agent.run("帮我规划北京三日游")
    """
    
    def __init__(
        self,
        name: str,
        llm=None,
        system_prompt: str = "",
        provider: str = "openai",
        model: str = None,
        temperature: float = 0.7,
        verbose: bool = True
    ):
        """
        初始化 SimpleAgent
        
        Args:
            name: Agent 名称
            llm: 预初始化的 LLM 实例（可选）
            system_prompt: 系统提示词，定义 Agent 的角色和行为
            provider: LLM 提供商（openai/aliyun）
            model: 模型名称
            temperature: 温度参数，控制输出随机性
            verbose: 是否输出详细日志
        """
        self.name = name
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.verbose = verbose
        
        # 使用预初始化的 LLM 或创建新的
        if llm is not None:
            self.llm = llm
        else:
            self.llm = get_llm(provider=provider, model=model, temperature=temperature)
        
        if self.verbose:
            print(f"✅ 创建 SimpleAgent: {self.name}")
    
    def run(self, query: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        执行推理任务
        
        Args:
            query: 用户查询
            context: 额外上下文信息（可选）
            
        Returns:
            LLM 生成的响应
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Agent: {self.name}")
            print(f"Query: {query}")
            if context:
                print(f"Context: {context}")
            print(f"{'='*60}")
        
        # 构建消息列表
        messages = []
        
        # 添加系统提示词
        if self.system_prompt:
            messages.append(SystemMessage(content=self.system_prompt))
        
        # 构建用户消息（包含上下文）
        user_message = query
        if context:
            context_str = "\n".join([f"{k}: {v}" for k, v in context.items()])
            user_message = f"上下文信息：\n{context_str}\n\n用户问题：{query}"
        
        messages.append(HumanMessage(content=user_message))
        
        # 调用 LLM
        response = self.llm.invoke(messages)
        result = response.content
        
        if self.verbose:
            print(f"\n响应: {result}")
            print(f"{'='*60}\n")
        
        return result
    
    async def arun(self, query: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        异步执行推理任务
        
        Args:
            query: 用户查询
            context: 额外上下文信息（可选）
            
        Returns:
            LLM 生成的响应
        """
        # 构建消息列表
        messages = []
        
        if self.system_prompt:
            messages.append(SystemMessage(content=self.system_prompt))
        
        user_message = query
        if context:
            context_str = "\n".join([f"{k}: {v}" for k, v in context.items()])
            user_message = f"上下文信息：\n{context_str}\n\n用户问题：{query}"
        
        messages.append(HumanMessage(content=user_message))
        
        response = await self.llm.ainvoke(messages)
        return response.content
    
    def __repr__(self) -> str:
        return f"SimpleAgent(name='{self.name}')"


def create_simple_agent(
    name: str,
    system_prompt: str = "",
    provider: str = "openai",
    model: str = None,
    temperature: float = 0.7,
    verbose: bool = True
) -> SimpleAgent:
    """
    创建 SimpleAgent 的便捷函数
    
    Args:
        name: Agent 名称
        system_prompt: 系统提示词
        provider: LLM 提供商
        model: 模型名称
        temperature: 温度参数
        verbose: 是否输出日志
        
    Returns:
        SimpleAgent 实例
    """
    return SimpleAgent(
        name=name,
        system_prompt=system_prompt,
        provider=provider,
        model=model,
        temperature=temperature,
        verbose=verbose
    )
