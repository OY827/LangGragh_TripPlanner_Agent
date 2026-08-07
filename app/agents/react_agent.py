from typing import List, Optional, Dict, Any, Callable
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from app.llm import get_llm
from app.monitoring import get_metrics, get_logger, Timer


class ReActAgent:
    """
    ReAct Agent - 支持工具调用的推理 Agent
    
    使用 ReAct（Reasoning + Acting）范式：
    1. 思考：分析问题，决定是否需要调用工具
    2. 行动：调用工具执行操作
    3. 观察：获取工具返回结果
    4. 总结：基于结果给出最终回答
    
    适用于：
    - 需要调用外部工具的任务
    - 多步骤推理任务
    - 需要获取实时信息的任务
    
    使用示例：
        tools = [CalculatorTool(), SearchTool()]
        agent = ReActAgent(
            name="景点搜索专家",
            tools=tools,
            system_prompt="你是景点搜索专家..."
        )
        result = agent.run("帮我搜索北京的景点")
    """
    
    def __init__(
        self,
        name: str,
        tools: List[BaseTool],
        llm=None,
        system_prompt: str = "",
        provider: str = "openai",
        model: str = None,
        temperature: float = 0.7,
        max_iterations: int = 10,
        verbose: bool = True,
        enable_monitoring: bool = True
    ):
        """
        初始化 ReActAgent
        
        Args:
            name: Agent 名称
            tools: 工具列表
            llm: 预初始化的 LLM 实例（可选）
            system_prompt: 系统提示词
            provider: LLM 提供商
            model: 模型名称
            temperature: 温度参数
            max_iterations: 最大迭代次数
            verbose: 是否输出详细日志
            enable_monitoring: 是否启用监控
        """
        self.name = name
        self.tools = tools
        self.tool_names = {tool.name: tool for tool in tools}
        self.system_prompt = system_prompt
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.enable_monitoring = enable_monitoring
        
        if llm is not None:
            self.llm = llm
        else:
            self.llm = get_llm(provider=provider, model=model, temperature=temperature)
        
        # 构建默认的 ReAct 提示词
        self._build_react_prompt()
        
        # 初始化监控模块
        if self.enable_monitoring:
            self._metrics = get_metrics()
            self._logger = get_logger(self.name)
        
        if self.verbose:
            print(f"✅ 创建 ReActAgent: {self.name}")
            print(f"   工具: {list(self.tool_names.keys())}")
    
    def _build_react_prompt(self):
        """构建 ReAct 风格的系统提示词"""
        tools_desc = self._format_tools_description()
        tool_names = ", ".join(self.tool_names.keys())
        
        default_prompt = f"""你是一个使用 ReAct（推理+行动）方法解决问题的AI助手。

你可以使用以下工具：
{tools_desc}

请使用以下格式：

问题：你需要回答的输入问题
思考：你应该思考该做什么
行动：要执行的操作，应该是[{tool_names}]中的一个
行动输入：行动的输入参数
观察：行动的结果
...（这个思考/行动/行动输入/观察可以重复N次）
思考：我现在知道最终答案了
最终答案：原始输入问题的最终答案

开始！"""
        
        # 如果有自定义系统提示词，添加到默认提示词前面
        if self.system_prompt:
            self.react_prompt = f"{self.system_prompt}\n\n{default_prompt}"
        else:
            self.react_prompt = default_prompt
    
    def _format_tools_description(self) -> str:
        """格式化工具描述"""
        tools_desc = []
        for tool in self.tools:
            tools_desc.append(f"{tool.name}: {tool.description}")
        return "\n".join(tools_desc)
    
    def _parse_llm_output(self, text: str) -> Dict[str, str]:
        """解析 LLM 输出"""
        lines = text.strip().split("\n")
        result = {
            "thought": "",
            "action": "",
            "action_input": "",
            "final_answer": ""
        }
        
        in_final_answer = False
        final_answer_lines = []
        
        for line in lines:
            line = line.strip()
            
            if line.startswith("思考："):
                result["thought"] = line.replace("思考：", "").strip()
            elif line.startswith("行动："):
                result["action"] = line.replace("行动：", "").strip()
            elif line.startswith("行动输入："):
                result["action_input"] = line.replace("行动输入：", "").strip()
            elif line.startswith("最终答案："):
                in_final_answer = True
                final_answer_lines.append(line.replace("最终答案：", "").strip())
            elif in_final_answer:
                final_answer_lines.append(line)
        
        if final_answer_lines:
            result["final_answer"] = "\n".join(final_answer_lines).strip()
        
        return result
    
    def _execute_action(self, action_name: str, action_input: str) -> str:
        """执行工具调用"""
        if action_name not in self.tool_names:
            return f"错误：未找到工具 '{action_name}'。可用工具：{list(self.tool_names.keys())}"
        
        tool = self.tool_names[action_name]
        try:
            import json
            try:
                input_dict = json.loads(action_input)
                if isinstance(input_dict, dict):
                    from langchain_core.tools import StructuredTool
                    if isinstance(tool, StructuredTool):
                        result = tool.invoke(input_dict)
                    else:
                        result = tool._run(**input_dict)
                else:
                    result = tool._run(action_input)
            except json.JSONDecodeError:
                result = tool._run(action_input)
            
            return str(result)
        except Exception as e:
            return f"执行工具时出错：{str(e)}"
    
    def run(self, query: str) -> str:
        """执行推理任务（支持工具调用）"""
        history = ""
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Agent: {self.name}")
            print(f"问题：{query}")
            print(f"{'='*60}")
        
        if self.enable_monitoring:
            self._metrics.increment_concurrent_requests()
            self._logger.log_request("unknown", query)
        
        total_timer = Timer("total_latency")
        total_timer.__enter__()
        
        try:
            for iteration in range(self.max_iterations):
                if self.verbose:
                    print(f"\n[迭代 {iteration + 1}/{self.max_iterations}]")
                
                prompt = f"{self.react_prompt}\n\n问题：{query}\n思考：{history}"
                messages = [HumanMessage(content=prompt)]
                
                llm_timer = Timer("llm_latency")
                llm_timer.__enter__()
                try:
                    response = self.llm.invoke(messages)
                    response_text = response.content
                    if self.enable_monitoring:
                        self._metrics.record_api_call("llm")
                        self._logger.log_llm_call(
                            model=self.model or "unknown",
                            prompt_tokens=0,
                            completion_tokens=0,
                            latency=llm_timer.elapsed()
                        )
                finally:
                    llm_timer.__exit__(None, None, None)
                
                if self.verbose:
                    print(f"LLM 响应:\n{response_text}")
                
                parsed = self._parse_llm_output(response_text)
                
                if parsed["final_answer"]:
                    if self.verbose:
                        print(f"\n{'='*60}")
                        print(f"最终答案：{parsed['final_answer']}")
                        print(f"{'='*60}\n")
                    if self.enable_monitoring:
                        self._metrics.increment_success()
                        self._logger.log_agent_step(
                            agent_name=self.name,
                            step=f"iteration_{iteration}",
                            action="finish",
                            input_data=query,
                            output_data=parsed["final_answer"]
                        )
                    return parsed["final_answer"]
                
                if parsed["action"] and parsed["action_input"]:
                    if self.verbose:
                        print(f"行动：{parsed['action']}")
                        print(f"行动输入：{parsed['action_input']}")
                    
                    tool_timer = Timer("tool_latency")
                    tool_timer.__enter__()
                    try:
                        observation = self._execute_action(parsed["action"], parsed["action_input"])
                        if self.enable_monitoring:
                            self._metrics.record_api_call("tool")
                            is_success = not observation.startswith("错误") and not observation.startswith("执行工具时出错")
                            self._logger.log_tool_call(
                                tool_name=parsed["action"],
                                input_data=parsed["action_input"],
                                output_data=observation,
                                latency=tool_timer.elapsed(),
                                success=is_success
                            )
                    finally:
                        tool_timer.__exit__(None, None, None)
                    
                    if self.verbose:
                        print(f"观察：{observation}")
                    
                    history += f"{parsed['thought']}\n行动：{parsed['action']}\n行动输入：{parsed['action_input']}\n观察：{observation}\n"
                else:
                    if parsed["thought"]:
                        history += f"{parsed['thought']}\n"
                    else:
                        history += response_text + "\n"
            
            if self.verbose:
                print(f"\n{'='*60}")
                print(f"已达到最大迭代次数 ({self.max_iterations})")
                print(f"{'='*60}\n")
            
            if self.enable_monitoring:
                self._metrics.increment_error()
                self._logger.log_error("max_iterations", f"达到最大迭代次数 {self.max_iterations}")
            
            return f"我无法在 {self.max_iterations} 次迭代内完成任务。最后响应是：{response_text}"
        
        except Exception as e:
            if self.enable_monitoring:
                self._metrics.increment_error()
                self._logger.log_error("agent_error", str(e))
            raise
        finally:
            total_timer.__exit__(None, None, None)
            if self.enable_monitoring:
                self._metrics.decrement_concurrent_requests()
    
    def add_tool(self, tool: BaseTool):
        """添加工具"""
        self.tools.append(tool)
        self.tool_names[tool.name] = tool
        self._build_react_prompt()  # 重新构建提示词
    
    def remove_tool(self, tool_name: str):
        """移除工具"""
        if tool_name in self.tool_names:
            tool = self.tool_names.pop(tool_name)
            self.tools.remove(tool)
            self._build_react_prompt()  # 重新构建提示词
    
    def __repr__(self) -> str:
        return f"ReActAgent(name='{self.name}', tools={list(self.tool_names.keys())})"


def create_react_agent(
    name: str,
    tools: List[BaseTool],
    system_prompt: str = "",
    provider: str = "openai",
    model: str = None,
    temperature: float = 0.7,
    max_iterations: int = 10,
    verbose: bool = True,
    enable_monitoring: bool = True
) -> ReActAgent:
    """
    创建 ReActAgent 的便捷函数
    
    Args:
        name: Agent 名称
        tools: 工具列表
        system_prompt: 系统提示词
        provider: LLM 提供商
        model: 模型名称
        temperature: 温度参数
        max_iterations: 最大迭代次数
        verbose: 是否输出日志
        enable_monitoring: 是否启用监控
        
    Returns:
        ReActAgent 实例
    """
    return ReActAgent(
        name=name,
        tools=tools,
        system_prompt=system_prompt,
        provider=provider,
        model=model,
        temperature=temperature,
        max_iterations=max_iterations,
        verbose=verbose,
        enable_monitoring=enable_monitoring
    )
