from typing import List, Dict, Any
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from app.llm import get_llm
from app.monitoring import get_metrics, get_logger, Timer


def _message_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("text"):
                parts.append(str(block.get("text")))
        return "\n".join(parts)
    return str(content)


class ReActAgent:
    """
    带工具的推理 Agent，使用模型原生 function calling（bind_tools）。

    循环：
    1. llm.bind_tools(tools).invoke(messages)
    2. 若 AIMessage.tool_calls 非空，执行对应 BaseTool，把结果以 ToolMessage 回填
    3. 没有 tool_calls 时，把模型文本当作最终答案

    适用于需要实时数据的景点 / 天气 / 酒店专家。规划师用 SimpleAgent，不走这里。
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

        self._bind_llm()

        if self.enable_monitoring:
            self._metrics = get_metrics()
            self._logger = get_logger(self.name)

        if self.verbose:
            print(f"✅ 创建 ReActAgent: {self.name}")
            print(f"   工具: {list(self.tool_names.keys())}")
            print("   调用方式: 原生 tool calling")

    def _bind_llm(self):
        """bind_tools 返回新 runnable，不会改掉 get_llm 缓存的那份 ChatOpenAI。"""
        if self.tools:
            self.llm_with_tools = self.llm.bind_tools(self.tools)
        else:
            self.llm_with_tools = self.llm

    def _build_system_prompt(self) -> str:
        tools_desc = self._format_tools_description()
        protocol = (
            "你通过 API 原生 function calling 使用工具，不要输出「思考：」「行动：」这类文本格式。"
            "需要实时或外部数据时必须调用工具，不要编造。"
            "工具结果足够后，用中文直接给出最终答案。"
        )
        if self.tools:
            protocol += f"\n\n可用工具：\n{tools_desc}"
        if self.system_prompt:
            return f"{self.system_prompt.strip()}\n\n{protocol}"
        return protocol

    def _format_tools_description(self) -> str:
        return "\n".join(f"- {tool.name}: {tool.description}" for tool in self.tools)

    def _get_tool_calls(self, response: AIMessage) -> List[Dict[str, Any]]:
        calls = getattr(response, "tool_calls", None) or []
        if calls:
            return list(calls)

        extra = getattr(response, "additional_kwargs", None) or {}
        raw = extra.get("tool_calls")
        if not raw and extra.get("function_call"):
            raw = [{"id": extra.get("tool_call_id") or str(uuid4()), "function": extra["function_call"]}]
        parsed = []
        for item in raw or []:
            if not isinstance(item, dict):
                continue
            fn = item.get("function") or {}
            parsed.append({
                "name": item.get("name") or fn.get("name") or "",
                "args": item.get("args") if item.get("args") is not None else fn.get("arguments") or {},
                "id": item.get("id") or str(uuid4()),
            })
        return parsed

    def _execute_tool_call(self, call: Dict[str, Any]) -> str:
        name = (call.get("name") or "").strip()
        args = call.get("args")
        if isinstance(args, str):
            import json
            try:
                args = json.loads(args) if args else {}
            except json.JSONDecodeError:
                args = {"input": args}

        tool = self.tool_names.get(name)
        if tool is None:
            for tool_name, candidate in self.tool_names.items():
                if tool_name in name or name in tool_name:
                    tool = candidate
                    break
        if tool is None:
            return f"错误：未找到工具 '{name}'。可用工具：{list(self.tool_names.keys())}"

        try:
            if isinstance(args, dict):
                result = tool.invoke(args)
            elif args in (None, ""):
                result = tool.invoke({})
            else:
                result = tool.invoke(args)
            return str(result)
        except Exception as e:
            return f"执行工具时出错：{str(e)}"

    def run(self, query: str) -> str:
        """执行推理任务（原生 tool calling）。"""
        messages: List[Any] = [
            SystemMessage(content=self._build_system_prompt()),
            HumanMessage(content=query),
        ]

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
        last_text = ""

        try:
            for iteration in range(self.max_iterations):
                if self.verbose:
                    print(f"\n[迭代 {iteration + 1}/{self.max_iterations}]")

                llm_timer = Timer("llm_latency")
                llm_timer.__enter__()
                try:
                    response = self.llm_with_tools.invoke(messages)
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

                messages.append(response)
                text = _message_text(getattr(response, "content", "")).strip()
                last_text = text
                tool_calls = self._get_tool_calls(response)

                if self.verbose:
                    preview = text[:500] if text else "(无文本，可能仅有 tool_calls)"
                    print(f"LLM 响应:\n{preview}")
                    if tool_calls:
                        print(f"tool_calls: {[c.get('name') for c in tool_calls]}")

                if not tool_calls:
                    if text:
                        if self.verbose:
                            print(f"\n{'='*60}")
                            print(f"最终答案：{text}")
                            print(f"{'='*60}\n")
                        if self.enable_monitoring:
                            self._metrics.increment_success()
                            self._logger.log_agent_step(
                                agent_name=self.name,
                                step=f"iteration_{iteration}",
                                action="finish",
                                input_data=query,
                                output_data=text,
                            )
                        return text
                    break

                for call in tool_calls:
                    call_id = call.get("id") or call.get("tool_call_id") or str(uuid4())
                    call_name = call.get("name") or ""
                    if self.verbose:
                        print(f"行动：{call_name}")
                        print(f"行动输入：{call.get('args')}")

                    tool_timer = Timer("tool_latency")
                    tool_timer.__enter__()
                    try:
                        observation = self._execute_tool_call(call)
                        if self.enable_monitoring:
                            self._metrics.record_api_call("tool")
                            is_success = not observation.startswith("错误") and not observation.startswith("执行工具时出错")
                            self._logger.log_tool_call(
                                tool_name=call_name,
                                input_data=str(call.get("args")),
                                output_data=observation,
                                latency=tool_timer.elapsed(),
                                success=is_success,
                            )
                    finally:
                        tool_timer.__exit__(None, None, None)

                    if self.verbose:
                        print(f"观察：{observation[:500]}")

                    tool_msg_kwargs = {
                        "content": observation,
                        "tool_call_id": str(call_id),
                    }
                    if call_name:
                        tool_msg_kwargs["name"] = call_name
                    messages.append(ToolMessage(**tool_msg_kwargs))

            if self.verbose:
                print(f"\n{'='*60}")
                print(f"已达到最大迭代次数 ({self.max_iterations})")
                print(f"{'='*60}\n")

            if self.enable_monitoring:
                self._metrics.increment_error()
                self._logger.log_error("max_iterations", f"达到最大迭代次数 {self.max_iterations}")

            return last_text or f"我无法在 {self.max_iterations} 次迭代内完成任务。"

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
        """添加工具并重新 bind_tools。"""
        self.tools.append(tool)
        self.tool_names[tool.name] = tool
        self._bind_llm()

    def remove_tool(self, tool_name: str):
        """移除工具并重新 bind_tools。"""
        if tool_name in self.tool_names:
            tool = self.tool_names.pop(tool_name)
            self.tools.remove(tool)
            self._bind_llm()

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
