"""带工具的 Agent：LangGraph prebuilt create_react_agent。

循环在图里完成：模型节点产生 tool_calls 时进入工具节点，否则结束。
传入 response_format 时，图在循环结束后再调用 with_structured_output，
结果放在 structured_response。
"""

from __future__ import annotations

import json
from typing import Any, List, Optional, Union

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import BaseTool
from langgraph.errors import GraphRecursionError
from langgraph.prebuilt import create_react_agent as create_tool_agent
from pydantic import BaseModel

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
    """LangGraph 工具循环。run() 仍返回字符串，方便原有调用方解析。"""

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
        enable_monitoring: bool = True,
        response_format: Optional[Union[type[BaseModel], tuple]] = None,
    ):
        self.name = name
        self.tools = list(tools or [])
        self.tool_names = {tool.name: tool for tool in self.tools}
        self.system_prompt = system_prompt
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.enable_monitoring = enable_monitoring
        self.response_format = response_format

        if llm is not None:
            self.llm = llm
        else:
            self.llm = get_llm(provider=provider, model=model, temperature=temperature)

        self._graph = None
        self._compile()

        if self.enable_monitoring:
            self._metrics = get_metrics()
            self._logger = get_logger(self.name)

        if self.verbose:
            print(f"✅ 创建 ReActAgent: {self.name}")
            print(f"   工具: {list(self.tool_names.keys())}")
            print("   调用方式: LangGraph create_react_agent")

    def _build_system_prompt(self) -> str:
        tools_desc = "\n".join(f"- {tool.name}: {tool.description}" for tool in self.tools)
        protocol = (
            "你通过 API 原生 function calling 使用工具，不要输出「思考：」「行动：」这类文本格式。"
            "需要实时或外部数据时必须调用工具，不要编造。"
            "工具结果足够后，用中文直接给出最终答案。"
        )
        if tools_desc:
            protocol += f"\n\n可用工具：\n{tools_desc}"
        if self.system_prompt:
            return f"{self.system_prompt.strip()}\n\n{protocol}"
        return protocol

    def _compile(self) -> None:
        if not self.tools:
            self._graph = None
            return
        kwargs = {
            "model": self.llm,
            "tools": self.tools,
            "prompt": self._build_system_prompt(),
            "name": self.name,
        }
        if self.response_format is not None:
            kwargs["response_format"] = self.response_format
        self._graph = create_tool_agent(**kwargs)

    def _recursion_limit(self) -> int:
        # 每一轮是模型节点加工具节点，结构化输出再占一步。
        return max(8, self.max_iterations * 3 + 2)

    def _result_text(self, result: dict) -> str:
        structured = result.get("structured_response")
        if isinstance(structured, BaseModel):
            return json.dumps(structured.model_dump(), ensure_ascii=False)
        if isinstance(structured, dict):
            return json.dumps(structured, ensure_ascii=False)
        messages = result.get("messages") or []
        for message in reversed(messages):
            if isinstance(message, AIMessage):
                text = _message_text(message.content).strip()
                if text:
                    return text
        return ""

    def _trace_react(self, result: dict) -> str:
        """把每一轮的工具调用和工具返回整理成可读文本。"""
        messages = result.get("messages") or []
        lines = []
        step = 0
        for message in messages:
            tool_calls = getattr(message, "tool_calls", None) or []
            if tool_calls:
                thought = _message_text(getattr(message, "content", "")).strip()
                for call in tool_calls:
                    step += 1
                    args = call.get("args") if isinstance(call, dict) else {}
                    name = call.get("name") if isinstance(call, dict) else ""
                    lines.append(f"[ReAct] 第{step}轮")
                    if thought:
                        lines.append(f"  模型说明: {thought[:400]}")
                        thought = ""
                    lines.append(
                        f"  调用 {name} {json.dumps(args or {}, ensure_ascii=False)}"
                    )
                continue
            if getattr(message, "type", "") == "tool":
                content = _message_text(getattr(message, "content", "")).strip()
                if len(content) > 800:
                    content = content[:800] + "…"
                lines.append(f"  工具返回: {content}")
        return "\n".join(lines)

    def run(self, query: str) -> str:
        """执行工具循环，返回最终文本或结构化结果的 JSON。"""
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
            if self._graph is None:
                response = self.llm.invoke([
                    HumanMessage(content=f"{self._build_system_prompt()}\n\n{query}")
                ])
                return _message_text(getattr(response, "content", response)).strip()

            result = self._graph.invoke(
                {"messages": [HumanMessage(content=query)]},
                config={"recursion_limit": self._recursion_limit()},
            )
            text = self._result_text(result if isinstance(result, dict) else {})
            trace = self._trace_react(result if isinstance(result, dict) else {})
            if trace:
                print(trace)
            if self.verbose:
                print(f"\n{'='*60}")
                print(f"最终答案：{text}")
                print(f"{'='*60}\n")
            if self.enable_monitoring:
                self._metrics.increment_success()
                if trace:
                    self._logger.log_agent_step(
                        agent_name=self.name,
                        step="graph",
                        action="trace",
                        input_data=query,
                        output_data=trace,
                    )
                self._logger.log_agent_step(
                    agent_name=self.name,
                    step="graph",
                    action="finish",
                    input_data=query,
                    output_data=text,
                )
            return text
        except GraphRecursionError:
            if self.enable_monitoring:
                self._metrics.increment_error()
                self._logger.log_error("max_iterations", f"达到最大迭代次数 {self.max_iterations}")
            return f"我无法在 {self.max_iterations} 次迭代内完成任务。"
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
        """添加工具并重新编译图。"""
        self.tools.append(tool)
        self.tool_names[tool.name] = tool
        self._compile()

    def remove_tool(self, tool_name: str):
        """移除工具并重新编译图。"""
        if tool_name in self.tool_names:
            tool = self.tool_names.pop(tool_name)
            self.tools.remove(tool)
            self._compile()

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
    enable_monitoring: bool = True,
    response_format: Optional[Union[type[BaseModel], tuple]] = None,
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
        enable_monitoring=enable_monitoring,
        response_format=response_format,
    )
