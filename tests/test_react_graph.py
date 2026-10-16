"""LangGraph create_react_agent 包装：无工具调用时结束，并交出 structured_response。"""

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool

from app.agents.react_agent import ReActAgent
from app.services.structured_output import FillNameOutput


class _SilentModel(BaseChatModel):
    @property
    def _llm_type(self) -> str:
        return "silent"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="已完成"))])

    def with_structured_output(self, schema, **kwargs):
        return RunnableLambda(lambda _messages: schema(name="外滩"))


@tool
def lookup_spot(city: str) -> str:
    """按城市查一个景点标题。"""
    return "外滩"


def test_graph_returns_structured_name_without_calling_a_real_model():
    agent = ReActAgent(
        name="补点测试",
        tools=[lookup_spot],
        llm=_SilentModel(),
        system_prompt="补一个景点",
        max_iterations=2,
        verbose=False,
        enable_monitoring=False,
        response_format=FillNameOutput,
    )
    text = agent.run("上海还空着两小时")
    assert '"name": "外滩"' in text or '"name":"外滩"' in text.replace(" ", "")
