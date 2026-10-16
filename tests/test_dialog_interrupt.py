"""主图在对话节点 interrupt，恢复后带着新回复再次暂停。"""

from typing import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.types import Command, interrupt

from app.agents.langgraph_builder import LangGraphBuilder, read_interrupt


class _State(TypedDict, total=False):
    plan: str
    reply: str
    active: bool


def _talk(state):
    user = interrupt({"reply": state.get("reply") or "welcome", "session_id": "s1"})
    text = str(user or "").strip()
    if text == "结束":
        return {"reply": "bye", "active": False}
    return {"reply": "got " + text, "active": True, "plan": (state.get("plan") or "") + "+" + text}


def _route(state):
    return "talk" if state.get("active") else "final"


def test_redis_url_uses_the_same_host_port_and_password():
    from app.agents.langgraph_builder import build_redis_url

    assert build_redis_url("127.0.0.1", 6379, "") == "redis://127.0.0.1:6379/0"
    assert build_redis_url("127.0.0.1", 6380, "a/b") == "redis://:a%2Fb@127.0.0.1:6380/0"


def test_route_stays_on_dialog_while_active():
    builder = LangGraphBuilder(nodes=object())
    assert builder._route_after_dialog({"dialog_active": True, "dialog_input": ""}) == "continue"
    assert builder._route_after_dialog({"dialog_active": False, "dialog_input": "结束"}) == "final"


def test_plan_pauses_and_resume_returns_the_next_reply():
    graph = StateGraph(_State)
    graph.add_node("plan", lambda state: {"plan": "day1"})
    graph.add_node("talk", _talk)
    graph.set_entry_point("plan")
    graph.add_edge("plan", "talk")
    graph.add_conditional_edges("talk", _route, {"talk": "talk", "final": END})
    app = graph.compile(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "pause-1"}}

    first = app.invoke({}, config)
    paused = read_interrupt(first)
    assert first["plan"] == "day1"
    assert paused["reply"] == "welcome"
    assert paused["session_id"] == "s1"

    second = app.invoke(Command(resume="加迪士尼"), config)
    assert second["plan"] == "day1+加迪士尼"
    assert read_interrupt(second)["reply"] == "got 加迪士尼"

    done = app.invoke(Command(resume="结束"), config)
    assert done["reply"] == "bye"
    assert read_interrupt(done) is None
