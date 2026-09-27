"""LangChain trim_messages 滑动窗口。"""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.memory.sliding_window import SlidingWindowChatMemory, records_to_messages


def test_sliding_window_keeps_last_messages():
    memory = SlidingWindowChatMemory(window_size=4)
    records = []
    for i in range(10):
        records.append({"role": "user", "content": f"用户{i}"})
        records.append({"role": "assistant", "content": f"助手{i}"})

    windowed = memory.compress_records(records)
    assert len(windowed) <= 4
    assert windowed[0]["role"] == "user"
    assert "用户9" in windowed[-2]["content"] or "用户9" in windowed[-1]["content"] or any(
        "用户9" in item["content"] for item in windowed
    )
    assert any("助手9" in item["content"] for item in windowed)
    assert not any("用户0" in item["content"] for item in windowed)


def test_sliding_window_uses_langchain_messages():
    memory = SlidingWindowChatMemory(window_size=3)
    messages = [
        SystemMessage(content="系统"),
        HumanMessage(content="第一问"),
        AIMessage(content="第一答"),
        HumanMessage(content="第二问"),
        AIMessage(content="第二答"),
        HumanMessage(content="第三问"),
        AIMessage(content="第三答"),
    ]
    windowed = memory.compress_messages(messages)
    texts = [m.content for m in windowed]
    assert "第一问" not in texts
    assert "第三答" in texts
    assert any(isinstance(m, HumanMessage) for m in windowed)


def test_records_roundtrip():
    records = [
        {"role": "user", "content": "加迪士尼"},
        {"role": "assistant", "content": "已加入"},
    ]
    messages = records_to_messages(records)
    assert isinstance(messages[0], HumanMessage)
    assert isinstance(messages[1], AIMessage)


if __name__ == "__main__":
    test_sliding_window_keeps_last_messages()
    test_sliding_window_uses_langchain_messages()
    test_records_roundtrip()
    print("ok")
