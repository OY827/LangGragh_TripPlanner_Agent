"""
验证 RAGAttractionTool 集成 Rerank 后的效果（上海测试用例）

直接调用 LangGraph Agent 使用的 RAGAttractionTool，确认工具内部已启用 Rerank 精排。
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from app.tools.agent_tools import RAGAttractionTool


def test_tool_rerank():
    print("=" * 70)
    print("🔧 RAGAttractionTool + Rerank 集成验证（上海）")
    print("=" * 70)

    tool = RAGAttractionTool()

    test_cases = [
        {"city": "上海", "keywords": "历史文化"},
        {"city": "上海", "keywords": "亲子游"},
        {"city": "上海", "keywords": "免费景点"},
    ]

    for i, case in enumerate(test_cases, 1):
        print(f"\n{'─' * 70}")
        print(f"🧪 测试{i}: city={case['city']}, keywords='{case['keywords']}'")
        print(f"{'─' * 70}")

        t1 = time.time()
        result = tool._run(city=case["city"], keywords=case["keywords"])
        elapsed = time.time() - t1

        print(result)
        print(f"\n⏱️ 耗时: {elapsed:.2f}s")

    print(f"\n{'=' * 70}")
    print("✅ RAGAttractionTool + Rerank 集成验证完成")
    print("=" * 70)


if __name__ == "__main__":
    test_tool_rerank()
