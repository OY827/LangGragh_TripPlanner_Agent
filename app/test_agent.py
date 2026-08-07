import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_react_agent():
    print("测试 ReAct Agent...")
    
    try:
        from app.agent import ReActAgent, create_react_agent
        from app.tools import get_basic_tools, get_all_tools
        
        print("✓ 导入成功")
        
        tools = get_basic_tools()
        print(f"✓ 创建了 {len(tools)} 个工具: {[t.name for t in tools]}")
        
        agent = create_react_agent(
            tools=tools,
            provider="aliyun",
            max_iterations=5,
            verbose=True
        )
        
        print("✓ 使用阿里云提供商创建 ReAct Agent")
        
        question = "计算一下：25乘以4加上100等于多少？"
        print(f"\n{'='*60}")
        print(f"测试问题：{question}")
        print(f"{'='*60}")
        
        result = agent.run(question)
        
        print(f"\n最终结果：{result}")
        
        return True
        
    except ImportError as e:
        print(f"✗ 导入错误：{e}")
        return False
    except Exception as e:
        print(f"✗ 意外错误：{e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_react_agent()
    sys.exit(0 if success else 1)
