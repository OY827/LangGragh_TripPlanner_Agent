import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_aliyun_llm():
    print("Testing Aliyun DashScope LLM...")
    
    try:
        from app.llm import get_aliyun_llm
        from app.chain import create_basic_chain
        
        print("✓ Imports successful")
        
        llm = get_aliyun_llm()  # 从环境变量 DASHSCOPE_MODEL 读取模型名
        print("✓ Created Aliyun LLM instance")

        chain = create_basic_chain(provider="aliyun")  # 从环境变量读取模型名
        print("✓ Created chain with Aliyun provider")
        
        response = chain.invoke({"question": "你好，我是阿里云用户"})
        print(f"\nResponse from Aliyun Qwen: {response}")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    except ValueError as e:
        print(f"✗ Configuration error: {e}")
        print("  Please set DASHSCOPE_API_KEY in .env file")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = test_aliyun_llm()
    sys.exit(0 if success else 1)
