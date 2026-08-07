import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_langchain_imports():
    print("Testing LangChain imports...")
    
    try:
        import langchain
        print(f"✓ langchain: {langchain.__version__}")
    except ImportError as e:
        print(f"✗ langchain: {e}")
        return False
    
    try:
        from langchain_openai import ChatOpenAI
        print("✓ langchain_openai")
    except ImportError as e:
        print(f"✗ langchain_openai: {e}")
        return False
    
    try:
        from langchain_core.prompts import ChatPromptTemplate
        print("✓ langchain_core.prompts")
    except ImportError as e:
        print(f"✗ langchain_core.prompts: {e}")
        return False
    
    try:
        from langchain_core.output_parsers import StrOutputParser
        print("✓ langchain_core.output_parsers")
    except ImportError as e:
        print(f"✗ langchain_core.output_parsers: {e}")
        return False
    
    print("\n✓ All LangChain imports successful!")
    print("\nNote: To use OpenAI, set OPENAI_API_KEY in .env file")
    return True

if __name__ == "__main__":
    test_langchain_imports()