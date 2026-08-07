import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if __name__ == "__main__":
    from app.chain import create_basic_chain

    print("Testing LangChain setup...")
    chain = create_basic_chain()

    response = chain.invoke({"question": "What is LangChain?"})
    print(f"\nResponse: {response}")
