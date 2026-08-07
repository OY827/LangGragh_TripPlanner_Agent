import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_basic_chain():
    try:
        from app.chain import create_basic_chain

        print("Testing basic chain creation...")
        chain = create_basic_chain()
        print("✓ Chain created successfully")

        return True
    except Exception as e:
        print(f"✗ Failed to create chain: {e}")
        return False

if __name__ == "__main__":
    print("Running LangChain tests...")
    print("-" * 50)

    test_basic_chain()

    print("-" * 50)
    print("Tests completed.")
