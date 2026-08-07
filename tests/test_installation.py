import sys
import importlib

def test_import(module_name):
    try:
        module = importlib.import_module(module_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"✓ {module_name}: {version}")
        return True
    except ImportError as e:
        print(f"✗ {module_name}: {e}")
        return False
    except Exception as e:
        print(f"✗ {module_name}: {e}")
        return False

if __name__ == "__main__":
    print("Testing LangChain packages installation...")
    print("-" * 50)

    packages = [
        "langchain",
        "langchain_openai",
        "langchain_community",
        "langchain_qdrant",
        "langchain_neo4j",
        "openai",
        "qdrant_client",
        "neo4j"
    ]

    results = []
    for package in packages:
        results.append(test_import(package))

    print("-" * 50)
    if all(results):
        print("✓ All packages are installed successfully!")
    else:
        print("✗ Some packages are missing. Please install them.")
        sys.exit(1)
