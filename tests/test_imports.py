"""环境冒烟：依赖是否装齐、核心模块能否 import。"""

import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PACKAGES = [
    "langchain",
    "langchain_openai",
    "langchain_community",
    "langchain_core",
    "openai",
    "qdrant_client",
    "chromadb",
]


def _check(module_name: str) -> bool:
    try:
        module = importlib.import_module(module_name)
        version = getattr(module, "__version__", "ok")
        print(f"✓ {module_name}: {version}")
        return True
    except Exception as e:
        print(f"✗ {module_name}: {e}")
        return False


def test_package_imports() -> bool:
    print("Testing installed packages...")
    print("-" * 50)
    results = [_check(name) for name in PACKAGES]
    print("-" * 50)
    return all(results)


def test_project_imports() -> bool:
    print("Testing project imports...")
    print("-" * 50)
    ok = True
    try:
        from langchain_core.prompts import ChatPromptTemplate  # noqa: F401
        from langchain_core.output_parsers import StrOutputParser  # noqa: F401
        print("✓ langchain_core prompts/parsers")
    except Exception as e:
        print(f"✗ langchain_core: {e}")
        ok = False
    print("-" * 50)
    return ok


if __name__ == "__main__":
    pkg_ok = test_package_imports()
    proj_ok = test_project_imports()
    if pkg_ok and proj_ok:
        print("✓ Smoke tests passed")
        sys.exit(0)
    print("✗ Some imports failed")
    sys.exit(1)
