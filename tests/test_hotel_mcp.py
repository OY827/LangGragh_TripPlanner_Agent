"""
RollingGo 酒店 MCP 连通性测试（避免 import app 触发 torch）。
用法：
  .venv\\Scripts\\python.exe tests\\test_hotel_mcp.py
"""

import importlib.util
import sys
import types
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")


def _load(modname: str, path: Path):
    spec = importlib.util.spec_from_file_location(modname, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[modname] = module
    spec.loader.exec_module(module)
    return module


def _stub_app_package() -> None:
    app = types.ModuleType("app")
    tools = types.ModuleType("app.tools")
    services = types.ModuleType("app.services")
    sys.modules["app"] = app
    sys.modules["app.tools"] = tools
    sys.modules["app.services"] = services
    app.tools = tools
    app.services = services


def main() -> int:
    _stub_app_package()
    http_mod = _load("app.tools.mcp_http_client", ROOT / "app" / "tools" / "mcp_http_client.py")
    sys.modules["app.tools"].mcp_http_client = http_mod
    hotel_mod = _load("app.services.hotel_mcp_service", ROOT / "app" / "services" / "hotel_mcp_service.py")

    service = hotel_mod.get_hotel_mcp_service()
    if not service.is_configured():
        print("FAIL: HOTEL_API_KEY 未配置")
        return 1

    print(f"endpoint: {service.url}")
    print("calling searchHotels: city=杭州, keywords=经济型酒店 ...")
    try:
        hotels = service.search_hotels(
            city="杭州",
            keywords="经济型酒店",
            accommodation="经济型",
            check_in_date="2026-10-01",
            stay_nights=2,
            size=3,
        )
    except Exception as exc:
        print(f"FAIL: {exc}")
        return 2

    print(f"got {len(hotels)} hotels")
    if not hotels:
        print("WARN: 空列表")
        return 0

    print(service.format_for_agent(hotels[:3], city="杭州"))

    print("\nchecking hotel agent tools ...")
    tools_mod = _load("app.tools.agent_tools", ROOT / "app" / "tools" / "agent_tools.py")
    sys.modules["app.tools"].agent_tools = tools_mod
    hotel_tools = tools_mod.get_hotel_tools()
    names = [t.name for t in hotel_tools]
    print("hotel agent tools:", names)
    if names[:2] != ["hotel_search", "amap_hotel_search"]:
        print("FAIL: 酒店推荐 Agent 工具绑定不正确")
        return 3

    print("invoking RollingGoHotelTool as the hotel agent would ...")
    observation = hotel_tools[0].invoke(
        {
            "city": "杭州",
            "keywords": "经济型酒店",
            "check_in_date": "2026-10-01",
            "stay_nights": 2,
        }
    )
    print(str(observation)[:500])
    if "元/晚" not in str(observation):
        print("FAIL: 工具返回里没有价格")
        return 4
    print("PASS: 酒店推荐 Agent 主工具已接到 RollingGo")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
