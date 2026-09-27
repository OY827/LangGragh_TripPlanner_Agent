"""
高德 MCP（Streamable HTTP）连通性测试。

默认连官方托管 https://mcp.amap.com/mcp?key=...
可用 AMAP_MCP_URL 指向本地 sugarforever/amap-mcp-server（如 http://localhost:8001/mcp）。

不 import 整个 app，避免 torch。不打印 API Key。

用法（项目根目录）:
  .\\.venv\\Scripts\\python.exe tests\\test_amap_mcp.py
  .\\.venv\\Scripts\\python.exe tests\\test_amap_mcp.py --city 上海 --keywords 外滩
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import types
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")


def _flag(name: str) -> bool:
    return os.getenv(name, "false").strip().lower() == "true"


def _key_state(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        return "MISSING"
    return f"SET len={len(value)}"


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


def _preview(result) -> str:
    text = str(result)
    return text[:400]


def main() -> int:
    parser = argparse.ArgumentParser(description="测试高德 MCP Streamable HTTP")
    parser.add_argument("--city", default="上海")
    parser.add_argument("--keywords", default="外滩")
    args = parser.parse_args()

    print("=" * 60)
    print("高德 MCP 连通性（Streamable HTTP）")
    print("=" * 60)
    print(f"  AMAP_MAPS_API_KEY: {_key_state('AMAP_MAPS_API_KEY')}")
    print(f"  AMAP_MCP_URL:      {os.getenv('AMAP_MCP_URL', '').strip() or '(default official)'}")
    print(f"  ENABLE_MCP_TOOLS:  {_flag('ENABLE_MCP_TOOLS')}  (工具层开关)")
    print(f"  USE_MCP:           {_flag('USE_MCP')}  (AmapService 开关)")
    print(f"  python:            {sys.executable}")

    if not os.getenv("AMAP_MAPS_API_KEY", "").strip() and not os.getenv("AMAP_MCP_URL", "").strip():
        print("FAIL: 未配置 AMAP_MAPS_API_KEY 或 AMAP_MCP_URL")
        return 1

    print("\n1) 加载 AmapMCPService")
    _stub_app_package()
    http_mod = _load("app.tools.mcp_http_client", ROOT / "app" / "tools" / "mcp_http_client.py")
    sys.modules["app.tools"].mcp_http_client = http_mod
    amap_mod = _load("app.services.amap_mcp_service", ROOT / "app" / "services" / "amap_mcp_service.py")

    service = amap_mod.get_amap_mcp_service()
    if not service.is_configured():
        print("FAIL: AmapMCPService 未配置")
        return 1
    print(f"  endpoint: {amap_mod._masked_url(service.url)}")

    print("\n2) tools/list")
    try:
        names = service.list_tool_names()
    except Exception as exc:
        print(f"FAIL: tools/list 失败: {exc}")
        return 2
    print(f"  工具数: {len(names)}")
    for name in names[:20]:
        print(f"    - {name}")
    if names and "maps_weather" not in names and "maps_text_search" not in names:
        print("WARN: 列表里没有 maps_weather / maps_text_search，后续调用可能失败")

    print(f"\n3) maps_weather city={args.city}")
    weather = service.weather(args.city)
    if not weather:
        print("FAIL: maps_weather 无结果")
        return 3
    forecasts = weather.get("forecasts") or weather.get("lives") or []
    print(f"  keys: {list(weather.keys())[:12]}")
    print(f"  预报/实况条数: {len(forecasts) if isinstance(forecasts, list) else 'n/a'}")
    if isinstance(forecasts, list) and forecasts:
        first = forecasts[0]
        if isinstance(first, dict):
            print(f"  第一条: {first.get('date') or first.get('city')} {first.get('dayweather') or first.get('weather')}")

    print(f"\n4) maps_text_search city={args.city} keywords={args.keywords}")
    search = service.text_search(args.keywords, args.city)
    if not search:
        print("FAIL: maps_text_search 无结果")
        return 4
    pois = service.extract_pois(search)
    print(f"  POI 条数: {len(pois)}")
    for i, poi in enumerate(pois[:5], 1):
        print(f"    {i}. {poi.get('name')}  {poi.get('address', '')}")
    if not pois:
        print(f"  原文预览: {_preview(search)}")
        print("WARN: 调用成功但没有解析到 pois")
        return 0

    print("PASS: Streamable HTTP 能调通 maps_weather / maps_text_search")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
