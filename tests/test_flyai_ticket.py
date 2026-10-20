"""飞猪景点门票：解析 search-poi 结果。不访问飞猪。"""

import importlib.util
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load():
    app = types.ModuleType("app")
    services = types.ModuleType("app.services")
    sys.modules["app"] = app
    sys.modules["app.services"] = services
    app.services = services
    path = ROOT / "app" / "services" / "flyai_ticket_service.py"
    spec = importlib.util.spec_from_file_location("app.services.flyai_ticket_service", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["app.services.flyai_ticket_service"] = module
    spec.loader.exec_module(module)
    return module


svc = _load()

SAMPLE = {
    "status": 0,
    "message": "success",
    "data": {
        "itemList": [
            {
                "name": "上海自然博物馆",
                "address": "静安区北京西路",
                "jumpUrl": "https://example.com/museum",
                "ticketInfo": {
                    "price": 30,
                    "ticketName": "成人票",
                },
            }
        ]
    },
}


def test_price_matches_scenic_name():
    assert svc.pick_price(SAMPLE, "上海自然博物馆") == 30


def test_format_includes_price_and_link():
    text = svc.format_ticket_result(SAMPLE, city="上海", scenic_name="上海自然博物馆")
    assert "¥30" in text
    assert "成人票" in text
    assert "https://example.com/museum" in text


def test_null_price_says_detail_page():
    data = {
        "status": 0,
        "data": {
            "itemList": [
                {"name": "外滩", "ticketInfo": {"price": None, "ticketName": "观光票"}}
            ]
        },
    }
    text = svc.format_ticket_result(data, city="上海", scenic_name="外滩")
    assert "详情页" in text
    assert svc.pick_price(data, "外滩") is None


def test_missing_key_is_not_configured():
    service = svc.FlyaiTicketService(api_key="", runner=lambda args, key: "{}")
    assert service.is_configured() is False


def test_search_passes_city_and_keyword():
    calls = []

    def runner(args, key):
        calls.append((args, key))
        return '{"status":0,"data":{"itemList":[]}}'

    service = svc.FlyaiTicketService(api_key="test-key", runner=runner)
    service.search("上海", "自然博物馆")
    assert calls == [(
        ["search-poi", "--city-name", "上海", "--keyword", "自然博物馆"],
        "test-key",
    )]


if __name__ == "__main__":
    test_price_matches_scenic_name()
    test_format_includes_price_and_link()
    test_null_price_says_detail_page()
    test_missing_key_is_not_configured()
    test_search_passes_city_and_keyword()
    print("ok")
