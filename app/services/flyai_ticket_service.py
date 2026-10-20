"""飞猪景点门票查询。

官方命令是 `flyai search-poi`，说明见
https://github.com/alibaba-flyai/flyai-skill/blob/main/skills/flyai/references/search-poi.md
密钥用环境变量 FLYAI_API_KEY，对应控制台 API Key。
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from typing import Any, Callable, Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()

Runner = Callable[[List[str], str], str]


class FlyaiTicketService:
    """按城市和景点名调用 search-poi，只查门票，不下单。"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        runner: Optional[Runner] = None,
    ):
        key = api_key if api_key is not None else os.getenv("FLYAI_API_KEY", "")
        self.api_key = (key or "").strip()
        self._runner = runner or run_flyai

    def is_configured(self) -> bool:
        return bool(self.api_key)

    def search(self, city: str, keyword: str) -> Dict[str, Any]:
        city = (city or "").strip()
        keyword = (keyword or "").strip()
        if not city:
            raise ValueError("缺少城市")
        if not keyword:
            raise ValueError("缺少景点名称")
        if not self.api_key:
            raise RuntimeError("未配置 FLYAI_API_KEY")
        raw = self._runner(
            ["search-poi", "--city-name", city, "--keyword", keyword],
            self.api_key,
        )
        return _parse_json(raw)

    def price_for(self, city: str, name: str) -> Optional[int]:
        return quote_of(self.search(city, name), name).get("price")

    def quote_for(self, city: str, name: str) -> Dict[str, Any]:
        return quote_of(self.search(city, name), name)

    def search_and_format(self, city: str, scenic_name: str) -> str:
        data = self.search(city, scenic_name)
        return format_ticket_result(data, city=city, scenic_name=scenic_name)


def pick_price(data: Any, name: str) -> Optional[int]:
    return quote_of(data, name).get("price")


def quote_of(data: Any, name: str) -> Dict[str, Any]:
    item = _best_item(_items(data), name)
    if not item:
        return {"price": None, "jump_url": ""}
    return {
        "price": _price_of(item),
        "jump_url": str(item.get("jumpUrl") or ""),
    }


def format_ticket_result(data: Any, city: str = "", scenic_name: str = "") -> str:
    if isinstance(data, dict) and data.get("status") not in (None, 0, "0"):
        message = data.get("message") or data.get("systemMessage") or "查询失败"
        return f"飞猪景点搜索失败：{message}"
    items = _items(data)
    title = scenic_name or city or "景点"
    if not items:
        return f"未找到「{title}」的飞猪门票"
    matched = _best_item(items, scenic_name) if scenic_name else None
    ordered = [matched] if matched else []
    for item in items:
        if item is not matched:
            ordered.append(item)
    lines = [f"{city}{title}，找到 {len(ordered)} 条景点：", ""]
    for index, item in enumerate(ordered, 1):
        lines.append(f"{index}. {item.get('name') or '景点'}")
        ticket = item.get("ticketInfo") if isinstance(item.get("ticketInfo"), dict) else {}
        ticket_name = ticket.get("ticketName") or ""
        if ticket_name:
            lines.append(f"   票型: {ticket_name}")
        price = _price_of(item)
        if price is not None:
            lines.append(f"   价格: ¥{price}")
        elif ticket:
            lines.append("   价格: 飞猪未返回数字价格，请打开详情页查看")
        address = item.get("address") or ""
        if address:
            lines.append(f"   地址: {address}")
        jump = item.get("jumpUrl") or ""
        if jump:
            lines.append(f"   详情: {jump}")
    return "\n".join(lines)


def run_flyai(args: List[str], api_key: str) -> str:
    binary = shutil.which("flyai") or shutil.which("flyai.cmd")
    if binary:
        command = [binary, *args]
    else:
        npx = shutil.which("npx") or shutil.which("npx.cmd")
        if not npx:
            raise RuntimeError("未找到 flyai 或 npx。请先安装 Node.js，再执行 npm i -g @fly-ai/flyai-cli")
        command = [npx, "--yes", "@fly-ai/flyai-cli", *args]
    env = os.environ.copy()
    env["FLYAI_API_KEY"] = api_key
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=90,
        env=env,
        encoding="utf-8",
        errors="replace",
    )
    output = (completed.stdout or "").strip()
    if _has_result_json(output):
        return output
    if completed.returncode != 0:
        detail = (completed.stderr or output or "flyai 调用失败").strip()
        raise RuntimeError(detail[:300])
    if not output:
        raise RuntimeError((completed.stderr or "flyai 没有输出").strip()[:300])
    return output


def _has_result_json(raw: str) -> bool:
    try:
        data = _parse_json(raw)
    except Exception:
        return False
    return isinstance(data, dict) and ("data" in data or data.get("status") in (0, "0"))


def _parse_json(raw: str) -> Dict[str, Any]:
    text = (raw or "").strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise RuntimeError("飞猪返回的不是 JSON")
        data = json.loads(text[start:end + 1])
    if not isinstance(data, dict):
        raise RuntimeError("飞猪返回的不是 JSON 对象")
    return data


def _items(data: Any) -> List[Dict[str, Any]]:
    payload = data.get("data", data) if isinstance(data, dict) else {}
    if not isinstance(payload, dict):
        return []
    rows = payload.get("itemList") or payload.get("items") or []
    return [row for row in rows if isinstance(row, dict)]


def _best_item(items: List[Dict[str, Any]], name: str) -> Optional[Dict[str, Any]]:
    target = (name or "").strip()
    if not target:
        return items[0] if items else None
    for item in items:
        if (item.get("name") or "").strip() == target:
            return item
    for item in items:
        title = (item.get("name") or "").strip()
        if target in title or title in target:
            return item
    return items[0] if items else None


def _price_of(item: Dict[str, Any]) -> Optional[int]:
    ticket = item.get("ticketInfo") if isinstance(item.get("ticketInfo"), dict) else {}
    raw = ticket.get("price")
    if raw in (None, ""):
        raw = item.get("price")
    if raw in (None, ""):
        return None
    try:
        return int(float(str(raw).replace("元", "").replace("¥", "").strip()))
    except (TypeError, ValueError):
        return None


_service: Optional[FlyaiTicketService] = None


def get_flyai_ticket_service() -> FlyaiTicketService:
    global _service
    if _service is None:
        _service = FlyaiTicketService()
    return _service
