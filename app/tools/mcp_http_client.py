"""
Streamable HTTP MCP 客户端

给后端用，连远程 MCP Server（如 RollingGo 酒店）。
不是 FastAPI，也不是高德 REST。

优先走官方文档里的 JSON-RPC POST（tools/call）；
若服务端要求先握手，再降级到 Python MCP SDK 的 streamable HTTP 会话。
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional
from uuid import uuid4

import requests

logger = logging.getLogger(__name__)


class StreamableHttpMCPClient:
    """远程 MCP 客户端：URL + 请求头，同步 tools/list 与 tools/call。"""

    def __init__(
        self,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        timeout: float = 40.0,
    ):
        if not url:
            raise ValueError("MCP URL 不能为空")
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            **(headers or {}),
        }

    def list_tools(self) -> List[Dict[str, Any]]:
        try:
            return self._list_tools_jsonrpc()
        except Exception as jsonrpc_error:
            logger.debug("JSON-RPC tools/list 失败，尝试 SDK 会话: %s", jsonrpc_error)
            try:
                return self._list_tools_sdk()
            except Exception as sdk_error:
                raise RuntimeError(
                    f"MCP tools/list 失败（jsonrpc={jsonrpc_error}; sdk={sdk_error}）"
                ) from sdk_error

    def call_tool(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> Any:
        arguments = arguments or {}
        try:
            return self._call_jsonrpc(name, arguments)
        except Exception as jsonrpc_error:
            logger.debug("JSON-RPC 直调失败，尝试 SDK 会话: %s", jsonrpc_error)
            try:
                return self._call_sdk_session(name, arguments)
            except Exception as sdk_error:
                raise RuntimeError(
                    f"MCP 工具调用失败（jsonrpc={jsonrpc_error}; sdk={sdk_error}）"
                ) from sdk_error

    def _list_tools_jsonrpc(self) -> List[Dict[str, Any]]:
        payload = {
            "jsonrpc": "2.0",
            "id": str(uuid4()),
            "method": "tools/list",
            "params": {},
        }
        response = requests.post(
            self.url,
            headers=self.headers,
            json=payload,
            timeout=self.timeout,
        )
        if response.status_code >= 400:
            raise RuntimeError(f"HTTP {response.status_code}: {response.text[:300]}")
        data = self._decode_http_body(response)
        if isinstance(data, dict) and data.get("error"):
            raise RuntimeError(str(data["error"]))
        result = data.get("result", data) if isinstance(data, dict) else data
        return self._parse_tools_list(result)

    def _list_tools_sdk(self) -> List[Dict[str, Any]]:
        import asyncio

        return asyncio.run(self._list_tools_sdk_async())

    async def _list_tools_sdk_async(self) -> List[Dict[str, Any]]:
        from mcp import ClientSession
        from mcp.client.streamable_http import streamable_http_client
        from mcp.shared._httpx_utils import create_mcp_http_client

        auth_headers = {
            k: v
            for k, v in self.headers.items()
            if k.lower() not in {"content-type", "accept"}
        }
        http_client = create_mcp_http_client(headers=auth_headers)
        async with http_client:
            async with streamable_http_client(self.url, http_client=http_client) as streams:
                read, write = streams[0], streams[1]
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.list_tools()
                    return self._parse_tools_list(result)

    def _call_jsonrpc(self, name: str, arguments: Dict[str, Any]) -> Any:
        payload = {
            "jsonrpc": "2.0",
            "id": str(uuid4()),
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
        }
        response = requests.post(
            self.url,
            headers=self.headers,
            json=payload,
            timeout=self.timeout,
        )
        if response.status_code >= 400:
            raise RuntimeError(f"HTTP {response.status_code}: {response.text[:300]}")

        data = self._decode_http_body(response)
        if not isinstance(data, dict):
            return data

        if data.get("error"):
            raise RuntimeError(str(data["error"]))

        result = data.get("result", data)
        parsed = self._parse_call_tool_result(result)
        if self._looks_like_mcp_error(parsed):
            raise RuntimeError(str(parsed))
        return parsed

    def _call_sdk_session(self, name: str, arguments: Dict[str, Any]) -> Any:
        import asyncio

        return asyncio.run(self._call_sdk_session_async(name, arguments))

    async def _call_sdk_session_async(self, name: str, arguments: Dict[str, Any]) -> Any:
        from mcp import ClientSession
        from mcp.client.streamable_http import streamable_http_client
        from mcp.shared._httpx_utils import create_mcp_http_client

        auth_headers = {
            k: v
            for k, v in self.headers.items()
            if k.lower() not in {"content-type", "accept"}
        }
        http_client = create_mcp_http_client(headers=auth_headers)
        async with http_client:
            async with streamable_http_client(self.url, http_client=http_client) as streams:
                read, write = streams[0], streams[1]
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.call_tool(name, arguments)
                    return self._parse_call_tool_result(result)

    def _decode_http_body(self, response: requests.Response) -> Any:
        content_type = (response.headers.get("Content-Type") or "").lower()
        text = response.text or ""
        if "text/event-stream" in content_type or text.lstrip().startswith("event:"):
            return self._parse_sse(text)
        try:
            return response.json()
        except ValueError:
            return self._parse_sse(text) if "data:" in text else text

    @staticmethod
    def _parse_sse(text: str) -> Any:
        data_chunks = []
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if line.startswith("data:"):
                data_chunks.append(line[5:].strip())
        if not data_chunks:
            raise RuntimeError(f"无法解析 SSE 响应: {text[:300]}")

        last_error = None
        for chunk in reversed(data_chunks):
            if not chunk or chunk == "[DONE]":
                continue
            try:
                return json.loads(chunk)
            except json.JSONDecodeError as exc:
                last_error = exc
        if last_error:
            raise RuntimeError(f"SSE JSON 解析失败: {data_chunks[-1][:300]}")
        return data_chunks[-1]

    @staticmethod
    def _parse_tools_list(result: Any) -> List[Dict[str, Any]]:
        tools = []
        raw = None
        if result is None:
            raw = []
        elif hasattr(result, "tools"):
            raw = result.tools
        elif isinstance(result, dict):
            raw = result.get("tools", result.get("result", []))
        elif isinstance(result, list):
            raw = result
        else:
            raw = []

        for item in raw or []:
            if isinstance(item, dict) and item.get("name"):
                tools.append({
                    "name": item["name"],
                    "description": item.get("description") or "",
                })
            else:
                name = getattr(item, "name", None)
                if name:
                    tools.append({
                        "name": name,
                        "description": getattr(item, "description", "") or "",
                    })
        return tools

    @staticmethod
    def _parse_call_tool_result(result: Any) -> Any:
        if result is None:
            return None

        if hasattr(result, "content"):
            texts = []
            for item in result.content or []:
                text = getattr(item, "text", None)
                if text:
                    texts.append(text)
            if texts:
                joined = "\n".join(texts)
                try:
                    return json.loads(joined)
                except json.JSONDecodeError:
                    return joined
            return result.content

        if isinstance(result, dict):
            if "content" in result:
                texts = []
                for item in result.get("content") or []:
                    if isinstance(item, dict) and item.get("text"):
                        texts.append(item["text"])
                    elif hasattr(item, "text") and item.text:
                        texts.append(item.text)
                if texts:
                    joined = "\n".join(texts)
                    try:
                        return json.loads(joined)
                    except json.JSONDecodeError:
                        return joined
            if "structuredContent" in result and result["structuredContent"] is not None:
                return result["structuredContent"]
        return result

    @staticmethod
    def _looks_like_mcp_error(parsed: Any) -> bool:
        if isinstance(parsed, dict) and parsed.get("isError"):
            return True
        if isinstance(parsed, str) and "error" in parsed.lower() and len(parsed) < 500:
            return "MCP" in parsed or "invalid" in parsed.lower()
        return False
