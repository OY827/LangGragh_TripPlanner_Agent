"""
MCP 工具管理模块
用于管理和集成 MCP（Model Context Protocol）工具
"""

from typing import List, Dict, Optional, Any, Tuple, Type
from dotenv import load_dotenv
import os
import json
import asyncio
import concurrent.futures

load_dotenv()


class Settings:
    """
    配置类 - 用于管理 API 密钥等配置
    
    从环境变量读取配置：
    - AMAP_MAPS_API_KEY: 高德地图 API 密钥
    - WEATHER_API_KEY: 天气 API 密钥
    - SEARCH_API_KEY: 搜索 API 密钥
    """
    
    @property
    def amap_api_key(self) -> str:
        return os.getenv("AMAP_MAPS_API_KEY", "")
    
    @property
    def weather_api_key(self) -> str:
        return os.getenv("WEATHER_API_KEY", "")
    
    @property
    def search_api_key(self) -> str:
        return os.getenv("SEARCH_API_KEY", "")
    
    @property
    def enable_mcp_tools(self) -> bool:
        return os.getenv("ENABLE_MCP_TOOLS", "false").lower() == "true"


class MCPToolWrapper:
    """
    MCP 工具包装器 - 使用 mcp 标准库连接 MCP 服务器
    
    使用方法：
        wrapper = MCPToolWrapper(
            name="amap",
            description="高德地图服务",
            server_command=["python", "-m", "amap_mcp_server"],
            env={"AMAP_MAPS_API_KEY": "your_key"}
        )
        result = wrapper.invoke(name="maps_text_search", arguments={...})
    """
    
    def __init__(
        self,
        name: str,
        description: str,
        server_command: List[str],
        env: Optional[Dict[str, str]] = None,
        auto_expand: bool = True,
        expandable: bool = True
    ):
        self.name = name
        self.description = description
        self.server_command = server_command
        self.env = env or {}
        self.auto_expand = auto_expand
        self.expandable = expandable
        self._client_initialized = False
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    
    def invoke(self, **kwargs) -> Any:
        """调用工具"""
        name = kwargs.get("name")
        arguments = kwargs.get("arguments", {})
        
        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():
                return self._executor.submit(
                    asyncio.run, self._invoke_async(name, arguments)
                ).result()
            else:
                return asyncio.run(self._invoke_async(name, arguments))
        except RuntimeError:
            return asyncio.run(self._invoke_async(name, arguments))
    
    async def _invoke_async(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """异步调用工具"""
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
            
            # 创建服务器参数
            full_env = {**self._get_default_env(), **self.env}
            server_params = StdioServerParameters(
                command=self.server_command[0],
                args=self.server_command[1:],
                env=full_env
            )
            
            async with stdio_client(server=server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    # 初始化会话
                    await session.initialize()
                    
                    # 调用工具
                    result = await session.call_tool(tool_name, arguments)
                    return self._parse_response(result)
                
        except Exception as e:
            raise ImportError(
                f"MCP 工具调用失败：{str(e)}\n"
                "请确保已安装 mcp 和 amap-mcp-server 包"
            )
    
    def _get_default_env(self) -> Dict[str, str]:
        """获取默认环境变量"""
        import sys
        default_vars = (
            ["APPDATA", "HOMEDRIVE", "HOMEPATH", "LOCALAPPDATA", "PATH", 
             "PROCESSOR_ARCHITECTURE", "SYSTEMDRIVE", "SYSTEMROOT", "TEMP", 
             "USERNAME", "USERPROFILE"]
            if sys.platform == "win32"
            else ["HOME", "LOGNAME", "PATH", "SHELL", "TERM", "USER"]
        )
        env = {}
        for key in default_vars:
            value = os.environ.get(key)
            if value and not value.startswith("()"):
                env[key] = value
        return env
    
    def _parse_response(self, response) -> Any:
        """解析响应 - 处理 CallToolResult 对象"""
        import json
        
        # 处理 CallToolResult 对象（MCP 工具调用结果）
        if hasattr(response, 'content'):
            # 提取文本内容
            text_contents = []
            for item in response.content:
                if hasattr(item, 'text'):
                    text_contents.append(item.text)
            
            if text_contents:
                # 尝试解析为 JSON
                full_text = '\n'.join(text_contents)
                try:
                    return json.loads(full_text)
                except json.JSONDecodeError:
                    return full_text
            
            return response.content
        
        # 处理旧的 JSON-RPC 响应格式
        if hasattr(response, 'message'):
            msg = response.message
            if hasattr(msg, 'result'):
                return msg.result
            if hasattr(msg, 'error'):
                raise Exception(f"MCP 工具调用错误：{msg.error}")
        
        return str(response)
    
    def __repr__(self) -> str:
        return f"MCPToolWrapper(name='{self.name}', description='{self.description}')"


class MCPToolManager:
    """
    MCP 工具管理器 - 用于统一管理和集成多个 MCP 工具
    
    支持：
    - 创建共享的 MCP 工具实例（延迟加载）
    - 自动管理环境变量配置
    - 支持工具的动态注册和获取
    - 提供工具列表查询功能
    """
    
    _instance = None
    _tools: Dict[str, MCPToolWrapper] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MCPToolManager, cls).__new__(cls)
            cls._tools = {}
        return cls._instance
    
    def register_tool(
        self,
        name: str,
        description: str,
        server_command: List[str],
        env: Optional[Dict[str, str]] = None,
        auto_expand: bool = True,
        expandable: bool = True
    ) -> MCPToolWrapper:
        """
        注册一个 MCP 工具
        
        Args:
            name: 工具名称
            description: 工具描述
            server_command: 启动 MCP 服务器的命令
            env: 环境变量字典
            auto_expand: 是否自动扩展工具
            expandable: 是否可扩展
            
        Returns:
            注册的 MCPToolWrapper 实例
        """
        if name in self._tools:
            print(f"  - MCP 工具 '{name}' 已存在，返回现有包装器")
            return self._tools[name]
        
        print(f"  - 创建共享 MCP 工具包装器 '{name}'...")
        
        wrapper = MCPToolWrapper(
            name=name,
            description=description,
            server_command=server_command,
            env=env,
            auto_expand=auto_expand
        )
        wrapper.expandable = expandable
        self._tools[name] = wrapper
        print(f"  ✓ MCP 工具包装器 '{name}' 创建成功")
        return wrapper
    
    def get_tool(self, name: str, load: bool = False) -> Optional[MCPToolWrapper]:
        """
        获取已注册的 MCP 工具包装器
        
        Args:
            name: 工具名称
            load: 是否立即加载实际的 MCPTool 实例
            
        Returns:
            MCPToolWrapper 实例，如果不存在返回 None
        """
        return self._tools.get(name)
    
    def get_all_tools(self, load: bool = False) -> List[MCPToolWrapper]:
        """
        获取所有已注册的工具包装器列表
        
        Args:
            load: 是否立即加载所有实际的 MCPTool 实例
            
        Returns:
            所有 MCPToolWrapper 实例的列表
        """
        return list(self._tools.values())
    
    def get_tool_names(self) -> List[str]:
        """
        获取所有工具名称
        
        Returns:
            工具名称列表
        """
        return list(self._tools.keys())
    
    def invoke_tool(self, name: str, **kwargs) -> Any:
        """
        调用指定工具
        
        Args:
            name: 工具名称
            **kwargs: 工具参数
            
        Returns:
            工具执行结果
        """
        wrapper = self.get_tool(name)
        if not wrapper:
            raise ValueError(f"工具 '{name}' 未注册")
        
        return wrapper.invoke(**kwargs)
    
    def unregister_tool(self, name: str) -> bool:
        """
        注销工具
        
        Args:
            name: 工具名称
            
        Returns:
            是否成功注销
        """
        if name in self._tools:
            del self._tools[name]
            return True
        return False
    
    def clear_all_tools(self):
        """清除所有已注册的工具"""
        self._tools.clear()
    
    def __contains__(self, name: str) -> bool:
        return name in self._tools
    
    def __len__(self) -> int:
        return len(self._tools)


# 全局设置实例
settings = Settings()


def init_mcp_tools() -> MCPToolManager:
    """
    初始化 MCP 工具管理器并注册常用工具
    
    Returns:
        MCPToolManager 实例
    """
    manager = MCPToolManager()
    
    if not settings.enable_mcp_tools:
        print("MCP 工具未启用（ENABLE_MCP_TOOLS=false）")
        return manager
    
    print("初始化 MCP 工具...")
    
    # 注册高德地图工具
    if settings.amap_api_key:
        manager.register_tool(
            name="amap",
            description="高德地图服务 - 提供地理编码、逆地理编码、路径规划等功能",
            server_command=["python", "-m", "amap_mcp_server"],
            env={"AMAP_MAPS_API_KEY": settings.amap_api_key},
            auto_expand=True
        )
    
    # 注册天气服务工具（示例）
    if settings.weather_api_key:
        manager.register_tool(
            name="weather",
            description="天气服务 - 提供实时天气查询、天气预报等功能",
            server_command=["uvx", "weather-mcp-server"],
            env={"WEATHER_API_KEY": settings.weather_api_key},
            auto_expand=True
        )
    
    # 注册搜索服务工具（示例）
    if settings.search_api_key:
        manager.register_tool(
            name="search",
            description="搜索服务 - 提供网络搜索功能",
            server_command=["uvx", "search-mcp-server"],
            env={"SEARCH_API_KEY": settings.search_api_key},
            auto_expand=True
        )
    
    print(f"MCP 工具初始化完成，共注册 {len(manager)} 个工具")
    return manager


# 创建全局 MCP 工具管理器实例
mcp_tool_manager = MCPToolManager()


if __name__ == "__main__":
    # 示例用法
    print("=== MCP 工具管理器示例 ===")
    
    # 创建管理器
    manager = MCPToolManager()
    
    print(f"已注册工具：{manager.get_tool_names()}")
    print(f"工具数量：{len(manager)}")
    
    # 测试单例模式
    manager2 = MCPToolManager()
    print(f"单例测试 - 同一实例：{manager is manager2}")
    
    print("\n=== 使用示例 ===")
    print("""
# 完整使用示例：
from app.tools.mcp_tools import MCPToolManager, settings

# 创建管理器（单例）
manager = MCPToolManager()

# 注册高德地图工具
manager.register_tool(
    name="amap",
    description="高德地图服务",
    server_command=["python", "-m", "amap_mcp_server"],
    env={"AMAP_MAPS_API_KEY": settings.amap_api_key},
    auto_expand=True
)

# 获取工具
amap_tool = manager.get_tool("amap")

# 调用工具
result = amap_tool.invoke(
    name="maps_text_search",
    arguments={"keywords": "景点", "city": "北京"}
)
print(result)
    """)
