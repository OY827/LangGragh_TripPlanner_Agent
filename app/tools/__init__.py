"""
Tools层 - Agent工具和MCP工具管理

包含：
- agent_tools: Agent执行工具（计算器、搜索、天气查询、高德地图工具等）
- mcp_tools: MCP工具管理器（MCP协议集成）
"""

from app.tools.agent_tools import (
    CalculatorTool,
    SearchTool,
    DateTimeTool,
    WeatherTool,
    AmapAttractionTool,
    AmapWeatherTool,
    AmapHotelTool,
    RollingGoHotelTool,
    FlyaiTicketSearchTool,
    AmapRouteTool,
    get_basic_tools,
    get_all_tools,
    get_amap_tools,
    get_attraction_tools,
    get_weather_tools,
    get_hotel_tools,
    get_route_tools,
)

from app.tools.mcp_tools import (
    Settings,
    MCPToolWrapper,
    MCPToolManager,
    settings,
    init_mcp_tools,
    mcp_tool_manager,
)

__all__ = [
    # Agent工具
    "CalculatorTool",
    "SearchTool",
    "DateTimeTool",
    "WeatherTool",
    "AmapAttractionTool",
    "AmapWeatherTool",
    "AmapHotelTool",
    "RollingGoHotelTool",
    "FlyaiTicketSearchTool",
    "AmapRouteTool",
    "get_basic_tools",
    "get_all_tools",
    "get_amap_tools",
    "get_attraction_tools",
    "get_weather_tools",
    "get_hotel_tools",
    "get_route_tools",
    # MCP工具
    "Settings",
    "MCPToolWrapper",
    "MCPToolManager",
    "settings",
    "init_mcp_tools",
    "mcp_tool_manager",
]
