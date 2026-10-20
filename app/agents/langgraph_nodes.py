"""
LangGraph 节点入口。

具体职责拆在景点、天气、酒店、规划、路线、时间轴、预算和对话模块里。
GraphNodes 把这些能力组合在一起，供主图和现有调用方使用。
"""

from app.llm import get_llm
from app.rag.retriever import KnowledgeRetriever
from app.rag.hybrid_rag import HybridRAG, HybridRAGConfig
from app.services.amap_service import AmapService
from app.memory.memory_manager import TravelMemoryManager
from app.monitoring import get_metrics, get_logger
from app.prompts import get_prompt_manager

from app.agents.schedule import ScheduleMixin
from app.agents.route import RouteDistanceTool, RouteMixin
from app.agents.recommend import AmapPoolTool, RecommendMixin
from app.agents.weather import WeatherMixin
from app.agents.hotels import HotelMixin
from app.agents.budget import BudgetMixin
from app.agents.planner import PlannerMixin
from app.agents.dialog_node import DialogMixin

__all__ = ["GraphNodes", "AmapPoolTool", "RouteDistanceTool"]


class GraphNodes(
    RecommendMixin,
    WeatherMixin,
    HotelMixin,
    PlannerMixin,
    BudgetMixin,
    DialogMixin,
    RouteMixin,
    ScheduleMixin,
):
    """LangGraph 节点函数集合。"""

    def __init__(
        self,
        llm_provider: str = "aliyun",
        llm_model: str = None,
        enable_monitoring: bool = True
    ):
        self.llm = get_llm(provider=llm_provider, model=llm_model, temperature=0.7)
        self.retriever = KnowledgeRetriever()
        self.hybrid_rag = HybridRAG(
            retriever=self.retriever,
            llm_provider=llm_provider,
            llm_model=llm_model,
            config=HybridRAGConfig(
                max_query_variants=3,
                min_relevant_docs=5,
                max_retrieval_rounds=2,
            )
        )
        self.amap_service = AmapService()
        self.memory_manager = TravelMemoryManager()
        self.enable_monitoring = enable_monitoring
        self.metrics = get_metrics() if enable_monitoring else None
        self.logger = get_logger() if enable_monitoring else None

        self.prompt_manager = get_prompt_manager()
        self._agent_prompts = {}
        self._hotel_agent = None
