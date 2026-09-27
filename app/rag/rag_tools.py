"""
RAG 工具定义
将 RAG 功能封装为 Agent 可调用的工具
"""

from langchain_core.tools import BaseTool
from typing import Optional, Dict, Any, List, Type
from pydantic import BaseModel, Field

from app.rag import KnowledgeRetriever
from app.rag.generator import AnswerGenerator


# ==================== 工具参数Schema ====================

class RAGQuerySchema(BaseModel):
    """RAG通用查询参数"""
    query: str = Field(description="用户查询问题")
    top_k: int = Field(default=3, description="返回结果数量，默认3条")
    city: Optional[str] = Field(default=None, description="城市过滤条件")


class AttractionRecommendSchema(BaseModel):
    """景点推荐查询参数"""
    city: str = Field(description="目标城市")
    preferences: str = Field(description="用户偏好，逗号分隔，如'历史文化,自然风光'")
    travel_days: int = Field(default=1, description="旅行天数")


class AttractionDetailSchema(BaseModel):
    """景点详情查询参数"""
    attraction_name: str = Field(description="景点名称")
    city: str = Field(description="所在城市")


class GuideRecommendSchema(BaseModel):
    """旅行攻略推荐参数"""
    city: str = Field(description="目标城市")
    theme: str = Field(default="通用", description="旅行主题，如'亲子游'、'情侣游'、'深度游'")


# ==================== RAG 工具类 ====================

class RAGQueryTool(BaseTool):
    """
    RAG 通用查询工具

    适用场景：
    - 用户提出关于景点、城市的一般性问题
    - 需要从知识库中检索相关信息
    - 回答景点介绍、历史文化等问题
    """

    name: str = "rag_通用查询"
    description: str = """从知识库中检索相关信息并生成回答。

输入参数：
- query: 用户查询问题（必填）
- top_k: 返回结果数量，默认3条
- city: 城市过滤条件（可选）

使用示例：
- "查询北京有哪些历史景点" -> query="北京历史景点推荐", city="北京"
- "介绍一下故宫" -> query="故宫历史介绍", city="北京"
- "北京三日游推荐" -> query="北京三日游攻略", city="北京"
"""
    args_schema: Type[BaseModel] = RAGQuerySchema

    retriever: KnowledgeRetriever = None  # 运行时注入

    def __init__(self, retriever: Optional[KnowledgeRetriever] = None, **kwargs):
        super().__init__(**kwargs)
        if retriever:
            self.retriever = retriever
        else:
            self.retriever = KnowledgeRetriever()

    def _run(self, query: str, top_k: int = 3, city: Optional[str] = None) -> str:
        """执行RAG查询"""
        try:
            # 构建过滤条件
            filters = {}
            if city:
                filters["city"] = city

            # 检索知识
            results = self.retriever.retrieve(
                query=query,
                filters=filters if filters else None,
                top_k=top_k
            )

            if not results:
                return f"抱歉，知识库中暂无关于「{query}」的相关信息。"

            # 格式化输出
            output_parts = [f"📚 根据您的查询「{query}」，找到以下相关信息：\n"]

            for i, result in enumerate(results, 1):
                title = result["metadata"].get("title", "未知")
                content = result["content"]
                score = result.get("score", 0)

                output_parts.append(f"【{i}. {title}】(相关度: {score:.2f})")
                output_parts.append(f"   {content}\n")

            return "\n".join(output_parts)

        except Exception as e:
            return f"❌ RAG查询出错: {str(e)}"

    async def _arun(self, query: str, top_k: int = 3, city: Optional[str] = None) -> str:
        """异步执行RAG查询"""
        return self._run(query, top_k, city)


class AttractionRecommendTool(BaseTool):
    """
    景点推荐工具

    专门用于根据城市和偏好推荐景点，返回推荐列表和详细信息。
    """

    name: str = "景点推荐"
    description: str = """根据城市和偏好推荐景点，返回景点列表及详细信息。

输入参数：
- city: 目标城市（必填）
- preferences: 用户偏好，逗号分隔，如'历史文化,自然风光'（必填）
- travel_days: 旅行天数，默认1天（必填）

使用示例：
- "推荐北京的历史文化景点，2日游" -> city="北京", preferences="历史文化", travel_days=2
- "上海有哪些自然风光景点" -> city="上海", preferences="自然风光", travel_days=1
"""
    args_schema: Type[BaseModel] = AttractionRecommendSchema

    retriever: KnowledgeRetriever = None
    generator: AnswerGenerator = None

    def __init__(self, retriever: Optional[KnowledgeRetriever] = None, generator: Optional[AnswerGenerator] = None, **kwargs):
        super().__init__(**kwargs)
        if retriever:
            self.retriever = retriever
        else:
            self.retriever = KnowledgeRetriever()
        
        if generator:
            self.generator = generator
        else:
            self.generator = AnswerGenerator()

    def _run(self, city: str, preferences: str, travel_days: int = 1) -> str:
        """执行景点推荐（完整RAG流程）"""
        try:
            # 解析偏好列表
            pref_list = [p.strip() for p in preferences.split(",")]

            # 🔍 Step 1: 使用检索器获取景点知识
            results = self.retriever.retrieve_attractions(
                city=city,
                preferences=pref_list,
                travel_days=travel_days
            )

            if not results:
                return f"抱歉，知识库中暂无{city}的{preferences}类景点信息。"

            # 📝 Step 2: 使用生成器生成自然语言推荐
            recommendation = self.generator.generate_itinerary(
                city=city,
                days=travel_days,
                attractions=results,
                preferences=pref_list
            )

            return recommendation

        except Exception as e:
            return f"❌ 景点推荐出错: {str(e)}"

    async def _arun(self, city: str, preferences: str, travel_days: int = 1) -> str:
        return self._run(city, preferences, travel_days)


class AttractionDetailTool(BaseTool):
    """
    景点详情查询工具

    查询特定景点的详细信息，包括历史背景、开放时间、门票价格等。
    """

    name: str = "景点详情"
    description: str = """查询特定景点的详细信息。

输入参数：
- attraction_name: 景点名称（必填）
- city: 所在城市（必填）

使用示例：
- "查询故宫的详细信息" -> attraction_name="故宫", city="北京"
- "天坛在哪里" -> attraction_name="天坛", city="北京"
"""
    args_schema: Type[BaseModel] = AttractionDetailSchema

    retriever: KnowledgeRetriever = None

    def __init__(self, retriever: Optional[KnowledgeRetriever] = None, **kwargs):
        super().__init__(**kwargs)
        if retriever:
            self.retriever = retriever
        else:
            self.retriever = KnowledgeRetriever()

    def _run(self, attraction_name: str, city: str) -> str:
        """查询景点详情"""
        try:
            # 检索景点详情
            results = self.retriever.retrieve(
                query=f"{attraction_name} 详细介绍 历史 攻略",
                filters={"city": city, "type": "attraction"},
                top_k=3
            )

            # 过滤出目标景点
            target_result = None
            for r in results:
                if attraction_name in r["metadata"].get("title", ""):
                    target_result = r
                    break

            if not target_result and results:
                target_result = results[0]

            if not target_result:
                return f"抱歉，知识库中暂无{attraction_name}的详细信息。"

            # 格式化输出
            metadata = target_result["metadata"]
            content = target_result["content"]

            output_parts = [
                f"📍 {metadata.get('title', attraction_name)}",
                f"{'─' * 40}",
                f"🏷️ 类型: {metadata.get('category', '未知')}",
                f"⭐ 评分: {metadata.get('rating', 'N/A')}",
                f"💰 票价: {metadata.get('price', 'N/A')}",
                f"⏱️ 建议游览: {metadata.get('duration', 60)}分钟",
                "",
                f"📖 详细介绍:",
                f"{content}"
            ]

            return "\n".join(output_parts)

        except Exception as e:
            return f"❌ 景点详情查询出错: {str(e)}"

    async def _arun(self, attraction_name: str, city: str) -> str:
        return self._run(attraction_name, city)


class GuideRecommendTool(BaseTool):
    """
    旅行攻略推荐工具

    根据旅行主题推荐相关攻略，包括路线建议、注意事项等。
    """

    name: str = "旅行攻略"
    description: str = """推荐旅行攻略和路线建议。

输入参数：
- city: 目标城市（必填）
- theme: 旅行主题（必填），如'亲子游'、'情侣游'、'深度游'、'穷游'

使用示例：
- "北京亲子游攻略" -> city="北京", theme="亲子游"
- "上海情侣游推荐" -> city="上海", theme="情侣游"
"""
    args_schema: Type[BaseModel] = GuideRecommendSchema

    retriever: KnowledgeRetriever = None

    def __init__(self, retriever: Optional[KnowledgeRetriever] = None, **kwargs):
        super().__init__(**kwargs)
        if retriever:
            self.retriever = retriever
        else:
            self.retriever = KnowledgeRetriever()

    def _run(self, city: str, theme: str = "通用") -> str:
        """推荐旅行攻略"""
        try:
            results = self.retriever.retrieve_guides(
                query=f"{city} {theme} 旅行攻略 路线建议",
                city=city,
                top_k=3,
            )

            if not results:
                return f"抱歉，知识库中暂无{city}的{theme}相关攻略。"

            # 格式化输出
            output_parts = [
                f"🗺️ {city}{theme}旅行攻略\n"
            ]

            for i, result in enumerate(results, 1):
                title = result["metadata"].get("title", f"攻略{i}")
                content = result["content"]
                category = result["metadata"].get("category", "攻略")

                output_parts.append(f"{'─' * 40}")
                output_parts.append(f"📌 {title} [{category}]")
                output_parts.append(f"{content}")

            return "\n".join(output_parts)

        except Exception as e:
            return f"❌ 攻略推荐出错: {str(e)}"

    async def _arun(self, city: str, theme: str = "通用") -> str:
        return self._run(city, theme)


# ==================== 工具集合 ====================

class RAGToolkit:
    """
    RAG 工具包

    整合所有 RAG 工具，便于统一注册到 Agent
    """

    def __init__(self, retriever: Optional[KnowledgeRetriever] = None):
        self.retriever = retriever

        # 初始化所有工具
        self.tools = [
            RAGQueryTool(retriever=retriever),
            AttractionRecommendTool(retriever=retriever),
            AttractionDetailTool(retriever=retriever),
            GuideRecommendTool(retriever=retriever),
        ]

    def get_tools(self) -> List[BaseTool]:
        """获取所有工具"""
        return self.tools

    def get_tool_by_name(self, name: str) -> Optional[BaseTool]:
        """根据名称获取工具"""
        for tool in self.tools:
            if tool.name == name or name in tool.name:
                return tool
        return None

    def get_tools_dict(self) -> Dict[str, BaseTool]:
        """获取工具字典 {name: tool}"""
        return {tool.name: tool for tool in self.tools}


# ==================== 工厂函数 ====================

def get_rag_tools() -> List[BaseTool]:
    """获取默认的RAG工具列表"""
    toolkit = RAGToolkit()
    return toolkit.get_tools()


def get_rag_toolkit() -> RAGToolkit:
    """获取RAG工具包"""
    return RAGToolkit()
