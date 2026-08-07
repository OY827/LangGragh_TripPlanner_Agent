"""
生成器模块 - Generator
负责基于检索结果生成自然语言回答

核心功能：
1. 通用问答生成
2. 行程规划生成
3. 景点推荐文案生成
"""

from typing import List, Dict, Optional, Union
try:
    from langchain_openai import ChatOpenAI
except ImportError:
    from langchain_community.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate


class AnswerGenerator:
    """
    答案生成器
    
    基于检索到的知识和用户查询，调用LLM生成自然语言回答。
    """

    def __init__(
        self,
        provider: str = "aliyun",
        model: str = None,
        temperature: float = 0.7,
        max_tokens: int = 2048
    ):
        """
        初始化生成器
        
        Args:
            provider: LLM服务提供商 ("aliyun", "openai", "local")
            model: 模型名称
            temperature: 生成温度（0-1，越低越严谨）
            max_tokens: 最大生成token数
        """
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # 初始化LLM
        self.llm = self._init_llm()

    def _init_llm(self):
        """初始化LLM客户端"""
        try:
            if self.provider == "aliyun":
                from langchain_community.chat_models import Tongyi
                return Tongyi(
                    model_name=self.model,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
            elif self.provider == "openai":
                return ChatOpenAI(
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
            elif self.provider == "local":
                from langchain_community.chat_models import ChatOllama
                return ChatOllama(
                    model=self.model,
                    temperature=self.temperature
                )
            else:
                # 默认使用通义千问
                from langchain_community.chat_models import Tongyi
                import os
                model_name = self.model or os.getenv("DASHSCOPE_MODEL", "qwen-turbo")
                return Tongyi(
                    model_name=model_name,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
        except ImportError:
            # 降级为简单的字符串处理
            return None

    def generate(
        self,
        query: str,
        context: List[Dict],
        persona: str = "专业旅行顾问",
        max_context_length: int = 4000
    ) -> str:
        """
        生成回答
        
        Args:
            query: 用户查询
            context: 检索到的知识列表
            persona: AI角色设定
            max_context_length: 最大上下文长度
        
        Returns:
            自然语言回答
        """
        if not self.llm:
            return self._fallback_generate(query, context)

        # 构建上下文
        context_text = self._build_context(context, max_context_length)

        # 构建Prompt
        messages = self._build_prompt(query, context_text, persona)

        # 调用LLM
        try:
            response = self.llm.invoke(messages)
            return response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            return f"生成回答时出错: {str(e)}\n\n{self._fallback_generate(query, context)}"

    def generate_itinerary(
        self,
        city: str,
        days: int,
        attractions: List[Dict],
        preferences: Optional[List[str]] = None
    ) -> str:
        """
        生成行程规划
        
        Args:
            city: 目标城市
            days: 旅行天数
            attractions: 景点列表（检索结果）
            preferences: 用户偏好
        
        Returns:
            完整的行程规划文案
        """
        if not self.llm:
            return self._fallback_itinerary(city, days, attractions)

        # 格式化景点信息
        attraction_text = self._format_attractions(attractions)

        # 构建Prompt
        preference_str = ", ".join(preferences) if preferences else "不限"
        
        prompt = f"""
你是一位专业的旅行规划师，请帮用户规划{city}{days}日游行程。

用户偏好：{preference_str}

可选景点：
{attraction_text}

要求：
1. 每天安排3-5个景点（大型景区可2-3个），按visit_duration排满全天（约09:00-21:00）
2. 考虑景点的地理位置和游览时长，减少折返
3. 包含交通建议和时间安排（优先使用真实路线距离与时长）
4. 标注门票价格和预计游览时间
5. 使用友好、清晰的格式呈现
6. 给出实用的旅行小贴士

请输出完整的行程规划。
"""

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            return response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            return f"生成行程时出错: {str(e)}\n\n{self._fallback_itinerary(city, days, attractions)}"

    def generate_attraction_intro(
        self,
        attraction_name: str,
        city: str,
        context: List[Dict]
    ) -> str:
        """
        生成景点介绍
        
        Args:
            attraction_name: 景点名称
            city: 城市
            context: 检索到的景点知识
        
        Returns:
            生动的景点介绍文案
        """
        if not self.llm:
            return self._fallback_intro(attraction_name, context)

        context_text = self._build_context(context, 2000)

        prompt = f"""
请为{city}的{attraction_name}写一段生动的介绍。

参考资料：
{context_text}

要求：
1. 突出景点的特色和亮点
2. 使用吸引人的语言
3. 包含历史背景、特色景点、游玩建议
4. 适合向游客推荐
"""

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            return response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            return f"生成介绍时出错: {str(e)}\n\n{self._fallback_intro(attraction_name, context)}"

    def _build_context(self, context: List[Dict], max_length: int) -> str:
        """构建上下文文本"""
        context_parts = []
        current_length = 0

        for item in context:
            title = item["metadata"].get("title", "知识")
            content = item["content"]
            part = f"【{title}】\n{content}\n\n"
            
            if current_length + len(part) <= max_length:
                context_parts.append(part)
                current_length += len(part)

        return "".join(context_parts)

    def _build_prompt(
        self,
        query: str,
        context: str,
        persona: str
    ) -> List[Union[SystemMessage, HumanMessage]]:
        """构建Prompt消息"""
        system_msg = SystemMessage(content=f"""
你是{persona}，擅长根据用户需求提供专业的旅行建议。

要求：
1. 优先使用提供的知识库信息回答
2. 结合景点的评分、票价、游览时长等信息给出实用建议
3. 回答要专业、简洁、有条理
4. 如果知识库信息不足，诚实地说明"知识库暂无此信息"
5. 可以适当补充知识库之外的相关信息
""")

        human_msg = HumanMessage(content=f"""
知识库信息：
{context}

用户问题：{query}

请基于知识库信息回答用户问题。
""")

        return [system_msg, human_msg]

    def _format_attractions(self, attractions: List[Dict]) -> str:
        """格式化景点列表"""
        lines = []
        for i, attr in enumerate(attractions, 1):
            title = attr["metadata"].get("title", "未知景点")
            category = attr["metadata"].get("category", "未知")
            rating = attr["metadata"].get("rating", "N/A")
            price = attr["metadata"].get("ticket_price", "N/A")
            duration = attr["metadata"].get("visit_duration", 60)
            
            lines.append(
                f"{i}. {title} - {category}"
                f" | 评分: {rating}"
                f" | 票价: {price}元"
                f" | 游览时长: {duration}分钟"
            )
        
        return "\n".join(lines)

    # ============ 降级方案（无LLM时使用）============
    
    def _fallback_generate(self, query: str, context: List[Dict]) -> str:
        """降级生成方案"""
        if not context:
            return f"抱歉，知识库中暂无关于「{query}」的信息。"
        
        parts = [f"根据知识库，关于「{query}」的信息如下：\n"]
        for i, item in enumerate(context, 1):
            title = item["metadata"].get("title", f"知识{i}")
            parts.append(f"【{title}】")
            parts.append(item["content"])
        
        return "\n".join(parts)

    def _fallback_itinerary(self, city: str, days: int, attractions: List[Dict]) -> str:
        """降级行程规划"""
        parts = [f"{city}{days}日游行程推荐\n"]
        
        # 简单分配景点
        attractions_per_day = (len(attractions) + days - 1) // days
        
        for day in range(1, days + 1):
            start_idx = (day - 1) * attractions_per_day
            day_attractions = attractions[start_idx:start_idx + attractions_per_day]
            
            parts.append(f"\n📅 Day {day}:")
            for attr in day_attractions:
                title = attr["metadata"].get("title", "未知")
                price = attr["metadata"].get("price", "N/A")
                duration = attr["metadata"].get("duration", 60)
                price_str = price if price and ("元" in price or price == "免费") else f"{price}元"
                parts.append(f"  - {title}（票价: {price_str}，游览{duration}分钟）")
        
        return "\n".join(parts)

    def _fallback_intro(self, attraction_name: str, context: List[Dict]) -> str:
        """降级景点介绍"""
        if not context:
            return f"暂无{attraction_name}的详细信息。"
        
        # 使用第一个检索结果
        item = context[0]
        title = item["metadata"].get("title", attraction_name)
        content = item["content"]
        
        return f"📍 {title}\n\n{content}"


# ==================== 便捷函数 ====================

def create_generator(
    provider: str = "aliyun",
    model: str = None,
    **kwargs
) -> AnswerGenerator:
    """创建生成器实例"""
    return AnswerGenerator(provider=provider, model=model, **kwargs)


# ==================== 示例 ====================

if __name__ == "__main__":
    # 创建生成器
    generator = AnswerGenerator(provider="aliyun")
    
    # 示例1: 生成问答
    context = [
        {
            "content": "故宫是明清两代的皇家宫殿，位于北京中心。",
            "metadata": {"title": "故宫"}
        }
    ]
    answer = generator.generate("故宫是什么？", context)
    print("问答生成示例:\n", answer, "\n")
    
    # 示例2: 生成行程
    attractions = [
        {
            "metadata": {
                "title": "故宫",
                "category": "历史文化",
                "rating": 4.9,
                "ticket_price": 60,
                "visit_duration": 240
            },
            "content": "故宫是世界上现存规模最大的木质结构古建筑群。"
        },
        {
            "metadata": {
                "title": "天坛",
                "category": "历史文化",
                "rating": 4.8,
                "ticket_price": 35,
                "visit_duration": 120
            },
            "content": "天坛是明清两代皇帝祭天的场所。"
        }
    ]
    itinerary = generator.generate_itinerary("北京", 1, attractions)
    print("行程生成示例:\n", itinerary)
