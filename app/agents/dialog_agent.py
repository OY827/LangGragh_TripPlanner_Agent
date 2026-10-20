"""
对话修改 Agent

负责与用户多轮对话，根据修改需求增量更新旅行计划。
核心能力：
1. 意图识别：理解用户修改需求（预算/景点/酒店/天数等）
2. 增量修改：基于当前计划局部调整，而非全量重生成
3. 差异生成：自动对比新旧版本，生成人类可读的差异说明
4. 版本管理：通过 VersionManager 保存每次修改的版本快照

工作流程：
    用户输入 → 意图识别 →（改景点/酒店时工具搜索）→ 增量修改/全量重生成 → 差异计算 → 版本保存 → 回复用户
"""

import json
import re
from typing import Dict, Any, Optional, List

from app.agents.dialog_models import (
    PlanVersion, DialogSession, DialogMessage,
    ChangeIntent, PlanDiff,
    ChangeType, ModifyScope
)
from app.agents.version_manager import VersionManager
from app.llm import get_llm
from app.services.structured_output import (
    DialogPlanUpdate,
    IntentOutput,
    as_dict,
    invoke_structured,
)
from app.prompts import get_prompt_manager
from app.monitoring import traced
from app.memory.sliding_window import SlidingWindowChatMemory


class DialogAgent:
    """
    对话修改 Agent
    
    通过多轮对话帮助用户修改和完善旅行计划。
    每次修改都创建新版本，支持回退和对比。
    """
    
    def __init__(
        self,
        llm_provider: str = "aliyun",
        llm_model: str = None,
        use_redis: bool = True,
        context_window_size: int = 6,
        enable_rag: bool = True,
        enable_amap: bool = True,
        enable_memory: bool = True
    ):
        """
        初始化对话 Agent
        
        Args:
            llm_provider: LLM 服务商
            llm_model: 模型名称
            use_redis: 版本管理器是否使用 Redis（会话/计划版本）
            context_window_size: LangChain 滑动窗口保留的最近消息条数
            enable_rag: 是否启用知识库检索
            enable_amap: 是否启用高德地图搜索
            enable_memory: 是否启用 Chroma 长期记忆
        """
        self.llm = get_llm(provider=llm_provider, model=llm_model, temperature=0.3)
        self.version_manager = VersionManager(use_redis=use_redis)
        self.prompt_manager = get_prompt_manager()
        self.context_window_size = context_window_size
        self.chat_memory = SlidingWindowChatMemory(window_size=context_window_size)

        # 外部工具集成（带优雅降级）
        self.rag = None
        self.amap = None
        self.memory = None

        if enable_rag:
            try:
                from app.rag.hybrid_rag import HybridRAG, HybridRAGConfig
                self.rag = HybridRAG(
                    llm_provider=llm_provider,
                    llm_model=llm_model,
                    config=HybridRAGConfig()
                )
                print("✅ 对话Agent: 知识库检索工具已加载")
            except Exception as e:
                print(f"⚠️ 对话Agent: 知识库加载失败，将使用LLM默认值: {e}")

        if enable_amap:
            try:
                from app.services.amap_service import get_amap_service
                self.amap = get_amap_service()
                print("✅ 对话Agent: 高德地图工具已加载")
            except Exception as e:
                print(f"⚠️ 对话Agent: 高德地图加载失败: {e}")

        if enable_memory:
            try:
                from app.memory.memory_manager import TravelMemoryManager
                self.memory = TravelMemoryManager()
                print("✅ 对话Agent: 记忆系统工具已加载")
            except Exception as e:
                print(f"⚠️ 对话Agent: 记忆系统加载失败: {e}")
    
    # ========== 对话入口 ==========
    
    @traced(name="dialog_process_modification", run_type="chain", tags=["dialog"])
    def process_modification(
        self,
        session_id: str,
        user_input: str
    ) -> Dict[str, Any]:
        """
        处理用户的修改请求（对话 Agent 主入口）
        
        流程：
        1. 获取当前版本和对话历史
        2. 检测是否为版本控制命令（回退/对比）
        3. 指代消解（"那个景点"→具体景点）
        4. 意图识别（注入对话上下文）
        5. 意图二次校验
        6. 执行修改（增量/全量）
        7. 计算差异、创建新版本
        8. 生成回复 + 主动推荐
        
        Args:
            session_id: 会话ID
            user_input: 用户的修改需求（自然语言）
            
        Returns:
            包含以下字段的字典：
            - reply: 回复内容
            - new_version_id: 新版本ID
            - diff: 差异信息
            - changes: 变更描述列表
            - suggestions: 主动推荐列表
        """
        # 1. 获取当前版本
        current_version = self.version_manager.get_current_version(session_id)
        if not current_version:
            return {
                "reply": "未找到当前旅行计划，请先创建一个计划。",
                "new_version_id": None,
                "diff": None,
                "changes": [],
                "suggestions": []
            }
        
        current_plan = current_version.plan_data

        # 2. 完整历史走 Redis 会话；送进 LLM 前用 LangChain trim_messages 滑窗压缩
        raw_history = self.version_manager.get_conversation_history(session_id, limit=100)
        conversation_history = self.chat_memory.compress_records(raw_history)
        version_context = self.version_manager.format_versions_for_llm(session_id)

        # 3. 检测版本控制命令（回退/对比）
        version_cmd = self._detect_version_control_command(user_input)
        if version_cmd:
            return self._handle_version_control(
                session_id, version_cmd, conversation_history
            )

        # 4. 指代消解：将"那个景点"等指代替换为具体内容
        resolved_input = self._resolve_reference(user_input, conversation_history, current_plan)

        # 5. 意图识别（注入对话上下文）
        intent = self._analyze_intent(
            resolved_input, current_plan, conversation_history, version_context
        )

        # 6. 意图二次校验
        intent = self._validate_intent(intent, resolved_input)

        # 7. 记录用户消息
        self.version_manager.add_message(
            session_id=session_id,
            role="user",
            content=user_input
        )

        # 8. 根据意图执行修改
        if intent.change_type == ChangeType.NO_CHANGE.value:
            reply = self._generate_no_change_reply(user_input, current_plan)
            suggestions = self._generate_proactive_suggestions(current_plan, session_id)
            if suggestions:
                reply += "\n\n" + suggestions
            self.version_manager.add_message(
                session_id=session_id,
                role="assistant",
                content=reply
            )
            return {
                "reply": reply,
                "new_version_id": None,
                "diff": None,
                "changes": [],
                "suggestions": suggestions.split("\n") if suggestions else []
            }

        # 9. 执行修改。改景点或酒店时，先让模型按用户要求调用搜索工具。
        if intent.scope == ModifyScope.FULL.value:
            new_plan = self._regenerate_plan(current_plan, intent, resolved_input)
        else:
            self._research_for_modification(current_plan, intent, resolved_input)
            new_plan = self._apply_incremental_change(current_plan, intent)

        if not new_plan:
            return {
                "reply": "抱歉，我无法处理您的修改请求，请尝试更具体的描述。",
                "new_version_id": None,
                "diff": None,
                "changes": [],
                "suggestions": []
            }

        # 10. 计算差异
        diff = self._compute_diff(current_plan, new_plan, current_version.version_id)

        # 11. 创建新版本
        new_version = self.version_manager.create_modified_version(
            session_id=session_id,
            plan_data=new_plan,
            change_intent=intent,
            diff=diff,
            changes=diff.change_descriptions
        )

        # 12. 生成回复 + 主动推荐
        reply = self._generate_reply(intent, diff, new_plan)
        suggestions = self._generate_proactive_suggestions(new_plan, session_id)
        if suggestions:
            reply += "\n\n" + suggestions

        # 13. 记录助手消息
        self.version_manager.add_message(
            session_id=session_id,
            role="assistant",
            content=reply,
            version_id=new_version.version_id,
            intent=intent,
            diff=diff
        )

        return {
            "reply": reply,
            "new_version_id": new_version.version_id,
            "diff": diff.to_dict(),
            "changes": diff.change_descriptions,
            "suggestions": suggestions.split("\n") if suggestions else []
        }
    
    # ========== 意图识别 ==========
    
    def _analyze_intent(
        self,
        user_input: str,
        current_plan: Dict[str, Any],
        conversation_history: List[Dict[str, Any]] = None,
        version_context: str = "",
    ) -> ChangeIntent:
        """
        使用 LLM 分析用户的修改意图（注入对话上下文）
        
        Args:
            user_input: 用户输入
            current_plan: 当前计划
            conversation_history: 已经过滑动窗口压缩的对话历史
            version_context: Redis 中的计划版本摘要，供查询/回退
            
        Returns:
            ChangeIntent 修改意图对象
        """
        prompt = self._build_intent_prompt(
            user_input, current_plan, conversation_history, version_context
        )
        
        try:
            parsed = invoke_structured(self.llm, IntentOutput, prompt)
            intent_data = as_dict(parsed)
            if isinstance(parsed, IntentOutput):
                details = {
                    key: value
                    for key, value in parsed.details.model_dump().items()
                    if value not in ("", None)
                }
                return ChangeIntent(
                    change_type=parsed.change_type or ChangeType.NO_CHANGE.value,
                    scope=parsed.scope or ModifyScope.INCREMENTAL.value,
                    description=parsed.description,
                    target_field=parsed.target_field,
                    details=details,
                    confidence=float(parsed.confidence),
                )
            if intent_data:
                details = intent_data.get("details") or {}
                if not isinstance(details, dict):
                    details = {}
                return ChangeIntent(
                    change_type=intent_data.get("change_type", ChangeType.NO_CHANGE.value),
                    scope=intent_data.get("scope", ModifyScope.INCREMENTAL.value),
                    description=intent_data.get("description", ""),
                    target_field=intent_data.get("target_field", ""),
                    details=details,
                    confidence=float(intent_data.get("confidence", 0.8)),
                )
        except Exception as e:
            print(f"意图识别失败，使用规则兜底: {e}")
        
        # 规则兜底：基于关键词匹配
        return self._rule_based_intent(user_input, current_plan)
    
    def _build_intent_prompt(
        self,
        user_input: str,
        current_plan: Dict[str, Any],
        conversation_history: List[Dict[str, Any]] = None,
        version_context: str = "",
    ) -> str:
        """构建意图识别的 LLM 提示词（包含滑动窗口后的对话上下文）"""
        plan_summary = {
            "city": current_plan.get("city", ""),
            "duration_days": current_plan.get("duration_days", 0),
            "attractions": current_plan.get("attractions", []),
            "budget": current_plan.get("budget", {})
        }
        
        context_str = self.chat_memory.format_for_prompt(conversation_history or [])
        
        context_section = ""
        if context_str:
            context_section = f"""
## 最近对话上下文（滑动窗口，最近{self.context_window_size}条）
{context_str}

请结合对话上下文理解用户的真实意图。例如：
- 如果用户说"那个不要了"，结合上下文确定"那个"指什么
- 如果用户说"再便宜点"，说明上一轮的降价幅度不够
"""
        version_section = ""
        if version_context:
            version_section = f"""
## 本会话计划版本（可回退/对比）
{version_context}
"""
        
        return f"""你是一个旅行计划修改意图分析器。请分析用户的修改需求。

## 当前计划摘要
{json.dumps(plan_summary, ensure_ascii=False, indent=2)}
{context_section}{version_section}
## 用户输入
{user_input}

## 请判断修改类型（返回JSON）
可选的 change_type：
- modify_budget: 调整预算
- modify_attractions: 修改景点
- modify_hotel: 修改酒店
- modify_duration: 修改天数
- add_item: 添加项目
- remove_item: 移除项目
- major_change: 重大变更（需要重新生成）
- no_change: 无需修改

可选的 scope：
- incremental: 增量修改（局部调整）
- full: 全量重新生成

注意事项：
- 如果用户提到具体天数（如"5天"、"延长到4天"），change_type 应为 modify_duration
- 如果用户提到"便宜/贵/预算/省钱"，change_type 应为 modify_budget
- 如果用户提到"酒店/住宿"，change_type 应为 modify_hotel
- 只有明确添加/移除具体景点时，才用 add_item / remove_item
- 模糊请求（如"多安排些景点"）应使用 major_change + full

请返回以下JSON格式（只返回JSON，不要其他内容）：
```json
{{
    "change_type": "add_item",
    "scope": "incremental",
    "description": "用户希望添加迪士尼乐园",
    "target_field": "attractions",
    "details": {{"action": "add", "item": "迪士尼"}},
    "confidence": 0.95
}}
```"""
    
    def _rule_based_intent(self, user_input: str, current_plan: Dict[str, Any]) -> ChangeIntent:
        """基于关键词的规则兜底意图识别"""
        input_lower = user_input.lower()
        
        # 预算相关
        if any(kw in user_input for kw in ["预算", "便宜", "贵", "省钱", "降低", "增加预算"]):
            action = "reduce" if any(kw in user_input for kw in ["便宜", "降低", "省钱"]) else "increase"
            return ChangeIntent(
                change_type=ChangeType.MODIFY_BUDGET.value,
                scope=ModifyScope.INCREMENTAL.value,
                description=user_input,
                target_field="budget",
                details={"action": action},
                confidence=0.7
            )
        
        # 添加景点
        if any(kw in user_input for kw in ["加上", "添加", "增加", "加入", "想去"]):
            return ChangeIntent(
                change_type=ChangeType.ADD_ITEM.value,
                scope=ModifyScope.INCREMENTAL.value,
                description=user_input,
                target_field="attractions",
                details={"action": "add"},
                confidence=0.7
            )
        
        # 替换景点（优先于单纯删除）
        if any(kw in user_input for kw in ["替换", "换成", "换一个", "换掉", "改成"]):
            return ChangeIntent(
                change_type=ChangeType.MODIFY_ATTRACTIONS.value,
                scope=ModifyScope.INCREMENTAL.value,
                description=user_input,
                target_field="attractions",
                details={"action": "replace"},
                confidence=0.8
            )

        # 移除景点
        if any(kw in user_input for kw in ["去掉", "删除", "不要", "取消", "去除", "移除"]):
            return ChangeIntent(
                change_type=ChangeType.REMOVE_ITEM.value,
                scope=ModifyScope.INCREMENTAL.value,
                description=user_input,
                target_field="attractions",
                details={"action": "remove"},
                confidence=0.7
            )
        
        # 修改天数
        if any(kw in user_input for kw in ["天数", "几天", "延长", "缩短", "多一天", "少一天"]):
            return ChangeIntent(
                change_type=ChangeType.MODIFY_DURATION.value,
                scope=ModifyScope.INCREMENTAL.value,
                description=user_input,
                target_field="duration_days",
                details={"action": "modify"},
                confidence=0.7
            )
        
        # 修改酒店
        if any(kw in user_input for kw in ["酒店", "住宿", "换酒店"]):
            return ChangeIntent(
                change_type=ChangeType.MODIFY_HOTEL.value,
                scope=ModifyScope.INCREMENTAL.value,
                description=user_input,
                target_field="hotel",
                details={"action": "modify"},
                confidence=0.7
            )
        
        # 默认：重大变更
        return ChangeIntent(
            change_type=ChangeType.MAJOR_CHANGE.value,
            scope=ModifyScope.FULL.value,
            description=user_input,
            confidence=0.5
        )
    
    def _validate_intent(self, intent: ChangeIntent, user_input: str) -> ChangeIntent:
        """
        意图二次校验：基于关键词规则修正 LLM 可能的误判
        
        解决问题：
        - "延长到5天" 被误判为 add_item
        - "酒店太贵" 被误判为 modify_budget 而非 modify_hotel
        - "降低酒店预算" 未正确设置 action 字段
        """
        desc = intent.description or user_input
        
        # 替换景点：去除/替换类请求统一为 modify_attractions
        replace_kws = ["替换", "换成", "换一个", "换掉", "改成", "换成别的", "换其他"]
        remove_kws = ["去掉", "删除", "不要", "取消", "去除", "移除"]
        if any(kw in user_input for kw in replace_kws) or (
            any(kw in user_input for kw in remove_kws)
            and any(kw in user_input for kw in ["其他", "别的", "换成", "替换"])
        ):
            intent.change_type = ChangeType.MODIFY_ATTRACTIONS.value
            intent.scope = ModifyScope.INCREMENTAL.value
            intent.target_field = "attractions"
            intent.details = intent.details or {}
            intent.details["action"] = "replace"
            intent.confidence = max(intent.confidence, 0.85)

        # 天数相关：如果输入包含天数关键词但意图不是 modify_duration
        if any(kw in user_input for kw in ["天", "延长", "缩短", "多一天", "少一天", "几天"]):
            # 检查是否有具体数字+天
            if re.search(r'\d+\s*天', user_input) or any(kw in user_input for kw in ["延长", "缩短"]):
                if intent.change_type != ChangeType.MODIFY_DURATION.value:
                    intent.change_type = ChangeType.MODIFY_DURATION.value
                    intent.target_field = "duration_days"
                    intent.details["action"] = "modify"
                    intent.confidence = min(intent.confidence + 0.1, 1.0)
        
        # 预算相关：如果输入包含预算关键词但意图不是 modify_budget/modify_hotel
        budget_kws = ["预算", "便宜", "贵", "省钱", "降低", "太贵"]
        hotel_kws = ["酒店", "住宿", "换酒店"]
        
        # 降价/提价关键词（用于确定 action）
        reduce_kws = ["便宜", "降低", "省钱", "太贵", "少点", "降", "减", "低", "省"]
        increase_kws = ["增加", "提高", "多点", "贵", "升", "增", "高", "多"]
        
        if any(kw in user_input for kw in budget_kws):
            if intent.change_type not in [ChangeType.MODIFY_BUDGET.value, ChangeType.MODIFY_HOTEL.value]:
                # 如果同时提到酒店，优先 modify_hotel
                if any(kw in user_input for kw in hotel_kws):
                    intent.change_type = ChangeType.MODIFY_HOTEL.value
                    intent.target_field = "hotel"
                else:
                    intent.change_type = ChangeType.MODIFY_BUDGET.value
                    intent.target_field = "budget"
            
            # 关键修复：无论 change_type 是否正确，都根据关键词设置 action
            has_reduce_kw = any(kw in user_input for kw in reduce_kws)
            has_increase_kw = any(kw in user_input for kw in increase_kws)
            
            if has_reduce_kw and not has_increase_kw:
                intent.details["action"] = "reduce"
            elif has_increase_kw and not has_reduce_kw:
                intent.details["action"] = "increase"
            
            intent.confidence = min(intent.confidence + 0.1, 1.0)
        
        # 添加景点：只有明确提到具体景点名时才保留 add_item
        if intent.change_type == ChangeType.ADD_ITEM.value:
            item = intent.details.get("item", "") or self._extract_attraction_name(desc)
            if not self._is_valid_attraction_name(item):
                # 不是具体景点名，改为 major_change
                intent.change_type = ChangeType.MAJOR_CHANGE.value
                intent.scope = ModifyScope.FULL.value
                intent.confidence = max(intent.confidence - 0.2, 0.3)
        
        return intent
    
    def _is_valid_attraction_name(self, name: str) -> bool:
        """
        判断提取的文本是否为有效的景点名称
        
        规则：
        - 长度 2-20 个字符
        - 不包含描述性词汇 / 整句意图描述
        - 不包含动词
        """
        if not name or len(name) < 2 or len(name) > 20:
            return False
        
        # 描述性词汇 / 意图句黑名单
        desc_words = [
            "更多", "一些", "有趣", "地方", "景点", "好玩", "特色",
            "推荐", "安排", "什么", "随便", "自由", "活动", "类型",
            "选择", "选项", "尽量", "其他", "希望", "用户", "替换",
            "去除", "去掉", "删除", "换成",
        ]
        if any(word in name for word in desc_words):
            return False
        
        # 动词黑名单
        verb_words = ["安排", "添加", "增加", "加入", "想去", "想要"]
        if any(word in name for word in verb_words):
            return False
        
        return True
    
    def _resolve_reference(
        self,
        user_input: str,
        conversation_history: List[Dict[str, Any]],
        current_plan: Dict[str, Any]
    ) -> str:
        """
        指代消解：将用户输入中的指代词替换为具体内容
        
        示例：
        - "那个景点不要了" → "迪士尼不要了"（从上一轮对话中提取）
        - "再便宜点" → "酒店预算再便宜点"（从上一轮对话中提取）
        - "回到之前的版本" → 保持原样（由版本控制处理）
        """
        if not conversation_history:
            return user_input
        
        # 检测指代词
        reference_words = ["那个", "这个", "刚才", "之前说", "上一个", "它"]
        has_reference = any(ref in user_input for ref in reference_words)
        
        if not has_reference:
            return user_input
        
        # 从最近的助手消息中提取上下文信息
        recent_assistant_msgs = [
            msg for msg in reversed(conversation_history)
            if msg.get("role") == "assistant"
        ]
        
        if not recent_assistant_msgs:
            return user_input
        
        last_assistant_msg = recent_assistant_msgs[0]
        last_content = last_assistant_msg.get("content", "")
        
        # 提取上一轮提到的景点
        attractions = current_plan.get("attractions", [])
        
        # 从 diff 中提取上一轮新增的景点
        last_diff = last_assistant_msg.get("diff", {})
        if last_diff:
            added = last_diff.get("added_attractions", [])
            if added and "那个" in user_input:
                # "那个景点" → 上一轮新增的景点
                return user_input.replace("那个景点", added[0]).replace("那个", added[0])
        
        # 从 diff 中提取上一轮移除的景点
        if last_diff:
            removed = last_diff.get("removed_attractions", [])
            if removed and "那个" in user_input:
                return user_input.replace("那个景点", removed[0]).replace("那个", removed[0])
        
        # 如果指代"便宜"，从上一轮意图中提取目标字段
        last_intent = last_assistant_msg.get("intent", {})
        if last_intent and "便宜" in user_input:
            target = last_intent.get("target_field", "")
            if target == "hotel":
                return user_input.replace("再便宜点", "酒店再便宜点")
            elif target == "budget":
                return user_input.replace("再便宜点", "预算再降低一点")
        
        return user_input
    
    # ========== 版本控制命令 ==========
    
    def _detect_version_control_command(self, user_input: str) -> Optional[Dict[str, Any]]:
        """
        检测用户是否在执行版本控制命令
        
        支持的命令：
        - 回退：("回到之前的版本", "撤销上次修改", "回退")
        - 对比：("对比一下", "有什么变化", "版本对比")
        - 历史：("版本历史", "修改记录", "之前都改了什么")
        
        Returns:
            {"action": "rollback"|"compare"|"history", "details": ...} 或 None
        """
        # 回退命令
        rollback_kws = ["回退", "回到之前", "撤销", "撤回", "恢复到", "还原"]
        if any(kw in user_input for kw in rollback_kws):
            # 尝试提取目标版本号
            numbers = re.findall(r'v?(\d+)', user_input)
            target_version = int(numbers[0]) if numbers else None
            return {"action": "rollback", "target_version": target_version}
        
        # 对比命令
        compare_kws = ["对比", "比较", "差异", "有什么变化", "改了什么"]
        if any(kw in user_input for kw in compare_kws):
            numbers = re.findall(r'v?(\d+)', user_input)
            if len(numbers) >= 2:
                return {"action": "compare", "v1": int(numbers[0]), "v2": int(numbers[1])}
            elif len(numbers) == 1:
                return {"action": "compare", "v1": int(numbers[0]), "v2": None}
            return {"action": "compare", "v1": None, "v2": None}
        
        # 历史命令
        history_kws = ["版本历史", "修改记录", "之前都改了", "改过哪些", "历史版本"]
        if any(kw in user_input for kw in history_kws):
            return {"action": "history"}
        
        return None
    
    def _handle_version_control(
        self,
        session_id: str,
        cmd: Dict[str, Any],
        conversation_history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """处理版本控制命令"""
        action = cmd["action"]
        
        if action == "rollback":
            return self._handle_rollback_command(session_id, cmd, conversation_history)
        elif action == "compare":
            return self._handle_compare_command(session_id, cmd, conversation_history)
        elif action == "history":
            return self._handle_history_command(session_id, conversation_history)
        
        return {
            "reply": "未识别的版本控制命令。",
            "new_version_id": None,
            "diff": None,
            "changes": [],
            "suggestions": []
        }
    
    def _handle_rollback_command(
        self,
        session_id: str,
        cmd: Dict[str, Any],
        conversation_history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """处理回退命令"""
        history = self.version_manager.get_version_history(session_id)
        
        if len(history) < 2:
            reply = "当前只有一个版本，无法回退。"
            self.version_manager.add_message(session_id, "user", "回退")
            self.version_manager.add_message(session_id, "assistant", reply)
            return {"reply": reply, "new_version_id": None, "diff": None, "changes": [], "suggestions": []}
        
        # 确定目标版本
        target_version = cmd.get("target_version")
        if target_version:
            # 查找指定版本号
            target_v = next((v for v in history if v.version_number == target_version), None)
            if not target_v:
                reply = f"未找到版本 v{target_version}。"
                self.version_manager.add_message(session_id, "user", "回退")
                self.version_manager.add_message(session_id, "assistant", reply)
                return {"reply": reply, "new_version_id": None, "diff": None, "changes": [], "suggestions": []}
        else:
            # 默认回退到上一个版本
            target_v = history[-2]  # 倒数第二个版本
        
        # 执行回退
        rollback_version = self.version_manager.rollback_to_version(session_id, target_v.version_id)
        
        if rollback_version:
            reply = f"已回退到版本 v{target_v.version_number}。\n\n当前计划：\n"
            plan = rollback_version.plan_data
            reply += f"  • 城市: {plan.get('city', '未知')}\n"
            reply += f"  • 天数: {plan.get('duration_days', 0)}天\n"
            reply += f"  • 景点: {', '.join(plan.get('attractions', []))}\n"
            budget = plan.get("budget", {})
            if isinstance(budget, dict) and budget.get("total"):
                reply += f"  • 预算: {budget['total']}元\n"
            
            self.version_manager.add_message(session_id, "user", "回退")
            self.version_manager.add_message(session_id, "assistant", reply)
            
            return {
                "reply": reply,
                "new_version_id": rollback_version.version_id,
                "diff": None,
                "changes": [f"回退到 v{target_v.version_number}"],
                "suggestions": []
            }
        
        return {
            "reply": "回退失败，请稍后重试。",
            "new_version_id": None,
            "diff": None,
            "changes": [],
            "suggestions": []
        }
    
    def _handle_compare_command(
        self,
        session_id: str,
        cmd: Dict[str, Any],
        conversation_history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """处理版本对比命令"""
        history = self.version_manager.get_version_history(session_id)
        
        if len(history) < 2:
            reply = "当前只有一个版本，无法对比。"
            self.version_manager.add_message(session_id, "user", "对比")
            self.version_manager.add_message(session_id, "assistant", reply)
            return {"reply": reply, "new_version_id": None, "diff": None, "changes": [], "suggestions": []}
        
        # 确定对比的两个版本
        v1_num = cmd.get("v1")
        v2_num = cmd.get("v2")
        
        if v1_num and v2_num:
            v1 = next((v for v in history if v.version_number == v1_num), None)
            v2 = next((v for v in history if v.version_number == v2_num), None)
        else:
            # 默认对比最后两个版本
            v1 = history[-2]
            v2 = history[-1]
        
        if not v1 or not v2:
            reply = "未找到指定的版本。"
            self.version_manager.add_message(session_id, "user", "对比")
            self.version_manager.add_message(session_id, "assistant", reply)
            return {"reply": reply, "new_version_id": None, "diff": None, "changes": [], "suggestions": []}
        
        diff = self.version_manager.compare_versions(v1.version_id, v2.version_id)
        
        if diff:
            lines = [f"📋 版本对比 (v{v1.version_number} → v{v2.version_number})：", ""]
            if diff.change_descriptions:
                for desc in diff.change_descriptions:
                    lines.append(f"  • {desc}")
            else:
                lines.append("  两个版本无差异。")
            lines.append("")
            lines.append(f"摘要: {diff.summary}")
            
            reply = "\n".join(lines)
            self.version_manager.add_message(session_id, "user", "对比")
            self.version_manager.add_message(session_id, "assistant", reply)
            
            return {
                "reply": reply,
                "new_version_id": None,
                "diff": diff.to_dict(),
                "changes": diff.change_descriptions,
                "suggestions": []
            }
        
        return {
            "reply": "对比失败，请稍后重试。",
            "new_version_id": None,
            "diff": None,
            "changes": [],
            "suggestions": []
        }
    
    def _handle_history_command(
        self,
        session_id: str,
        conversation_history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """处理版本历史查询命令"""
        history = self.version_manager.get_version_history(session_id)
        
        lines = [f"📋 版本历史（共 {len(history)} 个版本）：", ""]
        lines.append(f"{'版本':<8} {'创建者':<10} {'变更摘要'}")
        lines.append("-" * 50)
        
        for v in history:
            changes_str = "; ".join(v.changes[:2]) if v.changes else "初始版本"
            lines.append(f"v{v.version_number:<7} {v.created_by:<10} {changes_str}")
        
        lines.append("")
        lines.append('您可以说"回退到v2"或"对比v1和v3"。')
        
        reply = "\n".join(lines)
        self.version_manager.add_message(session_id, "user", "版本历史")
        self.version_manager.add_message(session_id, "assistant", reply)
        
        return {
            "reply": reply,
            "new_version_id": None,
            "diff": None,
            "changes": [],
            "suggestions": []
        }
    
    # ========== 主动推荐 ==========
    
    def _generate_proactive_suggestions(self, plan: Dict[str, Any], session_id: str = None) -> str:
        """
        根据当前计划状态生成主动推荐（集成记忆系统和知识库）
        
        检测维度：
        1. 行程密度：每天景点数过多或过少
        2. 预算分布：某项占比异常
        3. 个性化推荐：基于用户历史推荐未去过的景点
        """
        suggestions = []
        
        days = plan.get("duration_days", 1)
        attractions = plan.get("attractions", [])
        attraction_count = len(attractions)
        budget = plan.get("budget", {})
        city = plan.get("city", "")
        
        if not isinstance(budget, dict):
            budget = {}
        
        # 1. 检测景点数量
        attractions_per_day = attraction_count / days if days > 0 else 0
        
        if attractions_per_day < 1.0 and days >= 2:
            suggestions.append(
                f"💡 您的行程共{days}天，但只有{attraction_count}个景点，"
                f"平均每天不到1个。可以考虑添加更多景点丰富行程。"
            )
        elif attractions_per_day > 3.0:
            suggestions.append(
                f"💡 您的行程共{days}天，安排了{attraction_count}个景点，"
                f"平均每天{attractions_per_day:.1f}个，可能比较赶。"
                f"建议适当减少景点或延长天数。"
            )
        
        # 2. 检测预算分布
        total = budget.get("total", 0)
        hotel = budget.get("total_hotels", 0)
        meals = budget.get("total_meals", 0)
        transport = budget.get("total_transportation", 0)
        
        if total > 0:
            hotel_ratio = hotel / total
            if hotel_ratio > 0.5:
                suggestions.append(
                    f"💡 酒店费用占比{hotel_ratio:.0%}，偏高。"
                    f"可以考虑换更经济的住宿来降低预算。"
                )
            
            meal_ratio = meals / total if total > 0 else 0
            if meal_ratio < 0.1 and total > 500:
                suggestions.append(
                    f"💡 餐饮预算占比仅{meal_ratio:.0%}，"
                    f"可能不足以覆盖每日三餐，建议适当增加。"
                )
        
        # 3. 个性化推荐：基于用户历史推荐未去过的景点
        if session_id and city:
            # 获取用户ID
            session = self.version_manager.get_session(session_id)
            if session and self.memory:
                visited = self._get_user_visited_attractions(session.user_id, city)
                unvisited_in_plan = [a for a in attractions if a not in visited]
                
                if visited:
                    suggestions.append(
                        f"💡 根据您的旅行历史，您在{city}已去过: {', '.join(visited[:3])}。"
                        f"本次计划中有{len(unvisited_in_plan)}个新景点。"
                    )
            
            # 从知识库推荐未去过的景点
            if self.rag and attraction_count < days * 2:
                current_attractions = set(attractions)
                recommendations = self._recommend_attractions_from_kb(city, exclude=list(current_attractions))
                
                if recommendations:
                    top_recs = recommendations[:3]
                    rec_str = "; ".join([
                        f"{r['name']}({r['ticket_price']}元)" 
                        for r in top_recs
                    ])
                    suggestions.append(
                        f"💡 知识库推荐{city}热门景点: {rec_str}"
                    )
        
        if not suggestions:
            return ""
        
        return "📌 智能推荐：\n" + "\n".join(suggestions)
    
    # ========== 工具集成方法 ==========
    
    def _search_attraction_from_kb(self, attraction_name: str, city: str) -> Optional[Dict[str, Any]]:
        """
        从知识库检索景点信息
        
        Returns:
            {"name": ..., "ticket_price": ..., "category": ..., "rating": ...} 或 None
        """
        if not self.rag:
            return None

        exact = self._lookup_attraction_exact(attraction_name, city)
        if exact:
            return exact
        
        try:
            docs, meta = self.rag.retrieve_with_validation(
                query=attraction_name,
                city=city,
                preferences=[]
            )
            
            if not docs:
                return None
            
            # 从检索结果中提取景点信息
            for doc in docs:
                doc_meta = doc.get("metadata", {})
                title = doc_meta.get("title", "")
                
                # 检查是否匹配目标景点
                if attraction_name in title or title in attraction_name:
                    return self._attraction_from_metadata(doc_meta, title, "知识库")
            
            # 未精确匹配，返回第一个结果作为参考
            if docs:
                doc = docs[0]
                doc_meta = doc.get("metadata", {})
                return self._attraction_from_metadata(
                    doc_meta,
                    doc_meta.get("title", attraction_name),
                    "知识库(模糊匹配)",
                )
        except Exception as e:
            print(f"知识库检索失败: {e}")
        
        return None

    def _lookup_attraction_exact(self, attraction_name: str, city: str) -> Optional[Dict[str, Any]]:
        """按标题直接读知识库，避免整段混合检索失败时丢掉游览时长。"""
        store = getattr(getattr(self.rag, "retriever", None), "vector_store", None)
        finder = getattr(store, "find_by_title", None)
        if finder is None:
            return None
        try:
            hits = finder(attraction_name, city=city, limit=3)
        except Exception as e:
            print(f"知识库精确查找失败: {e}")
            return None
        if not hits:
            return None
        meta = (hits[0].get("metadata") or {})
        return self._attraction_from_metadata(meta, attraction_name, "知识库")

    def _attraction_from_metadata(
        self,
        doc_meta: Dict[str, Any],
        fallback_name: str,
        source: str,
    ) -> Dict[str, Any]:
        price_str = doc_meta.get("price", "0")
        try:
            price = int(re.sub(r"[^\d]", "", str(price_str)) or 0)
        except (ValueError, TypeError):
            price = 0
        return {
            "name": doc_meta.get("title") or fallback_name,
            "ticket_price": price,
            "category": doc_meta.get("category", ""),
            "rating": doc_meta.get("rating", ""),
            "visit_duration": self._visit_minutes(doc_meta.get("visit_duration"), default=60),
            "address": doc_meta.get("address", ""),
            "location": self._location_text(doc_meta),
            "source": source,
        }

    @staticmethod
    def _visit_minutes(value: Any, default: int = 120) -> int:
        try:
            minutes = int(float(value))
        except (TypeError, ValueError):
            return default
        return minutes if minutes > 0 else default
    
    def _search_hotels_from_amap(self, city: str, budget_level: str = "medium") -> List[Dict[str, Any]]:
        """
        从高德地图搜索酒店信息
        
        Args:
            city: 城市名称
            budget_level: budget_level ("budget"经济 / "medium"中档 / "luxury"高档)
            
        Returns:
            酒店信息列表 [{"name": ..., "price": ..., "address": ...}]
        """
        if not self.amap:
            return []
        
        try:
            keywords_map = {
                "budget": "经济型酒店 快捷酒店",
                "medium": "酒店 商务酒店",
                "luxury": "五星酒店 豪华酒店"
            }
            keywords = keywords_map.get(budget_level, "酒店")
            
            pois = self.amap.search_poi(keywords=keywords, city=city, citylimit=True)
            
            hotels = []
            for poi in pois[:10]:
                hotel_info = {
                    "name": poi.name if hasattr(poi, 'name') else str(poi),
                    "address": poi.address if hasattr(poi, 'address') else "",
                    "tel": poi.tel if hasattr(poi, 'tel') else "",
                    "type": poi.type if hasattr(poi, 'type') else "",
                    "source": "高德地图"
                }
                hotels.append(hotel_info)
            
            return hotels
        except Exception as e:
            print(f"高德酒店搜索失败: {e}")
        
        return []
    
    def _get_user_visited_attractions(self, user_id: str, city: str) -> List[str]:
        """从记忆系统获取用户已去过的景点"""
        if not self.memory:
            return []
        
        try:
            return self.memory.get_visited_attractions(user_id, city)
        except Exception as e:
            print(f"获取历史景点失败: {e}")
            return []
    
    def _recommend_attractions_from_kb(self, city: str, exclude: List[str] = None) -> List[Dict[str, Any]]:
        """
        从知识库推荐景点（排除已去的）
        
        Returns:
            推荐景点列表 [{"name": ..., "ticket_price": ..., "category": ...}]
        """
        if not self.rag:
            return []
        
        try:
            docs, _ = self.rag.retrieve_with_validation(
                query=f"{city}热门景点推荐",
                city=city,
                preferences=[]
            )
            
            exclude_set = set(exclude or [])
            recommendations = []
            
            for doc in docs:
                doc_meta = doc.get("metadata", {})
                title = doc_meta.get("title", "")
                
                if title and title not in exclude_set and title not in [r["name"] for r in recommendations]:
                    price_str = doc_meta.get("price", "0")
                    try:
                        price = int(re.sub(r'[^\d]', '', str(price_str)) or 0)
                    except (ValueError, TypeError):
                        price = 0
                    
                    recommendations.append({
                        "name": title,
                        "ticket_price": price,
                        "category": doc_meta.get("category", ""),
                        "rating": doc_meta.get("rating", ""),
                        "source": "知识库"
                    })
                    
                    if len(recommendations) >= 5:
                        break
            
            return recommendations
        except Exception as e:
            print(f"知识库推荐失败: {e}")
        
        return []
    
    def _needs_live_search(self, change_type: str) -> bool:
        return change_type in (
            ChangeType.ADD_ITEM.value,
            ChangeType.MODIFY_ATTRACTIONS.value,
            ChangeType.MODIFY_HOTEL.value,
        )

    def _build_search_agent(self):
        """对话修改：景点搜索与推荐 Agent 使用同一个高德工具，酒店仍单独搜索。"""
        from app.agents.react_agent import ReActAgent
        from app.agents.recommend import AmapPoolTool
        from app.services.structured_output import SearchResultOutput
        from app.tools.agent_tools import RollingGoHotelTool

        search = getattr(self.amap, "search_poi", None)
        attraction_tool = AmapPoolTool(search_poi=search, pool=[])
        agent = ReActAgent(
            name="对话检索",
            tools=[attraction_tool, RollingGoHotelTool()],
            llm=self.llm,
            system_prompt=(
                "你负责按用户的修改要求重新搜索景点和酒店，不重写整份行程。\n"
                "改景点时调用 amap_attraction_search，一次只传一个关键词。\n"
                "数量不够或类型不符合用户要求时，换一个关键词再搜。\n"
                "改酒店时调用 hotel_search，带上城市、住宿关键词、入住日期和住宿晚数。\n"
                "attractions 和 hotel 只能填写对应工具返回过的原名，不要编造。"
            ),
            max_iterations=6,
            verbose=False,
            response_format=SearchResultOutput,
        )
        return agent, attraction_tool

    def _research_for_modification(
        self,
        plan: Dict[str, Any],
        intent: ChangeIntent,
        user_input: str,
    ) -> None:
        """让模型调用景点/酒店工具，把重新搜索的结果记到意图上。"""
        if not self._needs_live_search(intent.change_type):
            return
        city = plan.get("city", "")
        if not city:
            return
        days = int(plan.get("duration_days") or 1)
        stay_nights = max(1, days - 1)
        start_date = ""
        daily = plan.get("daily_itinerary")
        if isinstance(daily, dict):
            start_date = str(daily.get("start_date") or "")
        query = (
            f"城市：{city}\n"
            f"当前景点：{', '.join(plan.get('attractions') or []) or '无'}\n"
            f"当前住宿：{plan.get('accommodation') or '未指定'}\n"
            f"入住日期：{start_date or '未指定'}\n"
            f"住宿晚数：{stay_nights}\n"
            f"修改类型：{intent.change_type}\n"
            f"用户要求：{user_input}"
        )
        try:
            agent, attraction_tool = self._build_search_agent()
            text = agent.run(query)
        except Exception as e:
            print(f"对话检索失败: {e}")
            return
        parsed = self._parse_json_from_response(text) if text else None
        if not isinstance(parsed, dict):
            return
        attractions = parsed.get("attractions") or []
        if isinstance(attractions, str):
            attractions = [attractions]
        found = []
        seen = set()
        for item in attraction_tool.pool:
            name = str(item.get("name") or "").strip()
            if name and name not in seen:
                seen.add(name)
                found.append(name)
        attractions = [str(name).strip() for name in attractions if str(name).strip() in seen]
        if not attractions:
            attractions = found
        price = parsed.get("hotel_price")
        try:
            price = float(price) if price not in (None, "") else None
        except (TypeError, ValueError):
            price = None
        intent.details = dict(intent.details or {})
        intent.details["search_result"] = {
            "attractions": [str(name).strip() for name in attractions if str(name).strip()],
            "hotel": str(parsed.get("hotel") or "").strip(),
            "hotel_price": price,
            "summary": str(parsed.get("summary") or "").strip(),
        }

    # ========== 增量修改 ==========
    
    def _apply_incremental_change(
        self,
        current_plan: Dict[str, Any],
        intent: ChangeIntent
    ) -> Optional[Dict[str, Any]]:
        """
        根据意图对计划进行增量修改
        
        Args:
            current_plan: 当前计划
            intent: 修改意图
            
        Returns:
            修改后的计划，或 None 表示修改失败
        """
        import copy
        new_plan = copy.deepcopy(current_plan)
        
        change_type = intent.change_type
        details = intent.details
        
        if change_type == ChangeType.ADD_ITEM.value:
            # 检查是否能提取出具体景点名，如果不能则走全量重生成
            item = details.get("item", "") or self._extract_attraction_name(intent.description)
            if item and self._is_valid_attraction_name(item):
                return self._handle_add_item(new_plan, intent)
            else:
                # 模糊请求，走全量重生成
                intent.scope = ModifyScope.FULL.value
                return self._regenerate_plan(new_plan, intent, intent.description)
        
        elif change_type == ChangeType.REMOVE_ITEM.value:
            return self._handle_remove_item(new_plan, intent)
        
        elif change_type == ChangeType.MODIFY_BUDGET.value:
            return self._handle_modify_budget(new_plan, intent)
        
        elif change_type == ChangeType.MODIFY_ATTRACTIONS.value:
            return self._handle_modify_attractions(new_plan, intent)
        
        elif change_type == ChangeType.MODIFY_DURATION.value:
            return self._handle_modify_duration(new_plan, intent)
        
        elif change_type == ChangeType.MODIFY_HOTEL.value:
            return self._handle_modify_hotel(new_plan, intent)
        
        return None
    
    def _handle_add_item(self, plan: Dict[str, Any], intent: ChangeIntent) -> Dict[str, Any]:
        """处理添加项目（集成知识库获取真实门票价格，并同步日程）"""
        item = intent.details.get("item", "")
        
        if not item:
            item = self._extract_attraction_name(intent.description)

        if not self._is_valid_attraction_name(item):
            searched = (intent.details or {}).get("search_result") or {}
            for name in searched.get("attractions") or []:
                if name and name not in plan.get("attractions", []):
                    item = name
                    intent.details["item"] = item
                    intent.details["from_tool_search"] = True
                    break
        if not self._is_valid_attraction_name(item):
            # 无效名称：尝试知识库推荐一个
            city = plan.get("city", "")
            recs = self._recommend_attractions_from_kb(city, exclude=plan.get("attractions", []))
            if not recs:
                return plan
            item = recs[0]["name"]
            intent.details["item"] = item
            intent.details["auto_recommended"] = True
        
        if item and item not in plan.get("attractions", []):
            plan.setdefault("attractions", []).append(item)
            
            city = plan.get("city", "")
            ticket_price = 100
            kb_info = self._search_attraction_from_kb(item, city) or {}
            if kb_info:
                ticket_price = kb_info.get("ticket_price", 100)
                intent.details["kb_verified"] = True
                intent.details["kb_info"] = {
                    "name": kb_info.get("name", item),
                    "ticket_price": ticket_price,
                    "category": kb_info.get("category", ""),
                    "source": kb_info.get("source", "")
                }
            else:
                intent.details["kb_verified"] = False
            
            budget = plan.get("budget", {})
            if isinstance(budget, dict):
                current_attraction_cost = budget.get("total_attractions", 0)
                budget["total_attractions"] = current_attraction_cost + ticket_price
                budget["total"] = budget.get("total", 0) + ticket_price
                plan["budget"] = budget

            plan = self._sync_attractions_to_itinerary(
                plan,
                removed=[],
                added=[{
                    "name": item,
                    "ticket_price": ticket_price,
                    "category": kb_info.get("category", "景点") if kb_info else "景点",
                    "description": f"新增推荐：{item}",
                    "address": kb_info.get("address", "") if kb_info else "",
                    "location": kb_info.get("location", "") if kb_info else "",
                    "visit_duration": kb_info.get("visit_duration", 120) if kb_info else 120,
                }],
            )
        
        return plan
    
    def _handle_remove_item(self, plan: Dict[str, Any], intent: ChangeIntent) -> Dict[str, Any]:
        """处理移除项目（使用知识库价格精确扣减，并同步日程）"""
        item = intent.details.get("item", "") or intent.details.get("remove", "")
        
        if not item:
            item = self._match_attraction_in_plan(intent.description or "", plan.get("attractions", []))
        if not item:
            item = self._extract_attraction_name(intent.description)
        
        attractions = list(plan.get("attractions", []))
        matched = self._match_attraction_in_plan(item, attractions) if item else ""
        if matched:
            attractions.remove(matched)
            plan["attractions"] = attractions
            intent.details["item"] = matched
            intent.details["remove"] = matched
            
            city = plan.get("city", "")
            ticket_price = 100
            kb_info = self._search_attraction_from_kb(matched, city)
            if kb_info:
                ticket_price = kb_info.get("ticket_price", 100)
            
            budget = plan.get("budget", {})
            if isinstance(budget, dict):
                current_attraction_cost = budget.get("total_attractions", 0)
                budget["total_attractions"] = max(0, current_attraction_cost - ticket_price)
                budget["total"] = max(0, budget.get("total", 0) - ticket_price)
                plan["budget"] = budget

            plan = self._sync_attractions_to_itinerary(plan, removed=[matched], added=[])
        
        return plan
    
    def _handle_modify_budget(self, plan: Dict[str, Any], intent: ChangeIntent) -> Dict[str, Any]:
        """处理预算修改"""
        action = intent.details.get("action", "reduce")
        description = intent.description or ""

        # 关键词二次校验：确保 action 正确
        reduce_kws = ["便宜", "降低", "省钱", "太贵", "少点", "降", "减", "低", "省"]
        increase_kws = ["增加", "提高", "多点", "升", "增", "高", "多"]

        has_reduce_kw = any(kw in description for kw in reduce_kws)
        has_increase_kw = any(kw in description for kw in increase_kws)

        if has_reduce_kw and not has_increase_kw:
            action = "reduce"
        elif has_increase_kw and not has_reduce_kw:
            action = "increase"

        budget = plan.get("budget", {})

        if isinstance(budget, dict):
            current_total = budget.get("total", 0)

            if action == "reduce":
                # 降低酒店费用（主要调整项）
                hotel_cost = budget.get("total_hotels", 0)
                if hotel_cost > 0:
                    # 确保减少：至少降15%
                    new_hotel_cost = max(0, int(hotel_cost * 0.85))
                    # 如果原酒店费太高，直接减半
                    if hotel_cost > 500:
                        new_hotel_cost = min(new_hotel_cost, int(hotel_cost * 0.6))
                    reduction = hotel_cost - new_hotel_cost
                    budget["total_hotels"] = new_hotel_cost
                    budget["total"] = max(0, current_total - reduction)
                else:
                    # 酒店费用为0，降低餐饮
                    meal_cost = budget.get("total_meals", 0)
                    if meal_cost > 0:
                        new_meal_cost = max(0, int(meal_cost * 0.85))
                        budget["total_meals"] = new_meal_cost
                        budget["total"] = max(0, current_total - (meal_cost - new_meal_cost))
            else:
                # 增加预算
                budget["total"] = int(current_total * 1.2)

            plan["budget"] = budget

        return plan
    
    def _handle_modify_attractions(self, plan: Dict[str, Any], intent: ChangeIntent) -> Dict[str, Any]:
        """处理景点修改 / 替换：移除目标景点并推荐真实替代景点"""
        details = intent.details or {}
        action = details.get("action", "replace")
        city = plan.get("city", "")
        current = list(plan.get("attractions", []))
        text = f"{intent.description or ''} {details.get('item', '')} {details.get('remove', '')}"

        # 1) 定位要移除的景点（优先精确匹配当前计划）
        remove_name = details.get("remove") or details.get("old_item") or ""
        if not remove_name:
            remove_name = self._match_attraction_in_plan(text, current)
        if not remove_name:
            # 再尝试从用户原句提取后匹配
            remove_name = self._match_attraction_in_plan(
                self._extract_attraction_name(text), current
            )

        # 2) 定位要加入的景点（若用户已点名）
        add_name = details.get("item") or details.get("new_item") or ""
        if add_name and (
            add_name == remove_name
            or not self._is_valid_attraction_name(add_name)
            or add_name in ["其他景点", "别的景点", "其他", "别的"]
        ):
            add_name = ""

        # 用户没有点名时，用工具重新搜到的景点
        if not add_name:
            searched = (intent.details or {}).get("search_result") or {}
            for name in searched.get("attractions") or []:
                if name and name != remove_name and name not in current:
                    add_name = name
                    intent.details["from_tool_search"] = True
                    break

        # 用户只说“替换成其他”时，从知识库推荐
        if not add_name:
            exclude = list(current)
            recs = self._recommend_attractions_from_kb(city, exclude=exclude)
            # 若有 remove，确保推荐不是它
            for rec in recs:
                if rec["name"] != remove_name:
                    add_name = rec["name"]
                    intent.details["auto_recommended"] = True
                    intent.details["kb_info"] = rec
                    break

        if not remove_name and not add_name:
            # 无法理解，回退全量
            intent.scope = ModifyScope.FULL.value
            return self._regenerate_plan(plan, intent, intent.description or text)

        # 仅删除
        if remove_name and not add_name:
            intent.details["item"] = remove_name
            intent.details["action"] = "remove"
            return self._handle_remove_item(plan, intent)

        # 仅添加
        if add_name and not remove_name:
            intent.details["item"] = add_name
            intent.details["action"] = "add"
            return self._handle_add_item(plan, intent)

        # 替换：删旧 + 加新，并同步日程
        old_price = 100
        new_price = 100
        kb_old = self._search_attraction_from_kb(remove_name, city) if remove_name else None
        kb_new = self._search_attraction_from_kb(add_name, city) if add_name else None
        if kb_old:
            old_price = kb_old.get("ticket_price", 100)
        if kb_new:
            new_price = kb_new.get("ticket_price", 100)
            intent.details["kb_verified"] = True
            intent.details["kb_info"] = kb_new

        if remove_name in current:
            current.remove(remove_name)
        if add_name not in current:
            current.append(add_name)
        plan["attractions"] = current

        budget = plan.get("budget", {})
        if isinstance(budget, dict):
            budget["total_attractions"] = max(
                0, budget.get("total_attractions", 0) - old_price + new_price
            )
            budget["total"] = max(0, budget.get("total", 0) - old_price + new_price)
            plan["budget"] = budget

        intent.details["action"] = "replace"
        intent.details["remove"] = remove_name
        intent.details["item"] = add_name
        intent.details["replaced"] = f"{remove_name} → {add_name}"

        plan = self._sync_attractions_to_itinerary(
            plan,
            removed=[remove_name] if remove_name else [],
            added=[{
                "name": add_name,
                "ticket_price": new_price,
                "category": (kb_new or {}).get("category", "景点"),
                "description": f"替换「{remove_name}」为「{add_name}」",
                "address": (kb_new or {}).get("address", ""),
                "location": (kb_new or {}).get("location", ""),
                "visit_duration": (kb_new or {}).get("visit_duration", 120),
            }] if add_name else [],
            prefer_day_of=remove_name,
        )
        return plan

    def _match_attraction_in_plan(self, text: str, attractions: List[str]) -> str:
        """在当前计划景点中匹配用户提到的名称（支持子串）"""
        if not text or not attractions:
            return ""
        # 精确优先
        for name in attractions:
            if name and name in text:
                return name
        # 去空白后再比
        compact = re.sub(r"\s+", "", text)
        for name in attractions:
            if name and name in compact:
                return name
        return ""

    def _sync_attractions_to_itinerary(
        self,
        plan: Dict[str, Any],
        removed: List[str] = None,
        added: List[Any] = None,
        prefer_day_of: str = "",
    ) -> Dict[str, Any]:
        """
        将 flat attractions 变更同步到 daily_itinerary.days，
        保证前端右侧预览能看到真实行程变化。
        """
        removed = removed or []
        added = added or []
        daily = plan.get("daily_itinerary")
        if not isinstance(daily, dict):
            return plan

        days = daily.get("days")
        if not isinstance(days, list) or not days:
            return plan

        changed_days = set()
        target_day_idx = None
        for i, day in enumerate(days):
            attrs = day.get("attractions", [])
            if not isinstance(attrs, list):
                continue
            kept = []
            for a in attrs:
                name = a.get("name", "") if isinstance(a, dict) else str(a)
                if prefer_day_of and name == prefer_day_of:
                    target_day_idx = i
                if name not in removed:
                    kept.append(a)
                else:
                    changed_days.add(i)
            day["attractions"] = kept

        if target_day_idx is None:
            # 选景点最少的一天放入新增
            counts = [(i, len(d.get("attractions", []) or [])) for i, d in enumerate(days)]
            target_day_idx = min(counts, key=lambda x: x[1])[0] if counts else 0

        for item in added:
            if isinstance(item, str):
                attr_obj = {
                    "name": item,
                    "address": "",
                    "location": "",
                    "visit_duration": 120,
                    "description": f"推荐景点：{item}",
                    "category": "景点",
                    "ticket_price": 0,
                }
            elif isinstance(item, dict):
                attr_obj = {
                    "name": item.get("name", ""),
                    "address": item.get("address", ""),
                    "location": self._location_text(item),
                    "visit_duration": item.get("visit_duration", 120),
                    "description": item.get("description", ""),
                    "category": item.get("category", "景点"),
                    "ticket_price": item.get("ticket_price", 0),
                    "rating": item.get("rating"),
                }
            else:
                continue
            if not attr_obj["name"]:
                continue
            days[target_day_idx].setdefault("attractions", []).append(attr_obj)
            changed_days.add(target_day_idx)

        city = plan.get("city", "")
        transportation = plan.get("transportation") or "公共交通"
        hotel = self._hotel_anchor_from_plan(plan)
        used_names = set()
        for day in days:
            for item in day.get("attractions") or []:
                name = item.get("name", "") if isinstance(item, dict) else str(item)
                if name:
                    used_names.add(name)
        for index in sorted(changed_days):
            if hotel:
                days[index]["hotel"] = hotel
            self._replan_day_schedule(days[index], city, transportation, used_names)

        # 以日程为准重建 flat 列表，避免两边不一致
        flat = []
        for day in days:
            for a in day.get("attractions", []) or []:
                name = a.get("name", "") if isinstance(a, dict) else str(a)
                if name and name not in flat:
                    flat.append(name)
        plan["attractions"] = flat
        daily["days"] = days
        plan["daily_itinerary"] = daily
        return plan

    @staticmethod
    def _location_text(data: Dict[str, Any]) -> str:
        """把坐标收成高德要的「经度,纬度」。没有坐标就返回空串，改用地址或城市+店名。"""
        if not isinstance(data, dict):
            return ""
        location = data.get("location")
        if isinstance(location, str) and "," in location:
            return location.strip()
        if isinstance(location, dict):
            lon = location.get("longitude")
            lat = location.get("latitude")
        else:
            lon = data.get("longitude")
            lat = data.get("latitude")
        if lon in (None, "") or lat in (None, ""):
            return ""
        return f"{lon},{lat}"

    def _schedule_helper(self):
        """只借用规划节点的路程和时间表算法，不初始化整份规划图。"""
        helper = getattr(self, "_schedule_nodes", None)
        if helper is not None:
            return helper
        from app.agents.langgraph_nodes import GraphNodes
        from app.services.amap_service import get_amap_service

        helper = GraphNodes.__new__(GraphNodes)
        helper.amap_service = self.amap or get_amap_service()
        helper.logger = None
        helper.llm = getattr(self, "llm", None)
        helper.retriever = getattr(getattr(self, "rag", None), "retriever", None)
        helper._fill_agent = None
        self._schedule_nodes = helper
        return helper

    def _replan_day_schedule(
        self,
        day: Dict[str, Any],
        city: str,
        transportation: str,
        used: set = None,
    ) -> None:
        """按第一轮同一套规则重算当天路程、用餐和时间，空档补到 20:00，超过 20:00 可以。"""
        if not isinstance(day, dict):
            return
        helper = self._schedule_helper()
        fill = getattr(helper, "_fill_day_until_close", None)
        if fill is None:
            return
        fill(day, city, transportation or "公共交通", used if used is not None else set())

    @staticmethod
    def _hotel_anchor_from_plan(plan: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        hotel = plan.get("hotel")
        if isinstance(hotel, dict) and str(hotel.get("name") or "").strip():
            return hotel
        accommodation = plan.get("accommodation")
        if isinstance(accommodation, str) and accommodation.strip():
            return {"name": accommodation.strip()}
        return None

    def _replan_days_for_hotel(self, plan: Dict[str, Any]) -> None:
        """换了具体酒店后，各天从酒店出发、结束时返回酒店。"""
        hotel = self._hotel_anchor_from_plan(plan)
        if not hotel:
            return
        daily = plan.get("daily_itinerary")
        if not isinstance(daily, dict):
            return
        days = daily.get("days")
        if not isinstance(days, list):
            return
        city = plan.get("city") or ""
        transportation = plan.get("transportation") or "公共交通"
        used_names = set()
        for day in days:
            if not isinstance(day, dict):
                continue
            for item in day.get("attractions") or []:
                name = item.get("name", "") if isinstance(item, dict) else str(item)
                if name:
                    used_names.add(name)
        for day in days:
            if not isinstance(day, dict):
                continue
            day["hotel"] = hotel
            self._replan_day_schedule(day, city, transportation, used_names)
        flat = []
        for day in days:
            if not isinstance(day, dict):
                continue
            for item in day.get("attractions") or []:
                name = item.get("name", "") if isinstance(item, dict) else str(item)
                if name and name not in flat:
                    flat.append(name)
        if flat:
            plan["attractions"] = flat
        daily["days"] = days
        plan["daily_itinerary"] = daily
    
    def _handle_modify_duration(self, plan: Dict[str, Any], intent: ChangeIntent) -> Dict[str, Any]:
        """处理天数修改"""
        description = intent.description
        
        # 尝试提取目标天数
        numbers = re.findall(r'\d+', description)
        if numbers:
            new_days = int(numbers[0])
            old_days = plan.get("duration_days", 1)
            plan["duration_days"] = new_days
            
            # 更新预算
            budget = plan.get("budget", {})
            if isinstance(budget, dict):
                meal_per_day = 120
                transport_per_day = 50
                delta_days = new_days - old_days
                budget["total_meals"] = budget.get("total_meals", 0) + meal_per_day * delta_days
                budget["total_transportation"] = budget.get("total_transportation", 0) + transport_per_day * delta_days
                budget["total"] = (
                    budget.get("total_attractions", 0) +
                    budget.get("total_hotels", 0) +
                    budget.get("total_meals", 0) +
                    budget.get("total_transportation", 0)
                )
                plan["budget"] = budget
        
        return plan
    
    def _handle_modify_hotel(self, plan: Dict[str, Any], intent: ChangeIntent) -> Dict[str, Any]:
        """处理酒店修改（集成高德搜索获取真实酒店信息）"""
        action = intent.details.get("action", "reduce")
        description = intent.description or ""
        searched = (intent.details or {}).get("search_result") or {}
        hotel_name = (searched.get("hotel") or "").strip()
        hotel_price = searched.get("hotel_price")
        budget = plan.get("budget", {})
        days = int(plan.get("duration_days") or 1)
        nights = max(1, days - 1)

        if hotel_name:
            plan["accommodation"] = hotel_name
            plan["hotel"] = {
                "name": hotel_name,
                "price_per_night": hotel_price,
                "source": "hotel_search",
            }
            intent.details["hotel_from_tool"] = hotel_name

        if isinstance(budget, dict) and isinstance(hotel_price, (int, float)) and hotel_price > 0:
            budget["total_hotels"] = int(hotel_price) * nights
            budget["total"] = (
                budget.get("total_attractions", 0)
                + budget.get("total_hotels", 0)
                + budget.get("total_meals", 0)
                + budget.get("total_transportation", 0)
            )
            plan["budget"] = budget
            if hotel_name:
                self._replan_days_for_hotel(plan)
            return plan

        # 降价/提价关键词
        reduce_kws = ["便宜", "降低", "省钱", "太贵", "少点", "经济", "实惠", "低档", "经济型", "快捷", "降", "减", "低", "省"]
        increase_kws = ["增加", "提高", "多点", "高档", "豪华", "五星", "好一点", "升级", "更好", "升", "增", "高", "多"]

        # 从描述中判断修改方向（关键词检测优先级最高）
        has_reduce_kw = any(kw in description for kw in reduce_kws)
        has_increase_kw = any(kw in description for kw in increase_kws)

        if has_reduce_kw and not has_increase_kw:
            action = "reduce"
        elif has_increase_kw and not has_reduce_kw:
            action = "increase"
        elif action not in ["reduce", "increase"]:
            action = "reduce" if not has_increase_kw else "increase"

        budget = plan.get("budget", {})

        if isinstance(budget, dict):
            hotel_cost = budget.get("total_hotels", 0)
            days = plan.get("duration_days", 1)
            nights = max(1, days - 1)

            if action == "reduce":
                # 确保预算一定降低：至少减15%，或降到目标单价以下
                target_cost = nights * 200  # 经济型目标（约200/晚）
                if hotel_cost > 0:
                    new_hotel_cost = min(target_cost, int(hotel_cost * 0.7))
                    # 强制减少：如果以上计算未减少，强制降15%
                    if new_hotel_cost >= hotel_cost:
                        new_hotel_cost = int(hotel_cost * 0.85)
                else:
                    new_hotel_cost = target_cost
                budget["total_hotels"] = max(0, new_hotel_cost)
            else:
                # 提价
                target_cost = nights * 600
                new_hotel_cost = max(target_cost, int(hotel_cost * 1.3))
                budget["total_hotels"] = new_hotel_cost

            # 更新总预算
            budget["total"] = (
                budget.get("total_attractions", 0) +
                budget.get("total_hotels", 0) +
                budget.get("total_meals", 0) +
                budget.get("total_transportation", 0)
            )
            plan["budget"] = budget

        if hotel_name:
            self._replan_days_for_hotel(plan)
        return plan
    
    # ========== 全量重生成 ==========
    
    def _regenerate_plan(
        self,
        current_plan: Dict[str, Any],
        intent: ChangeIntent,
        user_input: str
    ) -> Optional[Dict[str, Any]]:
        """
        使用 LLM 全量重新生成计划（注入知识库上下文）
        """
        city = current_plan.get("city", "")
        
        # 从知识库获取城市景点信息作为参考
        kb_context = ""
        if self.rag:
            try:
                docs, _ = self.rag.retrieve_with_validation(
                    query=f"{city}景点推荐",
                    city=city,
                    preferences=[]
                )
                if docs:
                    kb_attractions = []
                    for doc in docs[:8]:
                        meta = doc.get("metadata", {})
                        title = meta.get("title", "")
                        price = meta.get("price", "未知")
                        category = meta.get("category", "")
                        if title:
                            kb_attractions.append(f"  - {title} (门票: {price}, 类型: {category})")
                    if kb_attractions:
                        kb_context = f"""
## 知识库可用景点（请从此列表中选择景点，不要编造不存在的景点）
{chr(10).join(kb_attractions)}
"""
            except Exception as e:
                print(f"知识库查询失败: {e}")
        
        prompt = f"""你是一个旅行计划修改助手。请根据用户的修改需求，调整以下旅行计划。

## 当前计划
{json.dumps(current_plan, ensure_ascii=False, indent=2)}
{kb_context}
## 用户修改需求
{user_input}

## 要求
1. 保留用户满意的部分，只修改用户提到的内容
2. 保持计划的结构不变（city, duration_days, attractions, budget, daily_itinerary 等字段）
3. 如果修改了景点或天数，相应调整预算
4. **重要：推荐景点时请优先使用知识库中列出的景点，不要编造不存在的景点**
5. 返回完整的修改后计划（JSON格式）

请返回修改后的完整JSON计划（只返回JSON，不要其他内容）："""
        
        try:
            parsed = invoke_structured(self.llm, DialogPlanUpdate, prompt)
            if isinstance(parsed, DialogPlanUpdate):
                new_plan = parsed.model_dump(exclude_defaults=True)
            else:
                new_plan = as_dict(parsed)
            if new_plan:
                merged = current_plan.copy()
                merged.update(new_plan)
                return merged
        except Exception as e:
            print(f"全量重生成失败: {e}")
        
        return None
    
    # ========== 差异计算 ==========
    
    def _compute_diff(
        self,
        old_plan: Dict[str, Any],
        new_plan: Dict[str, Any],
        old_version_id: str
    ) -> PlanDiff:
        """计算两个计划之间的差异"""
        diff = PlanDiff(old_version_id=old_version_id, new_version_id="")
        
        # 对比景点
        old_attractions = set(old_plan.get("attractions", []))
        new_attractions = set(new_plan.get("attractions", []))
        diff.added_attractions = list(new_attractions - old_attractions)
        diff.removed_attractions = list(old_attractions - new_attractions)
        
        if diff.added_attractions:
            diff.change_descriptions.append(f"新增景点: {', '.join(diff.added_attractions)}")
        if diff.removed_attractions:
            diff.change_descriptions.append(f"移除景点: {', '.join(diff.removed_attractions)}")
        # 一对一替换时给出更清晰说明
        if len(diff.added_attractions) == 1 and len(diff.removed_attractions) == 1:
            diff.change_descriptions.insert(
                0,
                f"景点替换: {diff.removed_attractions[0]} → {diff.added_attractions[0]}"
            )
        
        # 对比预算
        old_budget = self._extract_budget_total(old_plan.get("budget"))
        new_budget = self._extract_budget_total(new_plan.get("budget"))
        if old_budget is not None and new_budget is not None and old_budget != new_budget:
            diff.budget_change = {
                "old": old_budget,
                "new": new_budget,
                "delta": new_budget - old_budget
            }
            direction = "增加" if new_budget > old_budget else "节省"
            diff.change_descriptions.append(
                f"预算变化: {old_budget}元 → {new_budget}元 ({direction}{abs(new_budget - old_budget)}元)"
            )
        
        # 对比天数
        old_days = old_plan.get("duration_days")
        new_days = new_plan.get("duration_days")
        if old_days is not None and new_days is not None and old_days != new_days:
            diff.duration_change = {
                "old": old_days,
                "new": new_days,
                "delta": new_days - old_days
            }
            diff.change_descriptions.append(f"行程天数: {old_days}天 → {new_days}天")

        old_stay = old_plan.get("accommodation") or ""
        new_stay = new_plan.get("accommodation") or ""
        if old_stay != new_stay and new_stay:
            diff.hotel_change = {"old": old_stay, "new": new_stay}
            diff.change_descriptions.append(f"住宿: {old_stay or '未指定'} → {new_stay}")
        
        # 生成摘要
        if diff.has_changes():
            diff.summary = "；".join(diff.change_descriptions[:3])
            if len(diff.change_descriptions) > 3:
                diff.summary += f"等{len(diff.change_descriptions)}项变更"
        else:
            diff.summary = "无变化"
        
        return diff
    
    # ========== 回复生成 ==========
    
    def _generate_reply(self, intent: ChangeIntent, diff: PlanDiff, new_plan: Dict[str, Any]) -> str:
        """生成给用户的回复"""
        lines = []
        
        # 替换场景优先给出明确说明
        details = intent.details or {}
        if details.get("action") == "replace" and details.get("remove") and details.get("item"):
            lines.append(
                f"好的，已将「{details['remove']}」替换为「{details['item']}」。"
            )
            if details.get("auto_recommended"):
                lines.append("（该景点由知识库根据当前行程自动推荐）")
            lines.append("")
        else:
            lines.append("好的，我已根据您的需求进行了调整。")
            lines.append("")
        
        # 列出变更
        if diff.change_descriptions:
            lines.append("📋 变更明细：")
            for desc in diff.change_descriptions:
                lines.append(f"  • {desc}")
            lines.append("")
        
        # 当前计划概要
        lines.append("📌 当前计划概要：")
        lines.append(f"  • 城市: {new_plan.get('city', '未知')}")
        lines.append(f"  • 天数: {new_plan.get('duration_days', 0)}天")
        
        attractions = new_plan.get("attractions", [])
        if attractions:
            lines.append(f"  • 景点: {', '.join(attractions)}")
        
        budget = new_plan.get("budget", {})
        if isinstance(budget, dict) and budget.get("total"):
            lines.append(f"  • 预算: {budget['total']}元")
        
        lines.append("")
        lines.append("如需进一步调整，请告诉我。")
        
        return "\n".join(lines)
    
    def _generate_no_change_reply(self, user_input: str, current_plan: Dict[str, Any]) -> str:
        """生成无需修改时的回复"""
        return f"""我理解您的需求，但当前计划似乎已经满足了这个要求。

当前计划：
  • 城市: {current_plan.get('city', '未知')}
  • 天数: {current_plan.get('duration_days', 0)}天
  • 景点: {', '.join(current_plan.get('attractions', []))}

请告诉我您具体想修改什么，比如"添加迪士尼"、"降低酒店预算"、"延长到5天"等。"""
    
    # ========== 辅助方法 ==========
    
    def _extract_attraction_name(self, text: str) -> str:
        """从文本中提取景点名称"""
        if not text:
            return ""
        # 去除常见动词和修饰词
        text = re.sub(
            r'(帮我|请|我想|想要|能不能|可以|加上|添加|增加|加入|去掉|删除|不要|取消|换掉|去除|移除|替换|换成|其他景点|别的景点|其他|别的|去|玩|逛|用户希望)',
            '',
            text
        )
        text = text.strip()
        
        # 去除标点符号
        text = re.sub(r'[，。！？、\s]+$', '', text)
        text = re.sub(r'^[，。！？、\s]+', '', text)
        
        return text if text else ""
    
    def _extract_budget_total(self, budget: Any) -> Optional[float]:
        """从预算字段提取总金额"""
        if budget is None:
            return None
        if isinstance(budget, (int, float)):
            return float(budget)
        if isinstance(budget, dict):
            total = budget.get("total", budget.get("total_budget", 0))
            if total:
                return float(total)
            calculated = (
                budget.get("total_attractions", 0) +
                budget.get("total_hotels", 0) +
                budget.get("total_meals", 0) +
                budget.get("total_transportation", 0)
            )
            if calculated > 0:
                return float(calculated)
        return None
    
    def _parse_json_from_response(self, content: str) -> Optional[Dict[str, Any]]:
        """从 LLM 响应中解析 JSON"""
        try:
            if "```json" in content:
                start = content.find("```json") + 7
                end = content.find("```", start)
                json_str = content[start:end].strip()
            elif "```" in content:
                start = content.find("```") + 3
                end = content.find("```", start)
                if end != -1:
                    json_str = content[start:end].strip()
                else:
                    json_str = content
            elif "{" in content and "}" in content:
                start = content.find("{")
                end = content.rfind("}") + 1
                json_str = content[start:end]
            else:
                json_str = content
            
            return json.loads(json_str)
        except Exception:
            return None
