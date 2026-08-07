"""
对话 Agent 数据模型

定义多轮对话修改旅行计划所需的数据结构：
- PlanVersion: 计划版本快照，支持版本追溯
- DialogSession: 对话会话，管理版本历史和对话记录
- ChangeIntent: 用户修改意图，用于指导增量修改
- PlanDiff: 计划差异，记录版本间的变化

设计原则：
1. 与现有 TravelPlan 模型兼容，复用其字段结构
2. 支持版本链追溯（parent_version_id）
3. 支持增量修改（ChangeIntent）和差异对比（PlanDiff）
"""

import json
import uuid
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any, Union
from enum import Enum


class ChangeType(Enum):
    """修改类型枚举"""
    MODIFY_BUDGET = "modify_budget"          # 调整预算
    MODIFY_ATTRACTIONS = "modify_attractions" # 修改景点
    MODIFY_HOTEL = "modify_hotel"            # 修改酒店
    MODIFY_DURATION = "modify_duration"      # 修改天数
    MODIFY_TRANSPORT = "modify_transport"    # 修改交通
    ADD_ITEM = "add_item"                    # 添加项目
    REMOVE_ITEM = "remove_item"              # 移除项目
    MAJOR_CHANGE = "major_change"            # 重大变更（需重新生成）
    NO_CHANGE = "no_change"                  # 无需修改


class ModifyScope(Enum):
    """修改范围枚举"""
    INCREMENTAL = "incremental"  # 增量修改（局部调整）
    FULL = "full"                # 全量重新生成


@dataclass
class ChangeIntent:
    """
    用户修改意图
    
    由对话 Agent 分析用户输入后生成，指导后续的修改操作。
    """
    intent_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    change_type: str = ChangeType.NO_CHANGE.value  # ChangeType 枚举值
    scope: str = ModifyScope.INCREMENTAL.value     # ModifyScope 枚举值
    description: str = ""                           # 修改描述（自然语言）
    target_field: str = ""                          # 目标字段（如 "budget", "attractions"）
    details: Dict[str, Any] = field(default_factory=dict)  # 具体修改细节
    confidence: float = 1.0                         # 意图识别置信度 (0-1)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChangeIntent":
        return cls(
            intent_id=data.get("intent_id", uuid.uuid4().hex[:8]),
            change_type=data.get("change_type", ChangeType.NO_CHANGE.value),
            scope=data.get("scope", ModifyScope.INCREMENTAL.value),
            description=data.get("description", ""),
            target_field=data.get("target_field", ""),
            details=data.get("details", {}),
            confidence=data.get("confidence", 1.0)
        )


@dataclass
class PlanDiff:
    """
    计划差异记录
    
    记录两个版本之间的具体变化，用于展示给用户。
    """
    diff_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    old_version_id: str = ""
    new_version_id: str = ""
    
    # 变化摘要
    summary: str = ""  # 一句话描述变化
    
    # 具体变化
    added_attractions: List[str] = field(default_factory=list)
    removed_attractions: List[str] = field(default_factory=list)
    budget_change: Optional[Dict[str, float]] = None  # {"old": 1000, "new": 1500, "delta": 500}
    duration_change: Optional[Dict[str, int]] = None   # {"old": 3, "new": 5, "delta": 2}
    hotel_change: Optional[Dict[str, Any]] = None      # {"old": "...", "new": "..."}
    transport_change: Optional[Dict[str, Any]] = None
    itinerary_changes: List[Dict[str, Any]] = field(default_factory=list)  # 日程变化列表
    
    # 完整变化描述（人类可读）
    change_descriptions: List[str] = field(default_factory=list)
    
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PlanDiff":
        return cls(
            diff_id=data.get("diff_id", uuid.uuid4().hex[:8]),
            old_version_id=data.get("old_version_id", ""),
            new_version_id=data.get("new_version_id", ""),
            summary=data.get("summary", ""),
            added_attractions=data.get("added_attractions", []),
            removed_attractions=data.get("removed_attractions", []),
            budget_change=data.get("budget_change"),
            duration_change=data.get("duration_change"),
            hotel_change=data.get("hotel_change"),
            transport_change=data.get("transport_change"),
            itinerary_changes=data.get("itinerary_changes", []),
            change_descriptions=data.get("change_descriptions", []),
            created_at=data.get("created_at", datetime.now().isoformat())
        )
    
    def add_change(self, description: str):
        """添加一条变化描述"""
        self.change_descriptions.append(description)
    
    def has_changes(self) -> bool:
        """是否有实际变化"""
        return (
            len(self.added_attractions) > 0 or
            len(self.removed_attractions) > 0 or
            self.budget_change is not None or
            self.duration_change is not None or
            self.hotel_change is not None or
            self.transport_change is not None or
            len(self.itinerary_changes) > 0
        )


@dataclass
class PlanVersion:
    """
    计划版本快照
    
    每次生成或修改计划时创建一个版本快照，支持版本追溯和回退。
    版本通过 parent_version_id 形成链式结构。
    """
    version_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    plan_id: str = ""                              # 关联的旅行计划ID
    parent_version_id: Optional[str] = None        # 父版本ID（用于版本链追溯）
    version_number: int = 1                        # 版本号（递增）
    
    # 计划数据（完整的 TravelPlan 快照）
    plan_data: Dict[str, Any] = field(default_factory=dict)
    
    # 版本信息
    changes: List[str] = field(default_factory=list)  # 相比父版本的变化描述
    change_intent: Optional[Dict[str, Any]] = None    # 触发此版本的修改意图
    diff: Optional[Dict[str, Any]] = None             # 与父版本的差异（PlanDiff）
    
    # 元数据
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    created_by: str = "planner"  # "planner" (规划Agent) 或 "dialog" (对话Agent)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PlanVersion":
        return cls(
            version_id=data.get("version_id", uuid.uuid4().hex[:8]),
            plan_id=data.get("plan_id", ""),
            parent_version_id=data.get("parent_version_id"),
            version_number=data.get("version_number", 1),
            plan_data=data.get("plan_data", {}),
            changes=data.get("changes", []),
            change_intent=data.get("change_intent"),
            diff=data.get("diff"),
            created_at=data.get("created_at", datetime.now().isoformat()),
            created_by=data.get("created_by", "planner")
        )
    
    def is_initial_version(self) -> bool:
        """是否为初始版本"""
        return self.parent_version_id is None
    
    def get_plan_field(self, field_name: str, default: Any = None) -> Any:
        """从 plan_data 中获取字段值"""
        return self.plan_data.get(field_name, default)


@dataclass
class DialogMessage:
    """
    对话消息
    
    记录用户与对话Agent之间的每一条消息。
    """
    message_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    role: str = "user"  # "user" 或 "assistant"
    content: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # 关联信息
    version_id: Optional[str] = None  # 关联的计划版本ID
    intent: Optional[Dict[str, Any]] = None  # 关联的修改意图
    diff: Optional[Dict[str, Any]] = None  # 关联的差异
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DialogMessage":
        return cls(
            message_id=data.get("message_id", uuid.uuid4().hex[:8]),
            role=data.get("role", "user"),
            content=data.get("content", ""),
            timestamp=data.get("timestamp", datetime.now().isoformat()),
            version_id=data.get("version_id"),
            intent=data.get("intent"),
            diff=data.get("diff")
        )


@dataclass
class DialogSession:
    """
    对话会话
    
    管理一次完整的对话修改流程，包含版本历史和对话记录。
    一个会话对应一个旅行计划的多次修改。
    """
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    user_id: str = ""
    plan_id: str = ""                               # 关联的旅行计划ID
    current_version_id: str = ""                    # 当前版本ID
    version_history: List[str] = field(default_factory=list)  # 所有版本ID（按时间顺序）
    
    # 对话历史
    conversation_log: List[Dict[str, Any]] = field(default_factory=list)  # DialogMessage 列表
    
    # 会话状态
    is_active: bool = True                          # 会话是否活跃
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # 上下文摘要（用于长会话压缩）
    summary: Optional[str] = None  # 对话历史的LLM摘要
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DialogSession":
        return cls(
            session_id=data.get("session_id", uuid.uuid4().hex[:12]),
            user_id=data.get("user_id", ""),
            plan_id=data.get("plan_id", ""),
            current_version_id=data.get("current_version_id", ""),
            version_history=data.get("version_history", []),
            conversation_log=data.get("conversation_log", []),
            is_active=data.get("is_active", True),
            created_at=data.get("created_at", datetime.now().isoformat()),
            updated_at=data.get("updated_at", datetime.now().isoformat()),
            summary=data.get("summary")
        )
    
    def add_message(self, message: DialogMessage):
        """添加对话消息"""
        self.conversation_log.append(message.to_dict())
        self.updated_at = datetime.now().isoformat()
    
    def add_version(self, version: PlanVersion):
        """添加新版本"""
        self.version_history.append(version.version_id)
        self.current_version_id = version.version_id
        self.updated_at = datetime.now().isoformat()
    
    def get_message_count(self) -> int:
        """获取消息数量"""
        return len(self.conversation_log)
    
    def get_version_count(self) -> int:
        """获取版本数量"""
        return len(self.version_history)
    
    def get_recent_messages(self, n: int = 10) -> List[Dict[str, Any]]:
        """获取最近的N条消息"""
        return self.conversation_log[-n:] if n > 0 else []
    
    def get_version_chain(self) -> List[str]:
        """获取版本链（从初始到当前）"""
        return self.version_history.copy()
    
    def can_rollback(self) -> bool:
        """是否可以回退"""
        return len(self.version_history) > 1
    
    def close(self):
        """关闭会话"""
        self.is_active = False
        self.updated_at = datetime.now().isoformat()
