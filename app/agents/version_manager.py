"""
版本管理器

负责旅行计划版本的创建、存储、查询、对比和回退。
是对话Agent的核心基础设施，支持多轮修改中的版本控制。

核心职责：
1. 版本创建：为每次生成的计划创建版本快照
2. 版本存储：持久化版本数据（Redis 或文件）
3. 版本查询：按版本ID、计划ID查询版本历史
4. 版本对比：计算两个版本间的差异
5. 版本回退：回退到历史版本
6. 会话管理：管理对话会话的生命周期

存储结构（Redis）：
    travel:session:{session_id}                    → DialogSession JSON
    travel:session:{session_id}:versions           → 版本ID列表 (List)
    travel:version:{version_id}                    → PlanVersion JSON
    travel:user:{user_id}:sessions                 → 会话ID列表 (List)

存储结构（文件）：
    app/data/dialog_sessions/{user_id}/{session_id}.json  → DialogSession JSON
    app/data/dialog_sessions/{user_id}/versions/{version_id}.json → PlanVersion JSON
"""

import json
import uuid
import os
from datetime import datetime
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import asdict

from app.agents.dialog_models import (
    PlanVersion, DialogSession, DialogMessage,
    ChangeIntent, PlanDiff,
    ChangeType, ModifyScope
)


class VersionManager:
    """
    版本管理器
    
    管理旅行计划的版本链和对话会话，支持多轮修改中的版本控制。
    """
    
    def __init__(self, use_redis: bool = False, redis_host: str = None,
                 redis_port: int = None, redis_password: str = None):
        """
        初始化版本管理器
        
        Args:
            use_redis: 是否使用Redis存储
            redis_host: Redis主机
            redis_port: Redis端口
            redis_password: Redis密码
        """
        self.use_redis = use_redis
        
        if use_redis:
            try:
                import redis
                from redis import Redis
                from dotenv import load_dotenv
                load_dotenv()
                
                host = redis_host or os.getenv("REDIS_HOST", "localhost")
                port = redis_port or int(os.getenv("REDIS_PORT", 6379))
                password = redis_password or os.getenv("REDIS_PASSWORD", "")
                
                if password:
                    self.client = Redis(
                        host=host, port=port, password=password,
                        decode_responses=True, socket_timeout=10
                    )
                else:
                    self.client = Redis(
                        host=host, port=port,
                        decode_responses=True, socket_timeout=10
                    )
                self.client.ping()
            except Exception as e:
                print(f"Redis连接失败，回退到文件存储: {e}")
                self.use_redis = False
        
        if not self.use_redis:
            self.storage_path = "app/data/dialog_sessions"
            os.makedirs(self.storage_path, exist_ok=True)
    
    # ========== 存储键/路径生成 ==========
    
    def _get_session_key(self, session_id: str) -> str:
        return f"travel:session:{session_id}"
    
    def _get_session_versions_key(self, session_id: str) -> str:
        return f"travel:session:{session_id}:versions"
    
    def _get_version_key(self, version_id: str) -> str:
        return f"travel:version:{version_id}"
    
    def _get_user_sessions_key(self, user_id: str) -> str:
        return f"travel:user:{user_id}:sessions"
    
    def _get_session_file(self, user_id: str, session_id: str) -> str:
        user_folder = os.path.join(self.storage_path, user_id)
        os.makedirs(user_folder, exist_ok=True)
        return os.path.join(user_folder, f"{session_id}.json")
    
    def _get_version_file(self, version_id: str) -> str:
        version_folder = os.path.join(self.storage_path, "_versions")
        os.makedirs(version_folder, exist_ok=True)
        return os.path.join(version_folder, f"{version_id}.json")
    
    # ========== 会话管理 ==========
    
    def create_session(self, user_id: str, plan_id: str = "") -> DialogSession:
        """
        创建新的对话会话
        
        Args:
            user_id: 用户ID
            plan_id: 关联的旅行计划ID
            
        Returns:
            新创建的 DialogSession
        """
        session = DialogSession(user_id=user_id, plan_id=plan_id)
        self._save_session(session)
        
        # 将会话ID添加到用户的会话列表
        if self.use_redis:
            self.client.rpush(self._get_user_sessions_key(user_id), session.session_id)
        
        return session
    
    def get_session(self, session_id: str) -> Optional[DialogSession]:
        """
        获取会话信息
        
        Args:
            session_id: 会话ID
            
        Returns:
            DialogSession 或 None
        """
        if self.use_redis:
            data = self.client.get(self._get_session_key(session_id))
            if data:
                return DialogSession.from_dict(json.loads(data))
        else:
            # 文件存储：需要搜索所有用户文件夹
            for user_folder in os.listdir(self.storage_path):
                user_path = os.path.join(self.storage_path, user_folder)
                if os.path.isdir(user_path):
                    session_file = os.path.join(user_path, f"{session_id}.json")
                    if os.path.exists(session_file):
                        with open(session_file, 'r', encoding='utf-8') as f:
                            return DialogSession.from_dict(json.load(f))
        
        return None
    
    def _save_session(self, session: DialogSession):
        """保存会话到存储"""
        session.updated_at = datetime.now().isoformat()
        data = json.dumps(session.to_dict(), ensure_ascii=False)
        
        if self.use_redis:
            self.client.set(self._get_session_key(session.session_id), data)
        else:
            session_file = self._get_session_file(session.user_id, session.session_id)
            with open(session_file, 'w', encoding='utf-8') as f:
                f.write(data)
    
    def close_session(self, session_id: str) -> bool:
        """
        关闭会话
        
        Args:
            session_id: 会话ID
            
        Returns:
            是否成功关闭
        """
        session = self.get_session(session_id)
        if session:
            session.close()
            self._save_session(session)
            return True
        return False
    
    def get_user_sessions(self, user_id: str) -> List[Dict[str, Any]]:
        """
        获取用户的所有会话摘要
        
        Args:
            user_id: 用户ID
            
        Returns:
            会话摘要列表
        """
        sessions = []
        
        if self.use_redis:
            session_ids = self.client.lrange(self._get_user_sessions_key(user_id), 0, -1)
            for sid in session_ids:
                session = self.get_session(sid)
                if session:
                    sessions.append({
                        "session_id": session.session_id,
                        "plan_id": session.plan_id,
                        "version_count": session.get_version_count(),
                        "message_count": session.get_message_count(),
                        "is_active": session.is_active,
                        "created_at": session.created_at,
                        "updated_at": session.updated_at
                    })
        else:
            user_folder = os.path.join(self.storage_path, user_id)
            if os.path.exists(user_folder):
                for filename in os.listdir(user_folder):
                    if filename.endswith('.json'):
                        filepath = os.path.join(user_folder, filename)
                        with open(filepath, 'r', encoding='utf-8') as f:
                            session = DialogSession.from_dict(json.load(f))
                        sessions.append({
                            "session_id": session.session_id,
                            "plan_id": session.plan_id,
                            "version_count": session.get_version_count(),
                            "message_count": session.get_message_count(),
                            "is_active": session.is_active,
                            "created_at": session.created_at,
                            "updated_at": session.updated_at
                        })
        
        return sessions
    
    # ========== 版本管理 ==========
    
    def create_initial_version(self, session_id: str, plan_data: Dict[str, Any]) -> PlanVersion:
        """
        创建初始版本（由规划Agent生成计划后调用）
        
        Args:
            session_id: 会话ID
            plan_data: 完整的计划数据
            
        Returns:
            新创建的 PlanVersion
        """
        session = self.get_session(session_id)
        if not session:
            raise ValueError(f"会话不存在: {session_id}")
        
        version = PlanVersion(
            plan_id=session.plan_id,
            version_number=1,
            plan_data=plan_data,
            created_by="planner"
        )
        
        self._save_version(version)
        session.add_version(version)
        self._save_session(session)
        
        return version
    
    def create_modified_version(
        self,
        session_id: str,
        plan_data: Dict[str, Any],
        change_intent: ChangeIntent,
        diff: PlanDiff,
        changes: List[str] = None
    ) -> PlanVersion:
        """
        创建修改版本（由对话Agent修改计划后调用）
        
        Args:
            session_id: 会话ID
            plan_data: 修改后的完整计划数据
            change_intent: 修改意图
            diff: 与父版本的差异
            changes: 变更描述列表
            
        Returns:
            新创建的 PlanVersion
        """
        session = self.get_session(session_id)
        if not session:
            raise ValueError(f"会话不存在: {session_id}")
        
        current_version = self.get_version(session.current_version_id)
        if not current_version:
            raise ValueError(f"当前版本不存在: {session.current_version_id}")
        
        new_version = PlanVersion(
            plan_id=session.plan_id,
            parent_version_id=current_version.version_id,
            version_number=current_version.version_number + 1,
            plan_data=plan_data,
            changes=changes or diff.change_descriptions,
            change_intent=change_intent.to_dict(),
            diff=diff.to_dict(),
            created_by="dialog"
        )
        
        self._save_version(new_version)
        session.add_version(new_version)
        self._save_session(session)
        
        return new_version
    
    def _save_version(self, version: PlanVersion):
        """保存版本到存储"""
        data = json.dumps(version.to_dict(), ensure_ascii=False)
        
        if self.use_redis:
            self.client.set(self._get_version_key(version.version_id), data)
        else:
            version_file = self._get_version_file(version.version_id)
            with open(version_file, 'w', encoding='utf-8') as f:
                f.write(data)
    
    def get_version(self, version_id: str) -> Optional[PlanVersion]:
        """
        获取版本信息
        
        Args:
            version_id: 版本ID
            
        Returns:
            PlanVersion 或 None
        """
        if self.use_redis:
            data = self.client.get(self._get_version_key(version_id))
            if data:
                return PlanVersion.from_dict(json.loads(data))
        else:
            version_file = self._get_version_file(version_id)
            if os.path.exists(version_file):
                with open(version_file, 'r', encoding='utf-8') as f:
                    return PlanVersion.from_dict(json.load(f))
        
        return None
    
    def get_current_version(self, session_id: str) -> Optional[PlanVersion]:
        """
        获取会话的当前版本
        
        Args:
            session_id: 会话ID
            
        Returns:
            当前 PlanVersion 或 None
        """
        session = self.get_session(session_id)
        if not session or not session.current_version_id:
            return None
        return self.get_version(session.current_version_id)
    
    def get_version_history(self, session_id: str) -> List[PlanVersion]:
        """
        获取会话的完整版本历史
        
        Args:
            session_id: 会话ID
            
        Returns:
            版本列表（按版本号升序）
        """
        session = self.get_session(session_id)
        if not session:
            return []
        
        versions = []
        for vid in session.version_history:
            version = self.get_version(vid)
            if version:
                versions.append(version)
        
        return sorted(versions, key=lambda v: v.version_number)
    
    def rollback_to_version(self, session_id: str, target_version_id: str) -> Optional[PlanVersion]:
        """
        回退到指定版本
        
        将目标版本设为当前版本，但保留版本历史。
        回退后的新版本号继续递增。
        
        Args:
            session_id: 会话ID
            target_version_id: 目标版本ID
            
        Returns:
            回退后的 PlanVersion 或 None
        """
        session = self.get_session(session_id)
        if not session:
            return None
        
        target_version = self.get_version(target_version_id)
        if not target_version:
            return None
        
        # 创建一个基于目标版本的新版本（而非删除后续版本）
        current_version = self.get_version(session.current_version_id)
        new_version_number = (current_version.version_number + 1) if current_version else 1
        
        rollback_version = PlanVersion(
            plan_id=session.plan_id,
            parent_version_id=current_version.version_id if current_version else None,
            version_number=new_version_number,
            plan_data=target_version.plan_data.copy(),
            changes=[f"回退到版本 {target_version.version_number} 的内容"],
            created_by="dialog"
        )
        
        self._save_version(rollback_version)
        session.add_version(rollback_version)
        self._save_session(session)
        
        return rollback_version
    
    # ========== 版本对比 ==========
    
    def compare_versions(
        self,
        old_version_id: str,
        new_version_id: str
    ) -> Optional[PlanDiff]:
        """
        对比两个版本的差异
        
        Args:
            old_version_id: 旧版本ID
            new_version_id: 新版本ID
            
        Returns:
            PlanDiff 差异对象 或 None
        """
        old_version = self.get_version(old_version_id)
        new_version = self.get_version(new_version_id)
        
        if not old_version or not new_version:
            return None
        
        old_plan = old_version.plan_data
        new_plan = new_version.plan_data
        
        diff = PlanDiff(
            old_version_id=old_version_id,
            new_version_id=new_version_id
        )
        
        # 对比景点
        old_attractions = set(old_plan.get("attractions", []))
        new_attractions = set(new_plan.get("attractions", []))
        diff.added_attractions = list(new_attractions - old_attractions)
        diff.removed_attractions = list(old_attractions - new_attractions)
        
        if diff.added_attractions:
            diff.change_descriptions.append(f"新增景点: {', '.join(diff.added_attractions)}")
        if diff.removed_attractions:
            diff.change_descriptions.append(f"移除景点: {', '.join(diff.removed_attractions)}")
        
        # 对比预算
        old_budget = self._extract_budget_total(old_plan.get("budget"))
        new_budget = self._extract_budget_total(new_plan.get("budget"))
        if old_budget is not None and new_budget is not None and old_budget != new_budget:
            diff.budget_change = {
                "old": old_budget,
                "new": new_budget,
                "delta": new_budget - old_budget
            }
            diff.change_descriptions.append(
                f"预算变化: {old_budget}元 → {new_budget}元 "
                f"({'增加' if new_budget > old_budget else '节省'}{abs(new_budget - old_budget)}元)"
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
            diff.change_descriptions.append(
                f"行程天数: {old_days}天 → {new_days}天"
            )
        
        # 对比城市
        old_city = old_plan.get("city", "")
        new_city = new_plan.get("city", "")
        if old_city != new_city:
            diff.change_descriptions.append(f"城市变更: {old_city} → {new_city}")
        
        # 生成摘要
        if diff.has_changes():
            diff.summary = "；".join(diff.change_descriptions[:3])
            if len(diff.change_descriptions) > 3:
                diff.summary += f"等{len(diff.change_descriptions)}项变更"
        else:
            diff.summary = "无变化"
        
        return diff
    
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
            # 尝试求和
            calculated = (
                budget.get("total_attractions", 0) +
                budget.get("total_hotels", 0) +
                budget.get("total_meals", 0) +
                budget.get("total_transportation", 0)
            )
            if calculated > 0:
                return float(calculated)
        return None
    
    # ========== 消息管理 ==========
    
    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        version_id: str = None,
        intent: ChangeIntent = None,
        diff: PlanDiff = None
    ) -> DialogMessage:
        """
        添加对话消息到会话
        
        Args:
            session_id: 会话ID
            role: 角色 ("user" 或 "assistant")
            content: 消息内容
            version_id: 关联的版本ID
            intent: 关联的修改意图
            diff: 关联的差异
            
        Returns:
            创建的 DialogMessage
        """
        session = self.get_session(session_id)
        if not session:
            raise ValueError(f"会话不存在: {session_id}")
        
        message = DialogMessage(
            role=role,
            content=content,
            version_id=version_id,
            intent=intent.to_dict() if intent else None,
            diff=diff.to_dict() if diff else None
        )
        
        session.add_message(message)
        self._save_session(session)
        
        return message
    
    def get_conversation_history(
        self,
        session_id: str,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        获取对话历史
        
        Args:
            session_id: 会话ID
            limit: 返回的最大消息数
            
        Returns:
            消息列表
        """
        session = self.get_session(session_id)
        if not session:
            return []
        return session.get_recent_messages(limit)
    
    # ========== 查询辅助 ==========
    
    def get_version_chain(self, session_id: str) -> List[Dict[str, Any]]:
        """
        获取版本链信息（用于展示版本树）
        
        Args:
            session_id: 会话ID
            
        Returns:
            版本链信息列表
        """
        versions = self.get_version_history(session_id)
        return [
            {
                "version_id": v.version_id,
                "version_number": v.version_number,
                "parent_version_id": v.parent_version_id,
                "created_at": v.created_at,
                "created_by": v.created_by,
                "changes": v.changes,
                "is_current": v.version_id == self.get_session(session_id).current_version_id
            }
            for v in versions
        ]
    
    def get_session_summary(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        获取会话摘要信息
        
        Args:
            session_id: 会话ID
            
        Returns:
            会话摘要字典
        """
        session = self.get_session(session_id)
        if not session:
            return None
        
        current_version = self.get_version(session.current_version_id) if session.current_version_id else None
        
        return {
            "session_id": session.session_id,
            "user_id": session.user_id,
            "plan_id": session.plan_id,
            "is_active": session.is_active,
            "version_count": session.get_version_count(),
            "message_count": session.get_message_count(),
            "current_version_id": session.current_version_id,
            "current_plan": current_version.plan_data if current_version else None,
            "created_at": session.created_at,
            "updated_at": session.updated_at
        }
