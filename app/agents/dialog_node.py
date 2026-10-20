"""计划生成后的版本保存和对话暂停。"""

from typing import Any, Dict

class DialogMixin:
    """版本与对话。"""

    def save_version(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        保存初始版本到VersionManager

        在规划完成后，创建第一个版本快照，用于后续对话修改的版本追溯。
        """
        from app.agents.version_manager import VersionManager
        from app.agents.dialog_models import DialogSession

        user_id = state.get("user_id", "")
        city = state.get("city", "")
        parsed = state.get("trip_plan_parsed")
        session_id = state.get("session_id", "")

        if not user_id or not parsed:
            return {"current_phase": "final"}

        try:
            # 初始化版本管理器
            version_manager = VersionManager(use_redis=True)

            # 如果没有session_id，创建新会话
            if not session_id:
                session = version_manager.create_session(user_id=user_id, plan_id="")
                session_id = session.session_id
            else:
                session = version_manager.get_session(session_id)
                if not session:
                    session = version_manager.create_session(user_id=user_id, plan_id="")

            # 构建计划数据
            attractions = []
            for day in parsed.get("days", []):
                for attr in day.get("attractions", []):
                    if isinstance(attr, dict):
                        attractions.append(attr.get("name", ""))

            plan_data = {
                "city": city,
                "duration_days": state.get("travel_days", 1),
                "attractions": attractions,
                "budget": state.get("budget", {}),
                "daily_itinerary": parsed,
                "transportation": state.get("transportation", ""),
                "accommodation": state.get("accommodation", ""),
            }

            # 创建初始版本
            initial_version = version_manager.create_initial_version(
                session_id=session_id,
                plan_data=plan_data
            )

            if self.logger:
                self.logger.log_info("VERSION", msg=(
                    f"[版本节点] 初始版本已保存: v{initial_version.version_number}, "
                    f"session_id={session_id}"
                ))

            return {
                "session_id": session_id,
                "current_version_id": initial_version.version_id,
                "current_phase": "save_version"
            }

        except Exception as e:
            if self.logger:
                self.logger.log_error("VERSION", msg=f"[版本节点] 保存失败: {e}")
            return {"current_phase": "final"}

    # ==================== 对话Agent节点 ====================

    def _dialog_welcome(self, state: Dict[str, Any]) -> str:
        city = state.get("city", "")
        budget = state.get("budget", {})
        budget_total = budget.get("total", 0) if isinstance(budget, dict) else 0
        return f"""✅ 旅行计划已生成！

📍 城市：{city}
💰 预算总计：{budget_total}元

您可以输入修改指令，例如：
• "添加迪士尼"
• "降低酒店预算"
• "延长到5天"
• "对比一下变化"
• "回退到之前的版本"

或输入"结束"退出对话。"""

    def dialog(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        对话节点：用 interrupt 停住，等用户的下一条修改。

        第一次进入时暂停，把欢迎语交给调用方。
        Command(resume=用户输入) 之后从这里继续，改完计划再暂停，直到用户结束。
        """
        from langgraph.types import interrupt

        session_id = state.get("session_id", "")
        if not session_id:
            return {
                "dialog_reply": "错误：会话未初始化，请先规划旅行",
                "dialog_active": False,
                "current_phase": "final",
            }

        shown = state.get("dialog_reply") or self._dialog_welcome(state)
        user_text = interrupt({"reply": shown, "session_id": session_id})
        text = str(user_text or "").strip()

        if not text:
            return {
                "dialog_reply": shown,
                "dialog_input": "",
                "dialog_active": True,
                "current_phase": "dialog",
            }

        if text in ["结束", "退出", "完成", "不需要修改", "好的"]:
            return {
                "dialog_reply": "好的，旅行计划已确定。祝您旅途愉快！",
                "dialog_input": text,
                "dialog_changes": [],
                "dialog_active": False,
                "current_phase": "final",
            }

        from app.agents.dialog_agent import DialogAgent

        try:
            dialog_agent = DialogAgent(
                llm_provider="aliyun",
                use_redis=True,
                enable_rag=True,
                enable_amap=True,
                enable_memory=True,
            )
            result = dialog_agent.process_modification(
                session_id=session_id,
                user_input=text,
            )

            new_plan = None
            if result.get("new_version_id"):
                from app.agents.version_manager import VersionManager
                version_manager = VersionManager(use_redis=True)
                new_version = version_manager.get_version(result["new_version_id"])
                if new_version:
                    new_plan = new_version.plan_data

            if self.logger:
                self.logger.log_info("DIALOG", msg=(
                    f"[对话节点] 用户输入: {text}, "
                    f"新版本: {result.get('new_version_id', '无')}"
                ))

            updates = {
                "dialog_reply": result.get("reply", ""),
                "dialog_changes": result.get("changes", []),
                "dialog_input": text,
                "dialog_active": True,
                "current_phase": "dialog",
            }
            if new_plan:
                updates["current_version_id"] = result["new_version_id"]
                daily = new_plan.get("daily_itinerary")
                if isinstance(daily, dict) and daily.get("days"):
                    updates["trip_plan_parsed"] = daily
                elif isinstance(new_plan.get("days"), list):
                    updates["trip_plan_parsed"] = new_plan
                saved = updates.get("trip_plan_parsed")
                previous = state.get("trip_plan_parsed") if isinstance(state.get("trip_plan_parsed"), dict) else {}
                if isinstance(saved, dict) and not saved.get("hotel_options") and previous.get("hotel_options"):
                    saved["hotel_options"] = previous["hotel_options"]
                budget = new_plan.get("budget")
                if isinstance(budget, dict):
                    updates["budget"] = budget
                    updates["budget_total"] = budget.get("total", 0)
                if new_plan.get("duration_days"):
                    updates["travel_days"] = new_plan["duration_days"]
                if new_plan.get("accommodation"):
                    updates["accommodation"] = new_plan["accommodation"]
            return updates
        except Exception as e:
            if self.logger:
                self.logger.log_error("DIALOG", msg=f"[对话节点] 处理失败: {e}")
            return {
                "dialog_reply": f"抱歉，处理您的请求时出现错误：{str(e)}",
                "dialog_input": text,
                "dialog_active": True,
                "current_phase": "dialog",
            }

    # ==================== 条件路由函数 ====================
    
