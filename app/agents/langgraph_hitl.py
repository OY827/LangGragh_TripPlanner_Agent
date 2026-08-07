"""
Human-in-the-loop (HITL) 模块

实现人工审批节点，支持：
1. 关键决策点暂停执行
2. 查看当前状态快照
3. 人工修改/确认后恢复执行
4. 会话状态回溯
"""

from typing import Dict, Any, Optional, Callable, List


class HumanInTheLoopManager:
    """
    人机协同管理器
    
    在 LangGraph 的关键节点（如反思节点）暂停执行，
    等待人工确认或修改后再恢复。
    
    使用示例:
        hitl = HumanInTheLoopManager()
        
        # 注册审批回调
        hitl.register_approver("plan_review", my_approver_func)
        
        # 在图中调用
        def hitl_node(state):
            approved = hitl.wait_for_approval("plan_review", state)
            if approved:
                return state  # 继续执行
            else:
                return {"current_phase": "final", "errors": ["用户拒绝"]}
    """
    
    def __init__(self):
        self._approvers: Dict[str, Callable] = {}
        self._pending_approvals: Dict[str, Dict[str, Any]] = {}
    
    def register_approver(self, name: str, func: Callable):
        """
        注册审批回调函数
        
        Args:
            name: 审批点名称
            func: 回调函数，签名为 func(state: dict) -> dict
                  返回修改后的状态或空 dict 表示通过
        """
        self._approvers[name] = func
    
    def wait_for_approval(
        self,
        approval_name: str,
        state: Dict[str, Any],
        default_state: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        等待人工审批
        
        Args:
            approval_name: 审批点名称
            state: 当前状态
            default_state: 默认通过时的状态更新
        
        Returns:
            审批后的状态更新
        """
        if approval_name not in self._approvers:
            print(f"⚠️  审批点 '{approval_name}' 未注册，默认通过")
            return default_state or {}
        
        approver = self._approvers[approval_name]
        
        try:
            result = approver(state)
            
            if result is None or result == {}:
                print(f"✅ 审批 '{approval_name}' 通过")
                return default_state or {}
            else:
                print(f"📝 审批 '{approval_name}' 返回修改")
                return result
        except Exception as e:
            print(f"❌ 审批 '{approval_name}' 失败: {e}")
            return {"errors": [f"审批失败: {e}"]}
    
    def create_review_node(self, approval_name: str):
        """
        创建一个审批节点函数，可直接用于 LangGraph
        
        Args:
            approval_name: 审批点名称
        
        Returns:
            节点函数
        """
        def review_node(state: Dict[str, Any]) -> Dict[str, Any]:
            print(f"\n📋 === 人工审批点: {approval_name} ===")
            print(f"   当前阶段: {state.get('current_phase', 'unknown')}")
            print(f"   迭代次数: {state.get('iteration_count', 0)}")
            
            if state.get("reflection_feedback"):
                print(f"   反思反馈: {state['reflection_feedback']}")
            
            if state.get("trip_plan_parsed"):
                parsed = state["trip_plan_parsed"]
                days = parsed.get("days", [])
                print(f"   行程天数: {len(days)}")
                for day in days:
                    attractions = day.get("attractions", [])
                    print(f"     Day {day.get('day_index', '?')}: {len(attractions)} 个景点")
            
            result = self.wait_for_approval(approval_name, state)
            return result
        
        return review_node


def console_approver(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    控制台审批器（用于测试）
    
    在终端显示状态并等待用户输入。
    输入 y 表示通过，其他表示需要修改。
    """
    print("\n" + "="*50)
    print("🤖 旅行计划审批")
    print("="*50)
    
    if state.get("trip_plan_parsed"):
        parsed = state["trip_plan_parsed"]
        print(f"\n📅 城市: {parsed.get('city', '?')}")
        print(f"🗓️ 日期: {parsed.get('start_date', '?')} 至 {parsed.get('end_date', '?')}")
        
        for day in parsed.get("days", []):
            day_idx = day.get("day_index", 0)
            attractions = day.get("attractions", [])
            print(f"\n--- Day {day_idx + 1} ---")
            for attr in attractions[:5]:
                name = attr.get("name", "?") if isinstance(attr, dict) else str(attr)
                print(f"  📍 {name}")
    
    print("\n" + "="*50)
    choice = input("审批通过？(y/n): ").strip().lower()
    
    if choice == 'y':
        return {}
    else:
        modification = input("请输入修改意见: ").strip()
        return {
            "errors": [modification],
            "reflection_feedback": modification
        }
