"""结构化输出：function calling 结果，以及原文 JSON 兜底。"""

from app.services.structured_output import (
    AnswerCheck,
    IntentOutput,
    QueryVariants,
    TripPlanOutput,
    invoke_structured,
)


class _Msg:
    def __init__(self, content):
        self.content = content


class _StructuredLLM:
    def __init__(self, payload):
        self.payload = payload
        self.calls = 0

    def with_structured_output(self, schema, **kwargs):
        self.schema = schema
        self.kwargs = kwargs
        return self

    def invoke(self, prompt):
        self.calls += 1
        return self.payload


class _TextFallbackLLM:
    def __init__(self, content):
        self.content = content
        self.calls = 0

    def with_structured_output(self, schema, **kwargs):
        raise RuntimeError("no tool calling")

    def invoke(self, prompt):
        self.calls += 1
        return _Msg(self.content)


def test_structured_payload_is_returned_without_a_second_call():
    intent = IntentOutput(
        change_type="add_item",
        scope="incremental",
        description="添加迪士尼",
        target_field="attractions",
        details={"action": "add", "item": "迪士尼"},
        confidence=0.9,
    )
    llm = _StructuredLLM({"raw": _Msg(""), "parsed": intent, "parsing_error": None})
    got = invoke_structured(llm, IntentOutput, "添加迪士尼")
    assert isinstance(got, IntentOutput)
    assert got.details.item == "迪士尼"
    assert llm.calls == 1
    assert llm.kwargs["method"] == "function_calling"
    assert llm.kwargs["include_raw"] is True


def test_markdown_json_is_used_when_tool_calling_is_unavailable():
    llm = _TextFallbackLLM('说明\n```json\n{"queries": ["上海博物馆", "上海古迹"]}\n```')
    got = invoke_structured(llm, QueryVariants, "改写")
    assert isinstance(got, QueryVariants)
    assert got.queries == ["上海博物馆", "上海古迹"]
    assert llm.calls == 1


def test_trip_plan_coerces_messy_fields():
    plan = TripPlanOutput.model_validate({
        "city": "上海",
        "days": [{
            "day_index": "0",
            "attractions": [
                "外滩",
                {
                    "name": "上海迪士尼",
                    "visit_duration": "240分钟",
                    "ticket_price": "399元",
                    "location": "121.66,31.14",
                },
            ],
            "hotel": "全季酒店",
        }],
        "budget": 2060,
    })
    day = plan.days[0]
    assert day.attractions[0].name == "外滩"
    disney = day.attractions[1]
    assert disney.visit_duration == 240
    assert disney.ticket_price == 399
    assert disney.location is None
    assert day.hotel.name == "全季酒店"
    assert plan.budget.total == 2060


def test_trip_plan_clips_long_prose():
    plan = TripPlanOutput.model_validate({
        "introduction": "上海" * 500,
        "overall_suggestions": "建议" * 200,
        "days": [{
            "description": "这一天非常充实" * 20,
            "attractions": [{"name": "外滩", "description": "外滩夜景" * 30}],
        }],
    })
    assert len(plan.introduction) == 360
    assert len(plan.overall_suggestions) == 160
    assert len(plan.days[0].description) == 40
    assert len(plan.days[0].attractions[0].description) == 40


def test_candidate_facts_replace_model_prose():
    from app.agents.planner import PlannerMixin

    parsed = {
        "introduction": "短介绍",
        "days": [{
            "attractions": [{"name": "外滩", "description": "模型写了一大段", "visit_duration": 10}],
        }],
    }
    PlannerMixin._apply_candidate_facts(None, parsed, [{
        "name": "外滩",
        "address": "中山东一路",
        "location": "121.49,31.24",
        "visit_duration": 90,
        "description": "外滩夜景",
        "ticket_price": 0,
    }])
    attraction = parsed["days"][0]["attractions"][0]
    assert attraction["visit_duration"] == 90
    assert attraction["location"] == "121.49,31.24"
    assert attraction["description"] == "外滩夜景"


def test_answer_check_accepts_string_bool():
    check = AnswerCheck.model_validate({
        "grounded": "false",
        "completeness": "0.2",
        "alignment_score": "0.3",
        "issues": "编造了景点",
    })
    assert check.grounded is False
    assert check.completeness == 0.2
    assert check.issues == ["编造了景点"]
