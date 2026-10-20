"""token 用量解析。不发起模型调用。"""

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load():
    name = "token_usage_under_test"
    path = ROOT / "app" / "monitoring" / "token_usage.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


mod = _load()
mod._publish = lambda *args, **kwargs: None


class _Message:
    def __init__(self, usage=None, response_metadata=None):
        self.usage_metadata = usage
        self.response_metadata = response_metadata or {}


class _Gen:
    def __init__(self, message):
        self.message = message


class _Result:
    def __init__(self, llm_output=None, generations=None):
        self.llm_output = llm_output
        self.generations = generations or []


def test_reads_openai_token_usage():
    result = _Result(llm_output={"token_usage": {"prompt_tokens": 120, "completion_tokens": 40}})
    assert mod.usage_from_result(result) == {"input_tokens": 120, "output_tokens": 40}


def test_reads_message_usage_when_llm_output_missing():
    result = _Result(generations=[[_Gen(_Message({"input_tokens": 15, "output_tokens": 6}))]])
    assert mod.usage_from_result(result) == {"input_tokens": 15, "output_tokens": 6}


def test_request_scope_sums_calls_and_ignores_duplicate_run():
    with mod.track_llm_usage() as ledger:
        mod.note_llm_result(
            _Result(llm_output={"token_usage": {"prompt_tokens": 10, "completion_tokens": 2}}),
            run_id="run-1",
        )
        mod.note_llm_result(
            _Result(llm_output={"token_usage": {"prompt_tokens": 10, "completion_tokens": 2}}),
            run_id="run-1",
        )
        mod.note_llm_result(_Result(), run_id="run-2")
    assert ledger["input_tokens"] == 10
    assert ledger["output_tokens"] == 2
    assert ledger["calls"] == 2
    assert ledger["calls_without_usage"] == 1


if __name__ == "__main__":
    test_reads_openai_token_usage()
    test_reads_message_usage_when_llm_output_missing()
    test_request_scope_sums_calls_and_ignores_duplicate_run()
    print("ok")
