"""工具耗时、端到端等待，以及串行相对并行的降幅。"""

from typing import Any, Dict, Sequence

from evaluation.metrics import latency_reduction, mean


def summarize_latency(samples: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    tool_values = []
    e2e_values = []
    reductions = []
    for sample in samples:
        tool_values.extend(float(value) for value in (sample.get("tool_latency_s") or []))
        if sample.get("e2e_s") is not None:
            e2e_values.append(float(sample["e2e_s"]))
        serial_s = sample.get("serial_s")
        parallel_s = sample.get("parallel_s")
        if serial_s is not None and parallel_s is not None:
            reductions.append(latency_reduction(float(serial_s), float(parallel_s)))
    return {
        "mean_tool_s": mean(tool_values),
        "mean_e2e_s": mean(e2e_values),
        "wait_reduction": mean(reductions),
        "tool_sample_count": len(tool_values),
        "e2e_sample_count": len(e2e_values),
        "parallel_sample_count": len(reductions),
    }
