"""
P0 准确率改进验证脚本

验证项：
1. SmartAttractionTool 走 HybridRAG + BGE，输出含建议游览/坐标
2. LangGraph merge 正确读取 metadata，保留 rerank 顺序
3. Planner Prompt 含候选约束
4. 小样本 grounded Precision@5（对比有无 Rerank 的排序质量）
"""

import os
import sys
import re
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()


def _extract_titles(text: str):
    titles = []
    for line in text.splitlines():
        m = re.match(r"^(\d+)\.\s*(.+?)(?:\s*\(|$)", line.strip())
        if m:
            titles.append(m.group(2).strip())
    return titles


def test_smart_tool():
    print("\n=== 1) SmartAttractionTool (HybridRAG优先) ===")
    from app.tools.agent_tools import SmartAttractionTool, _get_hybrid_rag

    hybrid = _get_hybrid_rag()
    assert hybrid is not None, "HybridRAG 初始化失败"
    assert "bge-reranker" in (hybrid.config.rerank_model or "").lower(), (
        f"期望 bge-reranker，实际: {hybrid.config.rerank_model}"
    )
    print(f"  ✓ HybridRAG rerank_model={hybrid.config.rerank_model}")

    tool = SmartAttractionTool()
    t0 = time.time()
    out = tool._run(city="上海", keywords="历史文化")
    elapsed = time.time() - t0
    print(f"  耗时: {elapsed:.1f}s")

    assert out and "出错" not in out[:20], f"工具失败: {out[:200]}"
    titles = _extract_titles(out)
    assert len(titles) >= 3, f"景点过少: {titles}"
    assert "建议游览" in out or "visit_duration" in out, "缺少建议游览时长字段"
    assert "Hybrid+Rerank" in out or "HybridRAG" in out or "相关性" in out, (
        "未体现 Hybrid/Rerank 检索模式"
    )
    print(f"  ✓ 返回 {len(titles)} 个景点")
    print(f"  Top5: {titles[:5]}")
    has_duration = out.count("建议游览")
    has_coord = out.count("坐标:")
    print(f"  ✓ 建议游览字段出现 {has_duration} 次, 坐标出现 {has_coord} 次")
    return titles, out


def test_merge_metadata():
    print("\n=== 2) LangGraph merge_attraction_results ===")
    from app.agents.langgraph_nodes import GraphNodes

    nodes = GraphNodes(enable_monitoring=False)
    kb_docs = [
        {
            "content": "外滩简介",
            "metadata": {
                "title": "外滩",
                "category": "历史文化",
                "rating": 4.8,
                "address": "中山东一路",
                "visit_duration": 90,
                "location": {"longitude": 121.49, "latitude": 31.24},
            },
            "rerank_score": 0.91,
        },
        {
            "content": "豫园简介",
            "metadata": {
                "title": "豫园",
                "category": "历史文化",
                "rating": 4.7,
                "address": "安仁街",
                "visit_duration": 120,
                "location": {"longitude": 121.49, "latitude": 31.23},
            },
            "rerank_score": 0.85,
        },
        {
            "content": "迪士尼简介",
            "metadata": {
                "title": "上海迪士尼度假区",
                "category": "主题乐园",
                "rating": 4.9,
                "visit_duration": 480,
            },
            "rerank_score": 0.20,
        },
    ]
    state = {
        "kb_docs": kb_docs,
        "amap_results": "1. 东方明珠\n2. 外滩观景平台\n",
        "visited_attractions": [],
    }
    result = nodes.merge_attraction_results(state)
    merged = result["merged_results"]
    print(merged[:500])

    assert "外滩" in merged
    assert "建议游览: 90分钟" in merged
    assert "121.49,31.24" in merged
    # 应按 rerank_score 排序：外滩 > 豫园 > 迪士尼
    pos_w = merged.find("外滩")
    pos_y = merged.find("豫园")
    pos_d = merged.find("上海迪士尼")
    assert pos_w < pos_y < pos_d, f"未按 rerank 顺序: w={pos_w}, y={pos_y}, d={pos_d}"
    assert "高德API结果" not in merged, "不应再塞整块高德文本为假景点"
    print("  ✓ metadata 解析正确，Rerank 顺序保留")


def test_planner_grounding_prompt():
    print("\n=== 3) Planner 候选约束 Prompt ===")
    from pathlib import Path

    planner_md = Path("app/prompts/agents/planner.md").read_text(encoding="utf-8")
    assert "候选约束" in planner_md or "禁止编造" in planner_md
    print("  ✓ planner.md 含候选约束")

    planner_src = Path("app/agents/planner.py").read_text(encoding="utf-8")
    assert "不要编造候选里没有的景点" in planner_src
    assert "只能从上方「景点候选」中选择" in planner_src
    print("  ✓ LangGraph planner message 含 grounding 约束")


def test_precision_sample():
    print("\n=== 4) 小样本 Precision@5（历史文化） ===")
    from app.rag.hybrid_rag import HybridRAG, HybridRAGConfig

    relevant = {
        "外滩", "豫园", "静安寺", "玉佛禅寺", "龙华寺", "城隍庙",
        "七宝古镇", "朱家角古镇", "田子坊", "新天地", "上海博物馆",
        "中共一大会址", "思南公馆", "武康大楼", "上海老街", "1933老场坊",
        "孙中山故居", "宋庆龄故居", "鲁迅故居", "M50创意园",
    }

    cfg_on = HybridRAGConfig(
        enable_rerank=True,
        rerank_model="BAAI/bge-reranker-base",
        retrieval_top_k=30,
        rerank_top_k=10,
        top_k=10,
        max_retrieval_rounds=1,
        enable_query_rewrite=False,
        enable_retrieval_validation=False,
        enable_answer_validation=False,
        use_mmr=False,
    )
    cfg_off = HybridRAGConfig(
        enable_rerank=False,
        retrieval_top_k=10,
        top_k=10,
        max_retrieval_rounds=1,
        enable_query_rewrite=False,
        enable_retrieval_validation=False,
        enable_answer_validation=False,
        use_mmr=True,
    )

    def p_at_k(docs, k=5):
        titles = []
        for d in docs[:k]:
            t = (d.get("metadata") or {}).get("title", "")
            if t:
                titles.append(t)
        hit = sum(1 for t in titles if any(r in t or t in r for r in relevant))
        return hit / max(len(titles), 1), titles

    rag_on = HybridRAG(config=cfg_on)
    rag_off = HybridRAG(config=cfg_off)
    query = "上海 历史文化"
    docs_on, meta_on = rag_on.retrieve_with_validation(query=query, city="上海", preferences=["历史文化"])
    docs_off, meta_off = rag_off.retrieve_with_validation(query=query, city="上海", preferences=["历史文化"])

    p_on, titles_on = p_at_k(docs_on, 5)
    p_off, titles_off = p_at_k(docs_off, 5)
    print(f"  无Rerank Top5: {titles_off}  P@5={p_off:.2f}")
    print(f"  有Rerank Top5: {titles_on}  P@5={p_on:.2f}  enabled={meta_on.get('rerank_enabled')}")
    assert meta_on.get("rerank_enabled") is True, "Rerank 未启用"
    assert p_on >= 0.4, f"Rerank 后 P@5 过低: {p_on}"
    print("  ✓ Rerank 路径可用且 P@5 达标")


def main():
    failures = []
    for name, fn in [
        ("smart_tool", test_smart_tool),
        ("merge_metadata", test_merge_metadata),
        ("planner_prompt", test_planner_grounding_prompt),
        ("precision_sample", test_precision_sample),
    ]:
        try:
            fn()
            print(f"✅ PASS {name}")
        except Exception as e:
            failures.append((name, e))
            print(f"❌ FAIL {name}: {e}")

    print("\n" + "=" * 50)
    if failures:
        print(f"失败 {len(failures)} 项:")
        for n, e in failures:
            print(f"  - {n}: {e}")
        sys.exit(1)
    print("全部通过")
    sys.exit(0)


if __name__ == "__main__":
    main()
