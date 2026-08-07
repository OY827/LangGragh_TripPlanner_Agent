"""
诊断召回率低的根因：检查 Ground Truth 文档在知识库中的覆盖率
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from app.rag.retriever import KnowledgeRetriever


EVAL_DATASET = [
    {
        "query": "上海的历史文化景点",
        "city": "上海",
        "relevant_docs": [
            "外滩", "豫园", "静安寺", "玉佛禅寺", "龙华寺", "城隍庙",
            "七宝古镇", "朱家角古镇", "枫泾古镇", "新场古镇",
            "中共一大会址", "中共二大会址", "中共四大会址",
            "孙中山故居", "宋庆龄故居", "鲁迅故居", "茅盾故居", "蔡元培故居",
            "张园", "思南公馆", "武康大楼", "南京西路", "上海老街",
            "田子坊", "M50创意园", "新天地", "1933老场坊",
            "上海博物馆", "上海科技馆", "上海世博会博物馆",
            "上海犹太难民纪念馆", "上海城市规划展示馆",
            "圆明园路", "虎丘路", "北京西路", "成都南路", "多伦路",
            "徐光启墓", "黄道婆墓", "土山湾博物馆",
            "上海老城隍庙", "上海文庙", "白云观",
            "小普陀寺", "下海庙", "上海佛教居士林",
            "上海新场古镇", "上海枫泾古镇",
        ],
    },
    {
        "query": "上海适合亲子游玩的地方",
        "city": "上海",
        "relevant_docs": [
            "上海迪士尼度假区", "上海欢乐谷", "上海玛雅海滩水公园",
            "上海海昌海洋公园", "上海海洋水族馆", "上海长风海洋世界",
            "上海科技馆", "上海自然博物馆", "上海乐高探索中心",
            "上海世茂精灵之城主题乐园", "上海之巅观光厅",
            "上海辰山植物园", "上海野生动物园", "上海动物园",
            "东方绿舟", "东平国家森林公园", "共青国家森林公园",
            "上海湾湿地公园", "上海植物园",
            "金茂大厦", "东方明珠塔", "上海中心大厦",
            "疯狂动物城", "冰雪奇缘城堡",
        ],
    },
    {
        "query": "上海免费景点推荐",
        "city": "上海",
        "relevant_docs": [
            "外滩", "南京西路", "上海老街", "田子坊",
            "新天地", "1933老场坊", "M50创意园",
            "豫园", "城隍庙", "七宝古镇", "朱家角古镇",
            "思南公馆", "武康大楼", "安福路",
            "朵云书院", "上海书城",
            "圆明园路", "虎丘路", "北京西路", "成都南路",
            "多伦路", "甜爱路", "山阴路",
            "上海博物馆", "上海城市规划展示馆",
            "上海犹太难民纪念馆", "上海世博会博物馆",
            "中共一大会址", "中共二大会址", "中共四大会址",
            "孙中山故居", "宋庆龄故居", "鲁迅故居",
            "上海赏水杉", "上海湿地", "崇明岛",
            "金山城市沙滩", "共青国家森林公园",
        ],
    },
]


def check_coverage():
    print("=" * 70)
    print("🔍 Ground Truth 在知识库中的覆盖率诊断")
    print("=" * 70)

    retriever = KnowledgeRetriever()

    for item in EVAL_DATASET:
        print(f"\n{'─' * 70}")
        print(f"查询: '{item['query']}'")
        print(f"{'─' * 70}")

        gt_docs = item["relevant_docs"]
        city = item["city"]

        # 从知识库中获取所有该城市的景点标题
        all_kb_docs = retriever.retrieve(
            query=f"{city} 景点",
            filters={"city": city},
            top_k=500
        )
        kb_titles = set()
        for doc in all_kb_docs:
            title = doc.get("metadata", {}).get("title", "")
            if title:
                kb_titles.add(title)

        print(f"  知识库中 {city} 景点总数: {len(kb_titles)}")

        # 检查每个 Ground Truth 文档是否在知识库中
        found = []
        not_found = []
        for gt in gt_docs:
            # 精确匹配 + 子串匹配
            if gt in kb_titles:
                found.append(gt)
            else:
                # 尝试模糊匹配
                fuzzy = False
                for kb_title in kb_titles:
                    if gt in kb_title or kb_title in gt:
                        fuzzy = True
                        break
                if fuzzy:
                    found.append(f"{gt} (模糊匹配)")
                else:
                    not_found.append(gt)

        coverage = len(found) / len(gt_docs) * 100
        print(f"  Ground Truth 总数: {len(gt_docs)}")
        print(f"  在知识库中找到: {len(found)} ({coverage:.1f}%)")
        print(f"  不在知识库中: {len(not_found)}")

        if not_found:
            print(f"\n  ❌ 不在知识库中的文档:")
            for name in not_found:
                print(f"     - {name}")

        # 计算最大可能的召回率（top-10）
        max_recall_10 = min(10, len(found)) / len(gt_docs)
        max_recall_10_kb = min(10, len(found)) / len(found) if found else 0
        print(f"\n  📊 召回率上限分析 (Top-10):")
        print(f"     绝对召回率上限: {max_recall_10:.2%} (10/{len(gt_docs)})")
        print(f"     知识库内召回率上限: {max_recall_10_kb:.2%} (10/{len(found)})")

        # 用更大的 top_k 来评估
        for top_k in [10, 20, 50]:
            max_r = min(top_k, len(found)) / len(found) if found else 0
            print(f"     Top-{top_k} 知识库内召回上限: {max_r:.2%}")


if __name__ == "__main__":
    check_coverage()
