"""
验证 Ground Truth：上海历史文化景点的真实数量
采用多种方法交叉验证
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from app.rag.retriever import KnowledgeRetriever
from app.llm import get_llm


CITY = "上海"


def load_attractions_from_json(city):
    """从 JSON 文件加载所有景点"""
    path = f"d:/Projects/LangChain/app/knowledge_base/attractions/{city}.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def method1_category_filter(attractions):
    """方法1：仅用 category 字段过滤（当前评估脚本的方式）"""
    history_categories = {"历史文化", "博物馆", "艺术文化"}
    matched = []
    for attr in attractions:
        metadata = attr.get("metadata", {})
        category = metadata.get("category", "")
        if category in history_categories:
            matched.append({
                "title": metadata.get("title", ""),
                "category": category,
                "tags": metadata.get("tags", ""),
                "price": metadata.get("price"),
            })
    return matched


def method2_keyword_match(attractions):
    """方法2：关键词匹配（在 title + tags + description + content 中搜索）"""
    history_keywords = [
        "历史", "古迹", "遗址", "古镇", "故居", "博物馆", "纪念",
        "园林", "文物", "寺庙", "教堂", "旧址", "文化馆", "蜡像",
        "美术馆", "艺术", "展览", "抗战", "会址", "陵", "墓"
    ]
    matched = []
    for attr in attractions:
        metadata = attr.get("metadata", {})
        content = attr.get("content", "")
        text = f"{metadata.get('title', '')} {metadata.get('tags', '')} {metadata.get('description', '')} {content}"

        hits = [kw for kw in history_keywords if kw in text]
        if hits:
            matched.append({
                "title": metadata.get("title", ""),
                "category": metadata.get("category", ""),
                "tags": metadata.get("tags", ""),
                "price": metadata.get("price"),
                "matched_keywords": hits,
            })
    return matched


def method3_llm_judge(attractions, query):
    """方法3：LLM 辅助判断（批量）"""
    llm = get_llm(provider="aliyun", model=None, temperature=0.1)

    # 准备景点列表（只传关键信息，节省 token）
    items = []
    for i, attr in enumerate(attractions):
        metadata = attr.get("metadata", {})
        items.append({
            "id": i,
            "title": metadata.get("title", ""),
            "category": metadata.get("category", ""),
            "tags": metadata.get("tags", ""),
            "description": metadata.get("description", ""),
        })

    # 分批处理（每批 40 个）
    batch_size = 40
    all_scores = {}

    for start in range(0, len(items), batch_size):
        batch = items[start:start + batch_size]
        batch_json = json.dumps(batch, ensure_ascii=False, indent=2)

        prompt = f"""查询意图："{query}"

请判断以下景点是否与查询意图相关（0-3分）：
- 3分：强相关（完全符合"历史文化景点"的查询意图）
- 2分：相关（基本符合，属于历史文化类）
- 1分：弱相关（部分符合，有历史元素但不是主要类型）
- 0分：不相关

景点列表：
{batch_json}

请返回JSON格式：
[
  {{"id": 0, "score": 3, "reason": "原因"}},
  ...
]

注意：
- "历史文化"包括：古迹、遗址、古镇、故居、博物馆、纪念馆、园林、寺庙、教堂、文物保护单位等
- "上海博物馆"虽然category可能是"博物馆"，但也属于历史文化类
- "城隍庙"虽然category可能是"城市漫步"，但具有历史文化属性
- 纯自然风光、主题乐园、商业街区不算历史文化（除非有历史价值）
"""

        try:
            response = llm.invoke(prompt)
            text = response.content if hasattr(response, 'content') else str(response)

            # 清理可能的 markdown 标记
            text = text.strip()
            if text.startswith("```json"):
                text = text[7:]
            if text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

            scores = json.loads(text)
            for item in scores:
                all_scores[item["id"]] = {
                    "score": item.get("score", 0),
                    "reason": item.get("reason", "")
                }
        except Exception as e:
            print(f"  ⚠️ 批次 {start}-{start+batch_size} LLM 判断失败: {e}")

    # 转换为结果列表
    matched = []
    for i, attr in enumerate(attractions):
        if i in all_scores and all_scores[i]["score"] >= 1:
            metadata = attr.get("metadata", {})
            matched.append({
                "title": metadata.get("title", ""),
                "category": metadata.get("category", ""),
                "tags": metadata.get("tags", ""),
                "price": metadata.get("price"),
                "llm_score": all_scores[i]["score"],
                "llm_reason": all_scores[i]["reason"],
            })

    return matched


def method4_human_known_landmarks(attractions):
    """方法4：已知历史文化地标清单（人工整理的标准答案）"""
    # 上海公认的历史文化景点（基于权威榜单和大众认知）
    known_landmarks = {
        # 古镇古村
        "七宝古镇", "朱家角古镇景区", "枫泾古镇", "新场古镇", "南翔古镇",
        # 名胜古迹
        "豫园", "城隍庙旅游区", "上海城隍庙", "静安寺", "龙华寺", "玉佛禅寺",
        "外滩", "外白渡桥", "武康大楼", "武康路", "张园", "思南公馆", "思南路",
        "南京路步行街", "南京西路", "淮海路", "田子坊", "新天地", "上海老街",
        "人民广场", "陆家嘴",
        # 博物馆纪念馆
        "上海博物馆", "上海博物馆(东馆)", "上海博物馆东馆",
        "世博会博物馆", "上海城市规划展示馆",
        "上海四行仓库抗战纪念馆", "上海宋庆龄故居纪念馆",
        "中国共产党第一次全国代表大会会址", "中共一大会址",
        "上海邮政博物馆", "上海电影博物馆", "上海自然博物馆",
        "中国航海博物馆", "震旦博物馆", "上海观复博物馆", "中华艺术宫",
        "浦东美术馆", "上海犹太难民纪念馆",
        # 文化遗址
        "广富林文化遗址", "上海佘山国家旅游度假区",
        # 艺术文化
        "上海《不眠之夜》", "梅赛德斯-奔驰文化中心",
        "徐家汇书院",
    }

    matched = []
    for attr in attractions:
        metadata = attr.get("metadata", {})
        title = metadata.get("title", "")
        # 精确匹配或模糊匹配
        is_match = title in known_landmarks or any(
            title in landmark or landmark in title
            for landmark in known_landmarks
        )
        if is_match:
            matched.append({
                "title": title,
                "category": metadata.get("category", ""),
                "tags": metadata.get("tags", ""),
                "price": metadata.get("price"),
            })

    return matched


def main():
    print("=" * 72)
    print("🔬 验证 Ground Truth：上海历史文化景点的真实数量")
    print("=" * 72)

    # 加载景点
    attractions = load_attractions_from_json(CITY)
    print(f"\n📦 {CITY} 知识库景点总数：{len(attractions)}")

    # 方法1：分类过滤
    print(f"\n{'─' * 72}")
    print("方法1：仅用 category 字段过滤（当前评估脚本方式）")
    print(f"{'─' * 72}")
    m1 = method1_category_filter(attractions)
    print(f"结果：{len(m1)} 个")
    print(f"分类分布：{dict((c, sum(1 for x in m1 if x['category'] == c)) for c in set(x['category'] for x in m1))}")
    print(f"景点列表：")
    for i, attr in enumerate(m1, 1):
        print(f"  {i}. {attr['title']} [{attr['category']}]")

    # 方法2：关键词匹配
    print(f"\n{'─' * 72}")
    print("方法2：关键词匹配（title + tags + description + content）")
    print(f"{'─' * 72}")
    m2 = method2_keyword_match(attractions)
    print(f"结果：{len(m2)} 个")

    # 方法3：LLM 判断
    print(f"\n{'─' * 72}")
    print("方法3：LLM 辅助判断（批量打分）")
    print(f"{'─' * 72}")
    m3 = method3_llm_judge(attractions, "上海的历史文化景点")
    print(f"结果：{len(m3)} 个")
    print(f"LLM 评分分布：")
    score_dist = {}
    for attr in m3:
        s = attr.get("llm_score", 0)
        score_dist[s] = score_dist.get(s, 0) + 1
    for s in sorted(score_dist.keys(), reverse=True):
        print(f"  {s}分：{score_dist[s]}个")

    # 方法4：人工地标
    print(f"\n{'─' * 72}")
    print("方法4：已知历史文化地标清单（人工整理）")
    print(f"{'─' * 72}")
    m4 = method4_human_known_landmarks(attractions)
    print(f"结果：{len(m4)} 个")
    print(f"景点列表：")
    for i, attr in enumerate(m4, 1):
        print(f"  {i}. {attr['title']} [{attr['category']}]")

    # 交叉对比
    print(f"\n{'=' * 72}")
    print("📊 四种方法交叉对比")
    print(f"{'=' * 72}")

    titles_m1 = set(x["title"] for x in m1)
    titles_m2 = set(x["title"] for x in m2)
    titles_m3 = set(x["title"] for x in m3)
    titles_m4 = set(x["title"] for x in m4)

    print(f"\n  方法1（分类过滤）：{len(titles_m1)} 个")
    print(f"  方法2（关键词匹配）：{len(titles_m2)} 个")
    print(f"  方法3（LLM 判断）：{len(titles_m3)} 个")
    print(f"  方法4（人工地标）：{len(titles_m4)} 个")

    # 共识：四种方法都认为相关的
    consensus = titles_m1 & titles_m2 & titles_m3 & titles_m4
    print(f"\n  ✅ 四种方法共识（都认为相关）：{len(consensus)} 个")
    for t in sorted(consensus):
        print(f"     - {t}")

    # 至少3种方法认为相关
    all_titles = titles_m1 | titles_m2 | titles_m3 | titles_m4
    majority = set()
    for t in all_titles:
        votes = sum([t in titles_m1, t in titles_m2, t in titles_m3, t in titles_m4])
        if votes >= 3:
            majority.add(t)
    print(f"\n  🟡 多数共识（≥3种方法认为相关）：{len(majority)} 个")

    # 仅 LLM 认为相关但其他方法没覆盖的
    only_llm = titles_m3 - titles_m1 - titles_m2 - titles_m4
    if only_llm:
        print(f"\n  ⚠️ 仅 LLM 认为相关（其他方法未覆盖）：{len(only_llm)} 个")
        for t in sorted(only_llm):
            print(f"     - {t}")

    # 方法1有但其他方法都没有的
    only_m1 = titles_m1 - titles_m2 - titles_m3 - titles_m4
    if only_m1:
        print(f"\n  ⚠️ 仅方法1认为相关（可能是误判）：{len(only_m1)} 个")
        for t in sorted(only_m1):
            print(f"     - {t}")

    # 结论
    print(f"\n{'=' * 72}")
    print("🎯 结论")
    print(f"{'=' * 72}")
    print(f"\n  上海历史文化景点的真实数量估计：")
    print(f"    - 下界（四种方法共识）：{len(consensus)} 个")
    print(f"    - 中位数（多数共识）：{len(majority)} 个")
    print(f"    - 上界（任一方法命中）：{len(all_titles)} 个")
    print(f"\n  建议采用多数共识作为 Ground Truth：{len(majority)} 个")


if __name__ == "__main__":
    main()
