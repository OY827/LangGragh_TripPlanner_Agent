"""从上海知识库真实 title 重新标定评测 Ground Truth。"""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KB_PATH = ROOT / "app" / "knowledge_base" / "attractions" / "上海.json"
OUT_JSON = ROOT / "evaluation" / "gt_shanghai_kb_aligned.json"


def load_kb():
    data = json.loads(KB_PATH.read_text(encoding="utf-8"))
    items = []
    for item in data:
        m = item.get("metadata", item)
        items.append(
            {
                "title": m.get("title"),
                "category": m.get("category") or "",
                "price": m.get("price"),
                "tags": m.get("tags") or "",
                "desc": m.get("description") or "",
            }
        )
    return items


def uniq_in_kb(seq, kb_titles):
    seen = set()
    out = []
    for s in seq:
        if s and s in kb_titles and s not in seen:
            seen.add(s)
            out.append(s)
    return out


def build_history(items):
    """历史文化：category=历史文化 + 明确历史/文博/古镇类，排除演出体验。"""
    out = []
    for x in items:
        t, c = x["title"], x["category"]
        if "不眠之夜" in t:
            continue
        if c == "历史文化":
            out.append(t)
            continue
        keys = ["豫园", "外滩", "上海老街", "城隍庙", "博物馆", "会址", "故居", "古镇", "寺", "庙", "纪念馆", "遗址"]
        bads = ["乐高", "迪士尼", "海洋", "野生", "乐园", "游船", "旗袍体验", "烹饪", "宴", "足球", "不眠"]
        if any(k in t for k in keys) and not any(b in t for b in bads):
            out.append(t)
    return out


def build_family(items):
    """亲子：乐园/海洋馆/动物园/科技馆/植物园等。"""
    kws = [
        "乐园", "迪士尼", "乐高", "海洋", "水族", "野生动物", "动物园", "玛雅",
        "精灵", "昆虫", "自然博物馆", "科技馆", "东方绿舟", "植物园", "森林公园",
        "辰山", "海昌", "天文馆", "航海博物馆", "影视乐园", "INS LAND", "耀雪",
        "水世界", "冰雪",
    ]
    bads = ["宴", "汤泉", "旗袍", "烹饪", "不眠", "足球", "书院", "会展", "旗舰店"]
    out = []
    for x in items:
        t = x["title"]
        blob = f"{t} {x['category']} {x['tags']} {x['desc']}"
        if any(b in t for b in bads):
            continue
        if any(k in blob for k in kws):
            out.append(t)
    return out


def build_free(items):
    """
    免费：price 为 0/免费，并排除明显收费/体验类错误标价。
    （知识库里迪士尼等也有 price=0，不能只看价格字段）
    """
    bads = [
        "乐园", "迪士尼", "乐高", "野生动物", "游船", "体验", "烹饪", "不眠",
        "足球", "破笼", "观光厅", "环球金融", "金茂", "精灵", "展馆深度",
        "citywalk", "磁浮", "宴", "汤泉", "蜡像", "水世界", "冰雪", "玛雅",
        "海洋公园", "水族", "杜莎", "时空之旅", "白玉兰", "叙宴", "曲水兰亭",
        "疯狂动物城", "昆虫", "东方明珠", "科技馆", "美术馆", "艺术宫",
        "会展", "奔驰文化中心", "佘山",
    ]
    out = []
    for x in items:
        t = x["title"]
        p = str(x["price"]).strip() if x["price"] is not None else ""
        if p not in ("0", "0.0", "免费", "0元"):
            continue
        if any(b in t for b in bads):
            continue
        out.append(t)
    return out


def main():
    items = load_kb()
    kb_titles = {x["title"] for x in items if x["title"]}

    history = uniq_in_kb(build_history(items), kb_titles)
    family = uniq_in_kb(build_family(items), kb_titles)
    free = uniq_in_kb(build_free(items), kb_titles)

    dataset = [
        {
            "query": "上海的历史文化景点",
            "city": "上海",
            "intent_categories": ["历史文化"],
            "relevant_docs": history,
            "note": "KB title 对齐：category=历史文化 + 文博/古镇/外滩等",
        },
        {
            "query": "上海适合亲子游玩的地方",
            "city": "上海",
            "intent_categories": ["主题乐园", "自然风光"],
            "relevant_docs": family,
            "note": "KB title 对齐：乐园/海洋馆/动物园/科技馆/植物园等",
        },
        {
            "query": "上海免费景点推荐",
            "city": "上海",
            "intent_categories": ["历史文化", "自然风光", "城市漫步"],
            "relevant_docs": free,
            "note": "KB title 对齐：price=0/免费且排除错误标价的乐园体验类",
        },
    ]

    # 强校验：全部 GT 必须是 KB 精确标题
    for item in dataset:
        missing = [t for t in item["relevant_docs"] if t not in kb_titles]
        assert not missing, f"{item['query']} GT 不在 KB: {missing}"

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"KB titles: {len(kb_titles)}")
    for item in dataset:
        print(f"\n[{item['query']}] GT={len(item['relevant_docs'])}")
        for t in item["relevant_docs"]:
            print(f"  - {t}")
    print(f"\nWrote {OUT_JSON}")


if __name__ == "__main__":
    main()
