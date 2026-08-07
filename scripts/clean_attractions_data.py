"""
景点数据清洗脚本

功能：
1. 从CSV读取4个城市（北京、上海、广州、深圳）的景点数据
2. 数据清洗：
   - 去除重复景点（按景区名称去重）
   - 去除演唱会、演出、展览类景点
   - 去除无介绍或无名称的无效数据
3. 转换为标准JSON格式（与现有知识库结构一致）
4. 保存为JSON文件
5. 可选：存入Qdrant向量库
"""

import csv
import json
import os
import re
import sys
import ast
from typing import List, Dict, Any, Set

from dotenv import load_dotenv
load_dotenv()  # 加载.env环境变量

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# 景点数据目录
ATTRACTIONS_DIR = os.path.join(PROJECT_ROOT, "app", "knowledge_base", "attractions")

# 需要清洗的城市
TARGET_CITIES = ["北京", "上海", "广州", "深圳"]

# 需要过滤的关键词（演唱会、演出、展览等）
EXCLUDE_KEYWORDS = {
    "name_keywords": [
        "演唱会", "音乐会", "演出", "巡演", "LIVE", "live",
        "话剧", "音乐剧", "舞台剧", "戏剧", "歌剧",
        "展览", "展会", "博览", "展销",
        "音乐节", "艺术节", "文化节", "嘉年华",
        "门票", "票务", "预约",
        "体验馆", "体验中心",  # 通常是商业体验店
        "密室", "剧本杀", "逃脱",
        "电影院", "影城", "影院",
        "KTV", "ktv", "量贩",
        "酒吧", "club", "Club", "CLUB",
        "餐厅", "饭店", "美食",  # 餐饮类不在景点库
        # 新增：演出/表演类
        "脱口秀", "喜剧", "相声", "小品", "杂技", "马戏",
        "舞蹈剧场", "沉浸式", "环境式",
        "综艺", "选秀", "比赛", "联赛", "总决赛",
        "专场", "特别企划", "诗剧", "舞剧", "剧场",
        "体育中心", "体育馆", "体育场", "运动中心",
        "电竞", "游戏", "桌游",
        "放映", "点映", "首映", "观影",
        "签售", "见面会", "粉丝",
        "工作坊", "手工", "DIY", "diy",
        "禅修", "冥想", "瑜伽",
        "采摘", "农家乐",
        "蹦床", "跳跳", "攀岩", "潜水",
        "美容", "SPA", "spa", "按摩", "足疗",
        "摄影", "写真", "跟拍",
        "卡丁车", "赛车",
        "高尔夫", "保龄球", "台球",
        "温泉", "洗浴", "汗蒸",
        "机票", "酒店", "民宿", "客栈",
        "研学", "夏令营", "冬令营",
        "卡牌", "桌游", "电玩",
        "空中游览", "直升机",
        "俱乐部", "会所",
    ],
    "tag_keywords": [
        "演出", "演唱会", "音乐会", "话剧", "音乐剧",
        "展览", "展会", "展销",
        "密室", "剧本杀",
        "电影", "影院",
        "脱口秀", "喜剧", "相声",
        "综艺", "比赛", "电竞",
    ],
    "category_keywords": [
        "演唱会", "演出", "展览", "展会",
        "电影院", "KTV", "酒吧",
        "密室", "剧本杀",
        "脱口秀", "喜剧", "相声",
        "电竞", "游戏",
    ]
}


def parse_csv_field(field: str) -> Any:
    """解析CSV字段，处理列表类型字段"""
    if not field or field.strip() == "":
        return []

    field = field.strip()

    # 尝试解析为Python列表
    if field.startswith("[") and field.endswith("]"):
        try:
            return ast.literal_eval(field)
        except (ValueError, SyntaxError):
            # 解析失败，按逗号分割
            inner = field[1:-1].strip()
            if inner:
                items = [item.strip().strip("'\"") for item in inner.split(",")]
                return [item for item in items if item]
            return []

    return field


def should_exclude(row: Dict[str, str]) -> bool:
    """
    判断该景点是否应该被过滤掉

    过滤规则：
    1. 景区名称包含演唱会、演出等关键词
    2. 景区标签包含演出、展览等关键词
    3. 无景区名称或无介绍
    """
    name = row.get("景区名称", "").strip()
    intro = row.get("介绍", "").strip()
    tags = row.get("景区标签列表", "")
    category = row.get("区域名称", "")

    # 无效数据过滤
    if not name:
        return True
    if not intro or len(intro) < 20:
        return True

    # 名称关键词过滤
    name_lower = name.lower()
    for kw in EXCLUDE_KEYWORDS["name_keywords"]:
        if kw.lower() in name_lower:
            return True

    # 过滤 "城市 · 演出名" 模式（中间带 · 的通常是演出）
    if " · " in name and any(kw in name for kw in ["《", "》", "专场", "季", "企划"]):
        return True

    # 标签关键词过滤
    tag_list = parse_csv_field(tags)
    if isinstance(tag_list, list):
        for tag in tag_list:
            tag_str = str(tag)
            for kw in EXCLUDE_KEYWORDS["tag_keywords"]:
                if kw in tag_str:
                    return True

    # 区域/类别关键词过滤
    for kw in EXCLUDE_KEYWORDS["category_keywords"]:
        if kw in category:
            return True

    return False


def convert_row_to_doc(row: Dict[str, str], city: str) -> Dict[str, Any]:
    """将CSV行转换为标准知识库文档格式"""
    name = row.get("景区名称", "").strip()
    address = row.get("景区地址", "").strip()
    rating = row.get("景区评论得分", "").strip()
    price = row.get("景区门票价格", "").strip()
    is_free = row.get("景区是否免费", "").strip()
    open_time = row.get("开放时间", "").strip()
    intro = row.get("介绍", "").strip()
    features = row.get("景区特点描述", "").strip()
    tags = parse_csv_field(row.get("景区标签列表", ""))
    level = row.get("星级", "").strip()
    image = row.get("景区展示图片", "").strip()
    area = row.get("区域名称", "").strip()
    hot = row.get("景区热度", "").strip()
    phone = row.get("联系电话", "").strip()
    preferential = parse_csv_field(row.get("优待政策", ""))
    facilities = parse_csv_field(row.get("服务设施", ""))
    lat = row.get("景区纬度", "").strip()
    lon = row.get("景区经度", "").strip()

    # 处理门票价格
    try:
        price_float = float(price) if price else 0.0
    except ValueError:
        price_float = 0.0

    if is_free.lower() == "true" or price_float == 0.0:
        price_display = "免费"
        price_value = 0
    else:
        price_display = f"{int(price_float)}元"
        price_value = int(price_float)

    # 处理评分
    try:
        rating_float = float(rating) if rating else 0.0
    except ValueError:
        rating_float = 0.0

    # 处理经纬度
    try:
        longitude = float(lon) if lon else None
    except ValueError:
        longitude = None
    try:
        latitude = float(lat) if lat else None
    except ValueError:
        latitude = None

    # 构建content文本
    tag_str = ", ".join(tags) if isinstance(tags, list) else str(tags)
    feature_str = features if features else "暂无"

    content = (
        f"景点名称：{name}\n"
        f"城市：{city}\n"
        f"地址：{address}\n"
        f"评分：{rating}\n"
        f"门票价格：{price_display}\n"
        f"开放时间：{open_time}\n"
        f"景区特点：{feature_str}\n"
        f"景区标签：{tag_str}\n"
        f"景区介绍：{intro}\n"
        f"优待政策：{preferential}\n"
        f"服务设施：{facilities}"
    )

    # 构建metadata
    metadata = {
        "title": name,
        "city": city,
        "address": address,
        "location": {
            "longitude": longitude,
            "latitude": latitude
        } if longitude and latitude else None,
        "visit_duration": _estimate_visit_duration(tags, intro, features),
        "description": feature_str,
        "category": _determine_category(tags, features, area),
        "rating": rating_float,
        "price": price_value,
        "tags": tag_str,
        "open_time": open_time,
        "level": level,
        "image": image,
        "area": area,
        "hot": hot,
        "phone": phone,
        "preferential_policy": preferential if isinstance(preferential, list) else [str(preferential)],
        "service_facilities": facilities if isinstance(facilities, list) else [str(facilities)]
    }

    return {
        "content": content,
        "metadata": metadata
    }


def _estimate_visit_duration(tags: Any, intro: str, features: str) -> int:
    """根据景点类型估算游览时长（分钟）"""
    tag_str = str(tags) + str(features)

    duration_map = {
        "主题公园": 240,
        "乐园": 240,
        "度假区": 240,
        "博物馆": 120,
        "美术馆": 90,
        "展览馆": 90,
        "公园": 120,
        "园林": 90,
        "古镇": 150,
        "老街": 90,
        "胡同": 60,
        "寺庙": 60,
        "教堂": 30,
        "广场": 60,
        "塔": 60,
        "桥": 30,
        "山": 180,
        "湖": 120,
        "海滩": 120,
        "步行街": 90,
        "商业街": 90,
        "地标": 60,
    }

    for keyword, duration in duration_map.items():
        if keyword in tag_str:
            return duration

    # 根据介绍长度估算
    if len(intro) > 500:
        return 120
    elif len(intro) > 200:
        return 90
    else:
        return 60


def _determine_category(tags: Any, features: str, area: str) -> str:
    """根据标签和特点确定景点类别"""
    tag_str = str(tags) + str(features) + str(area)

    category_map = {
        "历史文化": ["历史", "古迹", "文物", "古建", "胡同", "古镇", "遗址", "陵", "宫", "庙", "寺", "塔", "教堂"],
        "自然风光": ["山", "湖", "河", "海", "海滩", "公园", "湿地", "瀑布", "森林", "岛", "景"],
        "主题乐园": ["乐园", "主题公园", "度假区", "游乐园", "水族馆", "动物园", "植物园"],
        "博物馆": ["博物馆", "美术馆", "展览馆", "科技馆", "纪念馆"],
        "城市漫步": ["步行街", "商业街", "老街", "广场", "地标", "夜景", "城市"],
        "艺术文化": ["艺术", "文化", "创意", "设计", "画", "音乐"],
    }

    for category, keywords in category_map.items():
        for kw in keywords:
            if kw in tag_str:
                return category

    return "其他"


def clean_city_data(city: str) -> List[Dict[str, Any]]:
    """
    清洗单个城市的景点数据

    Returns:
        清洗后的文档列表
    """
    csv_path = os.path.join(ATTRACTIONS_DIR, f"{city}.csv")

    if not os.path.exists(csv_path):
        print(f"  ⚠️ {city}.csv 不存在，跳过")
        return []

    print(f"  📖 读取 {city}.csv ...")

    # 读取CSV
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"     原始数据: {len(rows)} 条")

    # 清洗数据
    cleaned_docs = []
    seen_names: Set[str] = set()  # 用于去重
    excluded_count = 0
    duplicate_count = 0

    for row in rows:
        # 过滤无效/演出类数据
        if should_exclude(row):
            excluded_count += 1
            continue

        name = row.get("景区名称", "").strip()

        # 去重（按景点名称）
        if name in seen_names:
            duplicate_count += 1
            continue
        seen_names.add(name)

        # 转换为标准格式
        doc = convert_row_to_doc(row, city)
        cleaned_docs.append(doc)

    print(f"     ✅ 清洗后: {len(cleaned_docs)} 条")
    print(f"     🗑️ 过滤无效/演出: {excluded_count} 条")
    print(f"     🔁 去重: {duplicate_count} 条")

    return cleaned_docs


def save_to_json(city: str, docs: List[Dict[str, Any]]) -> str:
    """保存清洗后的数据为JSON文件"""
    json_path = os.path.join(ATTRACTIONS_DIR, f"{city}.json")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)

    print(f"  💾 已保存: {json_path} ({len(docs)} 条)")
    return json_path


def load_to_qdrant(all_docs: List[Dict[str, Any]]):
    """将清洗后的数据存入Qdrant向量库"""
    print("\n⏳ 正在连接Qdrant向量数据库...")

    try:
        from app.rag.vector_store import QdrantVectorStore

        # 优先使用环境变量中的云Qdrant配置
        qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")
        print(f"  连接: {qdrant_url[:50]}...")

        vector_store = QdrantVectorStore(url=qdrant_url, api_key=qdrant_api_key)
        print("✅ Qdrant连接成功")

        # 删除旧集合（如果存在）
        collection_name = vector_store.default_collection
        try:
            if vector_store.client.collection_exists(collection_name=collection_name):
                print(f"⏳ 删除旧集合: {collection_name}")
                vector_store.delete_collection(collection_name)
        except Exception as e:
            print(f"⚠️ 删除旧集合失败（可忽略）: {e}")

        # 添加新数据
        print(f"⏳ 正在向向量库添加 {len(all_docs)} 条文档...")
        vector_store.add_documents(all_docs)

        # 获取统计信息
        stats = vector_store.get_collection_stats()
        print(f"✅ 向量库更新完成！")
        print(f"   - 集合名称: {stats['collection_name']}")
        print(f"   - 向量数量: {stats['vector_count']}")
        print(f"   - 向量维度: {stats['dimension']}")

        # 测试检索
        print("\n🔍 测试检索...")
        test_queries = ["历史文化景点", "自然风光", "主题乐园", "免费景点"]
        for query in test_queries:
            results = vector_store.search(query, k=3)
            print(f"\n  查询: '{query}'")
            for i, result in enumerate(results, 1):
                title = result.get('metadata', {}).get('title', '未知')
                city = result.get('metadata', {}).get('city', '?')
                score = result.get('score', 0)
                print(f"    {i}. [{city}] {title} (相似度: {score:.4f})")

    except Exception as e:
        print(f"❌ Qdrant操作失败: {e}")
        print("请确保Qdrant服务已启动：docker run -p 6333:6333 qdrant/qdrant")


def main():
    """主函数"""
    print("=" * 70)
    print("🔧 景点数据清洗工具")
    print("=" * 70)
    print(f"目标城市: {', '.join(TARGET_CITIES)}")
    print(f"数据目录: {ATTRACTIONS_DIR}")
    print()

    all_cleaned_docs = []
    total_original = 0
    total_cleaned = 0

    for city in TARGET_CITIES:
        print(f"\n{'─' * 50}")
        print(f"📍 处理城市: {city}")
        print(f"{'─' * 50}")

        # 清洗数据
        cleaned_docs = clean_city_data(city)

        if cleaned_docs:
            # 保存JSON
            save_to_json(city, cleaned_docs)
            all_cleaned_docs.extend(cleaned_docs)
            total_cleaned += len(cleaned_docs)

    # 汇总
    print(f"\n{'=' * 70}")
    print(f"📊 清洗汇总")
    print(f"{'=' * 70}")
    print(f"  总城市数: {len(TARGET_CITIES)}")
    print(f"  清洗后总文档数: {total_cleaned}")

    if not all_cleaned_docs:
        print("\n❌ 没有清洗后的数据，退出")
        return

    # 询问是否存入向量库
    print(f"\n{'=' * 70}")
    print(f"是否将清洗后的数据存入Qdrant向量库？")
    print(f"{'=' * 70}")

    try:
        choice = input("请输入 y/n: ").strip().lower()
    except EOFError:
        choice = "y"  # 非交互模式下自动执行

    if choice == "y":
        load_to_qdrant(all_cleaned_docs)
    else:
        print("已跳过向量库导入。")
        print(f"JSON文件已保存到: {ATTRACTIONS_DIR}")
        print("后续可手动运行: python scripts/init_knowledge_base.py")

    print(f"\n{'=' * 70}")
    print(f"✅ 数据清洗完成！")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
