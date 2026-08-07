"""
数据采集器 - Data Collector
从多种数据源采集景点和旅游知识
"""

from typing import List, Dict, Optional
import json
import os


class AttractionDataCollector:
    """景点数据采集器"""

    def __init__(
        self,
        knowledge_base_path: str = "app/knowledge_base/attractions",
        amap_service=None
    ):
        """
        初始化数据采集器

        Args:
            knowledge_base_path: 知识库文件路径
            amap_service: 高德地图服务实例（可选）
        """
        self.knowledge_base_path = knowledge_base_path
        self.amap_service = amap_service

        # 景点分类关键词（综合category和tags）
        self.category_keywords = {
            "历史文化": ["博物馆", "寺庙", "古建筑", "遗址", "宫殿", "园林", "塔", "阁", 
                        "历史建筑", "古镇古村", "历史古迹", "名人故居", "文化遗址"],
            "自然风光": ["公园", "山", "湖", "海", "森林", "瀑布", "峡谷", "草原",
                        "自然山水", "园林花园", "赏花胜地", "海滨玩水"],
            "现代地标": ["广场", "体育馆", "剧院", "CBD", "摩天楼", "展览馆",
                        "地标观景", "城市漫步"],
            "休闲娱乐": ["游乐场", "温泉", "采摘园", "动物园", "植物园", "海洋馆",
                        "亲子同乐", "遛娃宝藏地", "亲近动物", "度假村", "夜游观景"],
            "宗教寺庙": ["寺庙", "道观", "教堂", "清真寺", "神社", "佛寺"],
            "名校学府": ["大学", "学院", "校园", "图书馆"],
            "美食": ["美食", "小吃", "餐厅", "夜市"],
            "购物": ["购物", "步行街", "商场"]
        }

    def collect_from_amap(
        self,
        city: str,
        categories: Optional[List[str]] = None,
        max_pois: int = 20
    ) -> List[Dict]:
        """
        从高德API采集景点数据

        Args:
            city: 城市名称
            categories: 景点类别列表（默认包含主要类别）
            max_pois: 每个类别最大POI数量

        Returns:
            景点数据列表
        """
        if categories is None:
            categories = ["风景名胜", "博物馆", "寺庙", "公园"]

        if self.amap_service is None:
            # 如果没有高德服务，返回空列表并提示
            print(f"⚠️ 未配置高德服务，请先设置 amap_service")
            return []

        all_attractions = []

        for category in categories:
            try:
                print(f"   - 搜索 {category}...")
                # 搜索POI
                pois = self.amap_service.search_poi(category, city)
                print(f"     找到 {len(pois)} 个POI")

                for idx, poi in enumerate(pois[:max_pois]):
                    # 获取详细信息
                    try:
                        print(f"     获取详情 {idx+1}/{min(len(pois), max_pois)}: {poi.name}...")
                        detail = self.amap_service.get_poi_detail(poi.id)

                        attraction = self._convert_to_knowledge(detail, city)
                        if attraction:
                            all_attractions.append(attraction)
                    except Exception as e:
                        # 单个POI失败不影响整体
                        print(f"       失败: {e}")
                        continue

            except Exception as e:
                print(f"❌ 采集 {city} {category} 失败: {e}")
                continue

        return all_attractions

    def load_from_json(self, city: str) -> List[Dict]:
        """
        从JSON文件加载景点数据（并重新计算分类、票价和游览时长）

        Args:
            city: 城市名称

        Returns:
            景点数据列表
        """
        filepath = os.path.join(self.knowledge_base_path, f"{city}.json")

        if not os.path.exists(filepath):
            print(f"⚠️ 文件不存在: {filepath}")
            return []

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            # 转换为标准格式，并补充分类、票价和游览时长
            attractions = []
            for item in data:
                metadata = item.get("metadata", {})
                
                # 获取分类、POI类型和tags
                original_category = metadata.get("category", "其他")
                poi_type = metadata.get("poi_type", "")
                tags = metadata.get("tags", "")
                
                title = metadata.get("title", "")
                
                # 综合category、tags和title重新分类
                category = self._reclassify_attraction(original_category, tags, title, poi_type)
                
                # 优先使用已有的票价信息，没有则估算
                price = metadata.get("price")
                if not price:
                    price, _ = self._estimate_price_and_duration(category, poi_type, title)
                
                # 优先使用已有的游览时长，没有则估算
                duration = metadata.get("duration")
                if not duration:
                    _, duration = self._estimate_price_and_duration(category, poi_type, title)
                
                # 更新metadata
                metadata["category"] = category
                metadata["price"] = price
                metadata["duration"] = duration
                
                attraction = {
                    "content": item.get("content", ""),
                    "metadata": metadata
                }
                attractions.append(attraction)

            print(f"✅ 从 {filepath} 加载了 {len(attractions)} 条数据（已重新分类并补充票价和游览时长）")
            return attractions

        except Exception as e:
            print(f"❌ 加载 {filepath} 失败: {e}")
            return []

    def _reclassify_attraction(self, original_category: str, tags: str, title: str, poi_type: str) -> str:
        """
        综合category、tags和title重新分类景点

        Args:
            original_category: 原始分类
            tags: 标签字符串（如"['历史建筑', '城市漫步']"）
            title: 景点名称
            poi_type: POI类型

        Returns:
            重新分类后的类别
        """
        text_to_check = f"{tags} {title} {poi_type} {original_category}"
        
        for category, keywords in self.category_keywords.items():
            for keyword in keywords:
                if keyword in text_to_check:
                    return category
        
        return original_category

    def load_from_knowledge_base(self, city: Optional[str] = None) -> List[Dict]:
        """
        从本地知识库加载数据

        Args:
            city: 城市名称（None表示加载所有城市）

        Returns:
            景点数据列表
        """
        all_data = []

        if city:
            # 加载指定城市
            return self.load_from_json(city)

        # 加载所有城市
        if not os.path.exists(self.knowledge_base_path):
            print(f"⚠️ 知识库目录不存在: {self.knowledge_base_path}")
            return []

        for filename in os.listdir(self.knowledge_base_path):
            if filename.endswith(".json"):
                city_name = filename.replace(".json", "")
                city_data = self.load_from_json(city_name)
                all_data.extend(city_data)

        return all_data

    def collect_all(
        self,
        cities: List[str],
        use_api: bool = False,
        save_to_local: bool = True
    ) -> List[Dict]:
        """
        综合采集多个城市的数据

        Args:
            cities: 城市列表
            use_api: 是否使用API采集（优先使用本地数据）
            save_to_local: 是否将采集的数据保存到本地

        Returns:
            所有景点数据
        """
        all_attractions = []

        for city in cities:
            print(f"\n📍 采集 {city} 的数据...")

            if use_api and self.amap_service:
                # 使用API采集
                attractions = self.collect_from_amap(city)
                
                # 如果采集到数据且需要保存到本地
                if attractions and save_to_local:
                    self.save_to_json(attractions, city)
            else:
                # 从本地加载
                attractions = self.load_from_json(city)

            if attractions:
                all_attractions.extend(attractions)
                print(f"   ✅ 成功采集 {len(attractions)} 条数据")
            else:
                print(f"   ⚠️ 无数据")

        print(f"\n📊 共采集 {len(all_attractions)} 条数据")
        return all_attractions

    def _convert_to_knowledge(self, poi, city: str) -> Optional[Dict]:
        """
        将POI数据转换为知识库格式（使用API获取的真实信息）

        Args:
            poi: POI对象
            city: 城市名称

        Returns:
            知识库格式的数据
        """
        try:
            # 构建内容文本
            content_parts = []

            if poi.name:
                content_parts.append(poi.name)

            if poi.address:
                content_parts.append(f"位于{poi.address}")

            if poi.type:
                content_parts.append(f"属于{poi.type}类别")

            if poi.tel:
                content_parts.append(f"联系电话：{poi.tel}")

            if poi.rating:
                content_parts.append(f"评分：{poi.rating}分")

            content = "。".join(content_parts) + "。"

            # 提取分类
            category = self._extract_category(poi.type or "")

            # 从API获取的信息提取票价（优先使用真实数据，没有则估算）
            price = self._parse_cost_from_api(poi)
            if not price:
                price, _ = self._estimate_price_and_duration(category, poi.type or "", poi.name or "")

            # 估算游览时长
            _, duration = self._estimate_price_and_duration(category, poi.type or "", poi.name or "")

            # 构建元数据
            metadata = {
                "title": poi.name or "未知景点",
                "city": city,
                "type": "attraction",
                "category": category,
                "rating": float(poi.rating) if poi.rating else 0.0,
                "location": str(poi.location) if poi.location else "",
                "address": poi.address or "",
                "poi_type": poi.type or "",
                "price": price,
                "duration": duration,
                "cost_level": poi.cost or "",  # 保存原始消费等级
                "tag": poi.tag or ""  # 保存标签信息
            }

            return {
                "content": content,
                "metadata": metadata
            }

        except Exception as e:
            print(f"❌ 转换POI失败: {e}")
            return None

    def _parse_cost_from_api(self, poi) -> Optional[str]:
        """
        从POI详情中解析票价信息

        Args:
            poi: POI对象

        Returns:
            票价字符串（如"免费"、"30元"等）
        """
        # 优先检查是否有直接的票价信息
        if hasattr(poi, 'cost') and poi.cost:
            cost = poi.cost
            # 根据消费等级估算票价
            if cost == "免费":
                return "免费"
            elif cost == "¥":
                return "20元"
            elif cost == "¥¥":
                return "50元"
            elif cost == "¥¥¥":
                return "100元"
            elif cost == "¥¥¥¥":
                return "200元"
            else:
                # 如果是其他格式，直接返回
                return cost
        
        return None

    def _estimate_price_and_duration(self, category: str, poi_type: str, title: str = "") -> tuple:
        """
        根据景点类型估算票价和游览时长（基于真实市场价）

        Args:
            category: 景点分类
            poi_type: POI类型字符串
            title: 景点名称（可选）

        Returns:
            (票价, 游览时长)
        """
        # 默认值
        default_price = "免费"
        default_duration = 60  # 默认60分钟

        # 根据分类估算（使用真实市场价）
        if category == "历史文化":
            if "故宫" in poi_type or "紫禁城" in poi_type or "故宫" in title:
                return "60元", 240  # 故宫旺季票价60元
            elif "天坛" in poi_type or "天坛" in title:
                return "35元", 120  # 天坛门票35元
            elif "颐和园" in poi_type or "颐和园" in title:
                return "30元", 180  # 颐和园门票30元
            elif "八达岭" in poi_type or "长城" in poi_type:
                return "40元", 180  # 八达岭长城40元
            elif "圆明园" in poi_type or "圆明园" in title:
                return "10元", 120  # 圆明园门票10元
            elif "恭王府" in poi_type or "恭王府" in title:
                return "40元", 90  # 恭王府门票40元
            elif "博物馆" in poi_type:
                if "国家博物馆" in poi_type or "自然博物馆" in poi_type:
                    return "免费", 120  # 国家级博物馆免费
                else:
                    return "20元", 90  # 其他博物馆约20元
            elif "世界遗产" in poi_type:
                return "50元", 180  # 世界遗产景点
            else:
                return "20元", 120  # 一般历史文化景点

        elif category == "自然风光":
            if "天坛" in poi_type or "天坛" in category:
                return "35元", 120  # 天坛门票35元
            elif "颐和园" in poi_type:
                return "30元", 180  # 颐和园门票30元
            elif "北海公园" in poi_type:
                return "10元", 90  # 北海公园10元
            elif "景山" in poi_type:
                return "2元", 60  # 景山公园2元
            elif "香山" in poi_type:
                return "10元", 180  # 香山门票10元
            elif "公园" in poi_type:
                return "免费", 60  # 城市公园通常免费
            elif "山" in poi_type or "森林" in poi_type or "国家森林公园" in poi_type:
                return "50元", 180  # 山岳类景点
            elif "植物园" in poi_type:
                return "10元", 120  # 植物园
            else:
                return "30元", 120

        elif category == "现代地标":
            if "东方明珠" in poi_type or "东方明珠" in title:
                return "199元", 120  # 东方明珠门票
            elif "金茂大厦" in poi_type or "金茂大厦" in title:
                return "120元", 60  # 金茂大厦观光厅
            elif "观光厅" in poi_type:
                return "120元", 60  # 高层观光厅
            elif "广场" in poi_type:
                return "免费", 30  # 广场通常免费
            elif "外滩" in poi_type:
                return "免费", 60  # 外滩免费
            elif "展览馆" in poi_type:
                return "50元", 120
            elif "豫园" in poi_type or "豫园" in title:
                return "40元", 120  # 豫园门票
            else:
                return "30元", 60

        elif category == "休闲娱乐":
            if "游乐场" in poi_type or "欢乐谷" in poi_type:
                return "299元", 360  # 大型游乐场
            elif "动物园" in poi_type:
                return "40元", 180  # 动物园
            elif "海洋馆" in poi_type or "水族馆" in poi_type:
                return "160元", 180  # 海洋馆
            else:
                return "100元", 120

        elif category == "宗教寺庙":
            if "雍和宫" in poi_type:
                return "25元", 60  # 雍和宫门票25元
            elif "潭柘寺" in poi_type:
                return "55元", 90  # 潭柘寺门票55元
            elif "戒台寺" in poi_type:
                return "45元", 90  # 戒台寺门票45元
            elif "玉佛寺" in poi_type:
                return "20元", 60  # 玉佛寺门票20元
            else:
                return "免费", 60  # 大部分寺庙免费

        elif category == "名校学府":
            return "免费", 90  # 校园免费参观

        else:
            # 默认情况下也检查景点名称
            if "恭王府" in poi_type or "恭王府" in title:
                return "40元", 90
            elif "东方明珠" in poi_type or "东方明珠" in title:
                return "199元", 120
            elif "金茂大厦" in poi_type or "金茂大厦" in title:
                return "120元", 60
            elif "豫园" in poi_type or "豫园" in title:
                return "40元", 120
            elif "静安寺" in poi_type or "静安寺" in title:
                return "20元", 60
            else:
                return default_price, default_duration

    def _extract_category(self, poi_type: str) -> str:
        """
        从POI类型中提取分类

        Args:
            poi_type: POI类型字符串

        Returns:
            分类名称
        """
        for category, keywords in self.category_keywords.items():
            for keyword in keywords:
                if keyword in poi_type:
                    return category

        return "其他"

    def save_to_json(self, data: List[Dict], city: str) -> bool:
        """
        保存数据到JSON文件

        Args:
            data: 景点数据列表
            city: 城市名称

        Returns:
            是否保存成功
        """
        # 确保目录存在
        os.makedirs(self.knowledge_base_path, exist_ok=True)

        filepath = os.path.join(self.knowledge_base_path, f"{city}.json")

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            print(f"✅ 已保存 {len(data)} 条数据到 {filepath}")
            return True

        except Exception as e:
            print(f"❌ 保存失败: {e}")
            return False


def get_default_collector() -> AttractionDataCollector:
    """获取默认配置的数据采集器"""
    return AttractionDataCollector(
        knowledge_base_path="app/knowledge_base/attractions"
    )


# ==================== 测试算例 ====================
if __name__ == "__main__":
    print("=" * 50)
    print("🧪 数据采集器测试")
    print("=" * 50)

    # 尝试初始化高德服务（需要配置环境变量 AMAP_MAPS_API_KEY）
    amap_service = None
    use_real_api = False
    
    try:
        from app.services.amap_service import AmapService
        amap_service = AmapService()
        use_real_api = amap_service._use_real_api
        print(f"\n🌐 高德服务状态: {'已配置API密钥' if use_real_api else '未配置API密钥'}")
    except Exception as e:
        print(f"\n⚠️ 无法初始化高德服务: {e}")

    # 初始化采集器
    collector = AttractionDataCollector(amap_service=amap_service)

    # 测试1: 加载本地JSON数据
    print("\n📂 测试1: 加载本地JSON数据")
    print("-" * 30)
    attractions = collector.load_from_json("北京")
    print(f"加载到 {len(attractions)} 条数据")

    if attractions:
        print("\n前3条数据预览:")
        for i, attr in enumerate(attractions[:3], 1):
            print(f"  {i}. {attr['metadata']['title']}")
            print(f"     分类: {attr['metadata']['category']}")
            print(f"     评分: {attr['metadata']['rating']}")
            print()

    # 测试2: 加载所有城市数据
    print("\n📂 测试2: 加载所有城市数据")
    print("-" * 30)
    all_data = collector.load_from_knowledge_base()
    print(f"共加载 {len(all_data)} 条数据")

    # 测试3: 综合采集（根据是否有API密钥决定是否调用API）
    print("\n📂 测试3: 综合采集多个城市")
    print("-" * 30)
    cities = ["北京", "上海"]
    
    if use_real_api:
        print("✅ 将使用真实API采集数据")
        collected = collector.collect_all(cities, use_api=True)
    else:
        print("⚠️ API密钥未配置，仅使用本地数据")
        collected = collector.collect_all(cities, use_api=False)
    
    print(f"采集完成，共 {len(collected)} 条数据")

    # 测试4: API采集（仅当有API密钥时）
    if use_real_api:
        print("\n📂 测试4: 真实API采集")
        print("-" * 30)
        try:
            api_data = collector.collect_from_amap("北京", categories=["风景名胜"], max_pois=5)
            print(f"从API采集到 {len(api_data)} 条数据")
            
            if api_data:
                print("\nAPI采集结果预览:")
                for i, attr in enumerate(api_data[:3], 1):
                    print(f"  {i}. {attr['metadata']['title']}")
                    print(f"     地址: {attr['metadata'].get('address', '未知')}")
                    print(f"     分类: {attr['metadata']['category']}")
                    print()
        except Exception as e:
            print(f"❌ API采集失败: {e}")

    # 测试5: 数据分类统计
    print("\n📊 测试5: 数据分类统计")
    print("-" * 30)
    category_count = {}
    for attr in all_data:
        cat = attr['metadata'].get('category', '其他')
        category_count[cat] = category_count.get(cat, 0) + 1

    for cat, count in sorted(category_count.items(), key=lambda x: x[1], reverse=True):
        print(f"  {cat}: {count} 条")

    print("\n" + "=" * 50)
    print("✅ 测试完成")
    print("=" * 50)
