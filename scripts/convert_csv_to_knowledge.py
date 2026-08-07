import csv
import json
import os
from typing import List, Dict
import ast


class CSVToKnowledgeConverter:
    def __init__(self, csv_dir: str = "app/knowledge_base/attractions"):
        self.csv_dir = csv_dir
        self.category_keywords = {
            "历史文化": ["博物馆", "寺庙", "古建筑", "遗址", "宫殿", "园林", "塔", "阁", "故居"],
            "自然风光": ["公园", "山", "湖", "海", "森林", "瀑布", "峡谷", "草原", "景区"],
            "现代地标": ["广场", "体育馆", "剧院", "CBD", "摩天楼", "展览馆", "中心"],
            "休闲娱乐": ["游乐场", "温泉", "采摘园", "动物园", "植物园", "海洋馆", "乐园"],
            "宗教寺庙": ["寺庙", "道观", "教堂", "清真寺", "佛寺"],
            "名校学府": ["大学", "学院", "校园", "图书馆"]
        }

    def _extract_category(self, tags: str, name: str) -> str:
        tag_list = []
        if tags:
            try:
                tag_list = ast.literal_eval(tags)
            except:
                pass
        
        all_text = f"{name} {' '.join(tag_list)}"
        
        for category, keywords in self.category_keywords.items():
            for keyword in keywords:
                if keyword in all_text:
                    return category
        return "其他"

    def _estimate_duration(self, category: str) -> int:
        duration_map = {
            "历史文化": 120,
            "自然风光": 180,
            "现代地标": 60,
            "休闲娱乐": 240,
            "宗教寺庙": 60,
            "名校学府": 90,
            "其他": 90
        }
        return duration_map.get(category, 90)

    def _convert_row(self, row: Dict, city: str) -> Dict:
        try:
            longitude = float(row.get("景区经度", 0)) if row.get("景区经度") else 0
            latitude = float(row.get("景区纬度", 0)) if row.get("景区纬度") else 0
        except:
            longitude, latitude = 0, 0

        try:
            rating = float(row.get("景区评论得分", 0)) if row.get("景区评论得分") else 0
        except:
            rating = 0

        try:
            price = float(row.get("景区门票价格", 0)) if row.get("景区门票价格") else 0
        except:
            price = 0

        tags = row.get("景区标签列表", "")
        category = self._extract_category(tags, row.get("景区名称", ""))
        duration = self._estimate_duration(category)

        description = row.get("景区特点描述", "")
        try:
            desc_list = ast.literal_eval(description)
            description = " ".join(desc_list) if isinstance(desc_list, list) else description
        except:
            pass

        content = f"""
景点名称：{row.get('景区名称', '')}
城市：{city}
地址：{row.get('景区地址', '')}
评分：{rating}
门票价格：{price}元
开放时间：{row.get('开放时间', '')}
景区特点：{description}
景区标签：{tags}
景区榜单：{row.get('景区榜单', '')}
景区介绍：{row.get('介绍', '')}
优待政策：{row.get('优待政策', '')}
服务设施：{row.get('服务设施', '')}
"""

        metadata = {
            "title": row.get("景区名称", ""),
            "city": city,
            "address": row.get("景区地址", ""),
            "location": {"longitude": longitude, "latitude": latitude},
            "visit_duration": duration,
            "description": description,
            "category": category,
            "rating": rating,
            "price": price,
            "tags": tags,
            "open_time": row.get("开放时间", ""),
            "level": row.get("星级", ""),
            "image": row.get("景区展示图片", ""),
            "area": row.get("区域名称", ""),
            "hot": row.get("景区热度", 0),
            "phone": row.get("联系电话", ""),
            "preferential_policy": row.get("优待政策", ""),
            "service_facilities": row.get("服务设施", "")
        }

        return {"content": content.strip(), "metadata": metadata}

    def convert_city(self, city: str) -> List[Dict]:
        csv_path = os.path.join(self.csv_dir, f"{city}.csv")
        if not os.path.exists(csv_path):
            print(f"⚠️ 文件不存在: {csv_path}")
            return []

        attractions = []
        try:
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("景区名称"):
                        attraction = self._convert_row(row, city)
                        attractions.append(attraction)
            print(f"✅ 转换 {city}: {len(attractions)} 条数据")
        except Exception as e:
            print(f"❌ 转换 {city} 失败: {e}")
        return attractions

    def convert_all(self) -> Dict[str, List[Dict]]:
        result = {}
        csv_files = [f for f in os.listdir(self.csv_dir) if f.endswith(".csv")]
        
        for csv_file in csv_files:
            city = csv_file.replace(".csv", "")
            attractions = self.convert_city(city)
            if attractions:
                result[city] = attractions
        
        return result

    def save_to_json(self, data: Dict[str, List[Dict]]):
        for city, attractions in data.items():
            json_path = os.path.join(self.csv_dir, f"{city}.json")
            try:
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(attractions, f, ensure_ascii=False, indent=2)
                print(f"✅ 保存 {json_path}: {len(attractions)} 条数据")
            except Exception as e:
                print(f"❌ 保存 {json_path} 失败: {e}")


def main():
    converter = CSVToKnowledgeConverter()
    print("="*60)
    print("开始转换CSV数据到知识库格式")
    print("="*60)
    
    all_data = converter.convert_all()
    converter.save_to_json(all_data)
    
    print("\n" + "="*60)
    print("转换完成!")
    print("="*60)
    
    total_count = sum(len(v) for v in all_data.values())
    print(f"\n共转换 {len(all_data)} 个城市，{total_count} 条景点数据")


if __name__ == "__main__":
    main()