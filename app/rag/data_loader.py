import json
import os
from typing import List, Dict, Any


class KnowledgeBaseLoader:
    """知识库数据读取器"""
    
    def __init__(self, base_path: str = "app/knowledge_base"):
        self.base_path = base_path
        self._ensure_directories()
    
    def _ensure_directories(self):
        """确保必要的目录存在"""
        os.makedirs(self.base_path, exist_ok=True)
        os.makedirs(os.path.join(self.base_path, "attractions"), exist_ok=True)
        os.makedirs(os.path.join(self.base_path, "restaurants"), exist_ok=True)
        os.makedirs(os.path.join(self.base_path, "guides"), exist_ok=True)
    
    def load_attraction_data(self, city: str) -> List[Dict[str, Any]]:
        """加载指定城市的景点知识"""
        path = os.path.join(self.base_path, "attractions", f"{city}.json")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, list) else [data]
        return []
    
    def load_restaurant_data(self, city: str) -> List[Dict[str, Any]]:
        """加载指定城市的餐厅知识"""
        path = os.path.join(self.base_path, "restaurants", f"{city}.json")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, list) else [data]
        return []
    
    def load_guide_data(self, city: str) -> List[Dict[str, Any]]:
        """加载指定城市的旅行攻略"""
        path = os.path.join(self.base_path, "guides", f"{city}.json")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, list) else [data]
        return []
    
    def load_all_knowledge(self) -> List[Dict[str, Any]]:
        """加载所有知识库数据"""
        all_data = []
        
        for category in ["attractions", "restaurants", "guides"]:
            category_path = os.path.join(self.base_path, category)
            if not os.path.exists(category_path):
                continue
            
            for filename in os.listdir(category_path):
                if filename.endswith(".json"):
                    filepath = os.path.join(category_path, filename)
                    try:
                        with open(filepath, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            if isinstance(data, list):
                                all_data.extend(data)
                            else:
                                all_data.append(data)
                    except Exception as e:
                        print(f"Error loading {filepath}: {e}")
        
        return all_data
    
    def save_attraction_data(self, city: str, data: List[Dict[str, Any]]) -> None:
        """保存景点数据到知识库"""
        path = os.path.join(self.base_path, "attractions", f"{city}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def save_restaurant_data(self, city: str, data: List[Dict[str, Any]]) -> None:
        """保存餐厅数据到知识库"""
        path = os.path.join(self.base_path, "restaurants", f"{city}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def save_guide_data(self, city: str, data: List[Dict[str, Any]]) -> None:
        """保存旅行攻略数据"""
        path = os.path.join(self.base_path, "guides", f"{city}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def get_supported_cities(self) -> List[str]:
        """获取支持的城市列表"""
        cities = set()
        for category in ["attractions", "restaurants", "guides"]:
            category_path = os.path.join(self.base_path, category)
            if os.path.exists(category_path):
                for filename in os.listdir(category_path):
                    if filename.endswith(".json"):
                        city = filename[:-5]  # 去掉 .json
                        cities.add(city)
        return sorted(list(cities))