"""检查知识库中的tags字段格式"""

import json
import os

kb_path = "app/knowledge_base/attractions"

for filename in os.listdir(kb_path):
    if not filename.endswith(".json"):
        continue
    
    filepath = os.path.join(kb_path, filename)
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"\n=== {filename} ===")
    
    categories = {}
    tag_samples = []
    
    for item in data:
        metadata = item['metadata']
        category = metadata.get('category', '其他')
        tags = metadata.get('tags', '')
        
        categories[category] = categories.get(category, 0) + 1
        
        if tags and len(tag_samples) < 3:
            tag_samples.append(tags)
    
    print("分类统计:")
    for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
        print(f"  {cat}: {count}")
    
    print("\nTags示例:")
    for tag in tag_samples:
        print(f"  {tag[:100]}..." if len(tag) > 100 else f"  {tag}")