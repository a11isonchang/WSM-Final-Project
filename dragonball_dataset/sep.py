import json
import re
from tqdm import tqdm

INPUT_FILE = './dragonball_queries.jsonl'
FILE_ZH = './queries_zh.jsonl'
FILE_EN = './queries_en.jsonl'

def contains_chinese(text):
    # 檢查是否包含中文字符範圍
    return bool(re.search(r'[\u4e00-\u9fa5]', text))

def split_dataset():
    print("✂️ 正在進行語言分流...")
    count_zh = 0
    count_en = 0

    with open(INPUT_FILE, 'r', encoding='utf-8') as f_in, \
         open(FILE_ZH, 'w', encoding='utf-8') as f_zh, \
         open(FILE_EN, 'w', encoding='utf-8') as f_en:
        
        for line in tqdm(f_in):
            try:
                data = json.loads(line)
                # 取得 content，結構相容性處理同前
                content = ""
                if "query" in data and isinstance(data["query"], dict):
                    content = data["query"].get("content", "")
                elif "content" in data:
                    content = data["content"]
                
                if not content: continue

                # 分流邏輯
                if contains_chinese(content):
                    f_zh.write(line)
                    count_zh += 1
                else:
                    f_en.write(line)
                    count_en += 1
            except:
                continue

    print(f"✅ 分流完成！\n中文/混合: {count_zh} 筆 (存入 queries_zh.jsonl)\n純英文: {count_en} 筆 (存入 queries_en.jsonl)")

if __name__ == "__main__":
    split_dataset()
