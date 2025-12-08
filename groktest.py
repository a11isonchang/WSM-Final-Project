import json
import os
from openai import OpenAI
from tqdm import tqdm

# --- 設定區域 ---
INPUT_FILE = './dragonball_dataset/queries_show/test_queries_zh.jsonl'
OUTPUT_FILE = './database/database_test.jsonl'

# 填入你的 OpenRouter API Key
OPENROUTER_API_KEY = "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" 

# 在這裡填入你想用的 OpenRouter 模型名稱
# 例如 xAI 的 Grok (假設是 grok-2 或其他): "x-ai/grok-2-1212"
# 或是 Llama 3.3 70B: "meta-llama/llama-3.3-70b-instruct"
# 或是 Gemini 2.0 Flash (超快): "google/gemini-2.0-flash-exp:free"
MODEL_NAME = "x-ai/grok-4.1-fast"  # 請確認 OpenRouter 上的確切 ID

def extract_keywords_with_openrouter(client, query_text):
    """
    使用 OpenRouter API 提取關鍵字 (JSON Mode)
    """
    system_prompt = """
    You are an expert medical data analyst. 
    Extract key search terms from the query.
    Output purely in JSON format with a single key "keywords".
    Example: {"keywords": ["Hospital A", "Disease B", "2024"]}
    """

    user_prompt = f"Extract keywords from: {query_text}"

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            # 許多商業模型支援 response_format={"type": "json_object"}，這能保證 JSON 格式
            # 如果 Grok 暫時不支援此參數，可以拿掉這行，但通常 Prompt 夠強就沒問題
            response_format={"type": "json_object"}, 
            temperature=0.1, # 降低隨機性，越低越準
        )
        
        raw_output = response.choices[0].message.content.strip()
        
        # 解析 JSON
        json_res = json.loads(raw_output)
        keywords = json_res.get("keywords", [])
        
        return keywords
    except Exception as e:
        print(f"\n[Error] {e}")
        # 如果是 JSON 解析失敗，嘗試簡單的字串處理補救
        return []

def process_dataset():
    # 初始化 OpenAI Client，但指向 OpenRouter
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
        # OpenRouter 建議加這兩個 header 以便他們統計排名
        default_headers={
            "HTTP-Referer": "https://github.com/YourProject", 
            "X-Title": "WSM RAG Preprocessing" 
        }
    )
    
    if not os.path.exists(INPUT_FILE):
        print(f"找不到檔案: {INPUT_FILE}")
        return

    # 計算總行數以便顯示進度條
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)

    print(f"🚀 使用模型 [{MODEL_NAME}] 透過 OpenRouter 開始處理...")

    with open(INPUT_FILE, 'r', encoding='utf-8') as f_in, \
         open(OUTPUT_FILE, 'a', encoding='utf-8') as f_out:
        
        for line in tqdm(f_in, total=total_lines):
            try:
                line = line.strip()
                if not line: continue
                
                data = json.loads(line)
                
                # 資料結構解析 (加上容錯)
                content = None
                q_id = None
                q_type = None
                
                # 嘗試從標準結構讀取
                if "query" in data and isinstance(data["query"], dict):
                    content = data["query"].get("content")
                    q_id = data["query"].get("query_id")
                    q_type = data["query"].get("query_type")
                # 嘗試從扁平結構讀取 (如果有的話)
                elif "content" in data:
                    content = data["content"]
                    q_id = data.get("query_id")
                    q_type = data.get("query_type")

                if content:
                    # 呼叫 API
                    keywords = extract_keywords_with_openrouter(client, content)
                    
                    # Determine unsolve
                    unsolve = 0
                    if q_type in ["Irrelevant Unsolvable Question", "无关无解问"]:
                        unsolve = 1

                    # 建立新資料
                    new_record = {
                        "query_id": q_id,
                        "content": content,
                        "keywords": keywords,
                        "unsolve": unsolve
                    }
                    
                    f_out.write(json.dumps(new_record, ensure_ascii=False) + "\n")
                    
            except json.JSONDecodeError:
                continue
            except Exception as e:
                print(f"處理單行時發生未預期錯誤: {e}")
                continue

    print(f"\n✅ 處理完成！檔案已儲存至 {OUTPUT_FILE}")

if __name__ == "__main__":
    process_dataset()
