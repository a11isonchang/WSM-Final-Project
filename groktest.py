import json
import os
import concurrent.futures
from openai import OpenAI
from tqdm import tqdm

# --- 設定區域 ---
# ✅ 請確保這裡讀取的是英文檔案
INPUT_FILE = './dragonball_dataset/queries_en.jsonl'
OUTPUT_FILE = './database/database.jsonl'

# 填入你的 OpenRouter API Key
OPENROUTER_API_KEY = "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" 

# 模型選擇：建議使用 Llama-3, Mistral 或 Grok 等英文能力強的模型
MODEL_NAME = "x-ai/grok-4.1-fast" 

# 平行處理設定
MAX_WORKERS = 10 

def extract_keywords_with_openrouter(client, query_text, query_type):
    """
    使用 OpenRouter API 提取英文關鍵字
    """
    
    # --- 🇬🇧 English Optimized System Prompt ---
    system_prompt = """
    You are an expert **Keyword Extraction System** optimized for RAG (Retrieval-Augmented Generation).
    Your task is to convert natural language queries into a concise list of **high-signal search keywords** to maximize retrieval accuracy from financial and medical documents.

    ### ⚡ Extraction Protocol:
    1. **Target Entities:** Extract specific Company Names, People (keep capitalization), Locations, and Product Names.
    2. **Noun Phrases:** Prefer specific noun phrases over generic verbs (e.g., extract "appointment of CEO" rather than just "appoint").
    3. **Attribute Mapping (Crucial):**
       - Queries often ask "Which is higher?" or "Who is earlier?". You must extract the **underlying attribute**.
       - "Which company had higher operating income?" -> Extract **"operating income"**
       - "Which company acquired assets earlier?" -> Extract **"asset acquisitions"**, **"date"**
    
    ### ⛔ Negative Constraints (Noise Filtering):
    - **REMOVE** Stop words: "the", "a", "an", "in", "on", "at", "of", "to", "for".
    - **REMOVE** Question words: "When", "How", "What", "Which", "Who", "Where", "Why".
    - **REMOVE** Meta-words: "Based on", "According to", "Summarize", "Compare", "outline", "report".
    - **REMOVE** Comparative adjectives acting alone: "higher", "lower", "better", "earlier", "later" (unless part of a specific term like "higher education").

    ### 📝 Few-Shot Examples (Based on User's Ontology):

    **Type: Factual Question**
    Input: "When did Green Fields Agriculture Ltd. appoint a new CEO?"
    Output: {"keywords": ["Green Fields Agriculture Ltd.", "appoint new CEO", "date"]}

    **Type: Multi-hop Reasoning Question**
    Input: "How did the senior management changes in March 2021, including the appointment of a new CEO, contribute to Green Fields Agriculture Ltd.'s market competitiveness?"
    Output: {"keywords": ["Green Fields Agriculture Ltd.", "senior management changes", "March 2021", "appointment of new CEO", "market competitiveness"]}

    **Type: Summary Question**
    Input: "Based on the outline, summarize the key changes in the governance structure of Green Fields Agriculture Ltd. in 2021."
    Output: {"keywords": ["Green Fields Agriculture Ltd.", "2021", "governance structure", "key changes"]}

    **Type: Irrelevant Unsolvable Question**
    Input: "Based on Green Fields Agriculture Ltd.'s 2021 report, summarize the environmental and social responsibility measures taken by the company."
    Output: {"keywords": ["Green Fields Agriculture Ltd.", "2021 report", "environmental measures", "social responsibility measures"]}

    **Type: Multi-document Information Integration Question**
    Input: "What were the impacts of the dividend distributions made by CleanCo Housekeeping Services in 2018 and Retail Emporium in 2020?"
    Output: {"keywords": ["CleanCo Housekeeping Services", "2018", "Retail Emporium", "2020", "dividend distributions", "impacts"]}

    **Type: Multi-document Comparison Question**
    Input: "Compare the operating income of CleanCo Housekeeping Services in 2018 and Retail Emporium in 2020. Which company had higher operating income?"
    Output: {"keywords": ["CleanCo Housekeeping Services", "2018", "Retail Emporium", "2020", "operating income"]}

    **Type: Multi-document Time Sequence Question**
    Input: "Compare the major asset acquisitions by CleanCo Housekeeping Services in 2018 and Retail Emporium in 2020. Which company acquired assets earlier?"
    Output: {"keywords": ["CleanCo Housekeeping Services", "2018", "Retail Emporium", "2020", "major asset acquisitions", "date"]}

    **Type: Summarization Question (Medical)**
    Input: "According to the hospitalization records of Parker General Hospital, summarize the present illness of Y. Evans."
    Output: {"keywords": ["Parker General Hospital", "hospitalization records", "Y. Evans", "present illness"]}

    ### 📦 Output Format:
    Return **ONLY** a valid JSON object with a single key "keywords".
    """

    user_prompt = f"""
    Target Query Type: [{query_type}]
    Input Text: "{query_text}"
    """

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.0, 
        )
        
        raw_output = response.choices[0].message.content.strip()
        json_res = json.loads(raw_output)
        keywords = json_res.get("keywords", [])
        
        # 英文關鍵字轉小寫通常有助於檢索 (取決於你的 Tokenizer，這裡先保持原樣，讓使用者自己決定)
        # keywords = [k.lower() for k in keywords] 
        
        return keywords
    except Exception as e:
        print(f"\n[Error processing '{query_text[:20]}...']: {e}")
        return []

def process_single_line(line, client):
    """
    處理單行數據的輔助函數 (與中文版邏輯相同，僅針對英文 Type 判斷做微調)
    """
    try:
        line = line.strip()
        if not line: return None
        
        data = json.loads(line)
        
        content = None
        q_id = None
        q_type = "General"
        
        # --- JSON 解析 ---
        if "query" in data and isinstance(data["query"], dict):
            content = data["query"].get("content")
            q_id = data["query"].get("query_id")
            q_type = data["query"].get("query_type", "General")
        elif "content" in data:
            content = data["content"]
            q_id = data.get("query_id")
            q_type = data.get("query_type", "General")

        if content:
            keywords = extract_keywords_with_openrouter(client, content, q_type)
            
            # ✅ 針對英文的 Unsolvable Type 進行標記
            unsolve = 0
            if q_type in ["Irrelevant Unsolvable Question", "Unsolvable"]:
                unsolve = 1

            new_record = {
                "query_id": q_id,
                "content": content,
                "keywords": keywords,
                "unsolve": unsolve
            }
            return json.dumps(new_record, ensure_ascii=False)
            
    except Exception as e:
        return None
    return None

def process_dataset():
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
        default_headers={
            "HTTP-Referer": "https://github.com/YourProject", 
            "X-Title": "WSM RAG Preprocessing EN" 
        }
    )
    
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Input file not found: {INPUT_FILE}")
        return

    print(f"📂 Reading: {INPUT_FILE}")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    total_lines = len(lines)
    print(f"🚀 Processing {total_lines} queries with [{MODEL_NAME}]...")
    print(f"💾 Writing to: {OUTPUT_FILE}")

    with open(OUTPUT_FILE, 'a', encoding='utf-8') as f_out:
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [executor.submit(process_single_line, line, client) for line in lines]
            
            for future in tqdm(concurrent.futures.as_completed(futures), total=total_lines):
                result = future.result()
                if result:
                    f_out.write(result + "\n")
                    f_out.flush()

    print(f"\n✅ Processing Complete!")

if __name__ == "__main__":
    process_dataset()
