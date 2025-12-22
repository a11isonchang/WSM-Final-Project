import sys
import os
import json
import numpy as np
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from ollama import Client
from My_RAG.config import load_config

def get_ollama_client_and_model():
    try:
        config = load_config()
        ollama_cfg = config.get("ollama", {})
        host = ollama_cfg.get("host", "http://localhost:11434")
        model = ollama_cfg.get("model", "llama3")
        
        # Get embedding model from retrieval config
        retrieval_cfg = config.get("retrieval", {})
        embedding_model = retrieval_cfg.get("embedding_model_path", "qwen3-embedding:0.6b")
        
        client = Client(host=host)
        return client, model, embedding_model
    except Exception as e:
        print(f"Error loading config: {e}")
        return None, None, None

def load_database_embeddings(file_paths):
    data = []
    for path in file_paths:
        p = Path(path)
        if p.exists():
            print(f"Loading {p}...")
            with open(p, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        record = json.loads(line)
                        if 'embedding' in record and 'keywords' in record and 'content' in record:
                             record['embedding'] = np.array(record['embedding'], dtype=np.float32)
                             data.append(record)
                    except json.JSONDecodeError:
                        continue
    return data

def find_most_similar(query_embedding, database):
    if not database:
        return None
    
    query_vec = np.array(query_embedding, dtype=np.float32)
    norm_q = np.linalg.norm(query_vec)
    if norm_q > 0:
        query_vec /= norm_q
    else:
        return None
        
    best_score = -1.0
    best_item = None
    
    for item in database:
        vec = item['embedding']
        norm_v = np.linalg.norm(vec)
        if norm_v > 0:
             score = np.dot(query_vec, vec / norm_v)
             if score > best_score:
                 best_score = score
                 best_item = item
                 
    return best_item

def extract_keywords_zero_shot(client, model, query, language="en"):
    if language == "zh":
        prompt = f"""请从以下查询中提取关键字。仅输出关键字，用逗号分隔，不要包含其他文本。
查询: {query}
关键字:"""
    else:
        prompt = f"""Extract keywords from the following query. Output ONLY the keywords separated by commas, no other text.
Query: {query}
Keywords:"""

    try:
        response = client.generate(model=model, prompt=prompt, stream=False)
        return response['response'].strip()
    except Exception as e:
        return f"Error: {e}"

def extract_keywords_one_shot(client, model, query, language="en", embedding_model=None, database=None):
    example_text = ""
    
    if client and embedding_model and database:
        try:
            # Embed input query
            response = client.embed(model=embedding_model, input=query)
            if 'embeddings' in response and response['embeddings']:
                query_emb = response['embeddings'][0]
                
                similar_item = find_most_similar(query_emb, database)
                if similar_item:
                    ex_query = similar_item['content']
                    ex_keywords = ", ".join(similar_item['keywords'])
                    print(f"  [Found similar example: {ex_query}]")
                    
                    if language == "zh":
                        example_text = f"""
示例:
查询: {ex_query}
关键字: {ex_keywords}
"""
                    else:
                        example_text = f"""
Example:
Query: {ex_query}
Keywords: {ex_keywords}
"""
        except Exception as e:
            print(f"Error finding similar example: {e}")

    if not example_text:
        print("  [Using default example]")
        if language == "zh":
            example_text = """
示例:
查询: 苹果公司2023年的收入是多少？
关键字: 苹果公司, 2023年, 收入
"""
        else:
            example_text = """
Example:
Query: What is the revenue of Apple Inc. in 2023?
Keywords: Apple Inc., 2023, revenue
"""

    if language == "zh":
        prompt = f"""请从以下查询中提取关键字。仅输出关键字，用逗号分隔，不要包含其他文本。
{example_text}
查询: {query}
关键字:"""
    else:
        prompt = f"""Extract keywords from the following query. Output ONLY the keywords separated by commas, no other text.
{example_text}
Query: {query}
Keywords:"""

    try:
        response = client.generate(model=model, prompt=prompt, stream=False)
        return response['response'].strip()
    except Exception as e:
        return f"Error: {e}"

def main():
    client, model, embedding_model = get_ollama_client_and_model()
    if not client:
        print("Could not initialize Ollama client.")
        return

    print(f"Using generation model: {model}")
    print(f"Using embedding model: {embedding_model}")

    # Load Database
    root_dir = Path(__file__).resolve().parent.parent
    db_paths = [
        root_dir / "database" / "database_with_embeddings.jsonl",
        root_dir / "database" / "database_with_embeddings_part2.jsonl"
    ]
    print("Loading database for dynamic one-shot examples...")
    database = load_database_embeddings(db_paths)
    print(f"Loaded {len(database)} items.")

    queries = [
        ("zh", "根据夕照市人民医院和黄埔市妇幼保健院的住院病历，患者路某某和葛某某的初步诊断分别是什么？")
    ]

    for lang, query in queries:
        print(f"\n{'='*40}")
        print(f"Query ({lang}): {query}")
        print(f"{'-'*40}")
        
        print("Zero-Shot Result:")
        zs = extract_keywords_zero_shot(client, model, query, lang)
        print(zs)
        
        print(f"\nOne-Shot Result:")
        os = extract_keywords_one_shot(client, model, query, lang, embedding_model, database)
        print(os)

if __name__ == "__main__":
    main()
