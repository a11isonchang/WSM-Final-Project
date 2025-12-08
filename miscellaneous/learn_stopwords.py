import jsonlines
import jieba
import re
from collections import Counter
from pathlib import Path
import argparse

# Constants
EN_TOKEN_PATTERN = re.compile(r"[a-z0-9']+")

def tokenize_en(text):
    return set(EN_TOKEN_PATTERN.findall(text.lower()))

def tokenize_zh(text):
    # Filter out punctuation and whitespace, keep only meaningful words
    return set(word for word in jieba.cut(text) if word.strip() and not re.match(r'[^\w\s]', word))

def is_chinese(text):
    """Simple heuristic: if it contains Chinese characters, treat as Chinese context."""
    return any('\u4e00' <= char <= '\u9fff' for char in text)

def learn_stopwords(docs_path, query_path_en, query_path_zh, output_en, output_zh, top_k=300):
    print(f"Processing docs from: {docs_path}")
    print(f"Processing EN queries from: {query_path_en}")
    print(f"Processing ZH queries from: {query_path_zh}")
    
    counter_en = Counter()
    counter_zh = Counter()
    
    total_docs = 0
    
    # 1. Process Documents (Mixed Language)
    with jsonlines.open(docs_path) as reader:
        for obj in reader:
            content = obj.get('page_content', '') or obj.get('content', '')
            if not content:
                continue
            
            if is_chinese(content):
                tokens_zh = tokenize_zh(content)
                counter_zh.update(tokens_zh)
            else:
                tokens_en = tokenize_en(content)
                counter_en.update(tokens_en)
            
            total_docs += 1
            if total_docs % 1000 == 0:
                print(f"Processed {total_docs} documents...")

    # 2. Process EN Queries
    if query_path_en:
        with jsonlines.open(query_path_en) as reader:
            for obj in reader:
                content = obj.get('query', '')
                if isinstance(content, dict):
                    content = content.get('content', '')
                if content:
                    counter_en.update(tokenize_en(content))

    # 3. Process ZH Queries
    if query_path_zh:
        with jsonlines.open(query_path_zh) as reader:
            for obj in reader:
                content = obj.get('query', '')
                if isinstance(content, dict):
                    content = content.get('content', '')
                if content:
                    counter_zh.update(tokenize_zh(content))

    print(f"Total items processed: {total_docs}")

    # 3. Save Top-K words as stopwords
    print(f"Saving top {top_k} frequent words to {output_en}...")
    with open(output_en, 'w', encoding='utf-8') as f:
        # Get most common keys
        for word, count in counter_en.most_common(top_k):
            f.write(f"{word}\n")
            
    print(f"Saving top {top_k} frequent words to {output_zh}...")
    with open(output_zh, 'w', encoding='utf-8') as f:
        for word, count in counter_zh.most_common(top_k):
            f.write(f"{word}\n")

    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", default="./dragonball_dataset/dragonball_docs.jsonl")
    parser.add_argument("--queries_en", default="./dragonball_dataset/queries_show/queries_en.jsonl")
    parser.add_argument("--queries_zh", default="./dragonball_dataset/queries_show/queries_zh.jsonl")
    parser.add_argument("--out_en", default="./My_RAG/stopwords_learned_en.txt")
    parser.add_argument("--out_zh", default="./My_RAG/stopwords_learned_zh.txt")
    parser.add_argument("--top_k", type=int, default=200, help="Number of stopwords to learn")
    
    args = parser.parse_args()
    learn_stopwords(args.docs, args.queries_en, args.queries_zh, args.out_en, args.out_zh, args.top_k)
