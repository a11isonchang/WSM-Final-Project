import argparse
import json
import os
import sys
import numpy as np
from typing import List, Dict, Any
from tqdm import tqdm
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
import jieba
import re

# Path setup
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
try:
    from My_RAG.chunker import chunk_documents
except ImportError:
    sys.path.append('My_RAG')
    from chunker import chunk_documents

def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def preprocess_text_zh(text):
    tokens = jieba.cut(text)
    return [t.strip() for t in tokens if t.strip()]

def preprocess_text_en(text):
    tokens = re.findall(r'\w+', text.lower())
    return [t.strip() for t in tokens if t.strip()]

def evaluate_chunk_size(
    chunk_size: int,
    chunk_overlap: int,
    documents: List[Dict],
    queries: List[Dict],
    language: str,
    top_k: int = 5
):
    print(f"\n🔄 Testing Chunk Size: {chunk_size} | Overlap: {chunk_overlap}")
    
    # 1. Chunk Documents
    chunks = chunk_documents(
        documents, 
        language,
        chunk_size_en=chunk_size, 
        chunk_overlap_en=chunk_overlap,
        chunk_size_zh=chunk_size, 
        chunk_overlap_zh=chunk_overlap
    )
    print(f"   -> Generated {len(chunks)} chunks")
    
    # Convert to LangChain Documents
    lc_docs = [Document(page_content=c['page_content'], metadata=c.get('metadata', {})) for c in chunks]
    
    # 2. Build Retrievers (HF + BM25)
    # Use standard HF models that are close proxies to Gemma/Qwen
    if language == 'zh':
        model_name = "BAAI/bge-large-zh-v1.5"
        preprocess = preprocess_text_zh
    else:
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        preprocess = preprocess_text_en
        
    print(f"   -> Embedding with {model_name} (HF)...")
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    vectorstore = FAISS.from_documents(lc_docs, embeddings)
    dense_retriever = vectorstore.as_retriever(search_kwargs={"k": 100})
    
    bm25_retriever = BM25Retriever.from_documents(lc_docs, preprocess_func=preprocess)
    bm25_retriever.k = 100
    
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, dense_retriever],
        weights=[0.5, 0.5]
    )
    
    # 3. Evaluate Recall
    hits = 0
    total = 0
    
    for query_item in tqdm(queries, desc="Evaluating queries"):
        query_text = query_item["query"]["content"]
        ground_truth_doc_ids = set(query_item.get("ground_truth", {}).get("doc_ids", []))
        
        if not ground_truth_doc_ids:
            continue
            
        results = ensemble_retriever.invoke(query_text)
        top_results = results[:top_k]
        
        # Check if any retrieved chunk belongs to a ground truth doc
        retrieved_doc_ids = set()
        for res in top_results:
            doc_idx = res.metadata.get("doc_idx")
            if doc_idx is not None:
                retrieved_doc_ids.add(doc_idx)
        
        if not ground_truth_doc_ids.isdisjoint(retrieved_doc_ids):
            hits += 1
        total += 1
        
    recall = hits / total if total > 0 else 0
    print(f"   ✅ Recall@{top_k}: {recall:.4f}")
    return recall

def main():
    parser = argparse.ArgumentParser(description="Auto-tune Chunk Size for RAG (Colab/HF Version)")
    parser.add_argument('--docs_path', type=str, required=True, help='Path to dragonball_docs.jsonl')
    parser.add_argument('--queries_path', type=str, required=True, help='Path to dragonball_queries.jsonl')
    parser.add_argument('--language', type=str, required=True, choices=['en', 'zh'])
    parser.add_argument('--sizes', type=str, default="200,384,512,600,800,1000", help="Comma-separated chunk sizes to test")
    
    args = parser.parse_args()
    
    print(f"📂 Loading Data...")
    docs = load_jsonl(args.docs_path)
    queries = load_jsonl(args.queries_path)
    
    # Filter queries by language
    queries = [q for q in queries if q.get("language", args.language) == args.language]
    print(f"   Loaded {len(docs)} docs and {len(queries)} queries for {args.language}")
    
    chunk_sizes = [int(s) for s in args.sizes.split(',')]
    results = {}
    
    for size in chunk_sizes:
        overlap = int(size * 0.15)
        score = evaluate_chunk_size(size, overlap, docs, queries, args.language)
        results[size] = score
        
    print("\n🏆 ================= RESULTS ================= 🏆")
    best_size = max(results, key=results.get)
    print(f"Best Chunk Size for {args.language}: {best_size} (Recall: {results[best_size]:.4f})")
    print("-" * 40)
    for size, score in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"Size {size}: {score:.4f}")

if __name__ == "__main__":
    main()
