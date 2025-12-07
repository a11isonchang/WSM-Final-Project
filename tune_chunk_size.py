import argparse
import json
import os
import sys
import numpy as np
from typing import List, Dict, Any
from tqdm import tqdm

# Mock config for standalone run
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

try:
    from My_RAG.chunker import chunk_documents
    from My_RAG.retriever import create_retriever
except ImportError:
    # If running from root without module installation
    sys.path.append('My_RAG')
    from chunker import chunk_documents
    from retriever import create_retriever

def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

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
    # We use the same chunk_size for both en/zh params to ensure the test applies to the target lang
    chunks = chunk_documents(
        documents, 
        language,
        chunk_size_en=chunk_size, 
        chunk_overlap_en=chunk_overlap,
        chunk_size_zh=chunk_size, 
        chunk_overlap_zh=chunk_overlap
    )
    print(f"   -> Generated {len(chunks)} chunks")
    
    # 2. Build Retriever
    # Mock config to ensure dense is enabled
    mock_config = {
        "retrieval": {
            "dense": {"enabled": True},
            "parent_document_retrieval": {"enabled": False}, # We evaluate pure chunk retrieval precision
            "adaptive_threshold": False # Disable for fair comparison of recall
        },
        "ollama": {"host": "http://localhost:11434"} # Assumes Ollama is running
    }
    
    # Note: We pass None for parent_docs because we want to evaluate the CHUNKS' ability to be found.
    # If we enabled PDR, we'd be evaluating the parent doc mapping, which is a second step.
    retriever = create_retriever(chunks, language, mock_config.get("retrieval"))
    
    # 3. Evaluate Recall
    hits = 0
    total = 0
    
    for query_item in tqdm(queries, desc="Evaluating queries"):
        query_text = query_item["query"]["content"]
        ground_truth_doc_ids = set(query_item.get("ground_truth", {}).get("doc_ids", []))
        
        if not ground_truth_doc_ids:
            continue
            
        results, _ = retriever.retrieve(query_text, top_k=top_k)
        
        # Check if any retrieved chunk belongs to a ground truth doc
        retrieved_doc_ids = set()
        for res in results:
            doc_idx = res["metadata"].get("doc_idx")
            if doc_idx is not None:
                retrieved_doc_ids.add(doc_idx)
        
        # Hit if there is intersection
        if not ground_truth_doc_ids.isdisjoint(retrieved_doc_ids):
            hits += 1
        total += 1
        
    recall = hits / total if total > 0 else 0
    print(f"   ✅ Recall@{top_k}: {recall:.4f}")
    return recall

def main():
    parser = argparse.ArgumentParser(description="Auto-tune Chunk Size for RAG")
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
        # Rule of thumb: overlap is ~15-20% of chunk size
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
