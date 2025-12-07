import os
import faiss
import numpy as np
import argparse
import sys
from tqdm import tqdm
from utils import load_jsonl
from chunker import chunk_documents
from config import load_config
from retriever import create_retriever  # We'll use parts of this or duplicate logic

def build_and_save_index(docs_path, language, output_dir):
    """
    Builds a FAISS index using the Dense Retriever model configured in config.yaml
    and saves it to disk.
    """
    # 1. Load Config
    config = load_config()
    retrieval_config = config.get("retrieval", {})
    
    # Inject ollama config if needed
    if "ollama" in config:
        retrieval_config["ollama"] = config["ollama"]

    dense_config = retrieval_config.get("dense", {})
    if not dense_config.get("enabled", True):
        print("❌ Dense retrieval is disabled in config. Cannot build index.")
        return

    chunk_size = retrieval_config.get("chunk_size", 1000)
    chunk_overlap = retrieval_config.get("chunk_overlap", 200)

    print(f"\n{'='*60}")
    print(f"Building FAISS Index for {language}")
    print(f"Model: {dense_config.get('model_zh') if language == 'zh' else dense_config.get('model_en')}")
    print(f"{ '='*60}\n")

    # 2. Load Documents
    print("📂 Loading documents...")
    try:
        docs = load_jsonl(docs_path)
    except Exception as e:
        print(f"❌ Error loading documents: {e}")
        sys.exit(1)
    
    # 3. Chunk Documents
    print("✂️  Chunking documents...")
    chunks = chunk_documents(
        docs,
        language,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        ollama_config=config.get("ollama"),
    )
    if not chunks:
        print(f"❌ No chunks created for language '{language}'")
        sys.exit(1)
    print(f"✓ Created {len(chunks)} chunks")

    # 4. Initialize Embedding Model
    print("🧠 Initializing embedding model...")
    # We re-use logic from retriever.py effectively by instantiating just the model part
    # Or simpler: we create a temporary retriever just to access its model loading logic
    # BUT we need to be careful not to trigger the automatic index build in __init__
    
    # Let's manually load the model to be safe and explicit
    try:
        from sentence_transformers import SentenceTransformer
        
        if language == "zh":
            model_name = dense_config.get("model_zh") or dense_config.get("model")
        else:
            model_name = dense_config.get("model_en") or dense_config.get("model")
            
        device = "cuda" if dense_config.get("use_gpu", False) else "cpu"
        print(f"   Loading model: {model_name} on {device}")
        
        model = SentenceTransformer(model_name, device=device)
        
    except ImportError:
        print("❌ sentence-transformers not installed.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        sys.exit(1)

    # 5. Compute Embeddings
    print("zz Computing embeddings (this may take a while)...")
    passage_prefix = dense_config.get("passage_prefix", "passage: ")
    passages = [f"{passage_prefix}{chunk.get('page_content', '')}" for chunk in chunks]
    
    batch_size = dense_config.get("batch_size", 32)
    normalize = dense_config.get("normalize", True)
    
    embeddings = model.encode(
        passages,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
        show_progress_bar=True,
    )

    # 6. Build FAISS Index
    print("wd Building FAISS index...")
    d = embeddings.shape[1]
    
    # Use Inner Product (IP) for normalized embeddings (cosine similarity equivalent)
    # Use L2 for others
    metric = faiss.METRIC_INNER_PRODUCT if normalize else faiss.METRIC_L2
    
    if metric == faiss.METRIC_INNER_PRODUCT:
        index = faiss.IndexFlatIP(d)
    else:
        index = faiss.IndexFlatL2(d)
        
    index.add(embeddings)
    print(f"✓ Index built with {index.ntotal} vectors")

    # 7. Save Index
    os.makedirs(output_dir, exist_ok=True)
    index_path = os.path.join(output_dir, f"faiss_index_{language}.bin")
    
    print(f"💾 Saving index to {index_path}...")
    faiss.write_index(index, index_path)
    print("✓ Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--docs_path', required=True, help='Path to documents JSONL')
    parser.add_argument('--language', required=True, choices=['en', 'zh'], help='Language')
    parser.add_argument('--output_dir', default='My_RAG/indices', help='Directory to save index')
    args = parser.parse_args()
    
    build_and_save_index(args.docs_path, args.language, args.output_dir)
