import os
import faiss
import numpy as np
import argparse
import sys
from tqdm import tqdm
from My_RAG.utils import load_jsonl
from My_RAG.chunker import chunk_documents
from My_RAG.config import load_config
from ollama import Client
import numpy as np

class OllamaEmbeddings:
    """Wrapper for Ollama embeddings to match SentenceTransformer interface."""
    def __init__(self, model: str, base_url: str):
        self.client = Client(host=base_url)
        self.model = model

    def encode(self, sentences, batch_size=32, convert_to_numpy=True, normalize_embeddings=False, show_progress_bar=False):
        is_single = isinstance(sentences, str)
        inputs = [sentences] if is_single else sentences
        
        embeddings = []
        # show_progress_bar logic could be added here but keeping it simple
        iterator = tqdm(inputs, desc="Ollama Embedding") if show_progress_bar else inputs
        
        for text in iterator:
            try:
                if not text or not text.strip():
                    embeddings.append(None) 
                    continue

                response = self.client.embeddings(model=self.model, prompt=text)
                emb = response.get('embedding')
                if not emb:
                    embeddings.append(None)
                else:
                    embeddings.append(emb)
            except Exception as e:
                print(f"Warning: Embedding failed for text '{text[:20]}...': {e}")
                embeddings.append(None)
            
        # Fix None values
        valid_embs = [e for e in embeddings if e is not None]
        dim = len(valid_embs[0]) if valid_embs else 768
        
        final_embeddings = []
        for e in embeddings:
            if e is None:
                final_embeddings.append(np.zeros(dim))
            else:
                final_embeddings.append(e)
            
        result = np.array(final_embeddings)
        
        if normalize_embeddings:
            norm = np.linalg.norm(result, axis=1, keepdims=True)
            norm[norm == 0] = 1.0 
            result = result / norm

        if is_single:
            return result[0]
        return result

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
    model_name = dense_config.get('model_zh') if language == 'zh' else dense_config.get('model_en')
    print(f"Model: {model_name}")
    print(f"{'='*60}\n")

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
    
    try:
        if "embeddinggemma" in model_name or "qwen" in model_name or dense_config.get("type") == "ollama":
            host = config.get("ollama", {}).get("host") or "http://localhost:11434"
            print(f"   Loading Ollama model: {model_name} at {host}")
            model = OllamaEmbeddings(model=model_name, base_url=host)
        else:
            from sentence_transformers import SentenceTransformer
            device = "cuda" if dense_config.get("use_gpu", False) else "cpu"
            print(f"   Loading HF model: {model_name} on {device}")
            model = SentenceTransformer(model_name, device=device)
        
    except ImportError:
        print("❌ sentence-transformers or ollama not installed.")
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
