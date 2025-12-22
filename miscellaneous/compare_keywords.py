import jieba
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# ==========================================
# 1. Setup & Constants
# ==========================================
ENGLISH_STOP_WORDS_SET = set(ENGLISH_STOP_WORDS)
EN_TOKEN_PATTERN = re.compile(r"[a-z0-9']+")

# Use the local model path if it exists, otherwise fall back to the Hugging Face name
MODEL_PATH = "My_RAG/models/all_minilm_l6"
# If running locally without the model downloaded, you might need "all-MiniLM-L6-v2"
# We will try the local path first.

print(f"Loading embedding model from {MODEL_PATH} ...")
try:
    model = SentenceTransformer(MODEL_PATH)
except Exception as e:
    print(f"Could not load local model: {e}")
    print("Trying to load 'all-MiniLM-L6-v2' from Hub...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

# ==========================================
# 2. Simple Extraction (Token-based)
# ==========================================
def extract_simple(text, language="en", min_chars=3):
    print(f"\n--- Simple Extraction ({language}) ---")
    tokens = []
    
    if language == "zh":
        # Chinese: Use jieba
        tokens = [t.strip() for t in jieba.cut(text) if t.strip()]
        # (Optional) Remove stop words if you had a list
    else:
        # English: Regex split + lowercase + stopword filter
        raw_tokens = EN_TOKEN_PATTERN.findall(text.lower())
        tokens = [t for t in raw_tokens if t not in ENGLISH_STOP_WORDS_SET]

    # Filter by length
    keywords = {t for t in tokens if len(t) >= min_chars}
    
    print(f"Tokens found: {tokens}")
    print(f"Filtered Keywords: {sorted(list(keywords))}")
    return keywords

# ==========================================
# 3. Embedding Extraction (Semantic)
# ==========================================
def extract_semantic(text, language="en", min_chars=3, top_k=5):
    print(f"\n--- Embedding Extraction ({language}) ---")
    
    # 1. Generate Candidates (N-grams)
    if language == "zh":
        tokens = [t for t in jieba.cut(text) if t.strip()]
    else:
        tokens = [t for t in text.split() if t.strip()]
        
    candidates = set()
    # Generate 1-gram to 4-grams
    for n in range(1, 5):
        for i in range(len(tokens) - n + 1):
            if language == "zh":
                ngram = "".join(tokens[i : i + n])
            else:
                ngram = " ".join(tokens[i : i + n])
                
            if len(ngram.strip()) >= min_chars:
                candidates.add(ngram)
    
    candidates = list(candidates)
    if not candidates:
        print("No candidates found.")
        return set()
        
    print(f"Generated {len(candidates)} n-gram candidates (showing first 5): {candidates[:5]}")

    # 2. Embed Query and Candidates
    query_embedding = model.encode([text])
    candidate_embeddings = model.encode(candidates)
    
    # 3. Compute Cosine Similarity
    # Normalize vectors
    query_vec = query_embedding[0]
    query_vec /= np.linalg.norm(query_vec)
    
    cand_vecs = candidate_embeddings
    cand_norms = np.linalg.norm(cand_vecs, axis=1, keepdims=True)
    cand_vecs = cand_vecs / cand_norms
    
    # Dot product
    distances = np.dot(cand_vecs, query_vec)
    
    # 4. Rank and Select
    # Get top_k indices
    top_indices = np.argsort(distances)[-top_k:][::-1]
    
    results = []
    for idx in top_indices:
        score = distances[idx]
        word = candidates[idx]
        results.append((word, score))
        print(f"  Selected: '{word}' (Score: {score:.4f})")
        
    return {r[0] for r in results}

# ==========================================
# 4. Run Demonstration
# ==========================================
if __name__ == "__main__":
    # Example 1: English Query
    en_query = "What is the capital allocation strategy of Apple Inc.?"
    print(f"\nQuery: {en_query}")
    extract_simple(en_query, language="en")
    extract_semantic(en_query, language="en", top_k=3)

    # Example 2: Chinese Query
    zh_query = "绿源环保有限公司2017年的二氧化碳排放量是多少？"
    print(f"\nQuery: {zh_query}")
    extract_simple(zh_query, language="zh")
    extract_semantic(zh_query, language="zh", top_k=3)
