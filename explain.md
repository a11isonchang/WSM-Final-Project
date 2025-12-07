# Project Structure & RAG Strategy Guide

This document explains the architecture of your Retrieval-Augmented Generation (RAG) system and provides practical configuration combinations to optimize performance.

## 📂 Project Structure

```
/
├── run.sh                  # Main entry point. Runs the full pipeline (Inference -> Check -> Eval).
├── build_index.py          # Script to pre-build FAISS indices on GPU (saves time during run).
├── requirements.txt        # Python dependencies.
├── configs/
│   └── config_submit.yaml  # Central Brain. Controls ALL settings (models, weights, toggles).
└── My_RAG/
    ├── main.py             # Orchestrator. Connects Data -> Chunker -> Retriever -> Generator.
    ├── chunker.py          # Data Prep. splits documents into search-friendly pieces.
    ├── retriever.py        # The Core Engine. Finds relevant info using BM25, FAISS, or Cross-Encoders.
    ├── generator.py        # The Speaker. Uses LLM (Ollama) to answer questions based on retrieved info.
    └── utils.py            # Helper functions (loading files, etc.).
```

## ⚙️ How It Works (The Pipeline)

1.  **Semantic Chunking (`chunker.py`):**
    *   Instead of blindly cutting text every 500 characters, it uses an embedding model to split text where the *meaning* changes (e.g., between paragraphs or topics).
    *   It also tags every chunk with a `doc_idx` so we can find its "parent" later.

2.  **Retrieval (`retriever.py`):**
    *   **Stage 1: Candidate Generation (Recall):**
        *   Uses **BM25** (keyword matching) to quickly find the top 50-100 potential chunks.
        *   Optionally uses **FAISS** (Dense Retrieval) to find chunks based on meaning.
    *   **Stage 2: Reranking (Precision):**
        *   **Cross-Encoder:** A powerful model that reads the Query and Document *together* to give a highly accurate relevance score. It sorts the candidates to find the absolute best matches.
    *   **Stage 3: Parent Document Retrieval (Context):**
        *   Instead of giving the LLM the small snippet (chunk), it uses the ID to fetch the **Full Parent Document**. This ensures the LLM has complete context to answer complex questions.

3.  **Generation (`generator.py`):**
    *   **"Lost in the Middle" Fix:** Reorders documents so the most relevant ones are at the start and end (where LLMs pay most attention).
    *   **Strict Prompting:** Forces the LLM to answer *only* from the context, reducing hallucinations.

---

## 🚀 Practical Combinations (Combos)

Here are 3 recommended configurations to try in `configs/config_submit.yaml`.

### Combo 1: "The High-Scorer" (Current Setup)
*Best for: Maximum accuracy, difficult questions.*

*   **Logic:** Use BM25 to get a wide net of candidates, then use a heavy Cross-Encoder to pick the winners, and finally read the full document.
*   **Config:**
    ```yaml
    retrieval:
      chunk_size: 1000
      weights:
        bm25: 1   # Primary candidate generator
        tfidf: 0
        jm: 0
      
      candidate_multiplier: 50.0 # Fetch top ~150 chunks for reranking
      
      dense:
        enabled: false # Disabled to save memory/time, relies on Cross-Encoder
      
      cross_encoder:
        enabled: true
        model: "My_RAG/models/ms-marco-MiniLM-L-6-v2"
        device: "cuda" # Use GPU!
        
      parent_document_retrieval:
        enabled: true  # Read full context
    ```

### Combo 2: "The Balanced Speedster"
*Best for: Good accuracy with much faster performance than Combo 1.*

*   **Logic:** Combine Keyword Search (BM25) with Semantic Search (FAISS). No heavy Cross-Encoder.
*   **Prerequisite:** You must run `python build_index.py ...` first to generate the FAISS files.
*   **Config:**
    ```yaml
    retrieval:
      weights:
        bm25: 0.4   # Balance keywords
        tfidf: 0
        jm: 0
      
      candidate_multiplier: 3.0 # Standard multiplier
      
      dense:
        enabled: true  # Enable FAISS
        type: "faiss"
        use_gpu: true
        model_en: "embeddinggemma:300m"
        model_zh: "qwen3-embedding:0.6b"

      cross_encoder:
        enabled: false # Disable heavy reranker
        
      parent_document_retrieval:
        enabled: true
    ```

### Combo 3: "The Context-Heavy" (Hybrid + Rerank)
*Best for: "Needle in a haystack" problems.*

*   **Logic:** Use BOTH BM25 and FAISS to generate candidates (catching both keywords and subtle meanings), THEN rerank with Cross-Encoder.
*   **Config:**
    ```yaml
    retrieval:
      weights:
        bm25: 0.5
      
      dense:
        enabled: true # Contribute candidates via FAISS
      
      cross_encoder:
        enabled: true # Rerank EVERYONE
      
      parent_document_retrieval:
        enabled: true
    ```

## 🛠️ How to Switch
1.  Open `configs/config_submit.yaml`.
2.  Modify the `enabled: true/false` flags and `weights` as shown above.
3.  Run `./run.sh`.
