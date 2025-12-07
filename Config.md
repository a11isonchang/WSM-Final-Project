# Configuration Guide

This project reads configuration from `configs/config_local.yaml` (if present) or falls back to `configs/config_submit.yaml` (`My_RAG/config.py`). The YAML drives every stage of the RAG pipeline: chunking, retrieval, reranking, and generation.

## Top-Level
- `ollama.host` / `ollama.model`: Used by chunking (for semantic chunker embeddings), dense retrieval when using Ollama-based encoders, and final answer generation (`My_RAG/main.py`, `chunker.py`, `retriever.py`, `generator.py`).

## Retrieval Block (`retrieval`)
Consumed in `My_RAG/main.py` and passed to `retriever.create_retriever`, which maps directly into `HybridRetriever` (`My_RAG/retriever.py`).

- `chunk_size`, `chunk_overlap`: Control semantic/recursive chunker windowing (`chunker.py`).
- `top_k`: Final number of chunks returned to generation.
- `debug`: When `true`, prints retrieval breakdown per query.

### Lexical Hybrid Scoring
- `weights.tfidf`, `weights.bm25`, `weights.jm`: Mix scores (a*TF-IDF + b*BM25 + c*JM). BM25/TF-IDF/JM are normalized before combining.
- `bm25.k1`, `bm25.b`: BM25Okapi hyperparameters.
- `jm.lambda`: Jelinek–Mercer smoothing weight.
- `candidate_multiplier`: Size of the BM25/TF-IDF/JM candidate pool before reranking (`max(top_k, top_k * multiplier)`).

### Feedback and Expansion
- `prf_top_k`, `prf_term_count`: Pseudo-relevance feedback. Expands the query with top terms from the best lexical hits before rescoring.
- `hyde.enabled`: Generate a hypothetical document with the LLM; dense query uses that text.
- `query_expansion.enabled`: Rewrite/expand the query with the LLM; requires `ollama` block.
- `multi_query.enabled`, `multi_query.num_versions`: Generate multiple query variants with the LLM and fuse results via RRF.
- `keyword_boost.enabled`, `keywords`, `boost_factor`: Precompute document-level boosts when keywords are present.

### Dense Stage (Bi-Encoder / FAISS)
Controlled via `retrieval.dense`:
- `enabled`: Turn dense reranking/search on/off.
- `type`: `"faiss"` builds a FAISS index and uses `_dense_search_faiss`; otherwise encodes on the fly. `"ollama"` also routes encoding through OllamaEmbeddings.
- `use_gpu`: If FAISS GPU is available, move the index to GPU.
- `strategy`: `"rerank"` (default) reranks lexical candidates; `"dense_only"` ignores lexical ordering and returns FAISS top hits.
- `model_en`, `model_zh`, `model`: Language-specific fallback order for the dense encoder.
- `normalize`: L2-normalize embeddings; also selects FAISS metric (IP when true, L2 otherwise).
- `batch_size`: Batch size for passage encoding.
- `query_prefix`, `passage_prefix`: Prepended to text before encoding.

### Cross Encoder (optional)
If you add a `cross_encoder` block with `enabled: true` and `model`, it will rerank after lexical/dense. Device/batch_size are read if present.

## Generation
`generator.py` pulls `ollama.host` and `ollama.model` to generate answers. Retrieved chunks (already reranked) are reordered to mitigate “lost in the middle,” then sent to the model.

## Using BM25 + FAISS Hybrid
`configs/config_submit.yaml` already does this:
- Lexical: `weights.bm25: 1` (TF-IDF/JM 0) → picks candidates.
- Dense: `dense.enabled: true`, `type: "faiss"`, `use_gpu: true`, models set per language, `normalize: true`.
- Candidate pool: `candidate_multiplier: 20.0`.
- Final results: `top_k: 3`.

Toggle dense-only by adding `retrieval.dense.strategy: "dense_only"`; otherwise the default path is lexical → candidates → FAISS rerank.
