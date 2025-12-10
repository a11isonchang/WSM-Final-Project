from rank_bm25 import BM25Okapi
import jieba
import json
import re
import numpy as np
import faiss
import os
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sentence_transformers import SentenceTransformer
from pathlib import Path
from functools import lru_cache
from typing import List, Dict, Any, Optional
from ollama import Client
from config import load_config
from kg_retriever import create_kg_retriever

EN_TOKEN_PATTERN = re.compile(r"[a-z0-9']+")
ENGLISH_STOP_WORDS_SET = set(ENGLISH_STOP_WORDS)

def load_ollama_config() -> dict:
    """
    讀取 config.yaml 內的 ollama 設定。
    預期結構：
    ollama:
      host: http://127.0.0.1:11434
      model: your-model-name
    """
    config = load_config()
    assert "ollama" in config, "Ollama configuration not found in config file."
    assert "host" in config["ollama"], "Ollama host not specified in config file."
    assert "model" in config["ollama"], "Ollama model not specified in config file."
    return config["ollama"]


@lru_cache()
def load_chinese_stop_words() -> set:
    from jieba import analyse

    stop_words_path = getattr(analyse, "STOP_WORDS_PATH", None)
    if not stop_words_path:
        stop_words_path = Path(analyse.__file__).resolve().parent / "stop_words.utf8"

    stop_words = set()
    stop_file = Path(stop_words_path)
    if stop_file.exists():
        with stop_file.open("r", encoding="utf-8") as f:
            stop_words = {line.strip() for line in f if line.strip()}
    return stop_words


class BM25Retriever:
    """Hybrid retriever (BM25 + Dense) with optional keyword-based re-ranking."""

    def __init__(
        self,
        chunks: List[Dict[str, Any]],
        language: str = "en",
        *,
        k1: float = 1.5,
        b: float = 0.75,
        candidate_multiplier: float = 3.0,
        keyword_boost: float = 0.0,
        min_keyword_characters: int = 3,
        keyword_extraction_method: str = "simple",
        embedding_model_path: str = "My_RAG/models/all_minilm_l6",
        embedding_provider: str = "local",
        ollama_host: str = "http://localhost:11434",
        keyword_file: str = None,
        dense_weight: float = 0.5, # Weight for dense retrieval score (0.0 to 1.0)
    ):
        self.chunks = chunks
        self.language = language
        self.corpus = [chunk.get("page_content", "") for chunk in chunks]
        self.predefined_keywords = {}
        self.unsolvable_queries = set()
        if keyword_file:
            self._load_predefined_keywords(keyword_file)

        self.min_keyword_characters = max(1, int(min_keyword_characters))
        self.candidate_multiplier = max(1.0, candidate_multiplier)
        self.keyword_boost = max(0.0, keyword_boost)
        self.keyword_extraction_method = keyword_extraction_method
        self.embedding_provider = embedding_provider
        self.dense_weight = dense_weight
        
        self.embedding_model = None
        self.ollama_client = None
        self.chunk_embeddings = None
        self.faiss_index = None

        # Initialize Embedding Model (for Dense Retrieval and/or Semantic Keywords)
        if self.embedding_provider == "ollama":
            try:
                self.ollama_client = Client(host=ollama_host)
                self.embedding_model = embedding_model_path
                print(f"Using Ollama embedding model: {self.embedding_model}")
            except Exception as e:
                print(f"Failed to initialize Ollama client: {e}.")
        else:
            try:
                self.embedding_model = SentenceTransformer(embedding_model_path)
                print(f"Loaded local embedding model: {embedding_model_path}")
            except Exception as e:
                print(f"Failed to load local embedding model: {e}.")

        # Handle FAISS Index and Embeddings
        index_dir = Path("My_RAG/indices")
        index_dir.mkdir(parents=True, exist_ok=True)
        index_path = index_dir / f"faiss_index_{self.language}.bin"

        if self.embedding_model:
            if index_path.exists():
                print(f"Loading existing FAISS index from {index_path}...")
                try:
                    self.faiss_index = faiss.read_index(str(index_path))
                    # Reconstruct embeddings from index for use in hybrid scoring
                    # Assuming IndexFlatIP, we can reconstruct the full matrix
                    if isinstance(self.faiss_index, faiss.IndexFlat):
                        # For IndexFlat, we can access the vectors directly
                        # But read_index returns a generic Index, need to check ntotal
                        ntotal = self.faiss_index.ntotal
                        if ntotal == len(self.corpus):
                            # Reconstruct all vectors: 0 to ntotal
                            # This might be memory intensive but fast for scoring
                            self.chunk_embeddings = self.faiss_index.reconstruct_n(0, ntotal)
                            
                            # Verify that the embedding dimension matches the current model
                            test_emb = self._compute_embeddings(["test"])
                            if test_emb and len(test_emb) > 0:
                                test_array = np.array(test_emb)
                                if test_array.ndim == 1:
                                    expected_dim = test_array.shape[0]
                                else:
                                    expected_dim = test_array.shape[-1]
                                actual_dim = self.chunk_embeddings.shape[1]
                                if expected_dim != actual_dim:
                                    print(f"Warning: Embedding dimension mismatch. Index has {actual_dim} dims, but current model produces {expected_dim} dims. Re-computing.")
                                    self.faiss_index = None
                                    self.chunk_embeddings = None
                                else:
                                    print(f"Loaded {ntotal} embeddings from FAISS index (dim={actual_dim}).")
                            else:
                                print(f"Warning: Could not verify embedding dimension. Re-computing.")
                                self.faiss_index = None
                                self.chunk_embeddings = None
                        else:
                            print(f"Warning: Index size ({ntotal}) does not match corpus size ({len(self.corpus)}). Re-computing.")
                            self.faiss_index = None
                            self.chunk_embeddings = None
                    else:
                         print("Loaded index is not IndexFlat, cannot reconstruct easily. Re-computing.")
                         self.faiss_index = None
                         self.chunk_embeddings = None
                except Exception as e:
                    print(f"Error loading FAISS index: {e}. Re-computing.")
                    self.faiss_index = None
                    self.chunk_embeddings = None

            if self.faiss_index is None:
                print("Computing chunk embeddings for dense retrieval...")
                embeddings = self._precompute_corpus_embeddings(self.corpus)
                if embeddings is not None and len(embeddings) > 0:
                    # Normalize for Cosine Similarity (IndexFlatIP)
                    embeddings = embeddings.astype(np.float32)
                    faiss.normalize_L2(embeddings)
                    self.chunk_embeddings = embeddings
                    
                    dim = embeddings.shape[1]
                    self.faiss_index = faiss.IndexFlatIP(dim)
                    self.faiss_index.add(embeddings)
                    
                    print(f"Saving FAISS index to {index_path}...")
                    faiss.write_index(self.faiss_index, str(index_path))
                    print(f"Finished computing and saving embeddings for {len(self.corpus)} chunks.")
                else:
                    print("Failed to compute embeddings.")

        self.porter_stemmer = PorterStemmer()
        self.chinese_stop_words = load_chinese_stop_words()
        self.tokenized_corpus = [self._tokenize(doc) for doc in self.corpus]
        self.bm25 = BM25Okapi(self.tokenized_corpus, k1=k1, b=b)

    def _load_predefined_keywords(self, path: str):
        p = Path(path)
        if p.exists():
            try:
                with open(p, 'r', encoding='utf-8') as f:
                    for line in f:
                        data = json.loads(line)
                        content = data.get("content", "").strip()
                        qid = data.get("query_id")
                        keywords = data.get("keywords", [])
                        unsolve = data.get("unsolve", 0)

                        if unsolve == 1:
                            if qid is not None:
                                self.unsolvable_queries.add(str(qid))
                            if content:
                                self.unsolvable_queries.add(content)

                        if keywords:
                            kw_set = set(keywords)
                            if content:
                                self.predefined_keywords[content] = kw_set
                            if qid is not None:
                                self.predefined_keywords[str(qid)] = kw_set
                print(f"Loaded keywords from {path}")
            except Exception as e:
                print(f"Error loading keywords from {path}: {e}")

    def _tokenize(self, text: str):
        if self.language == "zh":
            tokens = [tok.strip() for tok in jieba.cut(text) if tok.strip()]
            stop_words = self.chinese_stop_words
            return [tok for tok in tokens if tok not in stop_words]

        raw_tokens = EN_TOKEN_PATTERN.findall(text.lower())
        filtered = [token for token in raw_tokens if token and token not in ENGLISH_STOP_WORDS_SET]
        return [self._stem_english(token) for token in filtered]

    def _stem_english(self, token: str) -> str:
        return self.porter_stemmer.stem(token)

    def _compute_embeddings(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
            
        if self.embedding_provider == "ollama":
            # Ollama embed returns an object with .embeddings
            # Note: Ollama client might not support batching natively in one call for large lists depending on version,
            # but usually handles list input.
            try:
                response = self.ollama_client.embed(model=self.embedding_model, input=texts)
                return response.embeddings
            except Exception as e:
                print(f"Ollama embedding error: {e}")
                return []
        else:
            # SentenceTransformer returns ndarray or list of ndarrays
            return self.embedding_model.encode(texts)

    def _precompute_corpus_embeddings(self, texts: List[str], batch_size: int = 32):
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            embeddings = self._compute_embeddings(batch)
            if embeddings is not None and len(embeddings) > 0:
                all_embeddings.extend(embeddings)
            else:
                # Fallback if embedding fails: use zero vectors
                # Assuming dimension based on a test or default
                dim = 384 # Default for all-minilm-l6-v2, will adapt if first batch succeeds
                if all_embeddings:
                    dim = len(all_embeddings[0])
                all_embeddings.extend([[0.0] * dim] * len(batch))
        return np.array(all_embeddings)

    def _extract_keywords_semantic(self, query, top_k=5):
        if not self.embedding_model:
            return set()

        if self.language == "zh":
            tokens = [t for t in jieba.cut(query) if t.strip()]
        else:
            tokens = [t for t in query.split() if t.strip()]

        candidates = set()
        # Generate n-grams (1 to 4)
        for n in range(1, 5):
            for i in range(len(tokens) - n + 1):
                ngram = "".join(tokens[i:i+n]) if self.language == "zh" else " ".join(tokens[i:i+n])
                # Filter candidates: minimal length
                if len(ngram.strip()) >= self.min_keyword_characters:
                     candidates.add(ngram)
        
        candidates = list(candidates)
        if not candidates:
            return set()
            
        query_embedding = self._compute_embeddings([query])
        candidate_embeddings = self._compute_embeddings(candidates)
        
        
        # Check if embeddings are valid
        if not query_embedding or not candidate_embeddings:
            return set()

        # Manual cosine similarity calculation
        query_vec = np.array(query_embedding).astype(np.float32)
        query_norm = np.linalg.norm(query_vec)
        if query_norm > 1e-9:
            query_vec_norm = query_vec / query_norm
        else:
            return set() # Zero query vector

        cand_vecs = np.array(candidate_embeddings).astype(np.float32)
        cand_norms = np.linalg.norm(cand_vecs, axis=1, keepdims=True)
        cand_vecs_norm = np.divide(cand_vecs, cand_norms, where=cand_norms > 1e-9)
        
        # Compute dot product
        # query_vec_norm is (1, dim), cand_vecs_norm is (n_candidates, dim)
        # Result will be (n_candidates, 1), flatten to (n_candidates,)
        distances = np.dot(cand_vecs_norm, query_vec_norm.T).flatten()
        
        # Get top indices
        top_indices = np.argsort(distances)[-top_k:]
        
        keywords = {candidates[i] for i in top_indices}
        return keywords

    def _extract_keywords(self, tokens):
        keywords = set()
        for token in tokens:
            normalized = token if self.language == "zh" else token.lower()
            if len(normalized.strip()) >= self.min_keyword_characters:
                keywords.add(normalized)
        return keywords

    def _keyword_overlap(self, chunk, keywords):
        if not keywords:
            return 0.0
        text = chunk["page_content"]
        haystack = text if self.language == "zh" else text.lower()
        return sum(1 for kw in keywords if kw and kw in haystack)

    def retrieve(self, query, top_k=5, query_id=None):
        is_unsolvable = False
        if (query_id is not None and str(query_id) in self.unsolvable_queries) or \
           (query.strip() in self.unsolvable_queries):
            is_unsolvable = True

        if not self.chunks:
            return [], {
                "language": self.language,
                "top_k": top_k,
                "candidate_count": 0,
                "keyword_info": None,
                "results": [],
                "unsolvable": is_unsolvable
            }

        # 1. Sparse Retrieval (BM25)
        tokenized_query = self._tokenize(query)
        if not tokenized_query:
            bm25_scores = np.zeros(len(self.chunks))
        else:
            bm25_scores = np.array(self.bm25.get_scores(tokenized_query))

        bm25_scores = np.array(self.bm25.get_scores(tokenized_query))

        # 2. Dense Retrieval (Cosine Similarity)
        dense_scores = np.zeros(len(self.chunks))
        if self.chunk_embeddings is not None:
            query_emb = self._compute_embeddings([query])
            if query_emb is not None and len(query_emb) > 0:
                # Check for zero vector to avoid RuntimeWarning
                query_vec = np.array(query_emb).astype(np.float32) # Ensure consistent type
                query_norm = np.linalg.norm(query_vec)
                
                if query_norm > 1e-9:
                    # Normalize query vector
                    query_vec_norm = query_vec / query_norm
                    
                    # Compute Dot Product: (1, dim) @ (n_chunks, dim).T -> (1, n_chunks)
                    # chunk_embeddings are already normalized in __init__
                    try:
                        # self.chunk_embeddings is numpy array of shape (n_chunks, dim)
                        # query_vec_norm is (1, dim) or (dim,)
                        
                        # Reshape query if needed to (1, dim)
                        if query_vec_norm.ndim == 1:
                            query_vec_norm = query_vec_norm.reshape(1, -1)
                            
                        # Dot product
                        raw_dense_scores = np.dot(self.chunk_embeddings, query_vec_norm.T).flatten()
                        
                        # Replace NaNs (from zero vectors in chunks) with 0.0
                        dense_scores = np.nan_to_num(raw_dense_scores, nan=0.0)
                    except Exception as e:
                        print(f"Error in dense retrieval scoring: {e}")
                        dense_scores = np.zeros(len(self.chunks))
                else:
                    # Query embedding is effectively zero, skip dense retrieval
                    dense_scores = np.zeros(len(self.chunks))

        # 3. Score Normalization (Min-Max)
        def normalize(scores):
            if np.max(scores) == np.min(scores):
                return scores
            return (scores - np.min(scores)) / (np.max(scores) - np.min(scores))

        bm25_norm = normalize(bm25_scores)
        dense_norm = normalize(dense_scores)

        # 4. Hybrid Score
        # If no embeddings, pure BM25. If no BM25 tokens, pure Dense (or 0).
        hybrid_scores = (1 - self.dense_weight) * bm25_norm + self.dense_weight * dense_norm

        # 5. Candidate Selection (based on Hybrid Score)
        candidate_count = max(top_k, int(round(top_k * self.candidate_multiplier)))
        candidate_count = min(candidate_count, len(self.chunks))
        
        # Get indices of top hybrid scores
        top_indices = np.argsort(hybrid_scores)[::-1][:candidate_count]

        # 6. Keyword Boosting
        keyword_summary = None
        if self.keyword_boost > 0:
            predefined_source = "dynamic_simple"
            keywords_to_use = set()

            if query_id is not None and str(query_id) in self.predefined_keywords:
                keywords_to_use = self.predefined_keywords[str(query_id)]
                predefined_source = "query_id"
            elif query.strip() in self.predefined_keywords:
                keywords_to_use = self.predefined_keywords[query.strip()]
                predefined_source = "query_text"
            else:
                if self.keyword_extraction_method == "semantic" and self.embedding_model:
                     keywords_to_use = self._extract_keywords_semantic(query)
                     predefined_source = "dynamic_semantic"
                elif self.language == "zh":
                    keywords_to_use = self._extract_keywords(tokenized_query)
                else:
                    raw_tokens = EN_TOKEN_PATTERN.findall(query.lower())
                    keywords_to_use = {
                        t for t in raw_tokens 
                        if t not in ENGLISH_STOP_WORDS_SET 
                        and len(t) >= self.min_keyword_characters
                    }
                if not keywords_to_use and predefined_source == "dynamic_simple":
                     predefined_source = "dynamic_simple" # Kept same

            keyword_summary = {
                "keywords": sorted(list(keywords_to_use)),
                "boost": self.keyword_boost,
                "predefined_source": predefined_source,
                "query_id_provided": query_id is not None,
            }
            
            # Apply boost ONLY to the selected candidates
            # We re-sort the top_indices based on (Hybrid Score + Boost)
            # Note: hybrid_scores is array of all scores.
            
            boosted_scores = []
            for idx in top_indices:
                base_score = hybrid_scores[idx]
                overlap = self._keyword_overlap(self.chunks[idx], keywords_to_use)
                boosted_score = base_score + (self.keyword_boost * overlap)
                boosted_scores.append((idx, boosted_score))
            
            # Sort by boosted score descending
            boosted_scores.sort(key=lambda x: x[1], reverse=True)
            top_indices = [x[0] for x in boosted_scores]

        # 7. Final Selection
        selected = [self.chunks[idx] for idx in top_indices[:top_k]]

        retrieval_debug = {
            "language": self.language,
            "top_k": top_k,
            "candidate_count": candidate_count,
            "keyword_info": keyword_summary,
            "results": [
                {
                    "metadata": selected_chunk.get("metadata", {}),
                    "preview": selected_chunk.get("page_content", "")[:200],
                    "score": float(hybrid_scores[top_indices[idx]]), # Show hybrid score (pre-boost) or boosted? Let's show hybrid for clarity of base relevance
                }
                for idx, selected_chunk in enumerate(selected)
            ],
            "unsolvable": is_unsolvable
        }

        return selected, retrieval_debug


def create_retriever(chunks, language, config=None):
    """Creates a retriever from document chunks based on config."""
    config = config or {}
    retriever_type = config.get("type", "bm25").lower()
    if retriever_type != "bm25":
        raise ValueError(f"Unsupported retriever type '{retriever_type}'.")

    bm25_cfg = config.get("bm25", {})
    
    return BM25Retriever(
        chunks,
        language,
        k1=bm25_cfg.get("k1", 1.5),
        b=bm25_cfg.get("b", 0.75),
        candidate_multiplier=config.get("candidate_multiplier", 3.0),
        keyword_boost=config.get("keyword_boost", 0.0),
        min_keyword_characters=config.get("min_keyword_characters", 3),
        keyword_extraction_method=config.get("keyword_extraction_method", "simple"),
        embedding_model_path=config.get("embedding_model_path", "My_RAG/models/all_minilm_l6"),
        embedding_provider=config.get("embedding_provider", "local"),
        ollama_host=config.get("ollama_host", "http://ollama-gateway:11434"),
        keyword_file=config.get("keyword_file", "database/database.jsonl"),
    )

def analysis_retriever_result(
    query: str,
    context_chunks: List[Dict[str, Any]],
    language: str,
) -> str:
    """
    透過llm根據queries、retrive到的文章，判斷夠不夠、準不準，不夠的話再用kg
    
    Returns:
        "sufficient" 或 "use_kg"
    """
    if not context_chunks:
        return "use_kg"
    
    # 構建context字符串
    context_parts = []
    for i, chunk in enumerate(context_chunks, start=1):
        content = chunk.get("page_content", "").strip()
        if content:
            # 限制每個chunk的長度，避免prompt太長
            if len(content) > 500:
                content = content[:500] + "..."
            context_parts.append(f"[Chunk {i}]\n{content}")
    
    context = "\n\n".join(context_parts)
    
    # 創建prompt
    prompt = _create_prompt(query, context, language)
    
    try:
        print("🐯 analysis! query:"+query)

        cfg = load_ollama_config()
        client = Client(host=cfg["host"])
        response = client.generate(
            model=cfg["model"],
            prompt=prompt,
            stream=False,
            options={
                "temperature": 0.1,
                "num_ctx": 2048,
            },
        )
        
        result = (response.get("response", "") or "").strip().lower()
        print("🐯 analysis! result:"+result)
        
        # 檢查結果
        if "sufficient" in result:
            return "sufficient"
        elif "use_kg" in result or "kg" in result:
            return "use_kg"
        else:
            # 默認返回use_kg，確保安全
            return "use_kg"
    except Exception as e:
        print(f"Error in analysis_retriever_result: {e}")
        # 遇到錯誤時默認使用KG，確保不會遺漏信息
        return "use_kg"

def _create_prompt(query: str, context: str, language: str) -> str:
    """
    analysis_retriever_result用的prompt
    """
    if language == "zh":
        return f"""你是一个评估检索结果的助手。请根据以下【问题】和【检索到的内容】，判断这些内容是否足够且准确来回答问题。

问题：
{query}

检索到的内容：
{context}

请评估：
1. 这些内容是否足够回答这个问题？（是否包含回答问题所需的关键信息）
2. 这些内容是否准确相关？（是否与问题直接相关，没有太多无关信息）

请只输出以下两种标签之一：
- "sufficient"：内容足够且准确，可以直接回答问题
- "use_kg"：内容不够充分或不准确，需要使用知识图谱检索补充

只输出标签，不要输出其他文字。"""
    else:
        return f"""You are an assistant evaluating retrieval results. Please judge whether the retrieved content is sufficient and accurate to answer the question.

Question:
{query}

Retrieved Content:
{context}

Please evaluate:
1. Is the content sufficient to answer this question? (Does it contain the key information needed?)
2. Is the content accurate and relevant? (Is it directly related to the question without too much irrelevant information?)

Please output only one of the following labels:
- "sufficient": Content is sufficient and accurate, can directly answer the question
- "use_kg": Content is insufficient or inaccurate, need to use knowledge graph retrieval to supplement

Output only the label, no other text."""
def kg_retriever(
    query: str,
    language: str,
    all_chunks: List[Dict[str, Any]],
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """
    使用 My_RAG/kg_output.json 的三元組圖做 KG 檢索。

    流程：
    1. 用 KGRetriever 依 query 算出 KG-based doc 排序 (doc_id list)
    2. 依 doc_id 到 all_chunks 裡撈出對應 chunks
    3. 每個 doc 取 1~2 個代表 chunk，最後回傳最多 top_k 個 chunks
    """
    try:
        kg_ret = create_kg_retriever(
            kg_path="My_RAG/kg_output.json",
            language=language,
        )

        print("🐯 kg! " + query)

        # 1) 依 KG 拿到 doc_id 排序
        doc_ids = kg_ret.retrieve_doc_ids(query, top_k=top_k * 3)  # 多取一些 doc 以備選
        if not doc_ids:
            return []

        # 轉成 set 方便查
        doc_id_set = set(doc_ids)

        # 2) 先依 doc_id 在 all_chunks 裡聚 group
        #    doc_id -> [chunks...]
        doc_to_chunks: Dict[Any, List[Dict[str, Any]]] = {}
        for chunk in all_chunks:
            meta = chunk.get("metadata", {})
            doc_id = meta.get("doc_id")
            if doc_id in doc_id_set:
                doc_to_chunks.setdefault(doc_id, []).append(chunk)

        # 3) 按照 KG 排序的 doc_ids，對每個 doc 選 1~2 個代表 chunk
        selected_chunks: List[Dict[str, Any]] = []
        for doc_id in doc_ids:
            chunks = doc_to_chunks.get(doc_id)
            if not chunks:
                continue

            # 這裡可以做更聰明的挑選（例如選最長、包含最多關鍵詞的 chunk）
            # 目前先簡單取前 2 個
            for c in chunks[:2]:
                selected_chunks.append(c)
                if len(selected_chunks) >= top_k:
                    break
            if len(selected_chunks) >= top_k:
                break

        return selected_chunks[:top_k]

    except Exception as e:
        print(f"Error in kg_retriever: {e}")
        return []
