from rank_bm25 import BM25Okapi
import jieba
import re
import numpy as np
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
from functools import lru_cache
from typing import List, Dict, Any, Optional
from collections import Counter
from ollama import Client

try:
    import faiss  # type: ignore
except Exception:
    faiss = None

try:
    import opencc
except ImportError:
    opencc = None

EN_TOKEN_PATTERN = re.compile(r"[a-z0-9']+")
ENGLISH_STOP_WORDS_SET = set(ENGLISH_STOP_WORDS)

def load_english_learned_stopwords():
    learned_path = Path(__file__).parent / "stopwords_learned_en.txt"
    if learned_path.exists():
        with learned_path.open("r", encoding="utf-8") as f:
            learned = {line.strip() for line in f if line.strip()}
            ENGLISH_STOP_WORDS_SET.update(learned)

load_english_learned_stopwords()


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
            
    # Load learned stopwords if available
    learned_path = Path(__file__).parent / "stopwords_learned_zh.txt"
    if learned_path.exists():
        with learned_path.open("r", encoding="utf-8") as f:
            stop_words.update({line.strip() for line in f if line.strip()})
            
    return stop_words


class OllamaEmbeddings:
    """Wrapper for Ollama embeddings to match SentenceTransformer interface."""
    def __init__(self, model: str, base_url: str):
        self.client = Client(host=base_url)
        self.model = model

    def encode(self, sentences, batch_size=32, convert_to_numpy=True, normalize_embeddings=False, show_progress_bar=False):
        is_single = isinstance(sentences, str)
        inputs = [sentences] if is_single else sentences
        
        embeddings = []
        for text in inputs:
            try:
                if not text or not text.strip():
                    # Return zero vector for empty text to avoid errors
                    # We don't know dim yet, so we'll fix it after first valid one or default to 768
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
            
        # Fix None values by replacing with zero vectors of correct dimension
        valid_embs = [e for e in embeddings if e is not None]
        dim = len(valid_embs[0]) if valid_embs else 768 # Default to 768 if all fail
        
        final_embeddings = []
        for e in embeddings:
            if e is None:
                final_embeddings.append(np.zeros(dim))
            else:
                final_embeddings.append(e)
            
        result = np.array(final_embeddings)
        
        if normalize_embeddings:
            # Avoid divide by zero
            norm = np.linalg.norm(result, axis=1, keepdims=True)
            # Replace 0 norm with 1 to avoid div/0, the vector remains 0
            norm[norm == 0] = 1.0 
            result = result / norm

        if is_single:
            return result[0]
        return result


class HybridRetriever:
    """
    Hybrid retriever combining BM25, TF-IDF, and JM Smoothing with Pseudo-Relevance Feedback.

    Final Score = a*TF-IDF + b*BM25 + c*JM

    Reranking策略（優先順序）:
    1. Cross Encoder（如果有設定 cross_encoder.model）
    2. Dense Bi-Encoder（如果有設定 dense.model）
    3. 只用 hybrid 分數（沒有 reranker）
    """

    def __init__(
        self,
        chunks: List[Dict[str, Any]],
        language: str = "en",
        *,
        k1: float = 1.5,
        b: float = 0.75,
        jm_lambda: float = 0.1,
        weights: Dict[str, float] = None,
        candidate_multiplier: float = 3.0,
        prf_top_k: int = 0,
        prf_term_count: int = 5,
        dense_config: Optional[Dict[str, Any]] = None,
        cross_encoder_config: Optional[Dict[str, Any]] = None,
        ollama_config: Optional[Dict[str, Any]] = None,
        query_expansion_config: Optional[Dict[str, Any]] = None,
        hyde_config: Optional[Dict[str, Any]] = None,
        multi_query_config: Optional[Dict[str, Any]] = None,
        keyword_boost_config: Optional[Dict[str, Any]] = None,
        parent_docs: Optional[List[Dict[str, Any]]] = None,
        pdr_config: Optional[Dict[str, Any]] = None,
    ):
        self.chunks = chunks
        self.language = language
        self.parent_docs = parent_docs
        self.pdr_config = pdr_config or {}
        self.corpus = [chunk.get("page_content", "") for chunk in chunks]
        self.weights = weights or {"tfidf": 0, "bm25": 1, "jm": 0}
        self.candidate_multiplier = max(1.0, candidate_multiplier)
        self.prf_top_k = max(0, int(prf_top_k))
        self.prf_term_count = max(0, int(prf_term_count))

        # ===== Keyword Boost Config =====
        self.keyword_boost_config = keyword_boost_config or {}
        self.doc_boosts = np.ones(len(self.chunks), dtype=np.float32)
        self._init_keyword_boost()

        # ===== Dense Bi-Encoder 設定 =====
        self.dense_config = dense_config or {}
        self.dense_model = None
        self.actual_dense_model_name = None # Store actual model name for debug
        self.dense_batch_size = int(self.dense_config.get("batch_size", 32))
        self.dense_normalize = bool(self.dense_config.get("normalize", True))
        self.dense_query_prefix = self.dense_config.get("query_prefix", "query: ")
        self.dense_passage_prefix = self.dense_config.get("passage_prefix", "passage: ")
        self.dense_strategy = self.dense_config.get("strategy", "rerank")  # "rerank" or "dense_only"
        self.use_faiss = self.dense_config.get("type") == "faiss"
        self.use_faiss_gpu = bool(self.dense_config.get("use_gpu", False))
        self.faiss_index = None
        self.faiss_metric = "ip"

        # ===== Cross Encoder 設定 =====
        self.cross_encoder_config = cross_encoder_config or {}
        self.cross_encoder_model = None
        self.cross_encoder_batch_size = int(
            self.cross_encoder_config.get("batch_size", 32)
        )
        
        # ===== LLM Query Expansion / HyDE / Multi-Query Config =====
        self.ollama_config = ollama_config or {}
        self.query_expansion_config = query_expansion_config or {}
        self.hyde_config = hyde_config or {}
        self.multi_query_config = multi_query_config or {}
        self.llm_client = None

        self.porter_stemmer = PorterStemmer()
        self.chinese_stop_words = load_chinese_stop_words()

        # Initialize OpenCC for Chinese normalization (Must be before tokenization)
        if self.language == "zh" and opencc:
            try:
                self.cc_t2s = opencc.OpenCC('t2s')
            except Exception as e:
                print(f"Warning: Failed to initialize OpenCC: {e}")
                self.cc_t2s = None
        else:
            self.cc_t2s = None

        self.tokenized_corpus = [self._tokenize(doc) for doc in self.corpus]
        self.bm25 = BM25Okapi(self.tokenized_corpus, k1=k1, b=b)

        self.tfidf_vectorizer = TfidfVectorizer(
            tokenizer=self._tokenize_stats, token_pattern=None
        )
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.corpus)

        self.jm_lambda = jm_lambda
        self.doc_lens = [len(d) for d in self.tokenized_corpus]
        self.collection_stats = Counter()
        for doc in self.tokenized_corpus:
            self.collection_stats.update(doc)
        self.collection_len = sum(self.collection_stats.values())

        # 初始化 reranker & LLM
        self._init_dense_encoder()
        self._build_faiss_index()
        self._init_cross_encoder()
        self._init_llm_client()

    def _init_keyword_boost(self):
        """Pre-calculate document boost scores based on keywords."""
        if not self.keyword_boost_config.get("enabled", False):
            return

        keywords = self.keyword_boost_config.get("keywords", [])
        if not keywords:
            return

        boost_factor = float(self.keyword_boost_config.get("boost_factor", 1.2))
        print(f"Initializing keyword boost for {len(keywords)} keywords with factor {boost_factor}")

        # Iterate through all documents and keywords
        # This is done once during initialization
        for i, content in enumerate(self.corpus):
            # Normalize content for matching if needed (e.g. lowercase)
            content_lower = content.lower()
            
            for keyword in keywords:
                if keyword.lower() in content_lower:
                    self.doc_boosts[i] *= boost_factor


    def _init_llm_client(self):
        """Initialize Ollama client if query expansion is enabled."""
        if self.query_expansion_config.get("enabled", False):
            if not self.ollama_config.get("host"):
                print("Warning: Query expansion enabled but no Ollama host provided.")
                return
            try:
                self.llm_client = Client(host=self.ollama_config["host"])
            except Exception as e:
                print(f"Warning: Failed to initialize Ollama client: {e}")
                self.llm_client = None

    def _init_dense_encoder(self):
        """Load the dense encoder lazily if configured."""
        if not self.dense_config.get("enabled", True):
            self.dense_model = None
            return

        # Select model based on language
        if self.language == "zh":
            model_name = self.dense_config.get("model_zh") or self.dense_config.get("model")
        else:
            model_name = self.dense_config.get("model_en") or self.dense_config.get("model")

        if not model_name:
            self.dense_model = None
            self.actual_dense_model_name = None
            return
            
        print(f"Loading dense retriever for {self.language}: {model_name}")
        self.actual_dense_model_name = model_name

        if "embeddinggemma" in model_name or "qwen" in model_name or self.dense_config.get("type") == "ollama":
            host = self.ollama_config.get("host") or "http://localhost:11434"
            self.dense_model = OllamaEmbeddings(
                model=model_name,
                base_url=host
            )
            return

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - dependency guidance
            raise ImportError(
                "sentence-transformers is required for dense re-ranking. "
                "Install it via requirements.txt before enabling the feature."
            ) from exc

        device = self.dense_config.get("device")
        if device:
            self.dense_model = SentenceTransformer(model_name, device=device)
        else:
            self.dense_model = SentenceTransformer(model_name)

    def _build_faiss_index(self):
        """Pre-compute dense embeddings into a FAISS index if enabled."""
        if not self.use_faiss:
            return
        if not faiss:
            print("Warning: FAISS not installed; skipping dense FAISS index.")
            return
        
        # Check for pre-built index on disk
        index_dir = Path(__file__).parent / "indices"
        index_path = index_dir / f"faiss_index_{self.language}.bin"
        
        if index_path.exists():
            print(f"📂 Loading pre-built FAISS index from {index_path}...")
            try:
                self.faiss_index = faiss.read_index(str(index_path))
                self.faiss_metric = "ip" if self.dense_normalize else "l2" # Assume config matches build
                print(f"✓ Loaded index with {self.faiss_index.ntotal} vectors.")
                return
            except Exception as e:
                print(f"⚠️  Failed to load pre-built index: {e}. Rebuilding...")

        if not self.dense_model:
            print("Warning: Dense model not initialized; skipping FAISS index.")
            return

        try:
            print("🏗️  Building FAISS index from scratch...")
            passages = [
                f"{self.dense_passage_prefix}{chunk.get('page_content', '')}"
                for chunk in self.chunks
            ]
            if not passages:
                return

            embeddings = self.dense_model.encode(
                passages,
                batch_size=self.dense_batch_size,
                convert_to_numpy=True,
                normalize_embeddings=self.dense_normalize,
                show_progress_bar=False,
            )
            embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=0.0, neginf=0.0).astype(
                np.float32
            )

            self.faiss_metric = "ip" if self.dense_normalize else "l2"
            if self.faiss_metric == "ip":
                index = faiss.IndexFlatIP(embeddings.shape[1])
            else:
                index = faiss.IndexFlatL2(embeddings.shape[1])

            # Move to GPU if requested and available
            if self.use_faiss_gpu and hasattr(faiss, "StandardGpuResources"):
                try:
                    res = faiss.StandardGpuResources()
                    index = faiss.index_cpu_to_gpu(res, 0, index)
                except Exception as gpu_exc:
                    print(f"Warning: Failed to move FAISS index to GPU: {gpu_exc}")

            index.add(embeddings)
            self.faiss_index = index
        except Exception as e:
            print(f"Warning: Failed to build FAISS index: {e}")
            self.faiss_index = None

    def _init_cross_encoder(self):
        """Load the cross encoder lazily if configured."""
        if not self.cross_encoder_config.get("enabled", True):
            self.cross_encoder_model = None
            return

        model_name = self.cross_encoder_config.get("model")
        if not model_name:
            self.cross_encoder_model = None
            return

        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover - dependency guidance
            raise ImportError(
                "sentence-transformers is required for cross-encoder re-ranking. "
                "Install it via requirements.txt before enabling the feature."
            ) from exc

        device = self.cross_encoder_config.get("device")
        if device:
            self.cross_encoder_model = CrossEncoder(model_name, device=device)
        else:
            self.cross_encoder_model = CrossEncoder(model_name)

    def _normalize_chinese_text(self, text: str) -> str:
        """
        Normalize Chinese text:
        1. Traditional to Simplified (if OpenCC available).
        2. Full-width to Half-width conversion.
        """
        if self.cc_t2s:
            text = self.cc_t2s.convert(text)
            
        # Full-width space (U+3000) -> Half-width space (U+0020)
        text = text.replace('\u3000', ' ')
        
        # Range U+FF01 to U+FF5E (Full-width ASCII) -> U+0021 to U+007E (Half-width ASCII)
        chars = []
        for char in text:
            code = ord(char)
            if 0xFF01 <= code <= 0xFF5E:
                chars.append(chr(code - 0xFEE0))
            else:
                chars.append(char)
        return "".join(chars)

    def _tokenize(self, text: str):
        if self.language == "zh":
            text = self._normalize_chinese_text(text)
            tokens = [tok.strip() for tok in jieba.cut(text) if tok.strip()]
            stop_words = self.chinese_stop_words
            return [tok for tok in tokens if tok not in stop_words]

        raw_tokens = EN_TOKEN_PATTERN.findall(text.lower())
        filtered = [
            token
            for token in raw_tokens
            if token and token not in ENGLISH_STOP_WORDS_SET
        ]
        return [self._stem_english(token) for token in filtered]

    def _tokenize_stats(self, text: str):
        return self._tokenize(text)

    def _stem_english(self, token: str) -> str:
        return self.porter_stemmer.stem(token)

    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        if len(scores) == 0:
            return scores
        if np.min(scores) == np.max(scores):
            return np.ones_like(scores)
        return (scores - np.min(scores)) / (np.max(scores) - np.min(scores))

    def _get_jm_scores(self, tokenized_query):
        scores = []
        query_tokens = [t for t in tokenized_query if t in self.collection_stats]

        if not query_tokens:
            return np.zeros(len(self.chunks))

        for i, doc_tokens in enumerate(self.tokenized_corpus):
            score = 0.0
            doc_stats = Counter(doc_tokens)
            doc_len = self.doc_lens[i]

            if doc_len == 0:
                scores.append(-np.inf)
                continue

            for token in query_tokens:
                p_doc = doc_stats[token] / doc_len
                p_coll = self.collection_stats[token] / self.collection_len
                prob = (1 - self.jm_lambda) * p_doc + self.jm_lambda * p_coll

                if prob > 0:
                    score += np.log(prob)
                else:
                    score += -20.0
            scores.append(score)

        return np.array(scores)

    def _calculate_hybrid_scores(self, query_text, tokenized_query):
        # TF-IDF
        query_vec = self.tfidf_vectorizer.transform([query_text])
        if query_vec.nnz == 0:
            tfidf_raw = np.zeros(self.tfidf_matrix.shape[0])
        else:
            tfidf_raw = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        
        # BM25
        bm25_raw = np.array(self.bm25.get_scores(tokenized_query))
        
        # JM
        jm_raw = self._get_jm_scores(tokenized_query)

        # Normalize
        norm_tfidf = self._normalize_scores(tfidf_raw)
        norm_bm25 = self._normalize_scores(bm25_raw)
        norm_jm = self._normalize_scores(jm_raw)

        # Combine
        a = self.weights.get("tfidf", 0)
        b = self.weights.get("bm25", 1)
        c = self.weights.get("jm", 0)
        
        final_scores = (a * norm_tfidf) + (b * norm_bm25) + (c * norm_jm)
        
        # Apply Keyword Boost
        if self.keyword_boost_config.get("enabled", False):
            final_scores = final_scores * self.doc_boosts
        
        return final_scores, (bm25_raw, tfidf_raw, jm_raw)

    def _dense_rerank(self, query_text: str, candidate_indices: np.ndarray):
        if not self.dense_model or len(candidate_indices) == 0:
            return candidate_indices, {}

        clean_query = query_text.strip()
        if not clean_query:
            return candidate_indices, {}

        try:
            candidate_indices = np.asarray(candidate_indices)
            query_payload = f"{self.dense_query_prefix}{clean_query}"
            query_embedding = self.dense_model.encode(
                query_payload,
                convert_to_numpy=True,
                normalize_embeddings=self.dense_normalize,
                show_progress_bar=False,
            )

            passages = [
                f"{self.dense_passage_prefix}{self.chunks[idx].get('page_content', '')}"
                for idx in candidate_indices
            ]
            
            # Filter out empty passages to prevent encoding errors
            valid_indices = []
            valid_passages = []
            for i, passage in enumerate(passages):
                if passage.strip():
                    valid_indices.append(candidate_indices[i])
                    valid_passages.append(passage)
            
            if not valid_passages:
                return candidate_indices, {}
            
            doc_embeddings = self.dense_model.encode(
                valid_passages,
                batch_size=self.dense_batch_size,
                convert_to_numpy=True,
                normalize_embeddings=self.dense_normalize,
                show_progress_bar=False,
            )

            # SANITY CHECK: Replace NaNs and Infs with 0.0 to prevent runtime warnings/errors
            doc_embeddings = np.nan_to_num(doc_embeddings, nan=0.0, posinf=0.0, neginf=0.0)
            query_embedding = np.nan_to_num(query_embedding, nan=0.0, posinf=0.0, neginf=0.0)

            dense_scores = doc_embeddings @ query_embedding
            order = np.argsort(dense_scores)[::-1]
            reranked = np.array(valid_indices)[order]
            dense_score_map = {
                idx: float(score) for idx, score in zip(valid_indices, dense_scores)
            }
            return reranked, dense_score_map
            
        except Exception as e:
            # Log warning and fall back to original ranking
            print(f"Warning: Dense reranking failed: {e}. Falling back to hybrid scores.")
            return candidate_indices, {}

    def _dense_search_faiss(self, query_text: str, candidate_indices: Optional[np.ndarray], top_k: int):
        """
        Use FAISS to retrieve dense results. If candidate_indices is provided,
        filter FAISS hits to that subset; otherwise run dense-only.
        """
        if not self.faiss_index or not self.dense_model:
            return np.array([]), {}

        clean_query = query_text.strip()
        if not clean_query:
            return np.array([]), {}

        try:
            query_payload = f"{self.dense_query_prefix}{clean_query}"
            query_embedding = self.dense_model.encode(
                query_payload,
                convert_to_numpy=True,
                normalize_embeddings=self.dense_normalize,
                show_progress_bar=False,
            )
            query_embedding = np.nan_to_num(query_embedding, nan=0.0, posinf=0.0, neginf=0.0).astype(
                np.float32
            )
            if query_embedding.ndim == 1:
                query_embedding = query_embedding[None, :]

            # Search a bit deeper if filtering to candidates
            search_k = min(len(self.chunks), max(top_k * 5, top_k))
            scores, indices = self.faiss_index.search(query_embedding, search_k)
            faiss_scores = scores[0]
            faiss_indices = indices[0]

            results = []
            if candidate_indices is not None:
                candidate_set = set(int(i) for i in np.asarray(candidate_indices))
                for idx, score in zip(faiss_indices, faiss_scores):
                    if idx == -1:
                        continue
                    if idx in candidate_set:
                        results.append((idx, score))
                    if len(results) >= top_k:
                        break
                # If no overlap, fall back to candidates in original order
                if not results:
                    return np.asarray(candidate_indices)[:top_k], {}
            else:
                for idx, score in zip(faiss_indices, faiss_scores):
                    if idx == -1:
                        continue
                    results.append((idx, score))

            # Convert distances to similarity if needed
            dense_scores = {}
            ordered_indices = []
            for idx, score in results:
                sim = float(-score) if self.faiss_metric == "l2" else float(score)
                dense_scores[idx] = sim
                ordered_indices.append(idx)

            # Sort by similarity descending
            ordered_indices = sorted(ordered_indices, key=lambda i: dense_scores[i], reverse=True)
            return np.array(ordered_indices), dense_scores
        except Exception as e:
            print(f"Warning: FAISS dense search failed: {e}")
            return np.array([]), {}

    def _cross_encoder_rerank(self, query_text: str, candidate_indices: np.ndarray):
        """
        使用 CrossEncoder 對 candidate_indices 重新排序。
        """
        if not self.cross_encoder_model or len(candidate_indices) == 0:
            return candidate_indices, {}

        clean_query = query_text.strip()
        if not clean_query:
            return candidate_indices, {}

        candidate_indices = np.asarray(candidate_indices)

        pairs = []
        for idx in candidate_indices:
            passage = self.chunks[idx].get("page_content", "")
            pairs.append((clean_query, passage))

        # CrossEncoder.predict 會回傳一個 list / np.array 的分數
        scores = self.cross_encoder_model.predict(
            pairs,
            batch_size=self.cross_encoder_batch_size,
            show_progress_bar=False,
        )
        scores = np.asarray(scores)

        order = np.argsort(scores)[::-1]
        reranked = candidate_indices[order]

        score_map = {idx: float(score) for idx, score in zip(candidate_indices, scores)}
        return reranked, score_map
        
    def _rewrite_query(self, query: str) -> str:
        """
        Rewrite the query using LLM to expand keywords and improve retrieval.
        """
        if not self.llm_client or not self.query_expansion_config.get("enabled", False):
            return query

        try:
            # Basic prompt for query expansion
            if self.language == "zh":
                prompt = f"""你是一個搜索查詢優化助手。請重寫並擴展以下查詢，使其包含更多相關的關鍵詞，以便在文檔庫中進行更好的檢索。
查詢: {query}
僅輸出重寫後的查詢，不要包含任何解釋或其他文字。"""
            else:
                prompt = f"""You are a search query optimization assistant. Please rewrite and expand the following query to include more relevant keywords for better retrieval in a document corpus.
Query: {query}
Output ONLY the rewritten query, without any explanation."""

            response = self.llm_client.generate(
                model=self.ollama_config["model"],
                prompt=prompt,
                options={"temperature": 0.2}
            )
            
            rewritten = response["response"].strip()
            # Clean up quotes if present
            if rewritten.startswith('"') and rewritten.endswith('"'):
                rewritten = rewritten[1:-1]
            
            # If rewriting failed (e.g. empty), return original
            if not rewritten:
                return query
                
            # Concatenate original query to preserve original intent (safe bet)
            # We weight the original query more by putting it first? 
            # Actually for BM25, having more terms is good.
            return f"{query} {rewritten}"
            
        except Exception as e:
            print(f"Warning: Query expansion failed: {e}")
            return query


    def _generate_query_variations(self, query: str) -> List[str]:
        """Generate multiple variations of the query."""
        if not self.llm_client:
            return [query]
            
        num_versions = self.multi_query_config.get("num_versions", 3)
        
        try:
            if self.language == "zh":
                prompt = f"""你是一个AI语言模型助手。你的任务是为给定的用户问题生成{num_versions}个不同版本，以便从向量数据库中检索相关文档。
通过生成多视角的查询，你的目标是帮助用户克服基于距离的相似性搜索的一些局限性。
请只输出替代问题，每行一个。不要包含任何其他文字、编号或通过“好的”、“以下是”等开头。
原问题: {query}"""
            else:
                prompt = f"""You are an AI language model assistant. Your task is to generate {num_versions} different versions of the given user question to retrieve relevant documents from a vector database.
By generating multiple perspectives on the user question, your goal is to help the user overcome some of the limitations of distance-based similarity search.
Provide ONLY the alternative questions separated by newlines. Do not number them. Do not include any introductory text like "Here are...".
Original question: {query}"""

            response = self.llm_client.generate(
                model=self.ollama_config["model"],
                prompt=prompt,
                options={"temperature": 0.5}
            )
            
            raw_variations = [v.strip() for v in response["response"].strip().split('\n') if v.strip()]
            variations = []
            for v in raw_variations:
                # Basic filtering of conversational filler
                lower_v = v.lower()
                if lower_v.startswith("here are") or lower_v.startswith("sure,") or lower_v.startswith("certainly"):
                    continue
                # Remove numbering if model ignored instruction (e.g., "1. What is...")
                v = re.sub(r'^\d+[\.\)]\s*', '', v)
                variations.append(v)

            # Add original query if not present (Priority #1)
            if query not in variations:
                variations.insert(0, query)
                
            return variations[:num_versions+1]
            
        except Exception as e:
            print(f"Warning: Multi-query generation failed: {e}")
            return [query]

    def _reciprocal_rank_fusion(self, results_list: List[List[int]], k=60):
        """
        Fuse rank lists using Reciprocal Rank Fusion (RRF).
        results_list: List of lists, where each inner list contains document indices sorted by rank.
        """
        fused_scores = {}
        for i, rank_list in enumerate(results_list):
            # Boost the original query (index 0) to reduce noise impact from poor variations
            weight = 3.0 if i == 0 else 1.0
            
            for rank, doc_idx in enumerate(rank_list):
                if doc_idx not in fused_scores:
                    fused_scores[doc_idx] = 0
                fused_scores[doc_idx] += weight * (1 / (k + rank + 1))
        
        # Sort by fused score descending
        reranked_results = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        return reranked_results

    def retrieve(self, query, top_k=5):
        if not self.chunks:
            return [], {}

        # --- Multi-Query / RRF Logic ---
        if self.multi_query_config.get("enabled", False):
            queries = self._generate_query_variations(query)
            # print(f"Multi-Query Variations: {queries}")
            
            all_ranked_lists = []
            debug_results = []
            
            for q in queries:
                # Run single-query retrieval logic for each variation
                # Note: Recursion is dangerous if not careful, so we'll refactor the core logic 
                # to a separate method _retrieve_single(q) or just call existing logic inline
                
                # For now, let's just copy the core logic inline to avoid huge refactor risks
                # or simpler: temporarily disable multi_query to call self.retrieve(q) 
                # BUT that would recurse infinite loop.
                
                # BETTER: Extract single query logic below into _retrieve_single
                indices, _ = self._retrieve_single(q, top_k=max(top_k, 20)) # Get deeper list for fusion
                all_ranked_lists.append(indices)
            
            # Fuse results
            fused_results = self._reciprocal_rank_fusion(all_ranked_lists)
            final_top_indices = [idx for idx, score in fused_results[:top_k]]
            
            selected = [self.chunks[i] for i in final_top_indices]
            
            return selected, {
                "language": self.language,
                "rank_basis": "multi_query_rrf",
                "variations": queries,
                "results": [
                    {
                        "metadata": self.chunks[i].get("metadata", {}),
                        "preview": self.chunks[i].get("page_content", "")[:100],
                        "score": score
                    }
                    for i, score in fused_results[:top_k]
                ]
            }

        # --- Standard Single Query Logic (wrapped in helper for Multi-Query use) ---
        top_indices, retrieval_debug = self._retrieve_single(query, top_k)
        selected = [self.chunks[i] for i in top_indices]
        return selected, retrieval_debug

    def _retrieve_single(self, query, top_k=5):
        """
        Performs the standard Hybrid + Dense retrieval for a single query string.
        Returns: (top_indices_list, debug_info)
        """
        # --- Step 0: Query Processing (Expansion or HyDE) ---
        search_query = query # For BM25/Lexical
        dense_query_text = query # For Dense/Embedding

        # HyDE Logic
        if self.hyde_config.get("enabled", False):
            hypothetical_doc = self._generate_hypothetical_doc(query)
            dense_query_text = hypothetical_doc 
            search_query = query 
        elif self.query_expansion_config.get("enabled", False):
            search_query = self._rewrite_query(query)
            dense_query_text = search_query

        tokenized_query = self._tokenize(search_query)
        final_scores, (bm25_s, tfidf_s, jm_s) = self._calculate_hybrid_scores(search_query, tokenized_query)

        # --- Pseudo-Relevance Feedback (Improved) ---
        expanded_info = None
        if self.prf_top_k > 0 and self.prf_term_count > 0:
            # Get top docs from initial pass
            temp_top_indices = np.argsort(final_scores)[::-1][:self.prf_top_k]
            
            # Only apply PRF if top results have reasonable scores
            top_scores = [final_scores[i] for i in temp_top_indices]
            avg_top_score = np.mean(top_scores) if top_scores else 0
            
            if avg_top_score > 0.1: 
                feedback_tokens = []
                for idx in temp_top_indices:
                    feedback_tokens.extend(self.tokenized_corpus[idx])
                
                if feedback_tokens:
                    term_freq = Counter(feedback_tokens)
                    query_terms_set = set(tokenized_query)
                    scored_terms = []
                    for term, count in term_freq.items():
                        if term in query_terms_set: continue
                        doc_freq = sum(1 for doc in self.tokenized_corpus if term in doc)
                        if doc_freq > len(self.corpus) * 0.5: continue
                        idf = np.log(len(self.corpus) / (1 + doc_freq))
                        score = count * idf
                        scored_terms.append((term, score))
                    
                    scored_terms.sort(key=lambda x: x[1], reverse=True)
                    new_terms = [term for term, score in scored_terms[:self.prf_term_count]]
                    
                    if new_terms:
                        tokenized_query.extend(new_terms)
                        expanded_query_text = search_query + " " + " ".join(new_terms)
                        expanded_info = new_terms
                        # Re-calculate scores
                        final_scores, (bm25_s, tfidf_s, jm_s) = self._calculate_hybrid_scores(expanded_query_text, tokenized_query)

        # Candidate Selection
        candidate_target = max(top_k, int(top_k * self.candidate_multiplier))
        candidate_count = min(len(self.chunks), candidate_target)
        candidate_indices = np.argsort(final_scores)[::-1][:candidate_count]

        # Reranking
        dense_scores: Dict[int, float] = {}
        rank_basis = "hybrid"

        if self.cross_encoder_model and candidate_indices.size > 0:
            rank_basis = "cross_encoder"
            reranked_indices, dense_scores = self._cross_encoder_rerank(
                dense_query_text, candidate_indices
            )
        elif self.use_faiss and self.faiss_index:
            rank_basis = "dense_faiss"
            if self.dense_strategy == "dense_only":
                reranked_indices, dense_scores = self._dense_search_faiss(
                    dense_query_text, None, top_k
                )
            else:
                reranked_indices, dense_scores = self._dense_search_faiss(
                    dense_query_text, candidate_indices, top_k
                )
            if reranked_indices.size == 0:
                reranked_indices = candidate_indices
        elif self.dense_model and candidate_indices.size > 0:
            rank_basis = "dense"
            reranked_indices, dense_scores = self._dense_rerank(
                dense_query_text, candidate_indices
            )
        else:
            reranked_indices = candidate_indices

        # Final Selection
        # --- PARENT DOCUMENT RETRIEVAL LOGIC ---
        use_pdr = self.pdr_config.get("enabled", False)
        
        if use_pdr and self.parent_docs:
            selected_docs = []
            seen_parent_ids = set()
            final_top_indices = [] # Track which chunks triggered the parents
            
            # Iterate through ranked chunks and pick unique parents until we hit top_k
            count = 0
            for idx in reranked_indices:
                if count >= top_k:
                    break
                    
                chunk = self.chunks[idx]
                parent_id = chunk.get("metadata", {}).get("doc_idx")
                
                if parent_id is not None and 0 <= parent_id < len(self.parent_docs):
                    if parent_id not in seen_parent_ids:
                        seen_parent_ids.add(parent_id)
                        
                        # Get parent doc
                        p_doc = self.parent_docs[parent_id]
                        
                        # Normalize parent doc to have 'page_content' if it only has 'content'
                        # We create a shallow copy to avoid modifying original if possible, 
                        # but for performance/memory we might just wrap it.
                        # Assuming structure matches input jsonl
                        content = p_doc.get("content", "") or p_doc.get("page_content", "")
                        
                        # Create a standardized dict for generation
                        ret_doc = {
                            "page_content": content,
                            "metadata": p_doc, # Store full original doc as metadata
                            # inherited metadata from chunk? No, this is the parent.
                        }
                        
                        selected_docs.append(ret_doc)
                        final_top_indices.append(idx)
                        count += 1
            
            top_indices = np.array(final_top_indices) if final_top_indices else np.array([])
            selected = selected_docs
            rank_basis += "_parent_doc"

        else:
            top_indices = reranked_indices[:top_k]
            selected = [self.chunks[i] for i in top_indices]
        
        retrieval_debug = {
            "language": self.language,
            "top_k": top_k,
            "candidate_count": int(candidate_count),
            "candidate_multiplier": self.candidate_multiplier,
            "weights": self.weights,
            "prf_expanded_terms": expanded_info,
            "rank_basis": rank_basis, 
            "dense": {
                "enabled": bool(self.dense_model),
                "model": getattr(self, "actual_dense_model_name", self.dense_config.get("model")),
                "backend": "faiss" if self.use_faiss else "encoder",
            },
            "pdr": {
                "enabled": bool(use_pdr),
                "parent_docs_available": bool(self.parent_docs)
            },
            "results": [
                {
                    "metadata": self.chunks[i].get("metadata", {}) if i < len(self.chunks) else {},
                    "preview": selected[n].get("page_content", "")[:200] if use_pdr else self.chunks[i].get("page_content", "")[:200],
                    "score": dense_scores.get(i, float(final_scores[i])) if i in dense_scores or i < len(final_scores) else 0.0,
                    "chunk_preview": self.chunks[i].get("page_content", "")[:100] if use_pdr else None
                }
                for n, i in enumerate(top_indices)
            ],
        }
        
        return selected, retrieval_debug


def create_retriever(chunks, language, config=None, parent_docs=None):
    config = config or {}
    weights = config.get("weights", {"tfidf": 0, "bm25": 1, "jm": 0})
    dense_config = config.get("dense", {})
    cross_encoder_config = config.get("cross_encoder", {})
    ollama_config = config.get("ollama", {}) 
    
    return HybridRetriever(
        chunks,
        language,
        k1=config.get("bm25", {}).get("k1", 1.5),
        b=config.get("bm25", {}).get("b", 0.75),
        jm_lambda=config.get("jm", {}).get("lambda", 0.1),
        weights=weights,
        candidate_multiplier=config.get("candidate_multiplier", 3.0),
        prf_top_k=config.get("prf_top_k", 3),
        prf_term_count=config.get("prf_term_count", 5),
        dense_config=dense_config,
        cross_encoder_config=cross_encoder_config,
        ollama_config=ollama_config, 
        query_expansion_config=config.get("query_expansion", {}),
        hyde_config=config.get("hyde", {}),
        multi_query_config=config.get("multi_query", {}),
        keyword_boost_config=config.get("keyword_boost", {}),
        parent_docs=parent_docs,
        pdr_config=config.get("parent_document_retrieval", {}),
    )
