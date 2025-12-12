from rank_bm25 import BM25Okapi
import jieba
import json
import re
import numpy as np
import faiss
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sentence_transformers import SentenceTransformer
from pathlib import Path
from functools import lru_cache
from typing import List, Dict, Any, Optional
from ollama import Client


EN_TOKEN_PATTERN = re.compile(r"[a-z0-9']+")
ENGLISH_STOP_WORDS_SET = set(ENGLISH_STOP_WORDS)


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
    """Hybrid retriever (BM25 + Dense) with optional keyword / KG re-ranking."""

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
        kg_retriever=None,
        kg_boost: float = 0.0,
        dense_weight: float = 0.5,  # 0.0 ~ 1.0
    ):
        self.chunks = chunks
        self.language = language
        self.corpus = [chunk.get("page_content", "") for chunk in chunks]
        self.predefined_keywords: Dict[str, set] = {}
        self.unsolvable_queries: set = set()
        if keyword_file:
            self._load_predefined_keywords(keyword_file)

        # KG
        self.kg_retriever = kg_retriever
        self.kg_boost = max(0.0, kg_boost)
        self._debug_kg = False  # 由 create_retriever 設定

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

        # ===== Embedding Model =====
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

        # ===== FAISS Index & Corpus Embeddings =====
        index_dir = Path("My_RAG/indices")
        index_dir.mkdir(parents=True, exist_ok=True)
        index_path = index_dir / f"faiss_index_{self.language}.bin"

        if self.embedding_model:
            if index_path.exists():
                print(f"Loading existing FAISS index from {index_path}...")
                try:
                    self.faiss_index = faiss.read_index(str(index_path))

                    if isinstance(self.faiss_index, faiss.IndexFlat):
                        ntotal = self.faiss_index.ntotal
                        if ntotal == len(self.corpus):
                            self.chunk_embeddings = self.faiss_index.reconstruct_n(0, ntotal)

                            test_emb = self._compute_embeddings(["test"])
                            if test_emb and len(test_emb) > 0:
                                test_array = np.array(test_emb)
                                if test_array.ndim == 1:
                                    expected_dim = test_array.shape[0]
                                else:
                                    expected_dim = test_array.shape[-1]
                                actual_dim = self.chunk_embeddings.shape[1]
                                if expected_dim != actual_dim:
                                    print(
                                        f"Warning: Embedding dim mismatch. "
                                        f"Index={actual_dim}, model={expected_dim}. Recomputing."
                                    )
                                    self.faiss_index = None
                                    self.chunk_embeddings = None
                                else:
                                    print(f"Loaded {ntotal} embeddings from FAISS (dim={actual_dim}).")
                            else:
                                print("Warning: cannot verify embedding dim. Recomputing.")
                                self.faiss_index = None
                                self.chunk_embeddings = None
                        else:
                            print(
                                f"Warning: Index size ({ntotal}) != corpus size ({len(self.corpus)}). Recomputing."
                            )
                            self.faiss_index = None
                            self.chunk_embeddings = None
                    else:
                        print("Loaded index is not IndexFlat, recomputing.")
                        self.faiss_index = None
                        self.chunk_embeddings = None
                except Exception as e:
                    print(f"Error loading FAISS index: {e}. Recomputing.")
                    self.faiss_index = None
                    self.chunk_embeddings = None

            if self.faiss_index is None:
                print("Computing chunk embeddings for dense retrieval...")
                embeddings = self._precompute_corpus_embeddings(self.corpus)
                if embeddings is not None and len(embeddings) > 0:
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

        # ===== BM25 =====
        self.porter_stemmer = PorterStemmer()
        self.chinese_stop_words = load_chinese_stop_words()
        self.tokenized_corpus = [self._tokenize(doc) for doc in self.corpus]
        self.bm25 = BM25Okapi(self.tokenized_corpus, k1=k1, b=b)

    # ======================
    # Keyword file
    # ======================

    def _load_predefined_keywords(self, path: str):
        p = Path(path)
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
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

    # ======================
    # Tokenization
    # ======================

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

    # ======================
    # Embeddings
    # ======================

    def _compute_embeddings(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        if self.embedding_provider == "ollama":
            try:
                response = self.ollama_client.embed(model=self.embedding_model, input=texts)
                return response.embeddings
            except Exception as e:
                print(f"Ollama embedding error: {e}")
                return []
        else:
            return self.embedding_model.encode(texts)

    def _precompute_corpus_embeddings(self, texts: List[str], batch_size: int = 32):
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            embeddings = self._compute_embeddings(batch)
            if embeddings is not None and len(embeddings) > 0:
                all_embeddings.extend(embeddings)
            else:
                dim = 384
                if all_embeddings:
                    dim = len(all_embeddings[0])
                all_embeddings.extend([[0.0] * dim] * len(batch))
        return np.array(all_embeddings)

    # ======================
    # Keyword extraction
    # ======================

    def _extract_keywords_semantic(self, query, top_k=5):
        if not self.embedding_model:
            return set()

        if self.language == "zh":
            tokens = [t for t in jieba.cut(query) if t.strip()]
        else:
            tokens = [t for t in query.split() if t.strip()]

        candidates = set()
        for n in range(1, 5):
            for i in range(len(tokens) - n + 1):
                ngram = "".join(tokens[i : i + n]) if self.language == "zh" else " ".join(
                    tokens[i : i + n]
                )
                if len(ngram.strip()) >= self.min_keyword_characters:
                    candidates.add(ngram)

        candidates = list(candidates)
        if not candidates:
            return set()

        query_embedding = self._compute_embeddings([query])
        candidate_embeddings = self._compute_embeddings(candidates)
        if not query_embedding or not candidate_embeddings:
            return set()

        query_vec = np.array(query_embedding).astype(np.float32)
        query_norm = np.linalg.norm(query_vec)
        if query_norm <= 1e-9:
            return set()
        query_vec_norm = query_vec / query_norm

        cand_vecs = np.array(candidate_embeddings).astype(np.float32)
        cand_norms = np.linalg.norm(cand_vecs, axis=1, keepdims=True)
        cand_vecs_norm = np.divide(cand_vecs, cand_norms, where=cand_norms > 1e-9)

        distances = np.dot(cand_vecs_norm, query_vec_norm.T).flatten()
        top_indices = np.argsort(distances)[-top_k:]
        return {candidates[i] for i in top_indices}

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

    # ======================
    # KG: multi-hop 判斷 + boost
    # ======================

    def _is_multi_hop_query(self, query: str) -> bool:
        q = query.lower()

        multi_hop_markers_en = [
            "before and after",
            "over time",
            "timeline",
            "sequence of events",
            "first",
            "then",
            "after that",
            "afterwards",
            "between",
            "from",
            "until",
            "up to",
            "change in",
            "changes in",
            "evolution of",
            "history of",
        ]

        multi_hop_markers_zh = [
            "前后",
            "之前",
            "之後",
            "之后",
            "期間",
            "期间",
            "历程",
            "過程",
            "过程",
            "多次",
            "先後",
            "先后",
            "從",
            "从",
            "一直到",
        ]

        if any(m in q for m in multi_hop_markers_en):
            return True
        if any(m in query for m in multi_hop_markers_zh):
            return True

        two_event_markers = [" and ", " & "]
        event_words = [
            "resignation",
            "appointment",
            "merger",
            "acquisition",
            "default",
            "restructuring",
            "lawsuit",
            "investigation",
        ]
        if any(sep in q for sep in two_event_markers):
            if sum(w in q for w in event_words) >= 2:
                return True

        return False

    def _get_kg_boost_scores(self, query: str) -> Dict[int, float]:
        """
        Triples-first KG boost:
        - Ask KGRetriever to rank doc_ids with *scores* (0~1) and evidence.
        - Convert those scores to small additive boosts on the hybrid scores.

        Why:
        - Rank-based decay (old) ignores how strongly a doc matches the query.
        - Using KG scores tends to produce more accurate top chunks and more comparable references.
        """
        if not self.kg_retriever or self.kg_boost <= 0:
            self._last_kg_ranked_docs = None
            return {}

        try:
            use_multi_hop = self._is_multi_hop_query(query)
            ranked_docs = self.kg_retriever.rank_docs(
                query=query,
                top_k=50,  # keep small to avoid noisy boost
                use_multi_hop=use_multi_hop,
                return_debug=getattr(self, "_debug_kg", False),
            )

            self._last_kg_ranked_docs = ranked_docs  # for downstream debug / reference output

            if not ranked_docs:
                if getattr(self, "_debug_kg", False):
                    print(f"[KG Debug] No ranked docs for query: {query[:80]}...")
                return {}

            # If KG returns too many docs, treat as noisy
            if len(ranked_docs) > 50:
                if getattr(self, "_debug_kg", False):
                    print(f"[KG Debug] Too many KG docs ({len(ranked_docs)}), ignore.")
                return {}

            # Convert KG relevance score -> additive boost
            # Keep the boost small (otherwise it dominates dense/BM25).
            # Typical: kg_boost=1.5 => max additive ~0.25
            max_add = min(0.25, 0.15 + 0.07 * float(self.kg_boost))
            min_add = 0.02

            boost_scores: Dict[int, float] = {}
            for d in ranked_docs:
                doc_id = int(d["doc_id"])
                s = float(d.get("score", 0.0))
                # non-linear shaping: emphasize high-confidence docs
                shaped = s ** 1.8
                boost = min_add + shaped * (max_add - min_add)
                boost_scores[doc_id] = float(boost)

            if getattr(self, "_debug_kg", False):
                top3 = ranked_docs[:3]
                print(
                    f"[KG Debug] use_multi_hop={use_multi_hop}, docs={len(ranked_docs)}, "
                    f"top3={[d['doc_id'] for d in top3]}, boosts={[boost_scores[int(d['doc_id'])] for d in top3]}"
                )

            return boost_scores

        except Exception as e:
            print(f"Warning: KG retrieval failed: {e}")
            self._last_kg_ranked_docs = None
            return {}

    # ======================
    # Main retrieve
    # ======================

    def retrieve(self, query: str, top_k: int = 5, query_id=None):
        is_unsolvable = False
        if (query_id is not None and str(query_id) in self.unsolvable_queries) or (
            query.strip() in self.unsolvable_queries
        ):
            is_unsolvable = True

        if not self.chunks:
            return [], {
                "language": self.language,
                "top_k": top_k,
                "candidate_count": 0,
                "keyword_info": None,
                "kg_info": None,
                "results": [],
                "unsolvable": is_unsolvable,
            }

        # 1. Sparse retrieval (BM25)
        tokenized_query = self._tokenize(query)
        if not tokenized_query:
            bm25_scores = np.zeros(len(self.chunks))
        else:
            bm25_scores = np.array(self.bm25.get_scores(tokenized_query))

        # 2. Dense retrieval
        dense_scores = np.zeros(len(self.chunks))
        if self.chunk_embeddings is not None:
            query_emb = self._compute_embeddings([query])
            if query_emb is not None and len(query_emb) > 0:
                query_vec = np.array(query_emb).astype(np.float32)
                query_norm = np.linalg.norm(query_vec)

                if query_norm > 1e-9:
                    query_vec_norm = query_vec / query_norm
                    if query_vec_norm.ndim == 1:
                        query_vec_norm = query_vec_norm.reshape(1, -1)

                    try:
                        raw_dense_scores = np.dot(self.chunk_embeddings, query_vec_norm.T).flatten()
                        dense_scores = np.nan_to_num(raw_dense_scores, nan=0.0)
                    except Exception as e:
                        print(f"Error in dense retrieval scoring: {e}")
                        dense_scores = np.zeros(len(self.chunks))

        # 3. Normalize
        def normalize(scores):
            if np.max(scores) == np.min(scores):
                return scores
            return (scores - np.min(scores)) / (np.max(scores) - np.min(scores))

        bm25_norm = normalize(bm25_scores)
        dense_norm = normalize(dense_scores)

        # 4. Hybrid score
        hybrid_scores = (1 - self.dense_weight) * bm25_norm + self.dense_weight * dense_norm

        # 5. Apply KG boost on hybrid_scores (after normalization)
        kg_boost_scores = self._get_kg_boost_scores(query)
        boosted_count = 0
        if kg_boost_scores:
            for idx, chunk in enumerate(self.chunks):
                doc_id = chunk.get("metadata", {}).get("doc_id")
                if doc_id is None or doc_id not in kg_boost_scores:
                    continue

                # 僅 boost 本來就略為相關的 chunk，避免硬拉無關文檔
                if bm25_norm[idx] < 0.2:
                    continue

                old_score = hybrid_scores[idx]
                hybrid_scores[idx] += kg_boost_scores[doc_id]
                boosted_count += 1

                if getattr(self, "_debug_kg", False) and boosted_count <= 3:
                    print(
                        f"[KG Debug] Chunk {idx} (doc_id={doc_id}) "
                        f"score: {old_score:.4f} -> {hybrid_scores[idx]:.4f} "
                        f"(boost={kg_boost_scores[doc_id]:.4f})"
                    )

        if getattr(self, "_debug_kg", False):
            if boosted_count > 0:
                print(f"[KG Debug] Applied KG boost to {boosted_count} chunks")
            else:
                print(f"[KG Debug] No chunks received KG boost (after BM25 threshold)")

        # 6. Candidate selection (by hybrid score)
        candidate_count = max(top_k, int(round(top_k * self.candidate_multiplier)))
        candidate_count = min(candidate_count, len(self.chunks))
        top_indices = np.argsort(hybrid_scores)[::-1][:candidate_count]

        # 7. Keyword boosting（只在 candidates 上 re-rank）
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
                        t
                        for t in raw_tokens
                        if t not in ENGLISH_STOP_WORDS_SET
                        and len(t) >= self.min_keyword_characters
                    }

            keyword_summary = {
                "keywords": sorted(list(keywords_to_use)),
                "boost": self.keyword_boost,
                "predefined_source": predefined_source,
                "query_id_provided": query_id is not None,
            }

            boosted_scores = []
            for idx in top_indices:
                base_score = hybrid_scores[idx]
                overlap = self._keyword_overlap(self.chunks[idx], keywords_to_use)
                boosted_score = base_score + (self.keyword_boost * overlap)
                boosted_scores.append((idx, boosted_score))

            boosted_scores.sort(key=lambda x: x[1], reverse=True)
            top_indices = [x[0] for x in boosted_scores]

        # 8. Final selection
        selected = [self.chunks[idx] for idx in top_indices[:top_k]]

        kg_info = None
        if self.kg_retriever and self.kg_boost > 0:
            try:
                # Debug 這邊可以視情況關 multi-hop
                kg_info = self.kg_retriever.get_entity_info(query, use_multi_hop=True)
            except Exception:
                kg_info = None

        retrieval_debug = {
            "language": self.language,
            "top_k": top_k,
            "candidate_count": candidate_count,
            "keyword_info": keyword_summary,
            "kg_info": kg_info,
            "kg_boost": self.kg_boost if self.kg_retriever else 0.0,
            "results": [
                {
                    "metadata": selected_chunk.get("metadata", {}),
                    "preview": selected_chunk.get("page_content", "")[:200],
                    "score": float(hybrid_scores[top_indices[idx]]),
                }
                for idx, selected_chunk in enumerate(selected)
            ],
            "unsolvable": is_unsolvable,
        }

        return selected, retrieval_debug


def create_retriever(chunks, language, config=None, docs_path=None):
    """Creates a retriever from document chunks based on config."""
    config = config or {}
    retriever_type = config.get("type", "bm25").lower()
    if retriever_type != "bm25":
        raise ValueError(f"Unsupported retriever type '{retriever_type}'.")

    bm25_cfg = config.get("bm25", {})

    # KG retriever
    kg_retriever = None
    kg_boost = config.get("kg_boost", 0.0)
    debug_kg = config.get("debug_kg", False)
    if kg_boost > 0:
        try:
            from kg_retriever import create_kg_retriever

            kg_path = config.get("kg_path", "My_RAG/kg_output.json")
            max_hops = config.get("kg_max_hops", 2)
            kg_docs_path = docs_path or config.get(
                "docs_path", "dragonball_dataset/dragonball_docs.jsonl"
            )
            kg_retriever = create_kg_retriever(kg_path, language, max_hops, kg_docs_path)
            print(f"✓ KG retriever initialized: boost={kg_boost}, max_hops={max_hops}, path={kg_path}")
            if debug_kg:
                print("  [KG Debug mode enabled]")
        except Exception as e:
            print(f"✗ Warning: Failed to initialize KG retriever: {e}")
            import traceback

            traceback.print_exc()
            kg_boost = 0.0
    else:
        print(f"  KG retriever disabled (kg_boost={kg_boost})")

    retriever = BM25Retriever(
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
        kg_retriever=kg_retriever,
        kg_boost=kg_boost,
        dense_weight=config.get("dense_weight", 0.5),
    )

    retriever._debug_kg = debug_kg
    return retriever
