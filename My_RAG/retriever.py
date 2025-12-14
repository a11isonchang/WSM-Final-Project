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
from pyserini.index.lucene import LuceneIndexer
from pyserini.search.lucene import LuceneSearcher
from pathlib import Path
import json
import os
import hashlib
from pathlib import Path
import json
import jieba

try:
    from sentence_transformers import CrossEncoder
except Exception:
    CrossEncoder = None

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


def _chunks_fingerprint(chunks: list) -> str:
    h = hashlib.md5()
    h.update(str(len(chunks)).encode("utf-8"))
    # 抽樣幾段，避免成本太高
    for i in (0, len(chunks)//2, max(0, len(chunks)-1)):
        if 0 <= i < len(chunks):
            h.update((chunks[i].get("page_content", "") or "")[:800].encode("utf-8", errors="ignore"))
    return h.hexdigest()[:10]


def _pretokenize_for_bm25(text: str, language: str) -> str:
    """zh: jieba 切詞後用空白 join，讓 Lucene whitespace tokenization work。"""
    text = text or ""
    if language.startswith("zh"):
        toks = [t.strip() for t in jieba.cut(text) if t.strip()]
        return " ".join(toks)
    return text

import shutil
from pyserini.index.lucene import LuceneIndexer
from pyserini.search.lucene import LuceneSearcher

def build_pyserini_index(chunks, index_dir: Path, language: str):
    index_dir = Path(index_dir)
    fp = _chunks_fingerprint(chunks)
    meta_path = index_dir / "_corpus_fp.txt"

    # reuse if possible
    if index_dir.exists() and meta_path.exists():
        if meta_path.read_text(encoding="utf-8").strip() == fp:
            try:
                _ = LuceneSearcher(str(index_dir))
                return
            except Exception:
                pass

    # force rebuild (乾淨刪掉整個資料夾最保險)
    if index_dir.exists():
        shutil.rmtree(index_dir, ignore_errors=True)
    index_dir.mkdir(parents=True, exist_ok=True)

    indexer = LuceneIndexer(index_dir=str(index_dir))

    for i, chunk in enumerate(chunks):
        text = chunk.get("page_content", "") or ""
        if language.startswith("zh"):
            text = " ".join(t.strip() for t in jieba.cut(text) if t.strip())
        indexer.add_doc_dict({"id": str(i), "contents": text})  # ✅重點：用 add_doc_dict

    indexer.close()
    meta_path.write_text(fp, encoding="utf-8")


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
        dense_weight: float = 0.7,  # 0.0 ~ 1.0
        # ===== Rerank =====
        rerank_enabled: bool = False,
        rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        rerank_top_n: int = 50,
        rerank_batch_size: int = 16,
        rerank_weight: float = 1.0,   # 0~1: rerank 分數融合權重
        rerank_device: str = None,

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

                    # Check if it's a flat index (IndexFlatIP or IndexFlatL2)
                    is_flat_index = (
                        isinstance(self.faiss_index, faiss.IndexFlatIP) or
                        isinstance(self.faiss_index, faiss.IndexFlatL2) or
                        hasattr(self.faiss_index, 'reconstruct_n')
                    )
                    if is_flat_index:
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
        # ===== Pyserini BM25 =====
        self.bm25_index_dir = Path("My_RAG/indices") / f"pyserini_bm25_{self.language}"
        build_pyserini_index(self.chunks, self.bm25_index_dir, self.language)

        self.bm25_searcher = LuceneSearcher(str(self.bm25_index_dir))
        self.bm25_searcher.set_bm25(k1, b)

        # ===== Reranker =====
        self.rerank_enabled = bool(rerank_enabled)
        self.rerank_model = rerank_model
        self.rerank_top_n = int(rerank_top_n)
        self.rerank_batch_size = int(rerank_batch_size)
        self.rerank_weight = float(rerank_weight)
        self.rerank_device = rerank_device

        self.reranker = None
        if self.rerank_enabled:
            try:
                self.reranker = CrossEncoderReranker(
                    model_name=self.rerank_model,
                    batch_size=self.rerank_batch_size,
                    device=self.rerank_device,
                )
                print(f"✓ Reranker enabled: {self.rerank_model}")
            except Exception as e:
                print(f"✗ Warning: failed to init reranker ({self.rerank_model}): {e}")
                self.reranker = None
                self.rerank_enabled = False



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
        tokenized_query = self._tokenize(query)
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

        # 1. Sparse retrieval (Pyserini BM25)
        bm25_scores = np.zeros(len(self.chunks), dtype=np.float32)

        # we only need BM25 for candidates; don't retrieve all docs.
        candidate_count = max(top_k, int(round(top_k * self.candidate_multiplier)))
        candidate_count = min(candidate_count, len(self.chunks))

        bm25_k = min(len(self.chunks), max(1000, candidate_count * 5))
        bm25_k = min(len(self.chunks), 1000)  # 或 candidate_count*5
        hits = self.bm25_searcher.search(query, k=bm25_k)

        for h in hits:
            doc_id = int(h.docid)
            bm25_scores[doc_id] = float(h.score)



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

        # 5. KG boost removed - no longer applying KG boost to hybrid scores

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

                # 8. Rerank on candidates (optional)
        rerank_info = None
        if self.rerank_enabled and self.reranker is not None and len(top_indices) > 1:
            rerank_n = min(len(top_indices), max(top_k, self.rerank_top_n))
            cand_indices = list(top_indices[:rerank_n])
            cand_texts = [self.chunks[i].get("page_content", "") or "" for i in cand_indices]

            try:
                rr_scores = np.array(self.reranker.rerank(query, cand_texts), dtype=np.float32)

                # normalize rerank scores to 0~1
                if rr_scores.max() != rr_scores.min():
                    rr_norm = (rr_scores - rr_scores.min()) / (rr_scores.max() - rr_scores.min())
                else:
                    rr_norm = rr_scores

                # fuse: final = (1-w)*hybrid + w*rerank
                w = float(np.clip(self.rerank_weight, 0.0, 1.0))
                fused = []
                for local_rank, idx in enumerate(cand_indices):
                    fused_score = (1 - w) * float(hybrid_scores[idx]) + w * float(rr_norm[local_rank])
                    fused.append((idx, fused_score))

                fused.sort(key=lambda x: x[1], reverse=True)

                # keep order: fused first, then remaining
                reranked = [i for i, _ in fused]
                rest = [i for i in top_indices if i not in set(reranked)]
                top_indices = reranked + rest

                rerank_info = {
                    "enabled": True,
                    "model": self.rerank_model,
                    "rerank_n": rerank_n,
                    "weight": w,
                    "top_rr": [float(rr_scores[i]) for i in np.argsort(-rr_scores)[:3]],
                }
            except Exception as e:
                rerank_info = {"enabled": False, "error": str(e)}

        # 9. Final selection
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
            "rerank_info": rerank_info,

        }

        return selected, retrieval_debug

class CrossEncoderReranker:
    def __init__(self, model_name: str, batch_size: int = 16, device: str | None = None):
        if CrossEncoder is None:
            raise ImportError("CrossEncoder not available. Please install/upgrade sentence-transformers.")
        self.model_name = model_name
        self.batch_size = int(batch_size)
        self.device = device
        self.model = CrossEncoder(model_name, device=device)

    def rerank(self, query: str, docs: list[str]) -> list[float]:
        pairs = [(query, d) for d in docs]
        scores = self.model.predict(pairs, batch_size=self.batch_size)
        # scores: np.ndarray or list
        return [float(s) for s in scores]

def analysis_retriever_result(query: str, context_chunks: List[Dict[str, Any]], language: str) -> str:
    """
    分析检索结果是否充分
    
    Args:
        query: 查询文本
        context_chunks: 检索到的chunks
        language: 语言类型
        
    Returns:
        "use_kg" 如果检索结果不够充分，否则返回 "ok"
    """
    # 如果检索到的chunks为空或太少，建议使用KG
    if not context_chunks or len(context_chunks) < 2:
        return "use_kg"
    
    # 检查chunks的内容是否与query相关
    # 简单策略：如果所有chunks的内容都很短，可能不够充分
    total_content_length = sum(len(chunk.get("page_content", "")) for chunk in context_chunks)
    if total_content_length < 100:  # 如果总内容长度小于100字符，可能不够充分
        return "use_kg"
    
    # 默认认为检索结果充分
    return "ok"


def kg_retriever(query: str, language: str, all_chunks: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
    """
    使用知识图谱检索相关的chunks
    
    Args:
        query: 查询文本
        language: 语言类型
        all_chunks: 所有可用的chunks
        top_k: 返回的chunks数量
        
    Returns:
        相关的chunks列表
    """
    try:
        from kg_retriever import create_kg_retriever
        
        # 创建KG检索器
        kg_path = "My_RAG/kg_output.json"
        kg_ret = create_kg_retriever(kg_path, language)
        
        # 获取相关的doc_ids
        related_doc_ids = kg_ret.retrieve_doc_ids(query, top_k=top_k * 2)  # 获取更多doc_ids以便筛选
        
        # 从all_chunks中找出对应的chunks
        # 建立doc_id到chunks的映射
        doc_id_to_chunks = {}
        for chunk in all_chunks:
            doc_id = chunk.get("metadata", {}).get("doc_id")
            if doc_id is not None:
                if doc_id not in doc_id_to_chunks:
                    doc_id_to_chunks[doc_id] = []
                doc_id_to_chunks[doc_id].append(chunk)
        
        # 收集相关的chunks
        kg_chunks = []
        seen_chunks = set()
        
        for doc_id in related_doc_ids:
            if doc_id in doc_id_to_chunks:
                for chunk in doc_id_to_chunks[doc_id]:
                    # 使用page_content作为唯一标识去重
                    content = chunk.get("page_content", "")
                    if content and content not in seen_chunks:
                        kg_chunks.append(chunk)
                        seen_chunks.add(content)
                        if len(kg_chunks) >= top_k:
                            break
            if len(kg_chunks) >= top_k:
                break
        
        return kg_chunks[:top_k]
        
    except Exception as e:
        print(f"Warning: KG retrieval failed: {e}")
        return []


def _ensure_float(value):
    """Ensure a value is a float, handling dict/list cases."""
    if isinstance(value, (int, float)):
        return float(value)
    elif isinstance(value, dict):
        # If it's a dict, try to get a numeric value or use default
        return 3.0
    elif isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return 3.0
    else:
        return 3.0


def create_retriever(chunks, language, config=None, docs_path=None):
    """Creates a retriever from document chunks based on config."""
    config = config or {}
    retriever_type = config.get("type", "bm25").lower()
    if retriever_type != "bm25":
        raise ValueError(f"Unsupported retriever type '{retriever_type}'.")

    bm25_cfg = config.get("bm25", {})
    rerank_enabled=config.get("rerank_enabled", False),
    rerank_model=config.get("rerank_model", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
    rerank_top_n=int(config.get("rerank_top_n", 50)),
    rerank_batch_size=int(config.get("rerank_batch_size", 16)),
    rerank_weight=float(config.get("rerank_weight", 1.0)),
    rerank_device=config.get("rerank_device", None),


    # KG retriever (always initialize for ToG reasoning, even if boost is disabled)
    kg_retriever = None
    kg_boost = config.get("kg_boost", 0.0)  # Keep for backward compatibility, but not used for boosting
    debug_kg = config.get("debug_kg", False)
    kg_path = config.get("kg_path", "My_RAG/kg_output.json")
    
    # Always try to initialize KG retriever for ToG reasoning
    try:
        # Try to import create_kg_retriever function
        try:
            from kg_retriever import create_kg_retriever
            max_hops = config.get("kg_max_hops", 2)
            kg_docs_path = docs_path or config.get(
                "docs_path", "dragonball_dataset/dragonball_docs.jsonl"
            )
            kg_retriever = create_kg_retriever(kg_path, language, max_hops, kg_docs_path)
            print(f"✓ KG retriever initialized for ToG reasoning: max_hops={max_hops}, path={kg_path}")
            if debug_kg:
                print("  [KG Debug mode enabled]")
        except ImportError:
            # If create_kg_retriever doesn't exist, try KGRetriever class
            try:
                from kg_retriever import KGRetriever
                max_hops = config.get("kg_max_hops", 2)
                kg_docs_path = docs_path or config.get(
                    "docs_path", "dragonball_dataset/dragonball_docs.jsonl"
                )
                kg_retriever = KGRetriever(
                    kg_path=kg_path,
                    language=language,
                    max_hops=max_hops,
                    docs_path=kg_docs_path,
                )
                print(f"✓ KG retriever initialized for ToG reasoning: max_hops={max_hops}, path={kg_path}")
                if debug_kg:
                    print("  [KG Debug mode enabled]")
            except ImportError:
                # KG retriever not available, continue without it
                kg_retriever = None
                print("✗ Warning: KG retriever module not available. Continuing without KG support.")
    except Exception as e:
        print(f"✗ Warning: Failed to initialize KG retriever: {e}")
        kg_retriever = None

    # Get dense_weight (support language-specific or global setting)
    dense_weight_config = config.get("dense_weight", 0.5)
    if isinstance(dense_weight_config, dict):
        # If dense_weight is a dict, get language-specific value
        # Handle language variants (e.g., "zh_CN" -> "zh")
        lang_key = language if language in dense_weight_config else (
            "zh" if language.startswith("zh") else "en"
        )
        dense_weight = dense_weight_config.get(lang_key, dense_weight_config.get("en", 0.5))
    else:
        # If dense_weight is a number, use it directly
        dense_weight = dense_weight_config

    retriever = BM25Retriever(
        chunks,
        language,
        k1=bm25_cfg.get("k1", 1.5),
        b=bm25_cfg.get("b", 0.75),
        candidate_multiplier=_ensure_float(config.get("candidate_multiplier", 3.0)),
        keyword_boost=config.get("keyword_boost", 0.0),
        min_keyword_characters=config.get("min_keyword_characters", 3),
        keyword_extraction_method=config.get("keyword_extraction_method", "simple"),
        embedding_model_path=config.get("embedding_model_path", "My_RAG/models/all_minilm_l6"),
        embedding_provider=config.get("embedding_provider", "local"),
        ollama_host=config.get("ollama_host", "http://ollama-gateway:11434"),
        keyword_file=config.get("keyword_file", "database/database.jsonl"),
        kg_retriever=kg_retriever,
        kg_boost=kg_boost,
        dense_weight=dense_weight,
    )

    retriever._debug_kg = debug_kg
    return retriever