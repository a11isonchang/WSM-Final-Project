# kg_retriever.py
# ============================================================
# This file contains:
# 1) KGRetriever + create_kg_retriever (ToG / Think-on-Graph)
# 2) BM25Retriever (Hybrid BM25 + Dense) + create_retriever
# ------------------------------------------------------------
# Why:
# - generator.py expects kg_retriever.rank_docs(...) for ToG
# - BM25Retriever expects kg_retriever.get_entity_info(...) for debug
# - create_retriever expects create_kg_retriever(...) factory
# ============================================================

from __future__ import annotations

import json
import re
import math
from dataclasses import dataclass
from pathlib import Path
from functools import lru_cache
from typing import List, Dict, Any, Optional, Tuple, Set

import numpy as np
import jieba
import faiss
from rank_bm25 import BM25Okapi
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from sentence_transformers import SentenceTransformer
from ollama import Client

EN_TOKEN_PATTERN = re.compile(r"[a-z0-9']+")
ENGLISH_STOP_WORDS_SET = set(ENGLISH_STOP_WORDS)


# ============================================================
# 0) Shared utils
# ============================================================

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


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _truncate(s: str, n: int) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else s[:n] + "..."


# ============================================================
# 1) ToG / Think-on-Graph: KGRetriever
# ============================================================

@dataclass
class KGTriple:
    head: str
    relation: str
    tail: str
    doc_id: int
    properties: Dict[str, Any]

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "KGTriple":
        return KGTriple(
            head=d.get("head", ""),
            relation=d.get("relation", ""),
            tail=d.get("tail", ""),
            doc_id=int(d.get("doc_id", -1)),
            properties=d.get("properties", {}) or {},
        )


class KGRetriever:
    """
    Minimal, working ToG-style retriever over a KG exported as kg_output.json.

    Provides:
    - rank_docs(query, top_k, use_multi_hop, return_debug)
    - get_entity_info(query, use_multi_hop)
    """

    def __init__(
        self,
        kg_data: Dict[str, Any],
        language: str = "en",
        max_hops: int = 2,
        docs_path: Optional[str] = None,
    ):
        self.language = language
        self.max_hops = max(1, int(max_hops))
        self.docs_path = docs_path  # optional (not required for rank_docs)

        # ---- Load entities ----
        self.entities: List[Dict[str, Any]] = kg_data.get("entities", []) or []
        self.entity_names: List[str] = [e.get("name", "") for e in self.entities if e.get("name")]
        self.entity_name_set: Set[str] = set(self.entity_names)

        # name -> list of ids (in case duplicates)
        self.entity_name_to_ids: Dict[str, List[str]] = {}
        for e in self.entities:
            name = e.get("name")
            eid = e.get("id")
            if name and eid:
                self.entity_name_to_ids.setdefault(name, []).append(eid)

        # ---- Load triples ----
        raw_triples = kg_data.get("triples", []) or []
        self.triples: List[KGTriple] = [KGTriple.from_dict(t) for t in raw_triples]

        # ---- doc mapping (optional optimization) ----
        # format: {"0": {"start_triple_index":..., "end_triple_index":...}, ...}
        self.doc_mapping = kg_data.get("doc_mapping", {}) or {}

        # ---- Build indices for graph traversal ----
        # adjacency by entity string (head/tail) -> triple indices
        self.entity_to_triple_idxs: Dict[str, List[int]] = {}
        for i, t in enumerate(self.triples):
            if t.head:
                self.entity_to_triple_idxs.setdefault(t.head, []).append(i)
            if t.tail:
                self.entity_to_triple_idxs.setdefault(t.tail, []).append(i)

        # also keep doc -> triple indices (fast evidence collection)
        self.doc_to_triple_idxs: Dict[int, List[int]] = {}
        for i, t in enumerate(self.triples):
            self.doc_to_triple_idxs.setdefault(t.doc_id, []).append(i)

        # stopwords
        self.zh_stop = load_chinese_stop_words()

    # --------------------------
    # Tokenization / query parse
    # --------------------------

    def _tokenize(self, text: str) -> List[str]:
        text = (text or "").strip()
        if not text:
            return []
        if self.language == "zh":
            toks = [tok.strip() for tok in jieba.cut(text) if tok.strip()]
            return [t for t in toks if t not in self.zh_stop]
        # en
        raw = EN_TOKEN_PATTERN.findall(text.lower())
        return [t for t in raw if t and t not in ENGLISH_STOP_WORDS_SET]

    def _match_entities_in_query(self, query: str) -> List[str]:
        """
        Naive but effective: substring match entity names in query.
        Returns matched entity NAMES (not ids).
        """
        q = (query or "").strip()
        if not q:
            return []

        matched = []
        # Prefer longer matches first to avoid short noisy entities.
        for name in sorted(self.entity_name_set, key=len, reverse=True):
            if not name:
                continue
            if name in q:
                matched.append(name)
                if len(matched) >= 8:  # cap
                    break
        return matched

    # --------------------------
    # Graph traversal (ToG core)
    # --------------------------

    def _seed_triples_by_keywords(self, query_tokens: List[str], limit: int = 200) -> List[int]:
        """
        If no entity match, seed by keyword overlap on (head/tail/relation).
        Returns triple indices.
        """
        if not query_tokens:
            return []
        toks = set(query_tokens)
        hits = []
        for i, t in enumerate(self.triples):
            hay = f"{t.head} {t.relation} {t.tail}".lower()
            overlap = sum(1 for x in toks if x and x in hay)
            if overlap > 0:
                hits.append((i, overlap))
        hits.sort(key=lambda x: x[1], reverse=True)
        return [i for i, _ in hits[:limit]]

    def _bfs_paths(
        self,
        seed_entities: List[str],
        seed_triple_idxs: List[int],
        use_multi_hop: bool,
    ) -> Tuple[Dict[int, List[List[int]]], Dict[int, Set[int]]]:
        """
        BFS over entity strings using triples as edges.
        Returns:
          - doc_id -> list of paths (each path is a list of triple indices)
          - doc_id -> set of evidence triple indices
        """
        max_hops = self.max_hops if use_multi_hop else 1

        # start frontier entities
        frontier_entities = set(seed_entities)
        visited_entities = set(seed_entities)

        # also include entities from seeded triples
        for ti in seed_triple_idxs:
            if 0 <= ti < len(self.triples):
                frontier_entities.add(self.triples[ti].head)
                frontier_entities.add(self.triples[ti].tail)

        # For path construction: keep (current_entity, path_triple_idxs)
        # Start with empty path at each seed entity.
        queue: List[Tuple[str, List[int]]] = [(e, []) for e in frontier_entities if e]

        doc_to_paths: Dict[int, List[List[int]]] = {}
        doc_to_evidence: Dict[int, Set[int]] = {}

        # BFS by hop count: each time we add 1 triple edge
        while queue:
            entity, path = queue.pop(0)
            hop = len(path)
            if hop >= max_hops:
                continue

            # explore incident triples
            for tri_idx in self.entity_to_triple_idxs.get(entity, []):
                t = self.triples[tri_idx]
                new_path = path + [tri_idx]

                doc_to_paths.setdefault(t.doc_id, []).append(new_path)
                doc_to_evidence.setdefault(t.doc_id, set()).add(tri_idx)

                # expand to next entity
                next_entity = t.tail if t.head == entity else t.head
                if not next_entity:
                    continue

                # prevent infinite loops; allow revisit but cap depth
                if next_entity not in visited_entities:
                    visited_entities.add(next_entity)
                    queue.append((next_entity, new_path))

        return doc_to_paths, doc_to_evidence

    # --------------------------
    # Scoring
    # --------------------------

    def _score_doc(
        self,
        doc_id: int,
        query: str,
        query_tokens: List[str],
        matched_entities: List[str],
        evidence_triple_idxs: Set[int],
        doc_paths: List[List[int]],
    ) -> float:
        """
        Simple, robust score:
        - entity match in evidence triples is strong
        - keyword overlap in triples contributes
        - multi-hop paths contribute a small bonus
        """
        if not evidence_triple_idxs:
            return 0.0

        ent_set = set(matched_entities)
        tok_set = set(query_tokens)

        score = 0.0
        for tri_idx in evidence_triple_idxs:
            t = self.triples[tri_idx]
            h = (t.head or "")
            r = (t.relation or "")
            ta = (t.tail or "")

            # entity hits
            if h in ent_set:
                score += 2.0
            if ta in ent_set:
                score += 2.0

            # keyword hits (relation split + substring)
            hay = f"{h} {r} {ta}".lower()
            overlap = sum(1 for x in tok_set if x and x in hay)
            score += 0.35 * overlap

            # small domain/time hints (optional)
            props = t.properties or {}
            if self.language == "zh":
                # if query mentions a year/month and triple tail contains it, small bonus
                if any(ch.isdigit() for ch in query) and any(ch.isdigit() for ch in ta):
                    score += 0.2
            else:
                if any(ch.isdigit() for ch in query) and any(ch.isdigit() for ch in ta):
                    score += 0.2

        # multi-hop path bonus
        if doc_paths:
            # reward having any path length >=2
            if any(len(p) >= 2 for p in doc_paths):
                score += 0.8
            # more distinct paths, small bonus but capped
            score += min(0.8, 0.15 * len(doc_paths))

        return score

    def _path_to_str(self, path: List[int]) -> str:
        if not path:
            return ""
        parts = []
        for tri_idx in path:
            t = self.triples[tri_idx]
            parts.append(f"{t.head} --[{t.relation}]--> {t.tail}")
        return " ; ".join(parts)

    # --------------------------
    # Public API
    # --------------------------

    def rank_docs(
        self,
        query: str,
        top_k: int = 10,
        use_multi_hop: bool = True,
        return_debug: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        ToG main API.
        Returns list[dict]:
          {
            "doc_id": int,
            "score": float (0~1),
            "paths": [str],
            "evidence_triples": [{"head","relation","tail","doc_id","properties"}...]
          }
        """
        top_k = max(1, int(top_k))

        q_tokens = self._tokenize(query)
        matched_entities = self._match_entities_in_query(query)

        # if no entity match, seed triples by keyword overlap
        seed_triple_idxs = []
        if not matched_entities:
            seed_triple_idxs = self._seed_triples_by_keywords(q_tokens, limit=200)

        doc_to_paths, doc_to_evidence = self._bfs_paths(
            seed_entities=matched_entities,
            seed_triple_idxs=seed_triple_idxs,
            use_multi_hop=use_multi_hop,
        )

        # score docs
        scored = []
        for doc_id, ev_idxs in doc_to_evidence.items():
            paths = doc_to_paths.get(doc_id, [])
            s = self._score_doc(
                doc_id=doc_id,
                query=query,
                query_tokens=q_tokens,
                matched_entities=matched_entities,
                evidence_triple_idxs=ev_idxs,
                doc_paths=paths,
            )
            if s > 0:
                scored.append((doc_id, s))

        if not scored:
            return []

        scored.sort(key=lambda x: x[1], reverse=True)

        # normalize to 0~1
        max_s = scored[0][1]
        min_s = scored[-1][1]
        denom = (max_s - min_s) if (max_s - min_s) > 1e-9 else 1.0

        results: List[Dict[str, Any]] = []
        for doc_id, raw_s in scored[: max(50, top_k)]:  # keep some headroom
            norm_s = (raw_s - min_s) / denom

            # build top paths
            paths_raw = doc_to_paths.get(doc_id, [])
            # prefer longer paths first (multi-hop)
            paths_raw_sorted = sorted(paths_raw, key=lambda p: (len(p),), reverse=True)
            path_strs = []
            for p in paths_raw_sorted[:3]:
                ps = self._path_to_str(p)
                if ps:
                    path_strs.append(ps)

            # evidence triples (limit)
            ev_idxs = list(doc_to_evidence.get(doc_id, set()))
            # prioritize triples that include matched entities
            ent_set = set(matched_entities)
            def tri_priority(i: int) -> Tuple[int, int]:
                t = self.triples[i]
                hit = int((t.head in ent_set) or (t.tail in ent_set))
                return (hit, len(t.relation or ""))
            ev_idxs.sort(key=tri_priority, reverse=True)

            evidence = []
            for i in ev_idxs[:5]:
                t = self.triples[i]
                evidence.append(
                    {
                        "head": t.head,
                        "relation": t.relation,
                        "tail": t.tail,
                        "doc_id": t.doc_id,
                        "properties": t.properties or {},
                    }
                )

            out = {
                "doc_id": int(doc_id),
                "score": float(norm_s),
                "paths": path_strs,
                "evidence_triples": evidence,
            }
            if return_debug:
                out["debug"] = {
                    "raw_score": float(raw_s),
                    "matched_entities": matched_entities[:8],
                    "query_tokens": q_tokens[:20],
                    "evidence_triple_count": len(ev_idxs),
                    "path_count": len(paths_raw),
                }
            results.append(out)

        # final top_k
        results.sort(key=lambda d: d.get("score", 0.0), reverse=True)
        return results[:top_k]

    def get_entity_info(self, query: str, use_multi_hop: bool = True) -> Dict[str, Any]:
        """
        Lightweight debug info for BM25Retriever.retrieve() (kg_info field).
        """
        matched_entities = self._match_entities_in_query(query)
        ranked = self.rank_docs(query, top_k=5, use_multi_hop=use_multi_hop, return_debug=False)
        return {
            "matched_entities": matched_entities,
            "top_docs": [{"doc_id": r["doc_id"], "score": r["score"]} for r in ranked],
            "top_evidence": (ranked[0]["evidence_triples"][:3] if ranked else []),
        }


def create_kg_retriever(
    kg_path: str,
    language: str = "en",
    max_hops: int = 2,
    docs_path: Optional[str] = None,
) -> KGRetriever:
    """
    Factory used by create_retriever(...).
    Loads kg_output.json and returns a KGRetriever object with rank_docs().
    """
    p = Path(kg_path)
    if not p.exists():
        raise FileNotFoundError(f"KG file not found: {kg_path}")

    with p.open("r", encoding="utf-8") as f:
        kg_data = json.load(f)

    return KGRetriever(kg_data=kg_data, language=language, max_hops=max_hops, docs_path=docs_path)


# ============================================================
# 2) Hybrid BM25 + Dense Retriever (your existing code)
#    (kept, minimal changes)
# ============================================================

class BM25Retriever:
    """Hybrid retriever (BM25 + Dense) with optional keyword / KG debug info."""

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
        kg_boost: float = 0.0,  # keep for backward compatibility
        dense_weight: float = 0.5,  # 0.0 ~ 1.0
    ):
        self.chunks = chunks
        self.language = language
        self.corpus = [chunk.get("page_content", "") for chunk in chunks]
        self.predefined_keywords: Dict[str, set] = {}
        self.unsolvable_queries: set = set()
        if keyword_file:
            self._load_predefined_keywords(keyword_file)

        # KG (used for debug info + ToG routing in generator)
        self.kg_retriever = kg_retriever
        self.kg_boost = max(0.0, kg_boost)
        self._debug_kg = False  # set by create_retriever

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

                    is_flat_index = (
                        isinstance(self.faiss_index, faiss.IndexFlatIP)
                        or isinstance(self.faiss_index, faiss.IndexFlatL2)
                        or hasattr(self.faiss_index, "reconstruct_n")
                    )
                    if is_flat_index:
                        ntotal = self.faiss_index.ntotal
                        if ntotal == len(self.corpus):
                            self.chunk_embeddings = self.faiss_index.reconstruct_n(0, ntotal)

                            test_emb = self._compute_embeddings(["test"])
                            if test_emb and len(test_emb) > 0:
                                test_array = np.array(test_emb)
                                expected_dim = test_array.shape[-1] if test_array.ndim > 1 else test_array.shape[0]
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

    # ----------------------
    # Keyword file
    # ----------------------

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

    # ----------------------
    # Tokenization
    # ----------------------

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

    # ----------------------
    # Embeddings
    # ----------------------

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

    # ----------------------
    # Keyword extraction
    # ----------------------

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
                ngram = "".join(tokens[i : i + n]) if self.language == "zh" else " ".join(tokens[i : i + n])
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

    # ----------------------
    # Main retrieve
    # ----------------------

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

        # 1) Sparse retrieval (BM25)
        tokenized_query = self._tokenize(query)
        if not tokenized_query:
            bm25_scores = np.zeros(len(self.chunks))
        else:
            bm25_scores = np.array(self.bm25.get_scores(tokenized_query))

        # 2) Dense retrieval
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

        # 3) Normalize
        def normalize(scores):
            if np.max(scores) == np.min(scores):
                return scores
            return (scores - np.min(scores)) / (np.max(scores) - np.min(scores))

        bm25_norm = normalize(bm25_scores)
        dense_norm = normalize(dense_scores)

        # 4) Hybrid score
        hybrid_scores = (1 - self.dense_weight) * bm25_norm + self.dense_weight * dense_norm

        # 5) Candidate selection
        candidate_count = max(top_k, int(round(top_k * self.candidate_multiplier)))
        candidate_count = min(candidate_count, len(self.chunks))
        top_indices = np.argsort(hybrid_scores)[::-1][:candidate_count]

        # 6) Keyword boosting（only re-rank candidates）
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

        # 7) Final selection
        selected = [self.chunks[idx] for idx in top_indices[:top_k]]

        # kg_info (debug only)
        kg_info = None
        if self.kg_retriever:
            try:
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
                    "preview": (selected_chunk.get("page_content", "") or "")[:200],
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

    # KG retriever (for ToG reasoning)
    kg_retriever_obj = None
    debug_kg = config.get("debug_kg", False)
    kg_path = config.get("kg_path", "My_RAG/kg_output.json")

    # Always try to initialize KG retriever for ToG reasoning
    try:
        max_hops = config.get("kg_max_hops", 2)
        kg_docs_path = docs_path or config.get("docs_path", "dragonball_dataset/dragonball_docs.jsonl")

        kg_retriever_obj = create_kg_retriever(
            kg_path=kg_path,
            language=language,
            max_hops=max_hops,
            docs_path=kg_docs_path,
        )
        print(f"✓ KG retriever initialized for ToG reasoning: max_hops={max_hops}, path={kg_path}")
        if debug_kg:
            print("  [KG Debug mode enabled]")
    except Exception as e:
        print(f"✗ Warning: Failed to initialize KG retriever: {e}")
        import traceback
        traceback.print_exc()
        kg_retriever_obj = None

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
        kg_retriever=kg_retriever_obj,
        kg_boost=config.get("kg_boost", 0.0),
        dense_weight=config.get("dense_weight", 0.5),
    )

    retriever._debug_kg = debug_kg
    return retriever
