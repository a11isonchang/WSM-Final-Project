from rank_bm25 import BM25Okapi
import jieba
from typing import List, Dict, Any, Optional, Tuple, Set
from pathlib import Path
import json
import re
from functools import lru_cache

import numpy as np
import faiss
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from sentence_transformers import SentenceTransformer
from ollama import Client

from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
try:
    from langchain.retrievers import EnsembleRetriever
except ImportError:
    from langchain_classic.retrievers.ensemble import EnsembleRetriever

try:
    from langchain_community.retrievers import BM25Retriever as LC_BM25Retriever
except ImportError:
    from langchain_community.retrievers.bm25 import BM25Retriever as LC_BM25Retriever

try:
    from langchain_community.vectorstores import FAISS
except ImportError:
    from langchain_community.vectorstores.faiss import FAISS

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
    """Lexical retriever with optional keyword-based re-ranking."""

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
    ):
        self.chunks = chunks
        self.language = language
        self.corpus = [chunk["page_content"] for chunk in chunks]
        self.min_keyword_characters = max(1, int(min_keyword_characters))
        self.candidate_multiplier = max(1.0, candidate_multiplier)
        self.keyword_boost = max(0.0, keyword_boost)
        self.tokenized_corpus = [self._tokenize(doc) for doc in self.corpus]
        self.bm25 = BM25Okapi(self.tokenized_corpus, k1=k1, b=b)

    def _tokenize(self, text: str):
        if self.language == "zh":
            return list(jieba.cut(text))
        return text.lower().split()

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

    def retrieve(self, query, top_k=5):
        if not self.chunks:
            return []
        tokenized_query = self._tokenize(query)
        if not tokenized_query:
            return self.chunks[:top_k]

        scores = self.bm25.get_scores(tokenized_query)
        candidate_count = max(top_k, int(round(top_k * self.candidate_multiplier)))
        candidate_count = min(candidate_count, len(self.chunks))
        top_indices = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)[:candidate_count]

        if self.keyword_boost > 0:
            keywords = self._extract_keywords(tokenized_query)
            top_indices = sorted(
                top_indices,
                key=lambda idx: scores[idx] + self.keyword_boost * self._keyword_overlap(self.chunks[idx], keywords),
                reverse=True,
            )

        return [self.chunks[idx] for idx in top_indices[:top_k]]


class LangchainBM25Retriever(BaseRetriever):
    """Langchain-compatible BM25 retriever."""

    lc_bm25_retriever: LC_BM25Retriever

    def __init__(
        self,
        chunks: List[Dict[str, Any]],
        language: str = "en",
        *,
        k1: float = 1.5,
        b: float = 0.75,
        **kwargs, # Accept additional kwargs for compatibility
    ):
        super().__init__(**kwargs)
        self.language = language

        # Convert chunks to LangChain Document objects
        docs = [Document(page_content=chunk["page_content"], metadata=chunk.get("metadata", {})) for chunk in chunks]
        
        # Initialize LangChain's BM25Retriever. It handles its own tokenization.
        self.lc_bm25_retriever = LC_BM25Retriever.from_documents(docs, k1=k1, b=b)

    def _get_relevant_documents(self, query: str, *, run_manager: Any = None) -> List[Document]:
        # The k parameter will be handled by the caller (EnsembleRetriever or create_retriever)
        return self.lc_bm25_retriever.get_relevant_documents(query)


class LangchainDenseRetriever(BaseRetriever):
    """Langchain-compatible Dense (FAISS) retriever."""

    def __init__(
        self,
        chunks: List[Dict[str, Any]],
        language: str = "en",
        *,
        embedding_model_path: str = "My_RAG/models/all_minilm_l6",
        embedding_provider: str = "local",
        ollama_host: str = "http://localhost:11434",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.chunks = chunks
        self.language = language
        self.corpus = [chunk.get("page_content", "") for chunk in chunks]

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
            return self.embedding_model.encode(texts).tolist() # Convert numpy array to list for FAISS

    def _precompute_corpus_embeddings(self, texts: List[str], batch_size: int = 32):
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            embeddings = self._compute_embeddings(batch)
            if embeddings is not None and len(embeddings) > 0:
                all_embeddings.extend(embeddings)
            else:
                dim = 384 # Default dimension
                if all_embeddings:
                    dim = len(all_embeddings[0])
                all_embeddings.extend([[0.0] * dim] * len(batch))
        return np.array(all_embeddings)

    def _get_relevant_documents(self, query: str, *, run_manager: Any = None) -> List[Document]:
        if self.faiss_index is None or self.chunk_embeddings is None:
            return []

        query_emb = self._compute_embeddings([query])
        if not query_emb:
            return []

        query_vec = np.array(query_emb).astype(np.float32)
        faiss.normalize_L2(query_vec)

        # Retrieve from FAISS
        # Assuming k is managed by the caller, e.g., EnsembleRetriever
        # For now, we'll hardcode a default k, or accept it as a param if passed
        k = 4 # Default k, will be overridden
        if hasattr(run_manager, "metadata") and "k" in run_manager.metadata:
            k = run_manager.metadata["k"] # Example of getting k if passed through run_manager

        distances, indices = self.faiss_index.search(query_vec, k)
        
        relevant_docs = []
        for i in indices[0]:
            if i != -1: # -1 indicates no match
                relevant_docs.append(Document(page_content=self.corpus[i])) # Reconstruct Document
        
        return relevant_docs


def create_retriever(chunks, language, config=None):
    """Creates a retriever from document chunks based on config."""
    config = config or {}
    retriever_type = config.get("type", "bm25").lower()
    top_k = config.get("top_k", 5) # Default top_k

    if retriever_type == "bm25":
        bm25_cfg = config.get("bm25", {})
        return BM25Retriever(
            chunks,
            language,
            k1=bm25_cfg.get("k1", 1.5),
            b=bm25_cfg.get("b", 0.75),
            candidate_multiplier=config.get("candidate_multiplier", 3.0),
            keyword_boost=config.get("keyword_boost", 0.0),
            min_keyword_characters=config.get("min_keyword_characters", 3),
        )
    elif retriever_type == "hybrid":
        # Configure BM25 Retriever
        bm25_cfg = config.get("bm25", {})
        bm25_retriever = LangchainBM25Retriever(
            chunks,
            language,
            k1=bm25_cfg.get("k1", 1.5),
            b=bm25_cfg.get("b", 0.75),
        )

        # Configure Dense Retriever
        dense_cfg = config.get("dense", {})
        dense_retriever = LangchainDenseRetriever(
            chunks,
            language,
            embedding_model_path=config.get("embedding_model_path", "My_RAG/models/all_minilm_l6"),
            embedding_provider=config.get("embedding_provider", "local"),
            ollama_host=config.get("ollama_host", "http://ollama-gateway:11434"),
        )

        # Configure Ensemble Retriever
        ensemble_weights = config.get("hybrid_weights", [0.5, 0.5])
        if not (len(ensemble_weights) == 2 and sum(ensemble_weights) == 1.0):
            print(f"Warning: hybrid_weights must be a list of 2 floats that sum to 1.0. Got {ensemble_weights}. Defaulting to [0.5, 0.5].")
            ensemble_weights = [0.5, 0.5]

        # Langchain's EnsembleRetriever
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, dense_retriever],
            weights=ensemble_weights
        )
        return ensemble_retriever
    else:
        raise ValueError(f"Unsupported retriever type '{retriever_type}'.")