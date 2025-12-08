from rank_bm25 import BM25Okapi
import jieba
import json
import re
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
from functools import lru_cache
from typing import List, Dict, Any
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
        keyword_extraction_method: str = "simple",
        embedding_model_path: str = "My_RAG/models/all_minilm_l6",
        embedding_provider: str = "local",
        ollama_host: str = "http://localhost:11434",
        keyword_file: str = None,
    ):
        self.chunks = chunks
        self.language = language
        self.corpus = [chunk.get("page_content", "") for chunk in chunks]
        self.predefined_keywords = {}
        if keyword_file:
            self._load_predefined_keywords(keyword_file)

        self.min_keyword_characters = max(1, int(min_keyword_characters))
        self.candidate_multiplier = max(1.0, candidate_multiplier)
        self.keyword_boost = max(0.0, keyword_boost)
        self.keyword_extraction_method = keyword_extraction_method
        self.embedding_provider = embedding_provider
        
        self.embedding_model = None
        self.ollama_client = None

        if self.keyword_boost > 0 and self.keyword_extraction_method == "semantic":
            if self.embedding_provider == "ollama":
                try:
                    self.ollama_client = Client(host=ollama_host)
                    # For ollama, embedding_model_path is treated as the model name
                    self.embedding_model = embedding_model_path 
                    print(f"Using Ollama embedding model: {self.embedding_model}")
                except Exception as e:
                    print(f"Failed to initialize Ollama client: {e}. Reverting to simple extraction.")
                    self.keyword_extraction_method = "simple"
            else:
                # Default to local SentenceTransformer
                try:
                    self.embedding_model = SentenceTransformer(embedding_model_path)
                    print(f"Loaded local embedding model: {embedding_model_path}")
                except Exception as e:
                    print(f"Failed to load local embedding model: {e}. Reverting to simple extraction.")
                    self.keyword_extraction_method = "simple"

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
            response = self.ollama_client.embed(model=self.embedding_model, input=texts)
            return response.embeddings
        else:
            # SentenceTransformer returns ndarray or list of ndarrays
            return self.embedding_model.encode(texts)

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
        
        distances = cosine_similarity(query_embedding, candidate_embeddings)
        # Get top indices
        top_indices = distances.argsort()[0][-top_k:]
        
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
        if not self.chunks:
            return [], {
                "language": self.language,
                "top_k": top_k,
                "candidate_count": 0,
                "keyword_info": None,
                "results": [],
            }
        tokenized_query = self._tokenize(query)
        if not tokenized_query:
            subset = self.chunks[:top_k]
            return subset, {
                "language": self.language,
                "top_k": top_k,
                "candidate_count": len(subset),
                "keyword_info": None,
                "results": [
                    {
                        "metadata": chunk.get("metadata", {}),
                        "preview": chunk.get("page_content", "")[:200],
                        "score": 0.0,
                    }
                    for chunk in subset
                ],
            }

        scores = self.bm25.get_scores(tokenized_query)
        candidate_count = max(top_k, int(round(top_k * self.candidate_multiplier)))
        candidate_count = min(candidate_count, len(self.chunks))
        top_indices = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)[:candidate_count]

        keyword_summary = None
        if self.keyword_boost > 0:
            predefined = None
            if query_id is not None:
                predefined = self.predefined_keywords.get(str(query_id))
            if not predefined:
                predefined = self.predefined_keywords.get(query.strip())

            if predefined:
                keywords = predefined
            elif self.keyword_extraction_method == "semantic" and self.embedding_model:
                keywords = self._extract_keywords_semantic(query)
            else:
                keywords = self._extract_keywords(tokenized_query)

            keyword_summary = {
                "keywords": sorted(keywords),
                "boost": self.keyword_boost,
            }
            top_indices = sorted(
                top_indices,
                key=lambda idx: scores[idx] + self.keyword_boost * self._keyword_overlap(self.chunks[idx], keywords),
                reverse=True,
            )

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
                    "score": scores[top_indices[idx]],
                }
                for idx, selected_chunk in enumerate(selected)
            ],
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
        keyword_file=config.get("keyword_file", "database/database_test.jsonl"),
    )
