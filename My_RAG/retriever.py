from rank_bm25 import BM25Okapi
import jieba
from typing import List, Dict, Any


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

    def extract_keywords_from_query(self, query: str) -> List[str]:
        """
        Simple keyword extraction: extract words longer than 3 characters after preprocessing.
        """
        tokens = self._tokenize(query)
        return [token for token in tokens if len(token) > 3]

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
    )
