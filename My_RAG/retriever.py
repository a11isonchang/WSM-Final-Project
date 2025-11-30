from rank_bm25 import BM25Okapi
import jieba
import re
import numpy as np
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
from functools import lru_cache
from typing import List, Dict, Any
from collections import Counter

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


class HybridRetriever:
    """
    Hybrid retriever combining BM25, TF-IDF, and JM Smoothing with Pseudo-Relevance Feedback.
    Final Score = a*TF-IDF + b*BM25 + c*JM
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
    ):
        self.chunks = chunks
        self.language = language
        self.corpus = [chunk.get("page_content", "") for chunk in chunks]
        self.weights = weights or {"tfidf": 0.33, "bm25": 0.33, "jm": 0.34}
        self.candidate_multiplier = max(1.0, candidate_multiplier)
        self.prf_top_k = max(0, int(prf_top_k))
        self.prf_term_count = max(0, int(prf_term_count))

        self.porter_stemmer = PorterStemmer()
        self.chinese_stop_words = load_chinese_stop_words()

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

    def _tokenize(self, text: str):
        if self.language == "zh":
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
        a = self.weights.get("tfidf", 0.33)
        b = self.weights.get("bm25", 0.33)
        c = self.weights.get("jm", 0.34)
        
        final_scores = (a * norm_tfidf) + (b * norm_bm25) + (c * norm_jm)
        
        return final_scores, (bm25_raw, tfidf_raw, jm_raw)

    def retrieve(self, query, top_k=5):
        if not self.chunks:
            return [], {}

        tokenized_query = self._tokenize(query)
        final_scores, (bm25_s, tfidf_s, jm_s) = self._calculate_hybrid_scores(query, tokenized_query)

        # --- Pseudo-Relevance Feedback ---
        expanded_info = None
        if self.prf_top_k > 0 and self.prf_term_count > 0:
            # Get top docs from initial pass
            temp_top_indices = np.argsort(final_scores)[::-1][:self.prf_top_k]
            
            feedback_tokens = []
            for idx in temp_top_indices:
                feedback_tokens.extend(self.tokenized_corpus[idx])
            
            if feedback_tokens:
                most_common = Counter(feedback_tokens).most_common(self.prf_term_count)
                new_terms = [term for term, count in most_common]
                
                # Expand Query
                tokenized_query.extend(new_terms)
                expanded_query_text = query + " " + " ".join(new_terms) # Approximate text for TF-IDF
                expanded_info = new_terms
                
                # Re-calculate scores with expanded query
                final_scores, (bm25_s, tfidf_s, jm_s) = self._calculate_hybrid_scores(expanded_query_text, tokenized_query)

        # Final Selection
        top_indices = np.argsort(final_scores)[::-1][:top_k]
        selected = [self.chunks[i] for i in top_indices]

        retrieval_debug = {
            "language": self.language,
            "top_k": top_k,
            "weights": self.weights,
            "prf_expanded_terms": expanded_info,
            "results": [
                {
                    "metadata": self.chunks[i].get("metadata", {}),
                    "preview": self.chunks[i].get("page_content", "")[:200],
                    "score": final_scores[i],
                    "components": {
                        "bm25": bm25_s[i],
                        "tfidf": tfidf_s[i],
                        "jm": jm_s[i]
                    },
                }
                for i in top_indices
            ],
        }

        return selected, retrieval_debug


def create_retriever(chunks, language, config=None):
    config = config or {}
    weights = config.get("weights", {"tfidf": 0.33, "bm25": 0.33, "jm": 0.34})

    return HybridRetriever(
        chunks,
        language,
        k1=config.get("bm25", {}).get("k1", 1.5),
        b=config.get("bm25", {}).get("b", 0.75),
        jm_lambda=config.get("jm", {}).get("lambda", 0.5),
        weights=weights,
        candidate_multiplier=config.get("candidate_multiplier", 3.0),
        prf_top_k=config.get("prf_top_k", 3),
        prf_term_count=config.get("prf_term_count", 5),
    )
