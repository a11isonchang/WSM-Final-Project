from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_core.documents import Document
import re
import jieba
from typing import List, Dict, Any, Optional
from pathlib import Path

class AdvancedRAGRetriever:
    """
    A custom retrieval engine that orchestrates hybrid search (Lexical + Semantic)
    and applies query optimization techniques.
    """
    
    def __init__(self, chunks: List[Dict[str, Any]], language: str, config: Optional[Dict] = None):
        self.chunks = chunks
        self.language = language
        self.config = config or {}
        self.retriever = None
        self._stopwords = self._load_stopwords()
        self._build_retrieval_engine()

    def _load_stopwords(self) -> set:
        """Loads language-specific stopwords from local files."""
        filename = 'stopwords_learned_zh.txt' if self.language == 'zh' else 'stopwords_learned_en.txt'
        path = Path(__file__).parent / filename
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                return {line.strip() for line in f}
        return set()

    def _preprocess_text(self, text: str) -> List[str]:
        """Custom tokenizer for BM25 that handles mixed CJK/Latin text."""
        is_chinese = any(u'\u4e00' <= c <= u'\u9fff' for c in text)
        if is_chinese:
            tokens = jieba.cut(text)
        else:
            tokens = re.findall(r'\w+', text.lower())
        return [t for t in tokens if t not in self._stopwords and t.strip()]

    def _optimize_query(self, query: str) -> str:
        """
        Heuristic Query Expansion: Boosts proper nouns and entity names
        by repeating them in the query string.
        """
        if self.language == 'zh':
            return query # Skip for Chinese for now
            
        # Extract capitalized words (potential entities) from English query
        # avoiding start of sentence capitalization if possible
        entities = re.findall(r'\b[A-Z][a-z0-9]+\b', query)
        
        # Filter out common question starters to avoid boosting "Who", "What"
        common_starters = {'Who', 'What', 'Where', 'When', 'Why', 'How', 'Is', 'Are', 'Do', 'Does'}
        entities = [e for e in entities if e not in common_starters]
        
        if entities:
            # Append entities to boost their BM25 score
            boost_str = " ".join(entities)
            # print(f"🚀 Boosting Query: '{query}' + [{boost_str}]")
            return f"{query} {boost_str}"
            
        return query

    def _build_retrieval_engine(self):
        """Initializes the underlying LangChain Ensemble components."""
        # 1. Prepare Documents
        docs = [Document(page_content=c['page_content'], metadata=c.get('metadata', {})) 
                for c in self.chunks]
        
        # 2. Lexical Search (BM25)
        bm25 = BM25Retriever.from_documents(
            docs, 
            preprocess_func=self._preprocess_text
        )
        bm25.k = 100 # Wide net
        
        # 3. Semantic Search (Dense)
        ollama_conf = self.config.get("ollama", {})
        dense_conf = self.config.get("dense", {})
        
        model_name = dense_conf.get("model_zh") if self.language == "zh" else dense_conf.get("model_en")
        if not model_name:
            model_name = "qwen3-embedding:0.6b" if self.language == "zh" else "embeddinggemma:300m"
            
        host = ollama_conf.get("host", "http://localhost:11434")
        
        embedding_model = OllamaEmbeddings(model=model_name, base_url=host)
        vectorstore = FAISS.from_documents(docs, embedding_model)
        dense = vectorstore.as_retriever(search_kwargs={"k": 100})
        
        # 4. Ensemble
        if self.language == "en":
            ensemble_weights = [0.6, 0.4] # Favor BM25 for English
        else:
            ensemble_weights = [0.4, 0.6] # Favor Dense for Chinese
            
        self.retriever = EnsembleRetriever(
            retrievers=[bm25, dense],
            weights=ensemble_weights
        )

    def retrieve(self, query: str, top_k: int = 5) -> tuple[List[Dict], Dict]:
        """
        Executes the retrieval pipeline:
        Query Optimization -> Ensemble Search -> Result Formatting
        """
        # Step 1: Query Optimization
        optimized_query = self._optimize_query(query)
        
        # Step 2: Execution
        # We request slightly more than top_k to handle potential duplicates/filtering if we added that later
        retrieved_docs = self.retriever.invoke(optimized_query)
        
        # Step 3: Formatting
        results = []
        for doc in retrieved_docs[:top_k]:
            results.append({
                'page_content': doc.page_content,
                'metadata': doc.metadata
            })
            
        debug_info = {
            "original_query": query,
            "optimized_query": optimized_query,
            "top_k": top_k
        }
        
        return results, debug_info

def create_retriever(chunks, language, config=None, parent_docs=None):
    """Factory function to instantiate the Advanced Retriever."""
    # Note: parent_docs is ignored as we moved to a pure-chunk strategy for precision
    return AdvancedRAGRetriever(chunks, language, config)
