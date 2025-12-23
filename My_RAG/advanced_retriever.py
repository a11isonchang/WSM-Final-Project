import json
import jieba
from ollama import Client
from config import load_config
# 修正 1: 引用正確的類別名稱
from rerank_model import GraphWeightedReranker

class AdvancedHybridRetriever:
    def __init__(self, original_retriever, kg_path="My_RAG/kg_output.json", language="en"):
        self.base_retriever = original_retriever
        self.language = language
        # 修正 1: 實例化正確的類別
        self.reranker = GraphWeightedReranker()
        self.kg_index = self._load_kg_index(kg_path)
        
        config = load_config()
        self.initial_k = config.get("retrieval", {}).get("initial_k", 50)
        print(f"AdvancedHybridRetriever initialized with KG boosting (language={language}, initial_k={self.initial_k}).")

    def _load_kg_index(self, kg_path):
        """建立 實體名稱 -> 相關文檔 ID 的映射"""
        entity_to_docs = {}
        try:
            with open(kg_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for triple in data.get('triples', []):
                    doc_id = str(triple.get('doc_id'))
                    head = triple.get('head', '').lower()
                    tail = triple.get('tail', '').lower()
                    
                    if head: 
                        if head not in entity_to_docs: entity_to_docs[head] = set()
                        entity_to_docs[head].add(doc_id)
                    if tail:
                        if tail not in entity_to_docs: entity_to_docs[tail] = set()
                        entity_to_docs[tail].add(doc_id)
        except Exception as e:
            print(f"KG Load Error: {e}")
        return entity_to_docs

    def _get_entity_boost_map(self, query):
        """分析 Query 中的實體，找出哪些文檔應該被加分"""
        boost_map = {} 
        tokens = jieba.lcut(query)
        for token in tokens:
            t = token.lower()
            if t in self.kg_index:
                for doc_id in self.kg_index[t]:
                    boost_map[doc_id] = boost_map.get(doc_id, 0) + 1.0
        
        if boost_map:
            max_boost = max(boost_map.values())
            for k in boost_map:
                boost_map[k] = boost_map[k] / max_boost 
        return boost_map

    def _rewrite_query(self, query: str) -> str:
        """Use LLM to rewrite the query for better retrieval."""
        try:
            config = load_config()
            ollama_cfg = config.get("ollama", {})
            host = ollama_cfg.get("host", "http://127.0.0.1:11434")
            model = "granite4:3b" # Enforce granite4:3b as requested

            client = Client(host=host)
            
            if self.language == "zh":
                prompt = f"""你是一个搜索优化助手。请重写以下查询，使其更适合搜索引擎检索。
保持原意不变，但可以补充相关的关键词或同义词，使其更完整。
直接输出重写后的查询，不要包含任何解释或其他文字。

原查询：{query}
重写后的查询："""
            else:
                prompt = f"""You are a search optimization assistant. Rewrite the following query to make it better for search engine retrieval.
Keep the original intent but add relevant keywords or synonyms to make it more comprehensive.
Output ONLY the rewritten query, without any explanation.

Original Query: {query}
Rewritten Query:"""

            response = client.generate(
                model=model,
                prompt=prompt,
                stream=False,
                options={
                    "temperature": 0.0, # Low temperature for deterministic output
                    "num_ctx": 1024,
                },
            )
            rewritten = response.get("response", "").strip()
            # If empty or too short, fallback to original
            if not rewritten or len(rewritten) < 2:
                return query
            
            # Remove quotes if the model added them
            if rewritten.startswith('"') and rewritten.endswith('"'):
                rewritten = rewritten[1:-1]
            
            return rewritten

        except Exception as e:
            print(f"[Query Rewrite Error] {e}")
            return query

    def retrieve(self, query, top_k=5, query_id=None):
        # 0. Rewrite Query
        rewritten_query = self._rewrite_query(query)
        if rewritten_query != query:
            print(f"[Query Rewrite] '{query}' -> '{rewritten_query}'")
        
        # 策略：先抓 50 篇給 Reranker 挑，最後只回傳 top_k
        initial_k = self.initial_k
        
        # 修正 3: 接收 base_retriever 回傳的 tuple (chunks, debug_info)
        # 使用重写后的查询进行检索
        candidates = self.base_retriever.retrieve(rewritten_query, top_k=initial_k)
        debug_info = {}

        
        if not candidates:
            return [], debug_info

        # 計算 KG Boost (使用重写后的查询)
        boost_map = self._get_entity_boost_map(rewritten_query)
        
        # 修正 2: 使用正確的參數名稱 kg_boost_map
        try:
           final_docs = self.reranker.rerank(rewritten_query, candidates, top_k=top_k, kg_boost_map=boost_map)
        except Exception as e:
            print(f"[Reranker Warning] Reranking failed (using base results): {e}")
            final_docs = candidates[:top_k]
        
        # Fallback: Just return top_k from base retriever
        # final_docs = candidates[:top_k]

        # 修正 3: 更新 debug_info 並回傳 tuple
        # 我們需要把重排序後的分數更新到 debug 資訊中，這樣 main.py 的 print 才會顯示正確分數
        new_debug_results = []
        for doc in final_docs:
            new_debug_results.append({
                "metadata": doc.get("metadata", {}),
                "preview": doc.get("page_content", "")[:160].replace("\n", " "),
                "score": doc.get("metadata", {}).get("score", 0.0) # Use base score
            })
        
        debug_info["results"] = new_debug_results
        debug_info["candidate_count"] = len(candidates) # 記錄原本召回了多少篇
        debug_info["top_k"] = top_k # 記錄最終回傳多少篇
        
        return final_docs, debug_info