import json
import jieba
# 引用新的 Reranker
from rerank_model import GraphWeightedReranker

class AdvancedHybridRetriever:
    def __init__(self, original_retriever, kg_path="My_RAG/kg_output.json"):
        self.base_retriever = original_retriever
        # 修正 1: 實例化正確的類別
        self.reranker = GraphWeightedReranker()
        self.kg_index = self._load_kg_index(kg_path)
        print("AdvancedHybridRetriever initialized with KG boosting.")

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

    def retrieve(self, query, top_k=5, query_id=None):
        # 策略：先抓 50 篇給 Reranker 挑，最後只回傳 top_k
        initial_k = 50 
        
        # 修正 3: 接收 base_retriever 回傳的 tuple (chunks, debug_info)
        # 你的 main.py 預期這裡有 query_id 參數
        candidates, debug_info = self.base_retriever.retrieve(query, top_k=initial_k, query_id=query_id)
        
        if not candidates:
            return [], debug_info

        # [DISABLED RERANKER]
        # 計算 KG Boost
        # boost_map = self._get_entity_boost_map(query)
        
        # 修正 2: 使用正確的參數名稱 kg_boost_map
        # try:
        #     final_docs = self.reranker.rerank(query, candidates, top_k=top_k, kg_boost_map=boost_map)
        # except Exception as e:
        #     print(f"[Reranker Warning] Reranking failed (using base results): {e}")
        #     final_docs = candidates[:top_k]
        
        # Fallback: Just return top_k from base retriever
        final_docs = candidates[:top_k]

        # 修正 3: 更新 debug_info 並回傳 tuple
        # 我們需要把重排序後的分數更新到 debug 資訊中，這樣 main.py 的 print 才會顯示正確分數
        new_debug_results = []
        for doc in final_docs:
            new_debug_results.append({
                "metadata": doc.get("metadata", {}),
                "preview": doc.get("page_content", "")[:160].replace("\n", " "),
                "score": doc.get("metadata", {}).get("final_score", 0.0) # 取重排序後的分數
            })
        
        debug_info["results"] = new_debug_results
        debug_info["candidate_count"] = len(candidates) # 記錄原本召回了多少篇
        debug_info["top_k"] = top_k # 記錄最終回傳多少篇
        
        return final_docs, debug_info
