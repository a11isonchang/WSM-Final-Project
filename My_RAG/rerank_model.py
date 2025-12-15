import requests
import numpy as np
import os
from config import load_config

class RemoteFlagReranker:

    def __init__(self, api_url: str):
        self.api_url = api_url
        self.batch_size = 32  # API 限制每次最多 32 對

    def compute_score(self, pairs, max_length=1024):
        all_scores = []
        
        # 自動分批處理 (Batching)
        for i in range(0, len(pairs), self.batch_size):
            batch_pairs = pairs[i : i + self.batch_size]
            
            # 構造 Payload
            payload = {"pairs": [{"text1": p[0], "text2": p[1]} for p in batch_pairs]}
            
            try:
                # 發送請求
                resp = requests.post(self.api_url, json=payload, timeout=60)
                if resp.status_code != 200:
                    raise RuntimeError(f"API request failed ({resp.status_code}): {resp.text}")
                scores = resp.json()["scores"]
                all_scores.extend(scores)

            except Exception as e:
                print(f"[Rerank Exception] {e}")
                raise

        return np.array(all_scores, dtype=float)

class GraphWeightedReranker:
    def __init__(self, model_path=None, device=None):
        """
        初始化重排序器 (API 版)。
        參數 model_path 和 device 保留是為了相容舊代碼調用，但在這裡不會被使用。
        """
        # 預設使用 Docker 內部 Gateway，但允許環境變數覆蓋 (方便本地測試)
        cfg = load_config()
        rrcfg = (cfg.get("retrieval", {}) or {}).get("rerank", {}) or {}
        default_url = rrcfg.get("api_url", "http://ollama-gateway:11434/rerank")
        self.api_url = os.getenv("RERANK_API_URL", default_url)
        
        self.model = RemoteFlagReranker(self.api_url)
        print(f"[Reranker] Initialized using Remote API: {self.api_url}")

    def rerank(self, query, docs, top_k=5, kg_boost_map=None):
        """
        執行重排序：(API 語義分數) + (KG 實體加分)
        """
        if not docs:
            return []

        # 1. 準備模型輸入 Pair [Query, Document]
        pairs = []
        valid_indices = []
        for i, doc in enumerate(docs):
            content = doc.get('page_content', '')
            if content:
                pairs.append([query, content])
                valid_indices.append(i)

        if not pairs:
            return docs[:top_k]

        # 2. 呼叫 API 獲取分數 (已包含分批邏輯)
        scores = self.model.compute_score(pairs)
        
        # 3. 融合分數 (API Score + KG Boost) - 你的獨家優勢
        final_results = []
        for idx, original_idx in enumerate(valid_indices):
            doc = docs[original_idx]
            
            # 確保分數存在
            base_score = float(scores[idx]) if idx < len(scores) else -999.0
            
            # 計算 KG 加分
            boost = 0.0
            doc_id = str(doc.get('metadata', {}).get('doc_id', ''))
            
            if kg_boost_map:
                # 支援 index map 或 doc_id map
                if original_idx in kg_boost_map:
                    boost = kg_boost_map[original_idx] * 2.0 # 係數可調
                elif doc_id in kg_boost_map:
                    boost = kg_boost_map[doc_id] * 2.0
            
            # 最終分數
            final_score = base_score + boost
            
            # 寫入 metadata 方便 main.py Debug
            if 'metadata' not in doc: doc['metadata'] = {}
            doc['metadata']['rerank_score'] = base_score
            doc['metadata']['kg_boost'] = boost
            doc['metadata']['final_score'] = final_score
            
            final_results.append(doc)

        # 4. 根據最終分數排序
        final_results.sort(key=lambda x: x['metadata']['final_score'], reverse=True)
        
        return final_results[:top_k]

# for local
'''
import torch
from sentence_transformers import CrossEncoder
import numpy as np
import os

class GraphWeightedReranker:
    def __init__(self, model_path="My_RAG/models/bge-reranker-v2-m3", device=None):
        """
        初始化重排序模型。
        :param model_path: 比賽環境中 bge-reranker-v2-m3 的本地路徑
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading Reranker from {model_path} on {self.device}...")
        
        # 如果本地路徑不存在，嘗試從 HuggingFace 下載 (比賽環境通常需要預先下載好)
        if not os.path.exists(model_path):
            print(f"Model path {model_path} not found, trying BAAI/bge-reranker-v2-m3...")
            model_path = "BAAI/bge-reranker-v2-m3"
            
        self.model = CrossEncoder(model_path, device=self.device, max_length=1024)

    def rerank(self, query, docs, top_k=5, kg_boost_map=None):
        """
        執行重排序，並加入 KG 實體加分。
        
        :param docs: List[Dict], 包含 'page_content' 和 'metadata'
        :param kg_boost_map: Dict {doc_id: boost_score}, 來自 AdvancedRetriever 的計算
        """
        if not docs:
            return []

        # 1. 構建模型輸入 (Query, Document) 對
        pairs = [[query, doc.get('page_content', '')] for doc in docs]
        
        # 2. Cross-Encoder 打分 (這步最準，但最慢，所以只對前 30-50 篇做)
        scores = self.model.predict(pairs)
        
        # 3. 融合分數 (Model Score + KG Boost)
        final_results = []
        for i, doc in enumerate(docs):
            base_score = float(scores[i])
            
            # --- 獨特性核心：KG 偏置注入 ---
            # 如果這篇文檔在 KG 路徑中被標記為重要，我們給它加權
            doc_id = str(doc.get('metadata', {}).get('doc_id', ''))
            boost = 0.0
            if kg_boost_map and doc_id in kg_boost_map:
                # 這裡的 2.0 係數是經驗值，可以根據 KG 的可信度調整
                boost = kg_boost_map[doc_id] * 2.0 
            
            # 使用 Sigmoid 歸一化模型分數以便與 Boost 相加 (可選，這裡直接相加更強勢)
            final_score = base_score + boost
            
            # 將分數寫回 metadata 以便 Debug
            doc['metadata']['original_score'] = base_score
            doc['metadata']['kg_boost'] = boost
            doc['metadata']['final_score'] = final_score
            
            final_results.append(doc)

        # 4. 根據最終分數重新排序
        final_results.sort(key=lambda x: x['metadata']['final_score'], reverse=True)
        
        return final_results[:top_k]
'''