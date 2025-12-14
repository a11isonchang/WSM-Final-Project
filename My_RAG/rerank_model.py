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
