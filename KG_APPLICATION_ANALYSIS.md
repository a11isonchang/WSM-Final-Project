# 知識圖譜 (KG) 應用流程分析

## 概述

本系統使用 **Think-on-Graph (ToG)** 方法進行知識圖譜推理，並將結果整合到混合檢索系統中。

## 完整流程

### 1. 知識圖譜構建階段 (`build_kg.py`)

**目的**: 從原始文檔中提取結構化的知識圖譜

**流程**:
1. 讀取文檔 (`dragonball_docs.jsonl`)
2. 對每個文檔使用 **OpenAI GPT-4o-mini** 進行實體和關係抽取
3. 生成結構化的 KG，包含：
   - `entities`: 實體列表 (Organization, Person, Event, etc.)
   - `triples`: 三元組列表 (head, relation, tail, doc_id)
4. 合併所有文檔的 KG 為統一的 `kg_output.json`

**輸出**: `My_RAG/kg_output.json`

---

### 2. 檢索階段 - KG 初始化 (`retriever.py`)

**位置**: `create_retriever()` 函數

**流程**:
```python
# 從 config 讀取 KG 設定
kg_boost = config.get("kg_boost", 0.0)  # 預設: 1.2
kg_path = config.get("kg_path", "My_RAG/kg_output.json")
max_hops = config.get("kg_max_hops", 2)  # 多跳推理深度

# 建立 KGRetriever
kg_retriever = create_kg_retriever(kg_path, language, max_hops, kg_docs_path)
```

**KGRetriever 初始化** (`kg_retriever.py`):
1. 載入 `kg_output.json`
2. 建立索引：
   - `_name_to_eids`: 實體名稱 → 實體 ID 映射
   - `_adj`: 鄰接表（圖結構）
   - `_doc_to_triples`: 文檔 ID → 三元組映射
3. **初始化 Ollama 客戶端** (granite4:3b) 用於 ToG 推理

---

### 3. 查詢處理階段 - ToG 推理 (`kg_retriever.py`)

當 `retriever.retrieve(query)` 被調用時：

#### 3.1 種子實體提取 (`_extract_seed_entities`)

**方法**:
1. **精確匹配**: 在查詢中尋找實體名稱的子字串匹配
2. **Token 重疊**: 使用 Jaccard 相似度計算查詢 tokens 與實體名稱的相似度

**輸出**: 種子實體列表 `[(entity_id, score), ...]`

#### 3.2 圖探索 - Beam Search (`rank_docs`)

**流程**:
```
初始 Beams: [種子實體]
↓
For hop in range(1, max_hops + 1):
    1. 擴展每個 beam (遍歷鄰接邊)
    2. 使用 ToG LLM Pruning 選擇最相關的路徑
    3. 保留 top beam_width 個路徑
```

**擴展邏輯** (`_expand`):
- 從當前節點遍歷所有出邊
- 避免循環（不回到前一個節點）
- 計算邊的相關性分數（基於 relation 和目標實體的 token 重疊）

#### 3.3 ToG Pruning (`_prune_with_llm`)

**使用 Ollama granite4:3b 進行路徑選擇**

**Prompt 範例** (中文):
```
你是一個知識圖譜推理系統。給定一個問題和一系列推理路徑，請選擇最相關的路徑來回答問題。

問題：{query}

推理路徑：
Path 1: 實體A -[關係1]-> 實體B
Path 2: 實體C -[關係2]-> 實體D
...

請返回最相關的 6 個路徑的編號（用逗號分隔），例如：1,3,5
```

**輸出**: 選中的路徑索引列表

#### 3.4 ToG Reasoning (`_reason_with_llm`)

**使用 Ollama granite4:3b 進行文檔評分**

**流程**:
1. 將所有路徑按 `doc_id` 分組
2. 為每個文檔構建路徑摘要
3. 使用 LLM 評估每個文檔與查詢的相關性

**Prompt 範例** (中文):
```
你是一個知識圖譜推理系統。給定一個問題和一系列文檔的推理路徑，請評估每個文檔與問題的相關性。

問題：{query}

文檔及其推理路徑：
Document 1:
  Paths: 實體A -[關係1]-> 實體B; 實體B -[關係2]-> 實體C
Document 2:
  Paths: 實體D -[關係3]-> 實體E
...

請為每個文檔評分（0-1之間），格式為：文檔編號:分數
```

**輸出**: `{doc_id: score}` 字典

#### 3.5 分數正規化

將 LLM 評分正規化到 [0, 1] 區間，並返回：
```python
{
    "doc_id": int,
    "score": float,  # 0-1
    "evidence_triples": [...],  # 證據三元組
    "paths": [...]  # 推理路徑
}
```

---

### 4. 混合檢索整合 (`retriever.py`)

#### 4.1 多種檢索方法

**BM25 檢索**:
- 基於詞頻的稀疏檢索
- 計算 `bm25_scores`

**Dense 檢索**:
- 使用 SentenceTransformer 或 Ollama embeddings
- FAISS 索引進行相似度搜索
- 計算 `dense_scores`

**KG Boost** (`_get_kg_boost_scores`):
- 調用 `kg_retriever.rank_docs(query)`
- 將 KG 分數轉換為 boost 分數

#### 4.2 分數融合

```python
# 1. 混合 BM25 和 Dense
hybrid_scores = (1 - dense_weight) * bm25_scores + dense_weight * dense_scores

# 2. 應用 Keyword Boost
boosted_scores = hybrid_scores + keyword_boost * keyword_overlap

# 3. 應用 KG Boost (按 doc_id)
for chunk_idx, chunk in enumerate(chunks):
    doc_id = chunk["metadata"]["doc_id"]
    if doc_id in kg_boost_scores:
        boosted_scores[chunk_idx] += kg_boost_scores[doc_id]
```

**KG Boost 計算邏輯**:
- `kg_boost` (config 中設定，預設 1.2)
- 最大 boost: `min(0.25, 0.15 + 0.07 * kg_boost)`
- 最小 boost: `0.02`
- 非線性 shaping: `boost = min_add + (score^1.8) * (max_add - min_add)`

#### 4.3 最終排序

按 `boosted_scores` 排序，返回 top_k 個 chunks

---

### 5. 答案生成階段 (`generator.py`)

使用檢索到的 chunks 和 Ollama granite4:3b 生成最終答案

---

## ToG 方法的核心優勢

### 1. **LLM-based Pruning**
- **傳統方法**: 基於啟發式分數（token 重疊、hop decay）
- **ToG 方法**: 使用 LLM 理解語義，選擇真正相關的路徑

### 2. **LLM-based Reasoning**
- **傳統方法**: 簡單的投票機制（路徑分數累加）
- **ToG 方法**: LLM 理解整個推理鏈，評估文檔的整體相關性

### 3. **多跳推理**
- 支援最多 `max_hops` 跳的關係探索
- 自動檢測多跳查詢（如 "timeline", "before and after"）

---

## 配置參數

### `config_local.yaml` 相關設定:

```yaml
retrieval:
  kg_boost: 1.2              # KG boost 權重
  kg_path: "My_RAG/kg_output.json"
  kg_max_hops: 2             # 多跳推理深度
  debug_kg: true             # 是否顯示 KG 調試信息

ollama:
  host: "http://localhost:11434"
  model: "granite4:3b"       # ToG 推理使用的模型
```

---

## 數據流圖

```
查詢 (Query)
    ↓
[1] 提取種子實體
    ↓
[2] Beam Search 探索圖結構
    ↓
[3] ToG Pruning (Ollama) → 選擇相關路徑
    ↓
[4] ToG Reasoning (Ollama) → 評分文檔
    ↓
[5] 正規化分數 → {doc_id: score}
    ↓
[6] 轉換為 chunk-level boost
    ↓
[7] 與 BM25 + Dense 分數融合
    ↓
[8] 返回 top_k chunks
    ↓
[9] 生成答案 (Ollama)
```

---

## 關鍵檔案

1. **`kg_retriever.py`**: ToG 推理實現
   - `_extract_seed_entities()`: 種子實體提取
   - `_prune_with_llm()`: LLM-based pruning
   - `_reason_with_llm()`: LLM-based reasoning
   - `rank_docs()`: 主要 API

2. **`retriever.py`**: 混合檢索整合
   - `_get_kg_boost_scores()`: KG boost 計算
   - `retrieve()`: 主檢索流程

3. **`build_kg.py`**: KG 構建
   - `extract_kg_from_doc()`: 單文檔 KG 抽取
   - `merge_knowledge_graphs()`: KG 合併

---

## 改進點與注意事項

### 1. **LLM 調用優化**
- 目前每次查詢會調用 LLM 2 次（pruning + reasoning）
- 可考慮批量處理或緩存

### 2. **Fallback 機制**
- 如果 Ollama 不可用或失敗，自動 fallback 到啟發式方法
- 確保系統穩定性

### 3. **多語言支援**
- 支援中英文 prompt
- 根據 `language` 參數自動切換

### 4. **調試模式**
- 設定 `debug_kg: true` 可查看詳細的 KG 推理過程
- 包括種子實體、推理路徑、文檔評分等

---

## 總結

本系統實現了完整的 ToG 方法：
1. ✅ 使用 Ollama granite4:3b 進行 LLM-based pruning
2. ✅ 使用 Ollama granite4:3b 進行 LLM-based reasoning
3. ✅ 支援多跳推理（最多 2 跳）
4. ✅ 與 BM25 + Dense 檢索無縫整合
5. ✅ 提供完整的 fallback 機制

KG 的應用方式為：**KG 不直接檢索 chunks，而是對 doc_ids 進行評分，然後將分數轉換為 chunk-level 的 boost，與其他檢索方法的分數融合**。

