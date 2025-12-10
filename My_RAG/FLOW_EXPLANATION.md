# RAG 系統流程說明

## 整體架構

從輸入 queries 到最終生成答案的完整流程如下：

```
輸入 Queries → Chunking → Retrieval (混合檢索) → Generation (答案生成)
```

---

## 詳細流程

### 1. 初始化階段 (`main.py`)

#### 1.1 載入配置和資料
- 從 `configs/config_local.yaml` 或 `config_submit.yaml` 載入配置
- 載入文檔 (`docs_path`) 和查詢 (`query_path`)

#### 1.2 文檔分塊 (Chunking)
```python
chunks = chunk_documents(
    docs=docs_for_chunking,
    language=language,
    chunk_size=chunk_size,      # 從配置讀取
    chunk_overlap=chunk_overlap # 從配置讀取
)
```
- 將長文檔切分成較小的 chunks
- 支援中英文不同的分塊策略

#### 1.3 建立檢索器 (Retriever)
```python
retriever = create_retriever(chunks, language, retrieval_config, docs_path)
```
- 建立 `BM25Retriever` 實例
- 可選：初始化知識圖譜檢索器 (`KGRetriever`)

---

### 2. 檢索階段 (Retrieval) - `retriever.py`

對每個 query，執行以下步驟：

#### 2.1 稀疏檢索 (BM25)
```python
tokenized_query = self._tokenize(query)  # 分詞
bm25_scores = self.bm25.get_scores(tokenized_query)  # BM25 評分
```
- **英文**：使用 Porter Stemmer + 停用詞過濾
- **中文**：使用 jieba 分詞 + 中文停用詞過濾

#### 2.2 密集檢索 (Dense Retrieval)
```python
query_emb = self._compute_embeddings([query])  # 計算 query embedding
dense_scores = np.dot(self.chunk_embeddings, query_vec_norm.T)  # 向量相似度
```
- 使用 SentenceTransformer 模型 (`all_minilm_l6` 或 `ms-marco-MiniLM-L-6-v2`)
- 預先計算所有 chunks 的 embeddings 並建立 FAISS 索引
- 計算 query 與 chunks 的 cosine similarity

#### 2.3 混合評分 (Hybrid Scoring)
```python
hybrid_scores = (1 - dense_weight) * bm25_norm + dense_weight * dense_norm
```
- 結合 BM25 和 Dense 分數
- `dense_weight` 從配置讀取（預設 0.5）

#### 2.4 知識圖譜增強 (KG Boost) - 可選
```python
kg_boost_scores = self._get_kg_boost_scores(query)
```
- 如果啟用 KG retriever (`kg_boost > 0`)：
  - 從 query 中提取實體
  - 透過知識圖譜找到相關的 `doc_id`
  - 對相關 chunks 進行分數提升
  - **多跳關係**：如果 query 包含時間序列、過程等關鍵詞，會啟用 multi-hop 檢索

#### 2.5 關鍵詞增強 (Keyword Boost) - 可選
```python
keywords_to_use = self._extract_keywords(query)
boosted_score = base_score + (keyword_boost * overlap)
```
- 提取 query 中的關鍵詞
- 計算 chunk 與關鍵詞的重疊度
- 對候選 chunks 進行重新排序

#### 2.6 候選選擇
```python
candidate_count = top_k * candidate_multiplier  # 通常取 3 * top_k
top_indices = np.argsort(hybrid_scores)[::-1][:candidate_count]
selected = [self.chunks[idx] for idx in top_indices[:top_k]]
```
- 先選出 `top_k * candidate_multiplier` 個候選
- 經過關鍵詞增強後，最終選出 `top_k` 個 chunks

---

### 3. 生成階段 (Generation) - `generator.py`

#### 3.1 相關性過濾 (可選)
```python
usable_chunks = _filter_chunks_by_relevance(query, context_chunks, language)
```
- 如果 chunks 數量 ≤ 6，使用 LLM 判斷每個 chunk 是否相關
- 過濾掉明顯不相關的 chunks（標記為 "C"）

#### 3.2 構建上下文 (Context Building)
```python
context_block = _build_context_block(query, context_chunks, language)
```
- 為每個 chunk 添加來源標籤：
  - `Source 1 [company=XXX | id=39]`
  - `Source 2 [court=YYY]`
  - `Source 3 [patient=ZZZ]`
- 將所有 chunks 組合成一個長字串

#### 3.3 生成 Prompt
根據語言選擇不同的 prompt 模板：

**英文 Prompt** (`_create_prompt_en`):
- 強調使用 **ONLY** context 中的資訊
- 注意時間、實體歸屬、數值單位
- 無法回答時回覆 "Unable to answer."

**中文 Prompt** (`_create_prompt_zh`):
- 強調完全依據上下文資料
- 注意時間點、實體對應、數值單位
- 無法回答時回覆 "无法回答。"

#### 3.4 調用 LLM 生成答案
```python
response = client.generate(
    model=cfg["model"],      # 從配置讀取
    prompt=prompt,
    stream=False,
    options={
        "temperature": 0.1,  # 低溫度，減少幻覺
        "num_ctx": 8192,     # 上下文長度
    }
)
```
- 使用 Ollama 客戶端調用本地 LLM
- 模型名稱和 host 從配置檔讀取

---

## 知識圖譜檢索器 (`kg_retriever.py`)

### 功能
- 從 `kg_output.json` 載入實體和關係
- 根據 query 提取實體，找到相關的 `doc_id`
- 支援多跳關係檢索（multi-hop）

### 檢索策略

1. **實體提取**
   - 英文：大小寫匹配 + 去除公司後綴（Ltd., Inc. 等）
   - 中文：jieba 分詞匹配

2. **文檔評分**
   - 直接匹配：`entity_name -> doc_id` (權重 10.0)
   - 實體關係：`entity_id -> doc_id` (權重 5.0)
   - 多跳關係：透過 BFS 找到相關實體 (權重遞減)
   - 關鍵詞匹配：relation type 和 properties 匹配 (權重 1.0-0.5)

3. **多跳檢索**
   - 只在 query 包含時間序列、過程等關鍵詞時啟用
   - 使用 BFS 遍歷實體圖，最多 `max_hops` 跳（預設 2）
   - 只對與起始實體有共同 `doc_id` 的實體計分，避免跨文件污染

---

## 配置參數

### Retrieval 配置
- `top_k`: 最終返回的 chunks 數量（可按語言配置）
- `dense_weight`: 密集檢索權重 (0.0-1.0)
- `candidate_multiplier`: 候選 chunks 倍數（預設 3.0）
- `keyword_boost`: 關鍵詞增強權重
- `kg_boost`: 知識圖譜增強權重
- `kg_max_hops`: 多跳關係最大跳數

### Generation 配置
- `ollama.host`: Ollama 服務地址
- `ollama.model`: 使用的 LLM 模型名稱

---

## 流程圖

```
┌─────────────┐
│ 輸入 Queries │
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│ 1. 載入文檔和配置 │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│ 2. 文檔分塊      │
│    (Chunking)    │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│ 3. 建立檢索器    │
│    (Retriever)   │
└──────┬──────────┘
       │
       ▼
┌─────────────────────────────────────┐
│ 4. 對每個 Query 進行檢索              │
│    ├─ BM25 稀疏檢索                  │
│    ├─ Dense 密集檢索                 │
│    ├─ 混合評分                        │
│    ├─ KG Boost (可選)                │
│    └─ Keyword Boost (可選)           │
└──────┬──────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│ 5. 生成答案                          │
│    ├─ 相關性過濾 (可選)              │
│    ├─ 構建上下文                      │
│    ├─ 生成 Prompt                    │
│    └─ 調用 LLM 生成                  │
└──────┬──────────────────────────────┘
       │
       ▼
┌─────────────┐
│ 輸出答案     │
└─────────────┘
```

---

## 關鍵設計特點

1. **混合檢索**：結合 BM25（稀疏）和 Embedding（密集）檢索
2. **知識圖譜增強**：透過實體關係提升相關文檔分數
3. **多跳推理**：支援透過實體關係鏈找到間接相關文檔
4. **相關性過濾**：使用 LLM 過濾明顯不相關的 chunks
5. **來源標註**：為每個 chunk 添加清晰的來源標籤，幫助 LLM 區分不同實體
6. **語言適配**：針對中英文使用不同的分詞和 prompt 策略

