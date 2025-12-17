# 專案架構與流程說明

## 專案概述

這是一個基於 RAG (Retrieval-Augmented Generation) 架構的問答系統，支援中英文雙語，使用混合檢索策略（詞彙檢索 + 密集向量檢索 + 知識圖譜增強）來提升檢索準確率，並結合大語言模型生成答案。

## 系統架構

### 整體架構圖

```
┌─────────────────────────────────────────────────────────────┐
│                      輸入層                                   │
│  - 查詢 (queries)                                           │
│  - 文檔 (documents)                                         │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                   文檔處理層 (Document Processing)            │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  1. Query-Aware Chunking (查詢感知分塊)               │  │
│  │     - 根據查詢類型動態調整 chunk_size/overlap         │  │
│  │     - 支援遞迴式文字分割 (RecursiveCharacterTextSplit)│  │
│  │     - 語言感知：中文(字符數) vs 英文(詞數)            │  │
│  └──────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  2. Knowledge Graph Construction (知識圖譜構建)       │  │
│  │     - 從文檔中抽取實體關係三元組                      │  │
│  │     - 儲存為 kg_output.json                          │  │
│  └──────────────────────────────────────────────────────┘  │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                   檢索層 (Retrieval Layer)                   │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  混合檢索器 (Hybrid Retriever)                        │  │
│  │  ┌──────────────────────────────────────────────┐   │  │
│  │  │ 1. Lexical Retrieval (詞彙檢索)              │   │  │
│  │  │    - BM25 (Best Matching 25)                 │   │  │
│  │  │    - 支援中英文停用詞過濾                      │   │  │
│  │  │    - 英文：Porter Stemmer                     │   │  │
│  │  │    - 中文：jieba 分詞                         │   │  │
│  │  └──────────────────────────────────────────────┘   │  │
│  │  ┌──────────────────────────────────────────────┐   │  │
│  │  │ 2. Dense Retrieval (密集向量檢索)            │   │  │
│  │  │    - Embedding 模型：                         │   │  │
│  │  │      * 英文：all-MiniLM-L6-v2                 │   │  │
│  │  │      * 中文：qwen3-embedding:0.6b (Ollama)   │   │  │
│  │  │    - FAISS 索引 (IndexFlatIP/L2)             │   │  │
│  │  │    - GPU 加速支援                            │   │  │
│  │  └──────────────────────────────────────────────┘   │  │
│  │  ┌──────────────────────────────────────────────┐   │  │
│  │  │ 3. Knowledge Graph Enhancement (KG增強)      │   │  │
│  │  │    - 實體匹配提升相關文檔分數                  │   │  │
│  │  │    - 支援多跳推理 (Multi-hop reasoning)      │   │  │
│  │  └──────────────────────────────────────────────┘   │  │
│  └──────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  重排序器 (Reranker)                                 │  │
│  │  - Cross-Encoder 語義相似度計算                     │  │
│  │  - KG Boost 分數融合                                │  │
│  │  - Remote API: ollama-gateway:11434/rerank         │  │
│  └──────────────────────────────────────────────────────┘  │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                   生成層 (Generation Layer)                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Answer Generator                                     │  │
│  │  - LLM: granite4:3b (via Ollama)                    │  │
│  │  - 上下文重排序 (避免 Lost in the Middle)            │  │
│  │  - 支援中英文 prompt 模板                            │  │
│  │  - 引用來源追蹤                                      │  │
│  └──────────────────────────────────────────────────────┘  │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                      輸出層                                   │
│  - 答案內容 (prediction.content)                            │
│  - 參考文檔 (prediction.references)                         │
│  - 元資料 (query_type, doc_ids, etc.)                      │
└─────────────────────────────────────────────────────────────┘
```

## 核心組件詳解

### 1. Query-Aware Chunking (查詢感知分塊)

**檔案**: `My_RAG/chunker.py`

**原理**:
- 根據查詢類型動態調整分塊策略
- 查詢類型分類使用正則表達式匹配：
  - `FACTUAL`: 事實性問題
  - `MULTI_DOC_INTEGRATION`: 多文檔整合
  - `MULTI_DOC_COMPARISON`: 多文檔比較
  - `MULTI_DOC_TIMESEQ`: 多文檔時間序列
  - `MULTIHOP`: 多跳推理
  - `SUMMARY`: 總結性問題

**技術細節**:
- 使用 `langchain.text_splitter.RecursiveCharacterTextSplitter`
- 中文以字符數計算 (`chunk_size: 128`, `chunk_overlap: 64`)
- 英文以詞數計算 (`chunk_size: 384`, `chunk_overlap: 20`)
- 相同查詢類型使用緩存機制，避免重複分塊

### 2. Hybrid Retriever (混合檢索器)

**檔案**: `My_RAG/retriever.py`, `My_RAG/advanced_retriever.py`

**架構**:
```
BM25Retriever (基礎檢索器)
    ↓
AdvancedHybridRetriever (進階混合檢索器)
    ↓
GraphWeightedReranker (圖權重重排序)
```

#### 2.1 Lexical Retrieval (BM25)

**技術**: `rank_bm25.BM25Okapi`

**參數**:
- `k1`: 1.5 (詞頻飽和度參數)
- `b`: 0.70 (文檔長度正規化參數)

**流程**:
1. 文本預處理：
   - 英文：小寫化、詞幹提取 (Porter Stemmer)、停用詞過濾
   - 中文：jieba 分詞、停用詞過濾
2. 構建 BM25 索引
3. 計算查詢與文檔的 BM25 分數
4. 候選池擴展：`candidate_count = top_k * candidate_multiplier` (預設 3.0)

#### 2.2 Dense Retrieval (密集向量檢索)

**技術**: `sentence-transformers` + `faiss`

**模型**:
- 英文：`all-MiniLM-L6-v2` (本地模型)
- 中文：`qwen3-embedding:0.6b` (Ollama API)

**流程**:
1. 文檔編碼：將 chunks 轉換為向量嵌入
2. FAISS 索引構建：
   - 使用 `IndexFlatIP` (內積) 或 `IndexFlatL2` (歐式距離)
   - 支援 GPU 加速 (`use_gpu: true`)
3. 查詢編碼：將查詢轉換為向量
4. 相似度搜索：使用 FAISS 快速檢索 top-k 候選

**分數融合**:
```python
final_score = (1 - dense_weight) * bm25_score + dense_weight * dense_score
```
- `dense_weight`: 中文 0.5, 英文 0.7

#### 2.3 Knowledge Graph Enhancement

**檔案**: `My_RAG/kg_retriever.py`, `My_RAG/build_kg.py`

**原理**:
- 從文檔中抽取實體-關係-實體三元組
- 構建實體到文檔 ID 的映射索引
- 在檢索時匹配查詢中的實體，提升相關文檔分數

**流程**:
1. 實體提取：使用 jieba 分詞提取查詢中的實體
2. 實體匹配：在 KG 索引中查找匹配的實體
3. 文檔加分：計算實體匹配的 boost 分數
4. 分數融合：`final_score = base_score + kg_boost * entity_match_score`

### 3. Reranker (重排序器)

**檔案**: `My_RAG/rerank_model.py`

**技術**: Cross-Encoder (遠程 API)

**API**: `http://ollama-gateway:11434/rerank`

**原理**:
- Cross-Encoder 比 Bi-Encoder 更準確（因為能同時看到查詢和文檔）
- 對候選文檔重新排序，結合語義相似度和 KG boost

**流程**:
1. 構建查詢-文檔對：`(query, document_content)`
2. 批次調用 API 獲取相似度分數
3. 融合 KG boost 分數
4. 按最終分數排序，返回 top-k

### 4. Answer Generator (答案生成器)

**檔案**: `My_RAG/generator.py`

**技術**: Ollama API + granite4:3b 模型

**流程**:
1. **上下文準備**:
   - 接收重排序後的 top-k chunks (預設 3)
   - 支援 pseudo chunks (假設性文檔，用於引導但不可引用)

2. **Prompt 構建**:
   - 根據語言選擇中英文模板
   - 包含指令、查詢、上下文、格式要求

3. **答案生成**:
   - 調用 Ollama API 生成答案
   - 支援流式輸出和完整輸出

4. **結果後處理**:
   - 提取答案內容
   - 整理引用來源 (`references`)
   - 記錄文檔 ID (`reference_doc_ids`)

**特殊機制**:
- **Lost in the Middle 緩解**: 對 chunks 重新排序，將重要內容放在中間位置
- **相關性過濾**: 使用 LLM 粗略判斷 chunk 是否相關

## 工作流程

### 完整執行流程

```
1. 初始化階段
   ├─ 載入配置文件 (config_local.yaml 或 config_submit.yaml)
   ├─ 載入文檔和查詢數據
   └─ 初始化檢索器和生成器

2. 文檔預處理 (緩存機制)
   ├─ 根據查詢類型分類
   ├─ 動態選擇 chunking 參數
   ├─ 執行 query-aware chunking
   └─ 構建/載入 FAISS 索引

3. 查詢處理循環 (對每個查詢)
   ├─ 查詢類型推斷
   │  └─ 使用正則表達式匹配查詢模式
   │
   ├─ 檢索階段
   │  ├─ Lexical Retrieval (BM25)
   │  │  └─ 擴展候選池 (top_k * multiplier)
   │  │
   │  ├─ Dense Retrieval (FAISS)
   │  │  ├─ 查詢編碼
   │  │  └─ 向量相似度搜索
   │  │
   │  ├─ 分數融合 (BM25 + Dense)
   │  │
   │  └─ KG Enhancement
   │     ├─ 實體提取
   │     └─ 文檔加分
   │
   ├─ 重排序階段
   │  ├─ Cross-Encoder 語義分數
   │  └─ KG boost 融合
   │
   └─ 生成階段
      ├─ 構建 prompt (包含 top-k chunks)
      ├─ LLM 生成答案
      └─ 後處理 (提取內容、整理引用)

4. 輸出保存
   └─ 將結果寫入 JSONL 文件
```

### 執行腳本流程

**檔案**: `run.sh`

```bash
1. 執行推理 (英文)
   python My_RAG/main.py --query_path ... --docs_path ... --language en --output ...

2. 格式檢查 (英文)
   python check_output_format.py --query_file ... --processed_file ...

3. 執行推理 (中文)
   python My_RAG/main.py --query_path ... --docs_path ... --language zh --output ...

4. 格式檢查 (中文)
   python check_output_format.py --query_file ... --processed_file ...

5. 評估 (英文)
   python rageval/evaluation/main.py --input_file predictions_en.jsonl --language en

6. 評估 (中文)
   python rageval/evaluation/main.py --input_file predictions_zh.jsonl --language zh

7. 處理評估結果
   python rageval/evaluation/process_intermediate.py
```

## 使用技術棧

### 核心框架與庫

1. **自然語言處理**:
   - `jieba`: 中文分詞
   - `nltk`: 英文文本處理 (停用詞、詞幹提取)
   - `langchain`: 文本分割和文檔處理

2. **檢索技術**:
   - `rank_bm25`: BM25 詞彙檢索
   - `sentence-transformers`: 文本嵌入模型
   - `faiss` (faiss-cpu/faiss-gpu): 向量相似度搜索索引
   - `scikit-learn`: 特徵提取 (TF-IDF)

3. **大語言模型**:
   - `ollama`: 本地 LLM API 客戶端
   - `granite4:3b`: 用於答案生成
   - `qwen3-embedding:0.6b`: 用於中文嵌入

4. **知識圖譜**:
   - 自定義實體關係抽取
   - JSON 格式存儲三元組

5. **工具庫**:
   - `numpy`: 數值計算
   - `jsonlines`: JSONL 文件處理
   - `tqdm`: 進度條顯示
   - `PyYAML`: 配置文件解析

### 外部服務

1. **Ollama Gateway**:
   - 主機: `http://ollama-gateway:11434`
   - 用途:
     - LLM 推理 (granite4:3b)
     - 嵌入向量生成 (qwen3-embedding:0.6b)
     - Cross-Encoder 重排序 (rerank API)

## 配置文件架構

### 配置文件結構

**檔案**: `configs/config_submit.yaml`

```yaml
ollama:
  host: "http://ollama-gateway:11434"
  model: "granite4:3b"

retrieval:
  # 分塊設定 (語言感知)
  chunking:
    zh:
      chunk_size: 128        # 字符數
      chunk_overlap: 64
    en:
      chunk_size: 384        # 詞數
      chunk_overlap: 20
  
  # 檢索參數
  top_k: 6                   # 最終返回的文檔數
  candidate_multiplier: 3.0  # 候選池擴展倍數
  keyword_boost: 0.15        # 關鍵詞加分權重
  min_keyword_characters: 3  # 最小關鍵詞長度
  
  # 密集檢索權重
  dense_weight:
    zh: 0.5
    en: 0.7
  
  # 嵌入模型設定
  embedding_provider: "ollama"
  embedding_model_path: "qwen3-embedding:0.6b"
  ollama_host: "http://ollama-gateway:11434"
  min_dense_similarity: 0.22  # 最小相似度閾值
  
  # 知識圖譜設定
  kg_boost: 0
  kg_path: "My_RAG/kg_output.json"
  
  # BM25 參數
  bm25:
    k1: 1.5
    b: 0.70
  
  # 重排序 API
  rerank:
    api_url: "http://ollama-gateway:11434/rerank"
  
  debug: false  # 檢索調試模式
```

## 核心設計原則

### 1. 混合檢索策略 (Hybrid Retrieval)

**原理**: 結合詞彙檢索和語義檢索的優勢
- **詞彙檢索 (BM25)**: 精確匹配關鍵詞，適合事實性問題
- **語義檢索 (Dense)**: 理解語義相似度，適合意圖理解

**實現**: 加權融合兩種分數，根據語言調整權重

### 2. Query-Aware Processing

**原理**: 根據查詢類型動態調整處理策略
- 不同查詢類型需要不同的 chunk 大小
- 事實性問題需要精確匹配 (小 chunk)
- 總結性問題需要上下文 (大 chunk)

**實現**: 使用正則表達式分類查詢，緩存相同類型的處理結果

### 3. 多階段檢索與重排序

**原理**: 先快速召回，再精確排序
1. **第一階段**: BM25 + Dense 快速召回候選 (擴大候選池)
2. **第二階段**: Cross-Encoder 精確重排序
3. **第三階段**: KG 增強提升相關文檔

**優勢**: 平衡效率和準確率

### 4. 知識圖譜增強

**原理**: 利用結構化知識提升檢索準確率
- 實體匹配可以發現語義相關但詞彙不匹配的文檔
- 支援多跳推理，發現間接相關的資訊

**實現**: 預先構建 KG，在檢索時進行實體匹配和加分

### 5. 緩存與優化

**原理**: 減少重複計算，提升效率
- Query-aware chunking 結果緩存
- FAISS 索引持久化
- 嵌入向量預計算

**實現**: 使用字典緩存和文件持久化

## 性能優化策略

1. **索引預構建**: FAISS 索引和嵌入向量預先計算並持久化
2. **批次處理**: 嵌入編碼和重排序 API 調用使用批次處理
3. **GPU 加速**: FAISS 索引支援 GPU 加速
4. **緩存機制**: 查詢類型分組和結果緩存
5. **並行處理**: 使用環境變數控制線程數，避免資源競爭

## 評估指標

系統使用 `rageval` 框架進行評估，主要指標包括：
- **ROUGE**: 答案與參考答案的重疊度
- **BLEU**: 翻譯質量指標
- **其他 RAG 特定指標**: 引用準確率、事實正確性等

## 總結

本專案實現了一個完整的 RAG 系統，核心特色包括：

1. **混合檢索**: BM25 + Dense + KG 三層檢索策略
2. **查詢感知**: 動態調整處理策略以適應不同查詢類型
3. **多階段重排序**: 粗排 + 精排兩階段提升準確率
4. **知識圖譜增強**: 利用結構化知識提升檢索質量
5. **雙語支援**: 完整的中英文處理流程

整個系統在保證檢索準確率的同時，通過緩存、索引預構建等優化策略提升了執行效率。

