# subq_engine.py
from typing import List, Dict, Any
import os
from config import load_config

def load_ollama_config() -> dict:
    """
    讀取 config.yaml 內的 ollama 設定。
    預期結構：
    ollama:
      host: http://127.0.0.1:11434
      model: your-model-name
    """
    config = load_config()
    assert "ollama" in config, "Ollama configuration not found in config file."
    assert "host" in config["ollama"], "Ollama host not specified in config file."
    assert "model" in config["ollama"], "Ollama model not specified in config file."
    return config["ollama"]

def try_subquestion_retrieve(query: str, chunks: List[Dict[str, Any]], language: str, top_k: int, ollama_host: str = None, llm_model: str = None):
    """
    回傳：List[chunks]（可直接丟進 generator）
    若環境沒有 llama_index，或初始化失敗 → 回傳 None
    """
    try:
        from llama_index.core import VectorStoreIndex, Document, Settings
        from llama_index.core.query_engine import SubQuestionQueryEngine
        from llama_index.core.tools import QueryEngineTool, ToolMetadata
        from llama_index.llms.ollama import Ollama
        from llama_index.embeddings.ollama import OllamaEmbedding
        from llama_index.core.question_gen import LLMQuestionGenerator
    except Exception as e:
        print(f"[WARN] Failed to import llama_index modules: {e}")
        return None

    try:
        # 從 config 讀取設定
        config = load_ollama_config()
        ollama_host_cfg = ollama_host or config["host"]
        llm_model_cfg = llm_model or config["model"]
        
        # 讀取 embedding 模型
        retrieval_config = load_config().get("retrieval", {})
        embed_model_name = retrieval_config.get("embedding_model_path", "qwen3-embedding:0.6b")

        # 把現成 chunks 轉成 llamaindex documents
        docs = []
        for c in chunks:
            meta = c.get("metadata", {}) or {}
            txt = c.get("page_content", "") or ""
            docs.append(Document(text=txt, metadata=meta))

        # 初始化 LLM（使用 granite4:3b）
        llm = Ollama(
            model=llm_model_cfg,
            base_url=ollama_host_cfg,
            request_timeout=600.0,
            temperature=0.1,
        )
        
        # 初始化 embedding
        embed = OllamaEmbedding(
            model_name=embed_model_name,
            base_url=ollama_host_cfg
        )

        # 建立索引和基礎查詢引擎
        index = VectorStoreIndex.from_documents(docs, embed_model=embed)
        base_qe = index.as_query_engine(
            similarity_top_k=min(max(top_k * 4, 20), 60),
            llm=llm
        )

        # 建立工具
        tool = QueryEngineTool(
            query_engine=base_qe,
            metadata=ToolMetadata(
                name="local_chunks",
                description="search over retrieved chunks"
            ),
        )

        # **關鍵修正：明確指定 question_gen 使用 Ollama LLM**
        question_gen = LLMQuestionGenerator.from_defaults(llm=llm)

        # 建立 SubQuestionQueryEngine，明確傳入 question_gen
        sqe = SubQuestionQueryEngine.from_defaults(
            query_engine_tools=[tool],
            question_gen=question_gen,  # 明確指定，避免預設去找 OpenAI
            llm=llm,
        )

        # 執行查詢
        resp = sqe.query(query)

        # 把結果轉成 pseudo-chunk
        text = str(resp) if resp is not None else ""
        if not text.strip():
            return None

        pseudo = [{
            "page_content": text,
            "metadata": {
                "source": "subquestion_engine",
                "language": language,
                "is_pseudo": True,
                "model": llm_model_cfg
            }
        }]
        return pseudo
        
    except Exception as e:
        print(f"[WARN] SubQuestion engine failed: {e}")
        return None