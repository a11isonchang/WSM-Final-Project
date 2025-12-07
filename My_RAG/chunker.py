from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from typing import Dict, Optional


# Default embedding models for semantic chunking
OLLAMA_EMBED_MODELS = {
    "en": "embeddinggemma:300m",
    "zh": "qwen3-embedding:0.6b",
}

FALLBACK_EMBED_MODELS = {
    "en": "sentence-transformers/all-MiniLM-L6-v2",
    "zh": "intfloat/multilingual-e5-large",
}


def get_embedding_model(language: str, ollama_config: Optional[Dict] = None):
    """
    Prefer local Ollama embeddings for semantic chunking, fall back to
    lightweight HF models if Ollama is unavailable.
    """
    lang_key = "zh" if language == "zh" else "en"
    model_name = OLLAMA_EMBED_MODELS[lang_key]
    host = None
    if ollama_config:
        host = ollama_config.get("host") or host

    try:
        return OllamaEmbeddings(model=model_name, base_url=host)
    except Exception as exc:
        print(f"Warning: Ollama embeddings '{model_name}' unavailable ({exc}); using HuggingFace fallback.")
        fallback_model = FALLBACK_EMBED_MODELS[lang_key]
        return HuggingFaceEmbeddings(
            model_name=fallback_model,
            model_kwargs={"device": "cpu"},
        )


def chunk_documents(docs, language, chunk_size=1000, chunk_overlap=200, ollama_config: Optional[Dict] = None):
    """
    Chunk using LangChain SemanticChunker (percentile=92 for ~sentence boundaries).
    Falls back to RecursiveCharacterTextSplitter if semantic fails.
    """
    chunks = []
    embedding_model = get_embedding_model(language, ollama_config)

    try:
        text_splitter = SemanticChunker(
            embedding_model,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=92,  # Tune for sentence-level
        )
    except Exception:
        # Fallback to old recursive
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        separators = ["\n\n", "\n", " ", ""] if language != "zh" else ["\n\n", "\n", "。", "！", "？", " ", ""]
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=separators
        )

    for doc_idx, doc in enumerate(docs):
        if 'content' not in doc or not isinstance(doc['content'], str) or doc.get('language') != language:
            continue

        text = doc['content']
        doc_meta = doc.copy()
        doc_meta.pop('content', None)

        split_texts = text_splitter.split_text(text)
        for i, chunk_content in enumerate(split_texts):
            chunk = Document(
                page_content=chunk_content,
                metadata={**doc_meta, 'chunk_index': i, 'doc_idx': doc_idx}
            )
            chunks.append(chunk.dict())  # Convert back to dict for compatibility

    return chunks
