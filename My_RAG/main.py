import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from tqdm import tqdm
from utils import load_jsonl, save_jsonl
from chunker import chunk_documents
from retriever import create_retriever
from generator import generate_answer
from config import load_config
import argparse


import kg_retriever

from chunker import get_query_aware_text_splitter
from advanced_retriever import AdvancedHybridRetriever 


def get_chunk_config(config: dict, language: str) -> dict:
    """
    根據語言取得對應的 chunking 設定
    """
    chunking_cfg = config.get("retrieval", {}).get("chunking", {})

    if language and language.startswith("zh"):
        return chunking_cfg["zh"]

    # 預設走英文
    return chunking_cfg.get(language, chunking_cfg.get("en"))


def main(query_path, docs_path, language, output_path):
    config = load_config()
    retrieval_config = config.get("retrieval", {})

    chunk_cfg = get_chunk_config(config, language)
    chunk_size = chunk_cfg["chunk_size"]
    chunk_overlap = chunk_cfg["chunk_overlap"]

    top_k = retrieval_config.get("top_k", 3)
    debug_retrieval = retrieval_config.get("debug", False)

    # 1. Load Data
    print("Loading documents...")
    docs_for_chunking = load_jsonl(docs_path)
    queries = load_jsonl(query_path)
    print(f"Loaded {len(docs_for_chunking)} documents.")
    print(f"Loaded {len(queries)} queries.")

    # 2. Chunk Documents
    print("Chunking documents...")
    print(f"[Chunk Config] language={language}, size={chunk_size}, overlap={chunk_overlap}")

    chunks = chunk_documents(
        docs=docs_for_chunking,
        language=language,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    print(f"Created {len(chunks)} chunks.")

    # 3. Create Retriever
    print("Preparing retriever cache...")
    _chunks_cache = {}
    _retriever_cache = {}

    def build_chunks_for_key(splitter, qres):
        # ✅ 不改 chunk 邏輯：直接回傳你剛剛 chunk_documents 產生的 chunks
        return chunks

    def get_retriever_for_query(query_text: str):
        splitter, qres = get_query_aware_text_splitter(
            language=language,
            query_text=query_text,
            base_chunk_size=chunk_size,
            base_chunk_overlap=chunk_overlap,
            query_context=None,
        )

        key = (
            language,
            qres.canonical_type,
            getattr(splitter, "_chunk_size", None),
            getattr(splitter, "_chunk_overlap", None),
        )

        if key not in _chunks_cache:
            _chunks_cache[key] = build_chunks_for_key(splitter, qres)

        if key not in _retriever_cache:
            base_retriever = create_retriever(
                _chunks_cache[key],
                language,
                retrieval_config,
            )
            _retriever_cache[key] = AdvancedHybridRetriever(
                base_retriever,
                kg_path="My_RAG/kg_output.json",
            )

        return _retriever_cache[key], qres, key

    # 4. Process Queries
    for query in tqdm(queries, desc="Processing Queries"):
        query_text = query["query"]["content"]
        query_id = query["query"].get("query_id")

        if "prediction" not in query or not isinstance(query["prediction"], dict):
            query["prediction"] = {}

        retriever, qres, key = get_retriever_for_query(query_text)

        ret = retriever.retrieve(
            query_text,
            top_k=top_k,
        )
        retrieved_chunks = ret[0] if isinstance(ret, tuple) else ret

        if debug_retrieval:
            print(f"\n[Retrieval Debug] Query ID {query_id}: {query_text}")
            print(f"  Language: {language} | top_k: {top_k} | chunks retrieved: {len(retrieved_chunks)}")
            for idx, chunk in enumerate(retrieved_chunks[:5], start=1):
                meta = chunk.get("metadata", {})
                preview = chunk.get("page_content", "").replace("\n", " ")[:160]
                print(f"    #{idx} meta={meta} preview={preview}")

        # 5. Generate Answer
        answer = generate_answer(
            query_text,
            retrieved_chunks,
            language,
            kg_retriever=kg_retriever
        )

        query["prediction"]["content"] = answer
        query["prediction"]["references"] = (
            [retrieved_chunks[0]["page_content"]] if retrieved_chunks else []
        )

    save_jsonl(output_path, queries)
    print(f"Predictions saved at '{output_path}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query_path", help="Path to the query file")
    parser.add_argument("--docs_path", help="Path to the documents file")
    parser.add_argument("--language", help="Language to filter queries (zh or en)")
    parser.add_argument("--output", help="Path to the output file")
    args = parser.parse_args()

    main(args.query_path, args.docs_path, args.language, args.output)
