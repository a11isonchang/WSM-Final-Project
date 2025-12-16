import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from tqdm import tqdm
from utils import load_jsonl, save_jsonl
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

    return chunking_cfg.get(language, chunking_cfg.get("en"))


def main(query_path, docs_path, language, output_path):
    config = load_config()
    retrieval_config = config.get("retrieval", {})

    # chunk config
    chunk_cfg = get_chunk_config(config, language)
    chunk_size = chunk_cfg["chunk_size"]
    chunk_overlap = chunk_cfg["chunk_overlap"]

    # top_k (support per-language)
    top_k_config = retrieval_config.get("top_k", 3)
    if isinstance(top_k_config, dict):
        top_k = top_k_config.get(language, top_k_config.get("en", 3))
    else:
        top_k = top_k_config

    debug_retrieval = retrieval_config.get("debug", False)

    # 1) Load data
    print("Loading documents...")
    docs_for_chunking = load_jsonl(docs_path)
    queries = load_jsonl(query_path)
    print(f"Loaded {len(docs_for_chunking)} documents.")
    print(f"Loaded {len(queries)} queries.")

    # 2) Query-aware chunking caches
    print("Preparing query-aware chunking cache...")

    # key = (language, canonical_type, chunk_size, chunk_overlap)
    _chunks_cache = {}
    _retriever_cache = {}

    def build_chunks_for_key(splitter, qres):
        chunks = []
        for doc in docs_for_chunking:
            if (
                isinstance(doc, dict)
                and isinstance(doc.get("content"), str)
                and doc.get("language") == language
            ):
                text = doc["content"]
                base_metadata = {k: v for k, v in doc.items() if k != "content"}

                base_metadata.update({
                    "query_type_pred": qres.dataset_label,
                    "query_type_canonical": qres.canonical_type,
                    "chunk_size_used": getattr(splitter, "_chunk_size", None),
                    "chunk_overlap_used": getattr(splitter, "_chunk_overlap", None),
                })

                split_docs = splitter.create_documents(
                    texts=[text],
                    metadatas=[base_metadata]
                )

                for i, sd in enumerate(split_docs):
                    sd.metadata["chunk_index"] = i
                    chunks.append({
                        "page_content": sd.page_content,
                        "metadata": sd.metadata
                    })

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
                docs_path
            )

            _retriever_cache[key] = AdvancedHybridRetriever(
                base_retriever,
                kg_path="My_RAG/kg_output.json"
            )

        return _retriever_cache[key], qres, key

    # 3) Process queries
    for query in tqdm(queries, desc="Processing Queries"):
        query_text = query["query"]["content"]
        query_id = query["query"].get("query_id")

        if "prediction" not in query or not isinstance(query["prediction"], dict):
            query["prediction"] = {}

        retriever, qres, key = get_retriever_for_query(query_text)

        retrieved_chunks, retrieval_debug = retriever.retrieve(
            query_text,
            top_k=top_k,
            query_id=query_id
        )

        if debug_retrieval:
            print(f"\n[QueryType] {qres.dataset_label} (canonical={qres.canonical_type}) key={key}")
            print(f"[Retrieval Debug] Query ID {query_id}: {query_text}")
            print(
                f"  Language: {retrieval_debug.get('language')} | "
                f"top_k: {retrieval_debug.get('top_k')} | "
                f"candidates considered: {retrieval_debug.get('candidate_count')}"
            )

            if retrieval_debug.get("kg_boost", 0) > 0:
                kg_info = retrieval_debug.get("kg_info") or {}
                print(
                    f"  KG boost={retrieval_debug.get('kg_boost')} | "
                    f"entities={len(kg_info.get('entities_found', []))}"
                )

            for idx, result in enumerate(retrieval_debug.get("results", []), start=1):
                meta = result.get("metadata", {})
                preview = (result.get("preview", "") or "").replace("\n", " ")[:160]
                score = result.get("score", 0.0)
                print(f"    #{idx} score={score:.4f} meta={meta} preview={preview}")

        # ===== Generate =====
        gen_k = 3
        answer = generate_answer(query_text, retrieved_chunks[:gen_k], language, kg_retriever=kg_retriever)


        query["prediction"]["content"] = answer

        query["prediction"]["references"] = [
            ch.get("page_content", "") for ch in retrieved_chunks[:gen_k]
        ]

        query["prediction"]["reference_doc_ids"] = []
        for ch in retrieved_chunks[:gen_k]:
            doc_id = (ch.get("metadata") or {}).get("doc_id")
            if doc_id is not None:
                query["prediction"]["reference_doc_ids"].append(doc_id)

        seen = set()
        dedup = []
        for d in query["prediction"]["reference_doc_ids"]:
            if d not in seen:
                seen.add(d)
                dedup.append(d)
        query["prediction"]["reference_doc_ids"] = dedup

        query["prediction"]["query_type_pred"] = qres.dataset_label
        query["prediction"]["query_type_canonical"] = qres.canonical_type

    save_jsonl(output_path, queries)
    print(f"Predictions saved at '{output_path}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query_path")
    parser.add_argument("--docs_path")
    parser.add_argument("--language")
    parser.add_argument("--output")
    args = parser.parse_args()

    main(args.query_path, args.docs_path, args.language, args.output)
