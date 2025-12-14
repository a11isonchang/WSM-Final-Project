from tqdm import tqdm
from utils import load_jsonl, save_jsonl
from chunker import chunk_documents
from retriever import create_retriever
from generator import generate_answer
from config import load_config
import argparse


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

    # ✅ NEW：從 chunking 設定取值
    chunk_cfg = get_chunk_config(config, language)
    chunk_size = chunk_cfg["chunk_size"]
    chunk_overlap = chunk_cfg["chunk_overlap"]

    # 获取top_k（支持按语言配置）
    top_k_config = retrieval_config.get("top_k", 3)
    if isinstance(top_k_config, dict):
        # 如果top_k是字典，根据语言获取对应的值
        top_k = top_k_config.get(language, top_k_config.get("en", 3))
    else:
        # 如果top_k是数字，直接使用
        top_k = top_k_config
    
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
    print("Creating retriever...")
    retriever = create_retriever(chunks, language, retrieval_config, docs_path)
    print("Retriever created successfully.")

    # 4. Process Queries
    for query in tqdm(queries, desc="Processing Queries"):
        query_text = query["query"]["content"]
        query_id = query["query"].get("query_id")

        # Retrieve relevant chunks
        retrieved_chunks, retrieval_debug = retriever.retrieve(query_text, top_k=top_k, query_id=query_id)
        
        if debug_retrieval:
            print(f"\n[Retrieval Debug] Query ID {query_id}: {query_text}")
            print(
                f"  Language: {retrieval_debug['language']} | "
                f"top_k: {retrieval_debug['top_k']} | "
                f"candidates considered: {retrieval_debug['candidate_count']}"
            )

            if retrieval_debug.get("keyword_info"):
                print(
                    f"  Keywords: {retrieval_debug['keyword_info']['keywords']} "
                    f"(boost={retrieval_debug['keyword_info']['boost']})"
                )
            
            # 显示知识图谱检索信息（包括多跳关系）
            if retrieval_debug.get("kg_boost", 0) > 0:
                kg_info = retrieval_debug.get("kg_info")
                if kg_info:
                    entities = kg_info.get("entities_found", [])
                    doc_ids = kg_info.get("related_doc_ids", [])
                    multi_hop = kg_info.get("multi_hop", {})
                    
                    print(
                        f"  KG: Found {len(entities)} entities, "
                        f"{len(doc_ids)} related docs (boost={retrieval_debug['kg_boost']})"
                    )
                    if entities:
                        entity_names = [e.get("name", "") for e in entities[:3]]
                        print(f"    Entities: {', '.join(entity_names)}")
                    
                    # 显示多跳关系信息
                    if multi_hop:
                        max_hops = multi_hop.get("max_hops", 0)
                        multi_hop_count = multi_hop.get("multi_hop_entities", 0)
                        if multi_hop_count > 0:
                            print(
                                f"    Multi-hop: {multi_hop_count} entities found "
                                f"through {max_hops}-hop relations"
                            )

            for idx, result in enumerate(retrieval_debug["results"], start=1):
                meta = result.get("metadata", {})
                preview = result.get("preview", "").replace("\n", " ")[:160]
                score = result.get("score", 0.0)
                print(f"    #{idx} score={score:.4f} meta={meta} preview={preview}")

        # 5. Generate Answer
        # Use top 3 chunks for generation to provide better context
        # Pass kg_retriever for ToG fallback when evidence is insufficient
        kg_retriever = getattr(retriever, 'kg_retriever', None)
        answer = generate_answer(query_text, retrieved_chunks[:3], language, kg_retriever=kg_retriever)

        query["prediction"]["content"] = answer

        # ✅ NEW: output references like ground_truth.references (doc_id list)
        refs = []
        for ch in retrieved_chunks[:3]:
            doc_id = (ch.get("metadata") or {}).get("doc_id")
            if doc_id is not None:
                refs.append(int(doc_id))

        # de-dup while preserving order
        seen = set()
        refs_unique = []
        for r in refs:
            if r in seen:
                continue
            seen.add(r)
            refs_unique.append(r)

        query["prediction"]["references"] = refs_unique
        # Save top 3 chunks as references for evaluation
        query["prediction"]["references"] = (
            [chunk["page_content"] for chunk in retrieved_chunks[:3]] if retrieved_chunks else []
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