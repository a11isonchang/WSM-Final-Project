from tqdm import tqdm
from utils import load_jsonl, save_jsonl
from chunker import chunk_documents
from retriever import create_retriever, analysis_retriever_result, kg_retriever
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

    # Extract top_k based on language (can be a dict or a number)
    top_k_config = retrieval_config.get("top_k", 3)
    if isinstance(top_k_config, dict):
        if language and language.startswith("zh"):
            top_k = top_k_config.get("zh", 3)
        else:
            top_k = top_k_config.get("en", top_k_config.get(language, 3))
    else:
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
    retriever = create_retriever(chunks, language, retrieval_config)
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

            for idx, result in enumerate(retrieval_debug["results"], start=1):
                meta = result.get("metadata", {})
                preview = result.get("preview", "").replace("\n", " ")[:160]
                score = result.get("score", 0.0)
                print(f"    #{idx} score={score:.4f} meta={meta} preview={preview}")

        # 5. 用analysis_retriever_result檢查retrieve結果
        analysis_result = analysis_retriever_result(
            query=query_text,
            context_chunks=retrieved_chunks,
            language=language
        )
        
        # 如果檢索結果不夠充分，使用KG檢索補充
        if analysis_result == "use_kg":
            kg_chunks = kg_retriever(
                query=query_text,
                language=language,
                all_chunks=chunks,
                top_k=top_k
            )
            
            # 合併原始檢索結果和KG檢索結果，去重（基於page_content）
            # 重要：將KG檢索的chunks插入到前面，確保它們會被使用
            if kg_chunks:
                seen_content = {chunk.get("page_content", "") for chunk in retrieved_chunks}
                new_kg_chunks = []
                for kg_chunk in kg_chunks:
                    kg_content = kg_chunk.get("page_content", "")
                    if kg_content and kg_content not in seen_content:
                        new_kg_chunks.append(kg_chunk)
                        seen_content.add(kg_content)
                
                # 將KG檢索的chunks插入到前面，優先使用
                if new_kg_chunks:
                    retrieved_chunks = new_kg_chunks + retrieved_chunks
                    if debug_retrieval:
                        print(f"  [KG Retrieval] Added {len(new_kg_chunks)} chunks from knowledge graph (inserted at front)")
        
        if debug_retrieval:
            print(f"  [Analysis Result] {analysis_result} | Final chunks: {len(retrieved_chunks)}")

        # 6. Generate Answer
        # Use top 3 chunks for generation to provide better context
        answer = generate_answer(query_text, retrieved_chunks[:3], language)

        query["prediction"]["content"] = answer
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
