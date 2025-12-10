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
            
            # 显示知识图谱检索信息
            if retrieval_debug.get("kg_boost", 0) > 0:
                kg_info = retrieval_debug.get("kg_info")
                if kg_info:
                    entities = kg_info.get("entities_found", [])
                    doc_ids = kg_info.get("related_doc_ids", [])
                    print(
                        f"  KG: Found {len(entities)} entities, "
                        f"{len(doc_ids)} related docs (boost={retrieval_debug['kg_boost']})"
                    )
                    if entities:
                        entity_names = [e.get("name", "") for e in entities[:3]]
                        print(f"    Entities: {', '.join(entity_names)}")

            for idx, result in enumerate(retrieval_debug["results"], start=1):
                meta = result.get("metadata", {})
                preview = result.get("preview", "").replace("\n", " ")[:160]
                score = result.get("score", 0.0)
                print(f"    #{idx} score={score:.4f} meta={meta} preview={preview}")

        # 5. Generate Answer
        # 5. Generate Answer（加入「相似度 gate / unsolvable」邏輯）
        is_unsolvable = retrieval_debug.get("unsolvable", False)
        have_context = bool(retrieved_chunks)

        if is_unsolvable or not have_context:
            # 🔴 Gate 擋掉：完全不給 context，直接回固定答案
            if language and language.startswith("zh"):
                answer = "无法回答。"
            else:
                answer = "Unable to answer."

            # 沒有用到任何文件，references 也留空
            query["prediction"]["references"] = []
        else:
            # ✅ 正常流程：只用前 3 個 chunk 當 context 給 LLM
            top_chunks = retrieved_chunks[:3]
            answer = generate_answer(query_text, top_chunks, language)

            # 把用到的 chunk 內容當作 reference 存起來（評分用）
            query["prediction"]["references"] = [
                chunk["page_content"] for chunk in top_chunks
            ]

        query["prediction"]["content"] = answer

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
