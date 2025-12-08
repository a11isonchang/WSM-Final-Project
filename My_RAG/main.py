from tqdm import tqdm
from utils import load_jsonl, save_jsonl
from chunker import chunk_documents
from retriever import create_retriever
from generator import generate_answer
from config import load_config
import argparse

def main(query_path, docs_path, language, output_path):
    config = load_config()
    retrieval_config = config.get("retrieval", {})
    chunk_size = retrieval_config.get("chunk_size", 1000)
    chunk_overlap = retrieval_config.get("chunk_overlap", 200)
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
    chunks = chunk_documents(
        docs_for_chunking,
        language,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    print(f"Created {len(chunks)} chunks.")

    # 3. Create Retriever
    print("Creating retriever...")
    retriever = create_retriever(chunks, language, retrieval_config)
    print("Retriever created successfully.")


    for query in tqdm(queries, desc="Processing Queries"):
        # 4. Retrieve relevant chunks
        query_text = query['query']['content']
        # print(f"\nRetrieving chunks for query: '{query_text}'")
        retrieved_chunks, retrieval_debug = retriever.retrieve(query_text, top_k=top_k)
        if debug_retrieval:
            print(f"\n[Retrieval Debug] Query ID {query['query']['query_id']}: {query_text}")
            print(f"  Language: {retrieval_debug['language']} | top_k: {retrieval_debug['top_k']} | "
                  f"candidates considered: {retrieval_debug['candidate_count']}")
            if retrieval_debug["keyword_info"]:
                print(f"  Keywords: {retrieval_debug['keyword_info']['keywords']} "
                      f"(boost={retrieval_debug['keyword_info']['boost']})")
            for idx, result in enumerate(retrieval_debug["results"], start=1):
                meta = result.get("metadata", {})
                preview = result.get("preview", "").replace("\n", " ")[:160]
                score = result.get("score", 0.0)
                print(f"    #{idx} score={score:.4f} meta={meta} preview={preview}")
        # print(f"Retrieved {len(retrieved_chunks)} chunks.")

        # 5. Generate Answer
        # print("Generating answer...")
        answer = generate_answer(query_text, retrieved_chunks, language)

        query["prediction"]["content"] = answer
        query["prediction"]["references"] = [retrieved_chunks[0]['page_content']] if retrieved_chunks else []

    save_jsonl(output_path, queries)
    print("Predictions saved at '{}'".format(output_path))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--query_path', help='Path to the query file')
    parser.add_argument('--docs_path', help='Path to the documents file')
    parser.add_argument('--language', help='Language to filter queries (zh or en), if not specified, process all')
    parser.add_argument('--output', help='Path to the output file')
    args = parser.parse_args()
    main(args.query_path, args.docs_path, args.language, args.output)
