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
        retrieved_chunks = retriever.retrieve(query_text, top_k=top_k)
        # print(f"Retrieved {len(retrieved_chunks)} chunks.")

        # 5. Generate Answer
        # print("Generating answer...")
        answer = generate_answer(query_text, retrieved_chunks)

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
