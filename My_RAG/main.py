from tqdm import tqdm
from utils import load_jsonl, save_jsonl
from chunker import chunk_documents
from retriever import create_retriever
from generator import generate_answer
from config import load_config
import argparse
import sys
from typing import Optional


def main(query_path: str, docs_path: str, language: str, output_path: str) -> None:
    """
    Main RAG pipeline with improved error handling and reference management.
    
    Args:
        query_path: Path to queries JSONL file
        docs_path: Path to documents JSONL file
        language: Target language ('en' or 'zh')
        output_path: Path to save predictions
    """
    try:
        # Load configuration
        config = load_config()
        retrieval_config = config.get("retrieval", {})
        
        # Inject ollama config into retrieval config for query expansion
        if "ollama" in config:
            retrieval_config["ollama"] = config["ollama"]
            
        chunk_size = retrieval_config.get("chunk_size", 1000)
        chunk_overlap = retrieval_config.get("chunk_overlap", 200)
        top_k = retrieval_config.get("top_k", 3)
        debug_retrieval = retrieval_config.get("debug", False)
        
        print(f"\n{'='*60}")
        print(f"RAG Pipeline Configuration")
        print(f"{'='*60}")
        print(f"Language: {language}")
        print(f"Chunk Size: {chunk_size}, Overlap: {chunk_overlap}")
        print(f"Top-K: {top_k}")
        print(f"{'='*60}\n")

        # 1. Load Data
        print("📂 Loading documents...")
        try:
            docs_for_chunking = load_jsonl(docs_path)
            queries = load_jsonl(query_path)
        except FileNotFoundError as e:
            print(f"❌ Error: File not found - {e}")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            sys.exit(1)
        
        print(f"✓ Loaded {len(docs_for_chunking)} documents")
        print(f"✓ Loaded {len(queries)} queries\n")

        # 2. Chunk Documents
        print("✂️  Chunking documents with semantic boundaries...")
        try:
            chunks = chunk_documents(
                docs_for_chunking,
                language,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        except Exception as e:
            print(f"❌ Error during chunking: {e}")
            sys.exit(1)
        
        if not chunks:
            print(f"❌ Error: No chunks created for language '{language}'")
            sys.exit(1)
        
        print(f"✓ Created {len(chunks)} semantic chunks\n")

        # 3. Create Retriever
        print("🔍 Creating hybrid retriever...")
        try:
            retriever = create_retriever(chunks, language, retrieval_config)
        except Exception as e:
            print(f"❌ Error creating retriever: {e}")
            sys.exit(1)
        
        print("✓ Retriever created successfully\n")

        # 4. Process Queries
        print("🤖 Processing queries...\n")
        successful = 0
        failed = 0
        
        for query_obj in tqdm(queries, desc="Processing Queries"):
            try:
                # Extract query text
                query_text = query_obj.get('query', {}).get('content', '')
                if not query_text:
                    print(f"⚠️  Warning: Empty query for ID {query_obj.get('query', {}).get('query_id')}")
                    query_obj["prediction"]["content"] = ""
                    query_obj["prediction"]["references"] = []
                    failed += 1
                    continue
                
                # 4a. Retrieve relevant chunks
                try:
                    retrieved_chunks, retrieval_debug = retriever.retrieve(query_text, top_k=top_k)
                except Exception as e:
                    print(f"⚠️  Retrieval error for query '{query_text[:50]}...': {e}")
                    retrieved_chunks = []
                    retrieval_debug = {}
                
                # Debug output
                if debug_retrieval and retrieval_debug:
                    _print_retrieval_debug(query_obj, query_text, retrieval_debug)
                
                # 4b. Generate Answer
                try:
                    answer = generate_answer(query_text, retrieved_chunks, language)
                except Exception as e:
                    print(f"⚠️  Generation error for query '{query_text[:50]}...': {e}")
                    answer = "Error generating answer." if language == "en" else "生成答案時發生錯誤。"
                
                # 4c. Update predictions with ALL retrieved chunks as references
                # Ensure prediction field exists
                if "prediction" not in query_obj:
                    query_obj["prediction"] = {}
                
                query_obj["prediction"]["content"] = answer
                query_obj["prediction"]["references"] = [
                    chunk['page_content'] for chunk in retrieved_chunks
                ] if retrieved_chunks else []
                
                # Note: All other original fields (ground_truth, domain, language, etc.) 
                # are automatically preserved since we only update the prediction field
                
                successful += 1
                
            except Exception as e:
                print(f"❌ Unexpected error processing query: {e}")
                # Ensure prediction field exists even on error
                if "prediction" not in query_obj:
                    query_obj["prediction"] = {}
                query_obj["prediction"]["content"] = ""
                query_obj["prediction"]["references"] = []
                # Note: ground_truth and other fields are preserved
                failed += 1
        
        # 5. Save Results
        print(f"\n💾 Saving predictions...")
        try:
            save_jsonl(output_path, queries)
        except Exception as e:
            print(f"❌ Error saving results: {e}")
            sys.exit(1)
        
        print(f"✓ Predictions saved to '{output_path}'")
        print(f"\n{'='*60}")
        print(f"Pipeline Summary")
        print(f"{'='*60}")
        print(f"Total queries: {len(queries)}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"Success rate: {successful/len(queries)*100:.1f}%")
        print(f"{'='*60}\n")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def _print_retrieval_debug(query_obj, query_text, retrieval_debug):
    """Helper function to print retrieval debug information"""
    print(f"\n{'─'*60}")
    print(f"[Debug] Query ID: {query_obj.get('query', {}).get('query_id')}")
    print(f"Query: {query_text}")
    print(f"Language: {retrieval_debug.get('language')} | Top-K: {retrieval_debug.get('top_k')} | "
          f"Candidates: {retrieval_debug.get('candidate_count')}")
    
    # Dense info
    dense_info = retrieval_debug.get("dense", {})
    if dense_info.get("enabled"):
        print(f"Dense Rerank: {dense_info.get('model')} (basis={dense_info.get('rank_basis')})")
    
    # PRF info
    prf_terms = retrieval_debug.get("prf_expanded_terms")
    if prf_terms:
        print(f"PRF Expanded Terms: {prf_terms}")
    
    # Results
    for idx, result in enumerate(retrieval_debug.get("results", []), start=1):
        meta = result.get("metadata", {})
        preview = result.get("preview", "").replace("\n", " ")[:100]
        score = result.get("score", 0.0)
        print(f"  #{idx} Score: {score:.4f}")
        print(f"      Metadata: {meta}")
        print(f"      Preview: {preview}...")
    print(f"{'─'*60}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--query_path', help='Path to the query file')
    parser.add_argument('--docs_path', help='Path to the documents file')
    parser.add_argument('--language', help='Language to filter queries (zh or en), if not specified, process all')
    parser.add_argument('--output', help='Path to the output file')
    args = parser.parse_args()
    main(args.query_path, args.docs_path, args.language, args.output)
