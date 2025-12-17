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

from chunker import get_query_aware_text_splitter

from advanced_retriever import AdvancedHybridRetriever 
from subq_engine import try_subquestion_retrieve

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

    # cache chunks & retrievers by policy key
    # key = (language, canonical_type, chunk_size_used, chunk_overlap_used)
    _chunks_cache = {}
    _retriever_cache = {}

    def build_chunks_for_key(splitter, qres):
        """
        Build chunks for this splitter policy (once), then cache.
        """
        chunks = []
        for doc in docs_for_chunking:
            if (
                isinstance(doc, dict)
                and isinstance(doc.get("content"), str)
                and doc.get("language") == language
            ):
                text = doc["content"]
                base_metadata = {k: v for k, v in doc.items() if k != "content"}

                # policy metadata for debug
                base_metadata.update({
                    "query_type_pred": qres.dataset_label,
                    "query_type_canonical": qres.canonical_type,
                    "chunk_size_used": getattr(splitter, "_chunk_size", None),
                    "chunk_overlap_used": getattr(splitter, "_chunk_overlap", None),
                })

                split_docs = splitter.create_documents(texts=[text], metadatas=[base_metadata])
                for i, sd in enumerate(split_docs):
                    sd.metadata["chunk_index"] = i
                    chunks.append({"page_content": sd.page_content, "metadata": sd.metadata})

        return chunks

    def get_retriever_for_query(query_text: str):
        """
        For each query:
        1) infer query type → get splitter & qres
        2) cache chunks by policy
        3) cache retriever by policy
        """
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
        '''
        if key not in _retriever_cache:
            #  retriever 綁定同一組 chunks
            _retriever_cache[key] = create_retriever(
                _chunks_cache[key],
                language,
                retrieval_config,
                docs_path
            )
        '''
        if key not in _retriever_cache:
            # 1. 先建立原本的基礎檢索器 (BM25/Vector)
            base_retriever = create_retriever(
                _chunks_cache[key],
                language,
                retrieval_config,
                docs_path
            )
            
            # 2. 用 AdvancedHybridRetriever 包裝它 (加入 KG 加權 + Cross-Encoder 重排序)
            # 這會無縫接管後面的 .retrieve() 呼叫
            _retriever_cache[key] = AdvancedHybridRetriever(
                base_retriever,
                kg_path="My_RAG/kg_output.json"  # 確保你的 KG 檔案路徑正確
            )


        return _retriever_cache[key], qres, key

    # 3) Process queries
    for query in tqdm(queries, desc="Processing Queries"):
        query_text = query["query"]["content"]
        query_id = query["query"].get("query_id")

        #  ensure prediction exists
        if "prediction" not in query or not isinstance(query["prediction"], dict):
            query["prediction"] = {}

        #  get policy-specific retriever
        retriever, qres, key = get_retriever_for_query(query_text)

        # Retrieve
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

            if retrieval_debug.get("keyword_info"):
                ki = retrieval_debug["keyword_info"]
                print(f"  Keywords: {ki.get('keywords')} (boost={ki.get('boost')})")

            if retrieval_debug.get("kg_boost", 0) > 0:
                kg_info = retrieval_debug.get("kg_info") or {}
                entities = kg_info.get("entities_found", [])
                doc_ids = kg_info.get("related_doc_ids", [])
                multi_hop = kg_info.get("multi_hop", {}) or {}
                print(
                    f"  KG: Found {len(entities)} entities, "
                    f"{len(doc_ids)} related docs (boost={retrieval_debug.get('kg_boost')})"
                )
                if multi_hop:
                    mh_cnt = multi_hop.get("multi_hop_entities", 0)
                    mh_hops = multi_hop.get("max_hops", 0)
                    if mh_cnt:
                        print(f"    Multi-hop: {mh_cnt} entities through {mh_hops}-hop relations")

            for idx, result in enumerate(retrieval_debug.get("results", []), start=1):
                meta = result.get("metadata", {})
                preview = (result.get("preview", "") or "").replace("\n", " ")[:160]
                score = result.get("score", 0.0)
                print(f"    #{idx} score={score:.4f} meta={meta} preview={preview}")

        # --- SubQuestionQueryEngine (pseudo evidence) ---
        # 只在「多跳」或「證據偏薄」時啟用，避免每題都跑導致變慢或不穩
        ctype = (qres.canonical_type or "").lower()
        is_multi_hop = bool(getattr(qres, "is_multi_hop", False)) or any(k in ctype for k in ["multi", "hop", "compare", "caus", "reason"])

        evidence_thin = (len(retrieved_chunks) < top_k)

        # 你也可以更嚴格：top1/2 chunk 太短時算薄（可選）
        # evidence_thin = evidence_thin or (retrieved_chunks and len(retrieved_chunks[0].get("page_content","")) < 120)

        use_subq = is_multi_hop or evidence_thin

        if use_subq:
            ollama_cfg = config.get("ollama", {}) or {}
            pseudo = try_subquestion_retrieve(
                query=query_text,
                chunks=retrieved_chunks,          # 用「已檢索到的 chunks」做子問題分解
                language=language,
                top_k=top_k,
                ollama_host=ollama_cfg.get("host"),
                llm_model=ollama_cfg.get("model"),
            )

            if pseudo:
                # 把 pseudo evidence 放在前面（讓它更容易被你後面 [:3] 吃到）
                retrieved_chunks = pseudo + retrieved_chunks

                if debug_retrieval:
                    print(f"[SubQ] enabled=True | multi_hop={is_multi_hop} thin={evidence_thin} | pseudo_chunks={len(pseudo)}")
            else:
                if debug_retrieval:
                    print(f"[SubQ] enabled=True but pseudo=None | multi_hop={is_multi_hop} thin={evidence_thin}")


        # Generate
        gen_k = 3
        kg_retriever = getattr(retriever, "kg_retriever", None)
        answer = generate_answer(query_text, retrieved_chunks[:gen_k], language, kg_retriever=kg_retriever)
        query["prediction"]["content"] = answer

        #  references: 存 top-3 chunks 的文本
        query["prediction"]["references"] = (
            [ch.get("page_content", "") for ch in retrieved_chunks[:gen_k]]
            if retrieved_chunks else []
        )

        # 存 doc_id 方便 debug
        query["prediction"]["reference_doc_ids"] = []
        for ch in retrieved_chunks[:gen_k]:
            doc_id = (ch.get("metadata") or {}).get("doc_id")
            if doc_id is not None:
                query["prediction"]["reference_doc_ids"].append(doc_id)

        # 去重 doc_id 並保序
        seen = set()
        dedup = []
        for d in query["prediction"]["reference_doc_ids"]:
            if d in seen:
                continue
            seen.add(d)
            dedup.append(d)
        query["prediction"]["reference_doc_ids"] = dedup

        # predicted query type（分析用）
        query["prediction"]["query_type_pred"] = qres.dataset_label
        query["prediction"]["query_type_canonical"] = qres.canonical_type


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
