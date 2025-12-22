from typing import List, Dict, Any, Optional
from ollama import Client
from config import load_config


def load_ollama_config() -> dict:
    config = load_config()
    assert "ollama" in config, "Ollama configuration not found in config file."
    assert "host" in config["ollama"], "Ollama host not specified in config file."
    assert "model" in config["ollama"], "Ollama model not specified in config file."
    return config["ollama"]


def _generate_with_ollama(prompt: str) -> str:
    cfg = load_ollama_config()
    client = Client(host=cfg["host"])
    response = client.generate(
        model=cfg["model"],
        prompt=prompt,
        stream=False,
        options={
            "temperature": 0.1,
            "num_ctx": 8192,
        },
    )
    return (response.get("response", "") or "").strip()


# ========= KG ONLY (EN): decide whether to use KG =========
def _should_use_kg_en(context_chunks: List[Dict[str, Any]]) -> bool:
    """
    Minimal rule: if retrieved context is empty or too short, use KG (EN only).
    """
    if not context_chunks:
        return True
    if len(context_chunks) <= 1:
        return True
    return False


# ========= KG ONLY (EN): ToG reasoning + build KG context =========
def _build_tog_context_en(
    query: str,
    kg_retriever,
    top_k: int = 10,
    max_docs_for_prompt: int = 5,
    max_paths_per_doc: int = 3,
    max_triples_per_doc: int = 3,
) -> str:
    """
    Use kg_retriever.rank_docs for ToG reasoning and format as English-only context.
    """
    ranked_docs = kg_retriever.rank_docs(
        query=query,
        top_k=top_k,
        use_multi_hop=True,
        return_debug=True,
    )

    if not ranked_docs:
        return ""

    parts: List[str] = []
    for i, doc_result in enumerate(ranked_docs[:max_docs_for_prompt], start=1):
        doc_id = doc_result.get("doc_id")
        score = doc_result.get("score", 0.0)
        paths = doc_result.get("paths", []) or []
        evidence = doc_result.get("evidence_triples", []) or []

        parts.append(f"### ToG Reasoning {i} (doc_id={doc_id}, score={score:.2f})")

        if paths:
            parts.append("Reasoning paths:")
            for p in paths[:max_paths_per_doc]:
                parts.append(f"- {p}")

        if evidence:
            parts.append("Evidence triples:")
            for t in evidence[:max_triples_per_doc]:
                h = t.get("head", "")
                r = t.get("relation", "")
                ta = t.get("tail", "")
                parts.append(f"- {h} --[{r}]--> {ta}")

        parts.append("")  # blank line

    return "\n".join(parts).strip()


def generate_answer(
    query: str,
    context_chunks: List[Dict[str, Any]],
    language: str = "en",
    kg_retriever: Optional[object] = None,
) -> str:
    """
    Backward compatible:
      - Old call: generate_answer(query, context_chunks) still works.
    KG behavior:
      - KG is used ONLY when language is English (NOT zh/zh-*).
      - If KG is used, we run ToG reasoning and answer based ONLY on ToG results.
    """
    # ----- normal retrieved context (original behavior) -----
    context = "\n\n".join([ch.get("page_content", "") for ch in (context_chunks or [])]).strip()

    # ----- KG path (EN ONLY) -----
    is_english = not (language or "").lower().startswith("zh")
    if is_english and kg_retriever is not None and _should_use_kg_en(context_chunks):
        try:
            tog_context = _build_tog_context_en(query, kg_retriever)
        except Exception:
            tog_context = ""

        if tog_context:
            prompt = f"""You are a Q&A system based on knowledge-graph reasoning (ToG).
Answer using ONLY the ToG reasoning results below. If insufficient, say exactly: "Unable to answer."

Question:
{query}

ToG reasoning results:
{tog_context}

Answer in English:"""
            return _generate_with_ollama(prompt)

    # ----- fallback (always available) -----
    prompt = f"""You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say that you don't know. \
Use three sentences maximum and keep the answer concise.\n\nQuestion: {query} \nContext: {context} \nAnswer:\n"""
    return _generate_with_ollama(prompt)


if __name__ == "__main__":
    # test
    query = "What is the capital of France?"
    context_chunks = [
        {"page_content": "France is a country in Europe. Its capital is Paris."},
        {"page_content": "The Eiffel Tower is located in Paris, the capital city of France."}
    ]
    print(generate_answer(query, context_chunks))  # no KG
