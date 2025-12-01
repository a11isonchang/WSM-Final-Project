from ollama import Client
from config import load_config


def load_ollama_config() -> dict:
    config = load_config()
    assert "ollama" in config, "Ollama configuration not found in config file."
    assert "host" in config["ollama"], "Ollama host not specified in config file."
    assert "model" in config["ollama"], "Ollama model not specified in config file."
    return config["ollama"]


def generate_answer(query, context_chunks, language="en"):
    context = "\n\n".join([chunk['page_content'] for chunk in context_chunks])

    prompt = f"""
You are an expert AI for a Retrieval-Augmented Generation (RAG) system.

Your task is to answer the question strictly based on the Retrieved Context.

Rules:
1. Use ONLY the information explicitly stated in the Retrieved Context.
2. Do NOT use any external knowledge.
3. Do NOT infer, assume, speculate, or complete missing information.
4. Do NOT perform comparisons unless ALL required values for comparison are explicitly present.
5. If the context does NOT contain enough information to fully and directly answer the question, reply exactly:
   "Insufficient information in the retrieved documents."
6. Use at most THREE sentences.
7. Be concise and factual.
8. If multiple passages conflict, briefly summarize the conflict.

Retrieved Context:
{context}

Question:
{query}

Answer:

"""

    ollama_config = load_ollama_config()
    client = Client(host=ollama_config["host"])
    response = client.generate(
        model=ollama_config["model"],
        prompt=prompt,
        options={"temperature": 0.2}  # improves factual stability
    )

    return response["response"].strip()


if __name__ == "__main__":
    # test the function
    query = "What is the capital of France?"
    context_chunks = [
        {"page_content": "France is a country in Europe. Its capital is Paris."},
        {"page_content": "The Eiffel Tower is located in Paris, the capital city of France."}
    ]
    answer = generate_answer(query, context_chunks)
    print("Gener：ated Answer:", answer)
