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
You are an expert AI participating in a Retrieval-Augmented Generation (RAG) competition.

Your task:
- Answer the question strictly using ONLY the retrieved context.
- Do NOT use any outside knowledge.
- Do NOT make assumptions or invent facts.
- If the answer is not explicitly stated in the context, reply exactly:
  "Insufficient information in the retrieved documents."

Rules:
1. Use at most three sentences.
2. Be concise and precise.
3. If multiple passages conflict, briefly summarize the conflict.

Retrieved Context:
{context}

Question:
{query}

Final Answer:
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
