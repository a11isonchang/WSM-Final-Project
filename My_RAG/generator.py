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

Your task is to answer the question **strictly based on the Retrieved Context**.

Before answering, detect the language of the user's question:
- If the question is in **Chinese**, use the **Chinese Answering Mode (strict & concise)**.
- If the question is in **English**, use the **English Answering Mode (CoT-light)**.

====================================================
# Chinese Answering Mode (for Chinese questions)

Rules:
1. **只使用 Retrieved Context 中明確提到的內容。**
2. **禁止使用外部知識、背景知識、世界知識。**
3. **禁止推論、猜測、補全、模糊寫作。**
4. **回答必須簡短貼題，不能加入無關資訊。**
5. **最多二到三句，越短越好。**
6. **不要重述題目，不要加解釋，不要加背景，不要舉例。**
7. **如果資料缺少必須資訊，請回答：**
   「Insufficient information in the retrieved documents.」
8. **如果資料彼此矛盾，只需簡短指出矛盾點，不要自行推測。**

Chinese style priorities:
- Factuality ＞ Completeness ＞ ROUGE ＞ Length
- 寧可回答較短，也不要加入任何猜測或廢話。

====================================================
# English Answering Mode (for English questions)

Rules:
1. **Use ONLY information explicitly stated in the Retrieved Context.**
2. **No external knowledge, no speculation, no assumptions.**
3. **Provide a direct answer to the question; do NOT restate the question.**
4. **Use at most 3 sentences, but they may be slightly more informative than the Chinese mode.**
5. **Be concise, factual, and focused on the asked information.**
6. **If required information is missing, reply exactly:**
   "Insufficient information in the retrieved documents."
7. **If context contains contradictions, briefly state the conflict.**

English style priorities:
- Completeness ＞ Factuality ＞ ROUGE ＞ Brevity
- Prefer a clean, grounded short answer over speculative elaboration.

====================================================
# Hidden Chain-of-Thought Instructions (Do NOT show)

- Internally analyze the retrieved context step-by-step.
- Extract only explicit facts.
- Detect missing info, contradictions, or ambiguity.
- Decide whether an answer can be strictly grounded.
- In the final output, ONLY output the final answer—never the reasoning.

====================================================

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
