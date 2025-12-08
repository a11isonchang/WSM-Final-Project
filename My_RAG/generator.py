from ollama import Client
from config import load_config


def load_ollama_config() -> dict:
    config = load_config()
    assert "ollama" in config, "Ollama configuration not found in config file."
    assert "host" in config["ollama"], "Ollama host not specified in config file."
    assert "model" in config["ollama"], "Ollama model not specified in config file."
    return config["ollama"]

def _create_prompt_en(query: str, context: str) -> str:
    """Create optimized English prompt"""
    return f"""You are an expert assistant for a Retrieval-Augmented Generation (RAG) system. Your task is to provide accurate, concise answers based strictly on the provided context.

**CRITICAL RULES:**
1. Base your answer ONLY on the Retrieved Context below
2. Do NOT use external knowledge, assumptions, or inferences
3. Do NOT make comparisons unless ALL necessary values are explicitly stated
4. If the context does NOT contain the answer, reply: "Insufficient information in the retrieved documents."
5. Synthesize information across passages when they complement each other
6. If passages conflict, acknowledge the conflict briefly
7. Be concise, factual, and comprehensive in covering the key points requested.

**Retrieved Context:**
{context}

**Question:**
{query}

**Answer:**
"""


def _create_prompt_zh(query: str, context: str) -> str:
    """Create optimized Chinese prompt"""
    return f"""你是一個專業的檢索增強生成（RAG）系統助手。你的任務是根據提供的上下文提供準確、簡潔的答案。

**重要規則：**
1. 答案必須嚴格基於下方的檢索上下文
2. 不可使用外部知識、假設或推測
3. 除非所有必要的數值都明確存在，否則不要進行比較
4. 如果上下文缺少足夠的資訊，請準確回答：
   "檢索文檔中資訊不足。"
5. 當多個段落互補時，請綜合資訊
6. 如果段落之間存在衝突，請簡要說明衝突
7. 保持答案簡潔、事實準確，並全面涵蓋所需的關鍵點。

**檢索上下文：**
{context}

**問題：**
{query}

**回答：**
"""


def generate_answer(query, context_chunks, language="en"):
    context = "\n\n".join([chunk['page_content'] for chunk in context_chunks])

    if language == "zh":
            prompt = _create_prompt_zh(query, context)
    else:
        prompt = _create_prompt_en(query, context)

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
