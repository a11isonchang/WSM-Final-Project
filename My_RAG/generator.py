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

    if language == "zh":
        prompt = f"""你是一個專業的問答系統。請根據以下「檢索到的文件」回答問題。

規則：
1. 只使用檢索到的文件中明確提到的內容
2. 禁止使用外部知識或推測
3. 回答要簡短精確（最多2-3句）
4. 如果資料不足，回答：「Insufficient information in the retrieved documents.」

檢索到的文件：
{context}

問題：
{query}

回答："""
    else:  # English
        prompt = f"""You are a professional question-answering system. Answer the question based ONLY on the Retrieved Documents below.

Rules:
1. Use ONLY information explicitly stated in the documents
2. No external knowledge or speculation
3. Be concise (max 3 sentences)
4. If information is missing, reply: "Insufficient information in the retrieved documents."

Retrieved Documents:
{context}

Question:
{query}

Answer:"""

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
