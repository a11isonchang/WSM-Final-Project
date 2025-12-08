from ollama import Client
from config import load_config


def load_ollama_config() -> dict:
    config = load_config()
    assert "ollama" in config, "Ollama configuration not found in config file."
    assert "host" in config["ollama"], "Ollama host not specified in config file."
    assert "model" in config["ollama"], "Ollama model not specified in config file."
    return config["ollama"]

def _create_prompt_en(query: str, context: str) -> str:
    """
    Optimized prompt for complex RAG tasks including:
    - Multi-hop reasoning
    - Cross-document comparison (dates, values)
    - Summarization
    - Strict unanswerable handling
    """
    return f"""You are an advanced analytical assistant for a RAG system. Your task is to synthesize information from the provided context to answer the user's question accurately and concisely.

**STRICT OPERATIONAL RULES:**

1. **NO OUTSIDE KNOWLEDGE:** Answer ONLY using the information within the <context> tags.
2. **HANDLING MISSING INFO:** - If the context does NOT contain sufficient information to answer the specific question, you MUST reply exactly: "Unable to answer". 
   - Do not attempt to summarize unrelated parts of the document just to provide an answer. (e.g., if asked about environmental measures but the text is about governance, say "Unable to answer").
3. **LOGICAL COMPARISONS:**
   - If asked to **compare** (e.g., "earlier," "higher," "better"), explicitly extract the values/dates for BOTH entities from the context first, then perform the comparison.
4. **MULTI-HOP SYNTHESIS:**
   - If the answer requires combining facts from different sentences or document sections (e.g., "How did A and B contribute to C?"), explicitly link the cause and effect.
5. **SUMMARIZATION:**
   - Capture all key points (dates, specific symptoms, event sequences) mentioned in the query scope.

**RETRIEVED CONTEXT:**
<context>
{context}
</context>

**USER QUESTION:**
{query}

**ANSWER:**
"""


def _create_prompt_zh(query: str, context: str) -> str:
    """Create optimized Chinese prompt"""
    return f"""你是一個具備邏輯推理能力的專業 RAG 助手。請根據下方提供的【檢索上下文】回答用戶的【問題】。

**核心指令：**

1. **嚴格基於上下文：** - 答案必須完全源自【檢索上下文】，嚴禁使用外部知識或自行編造。
   - 若上下文中沒有足夠資訊回答問題（包含部分缺失），請直接回答：「無法回答」。不要嘗試強行總結無關內容。

2. **邏輯比較與推理 (關鍵)：**
   - **數值/時間比較：** 若問題涉及「比較」（如：誰更早、誰金額更高），請先在腦中提取雙方的具體數值或日期，再進行比較。注意中文單位的換算（如：億 vs 萬）。
   - **多文檔整合：** 若答案散落在不同段落（如：A公司在文檔1，B公司在文檔2），請分別提取資訊後再合併回答，並清楚標示主體。

3. **回答風格：**
   - 保持客觀、簡潔。
   - 針對「總結類」問題，請涵蓋所有關鍵時間點與事件（Who, When, What）。

**檢索上下文：**
<context>
{context}
</context>

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
        options={"temperature": 0.1}  # improves factual stability
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
