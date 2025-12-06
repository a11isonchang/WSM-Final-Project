from ollama import Client
from config import load_config
from typing import List, Dict, Any


def load_ollama_config() -> dict:
    config = load_config()
    assert "ollama" in config, "Ollama configuration not found in config file."
    assert "host" in config["ollama"], "Ollama host not specified in config file."
    assert "model" in config["ollama"], "Ollama model not specified in config file."
    return config["ollama"]


def _rerank_context_for_generation(context_chunks: List[Dict[str, Any]]) -> List[str]:
    """
    Rerank context chunks to avoid 'Lost in the Middle' problem.
    Places most relevant chunks at beginning and end.
    
    Args:
        context_chunks: List of retrieved chunks (already sorted by relevance)
    
    Returns:
        List of reranked context strings
    """
    if not context_chunks:
        return []
    
    if len(context_chunks) <= 2:
        return [chunk['page_content'] for chunk in context_chunks]
    
    # Interleave: [most relevant, least relevant, 2nd most, 2nd least, ...]
    # This puts important info at start and end where LLMs pay most attention
    reranked = []
    chunks = [chunk['page_content'] for chunk in context_chunks]
    
    left = 0
    right = len(chunks) - 1
    start = True
    
    while left <= right:
        if start:
            reranked.append(chunks[left])
            left += 1
        else:
            reranked.append(chunks[right])
            right -= 1
        start = not start
    
    return reranked


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

**答案：**
"""


def generate_answer(query: str, context_chunks: List[Dict[str, Any]], language: str = "en") -> str:
    """
    Generate answer using LLM based on retrieved context.
    
    Args:
        query: User query
        context_chunks: Retrieved context chunks (sorted by relevance)
        language: Language code ('en' or 'zh')
    
    Returns:
        Generated answer string
    """
    if not context_chunks:
        if language == "zh":
            return "未找到相關文檔。"
        return "No relevant documents found."
    
    try:
        # Rerank context to optimize LLM attention
        reranked_contexts = _rerank_context_for_generation(context_chunks)
        
        # Limit context length to avoid token limits (approximately 3000 chars per chunk max)
        max_context_chars = 8000
        context_parts = []
        current_length = 0
        
        for idx, ctx in enumerate(reranked_contexts, 1):
            ctx_with_label = f"[Passage {idx}]\n{ctx}"
            ctx_length = len(ctx_with_label)
            
            if current_length + ctx_length > max_context_chars:
                break
            
            context_parts.append(ctx_with_label)
            current_length += ctx_length
        
        context = "\n\n".join(context_parts)
        
        # Create language-specific prompt
        if language == "zh":
            prompt = _create_prompt_zh(query, context)
        else:
            prompt = _create_prompt_en(query, context)
        
        # Generate answer
        ollama_config = load_ollama_config()
        client = Client(host=ollama_config["host"])
        
        response = client.generate(
            model=ollama_config["model"],
            prompt=prompt,
            options={
                "temperature": 0.1,  # Lower for more factual responses
                "top_p": 0.9,
                "top_k": 40,
            }
        )
        
        answer = response["response"].strip()
        
        # Post-process answer
        if not answer:
            if language == "zh":
                return "無法生成答案。"
            return "Unable to generate answer."
        
        return answer
        
    except Exception as e:
        print(f"Error in generate_answer: {e}")
        if language == "zh":
            return "生成答案時發生錯誤。"
        return "Error occurred while generating answer."


if __name__ == "__main__":
    # test the function
    query = "What is the capital of France?"
    context_chunks = [
        {"page_content": "France is a country in Europe. Its capital is Paris."},
        {"page_content": "The Eiffel Tower is located in Paris, the capital city of France."}
    ]
    answer = generate_answer(query, context_chunks)
    print("Gener：ated Answer:", answer)
