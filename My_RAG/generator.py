# generator.py

from typing import List, Dict, Any
from ollama import Client
from config import load_config


def load_ollama_config() -> dict:
    """
    讀取 config.yaml 內的 ollama 設定。
    預期結構：
    ollama:
      host: http://127.0.0.1:11434
      model: your-model-name
    """
    config = load_config()
    assert "ollama" in config, "Ollama configuration not found in config file."
    assert "host" in config["ollama"], "Ollama host not specified in config file."
    assert "model" in config["ollama"], "Ollama model not specified in config file."
    return config["ollama"]



def _is_chunk_clearly_relevant(query: str, chunk_text: str, language: str) -> bool:
    """
    使用 LLM 粗略判斷這個 chunk 是否「明顯跟問題無關」。
    - 回傳 True：看起來有幫助或可能有幫助 → 保留
    - 回傳 False：幾乎完全無關 → 可以考慮丟掉

    為了省成本：
    - 只看前一小段文字
    - 若遇到錯誤就一律當作「相關」，避免錯殺
    """
    snippet = (chunk_text or "").strip()
    if not snippet:
        return False

    if len(snippet) > 700:
        snippet = snippet[:700]

    if language == "zh":
        prompt = f"""请根据下列【问题】与【内容】，判断这段内容对回答问题有没有帮助。

问题：
{query}

内容：
{snippet}

请在下面三种标签中任选其一，并**只输出标签字母本身**：
- A：明显相关，直接有助于回答问题
- B：可能相关，里面有部分信息可能会用到
- C：几乎完全无关，对回答问题没有实际帮助

只输出 A / B / C 其中一个字母，不要输出任何其他文字。"""
    else:
        prompt = f"""You are given a user question and a candidate passage.

Question:
{query}

Passage:
{snippet}

Decide how useful this passage is for answering the question.
Choose exactly ONE label and output ONLY its letter:

- A: clearly relevant and directly useful
- B: possibly helpful or partially related
- C: almost completely irrelevant

Output only one character: A, B, or C."""

    try:
        cfg = load_ollama_config()
        client = Client(host=cfg["host"])
        resp = client.generate(
            model=cfg["model"],
            prompt=prompt,
            stream=False,
            options={
                "temperature": 0.0,
                "num_ctx": 2048,
            },
        )
        ans = (resp.get("response", "") or "").strip().upper()

        if ans.startswith("C"):
            return False
        if ans.startswith("A") or ans.startswith("B"):
            return True

        return True
    except Exception:
        return True


def _filter_chunks_by_relevance(
    query: str,
    context_chunks: List[Dict[str, Any]],
    language: str,
) -> List[Dict[str, Any]]:
    """
    使用 LLM 做一層「相關性過濾」：
    - chunk 數量很少時（例如 <= 6）才啟動，避免太貴
    - 全部都被判不相關 → fallback 回原本的 chunks（避免空集合）
    """
    if not context_chunks:
        return []

    if len(context_chunks) > 6:
        return context_chunks

    filtered = []
    for ch in context_chunks:
        text = ch.get("page_content", "")
        if not text or not text.strip():
            continue
        if _is_chunk_clearly_relevant(query, text, language):
            filtered.append(ch)

    return filtered or context_chunks


def _format_metadata(meta: Dict[str, Any]) -> str:
    """
    針對不同 domain 的欄位，做一個簡短標籤：
    - Finance: company_name
    - Law: court_name
    - Medical: hospital_patient_name
    再加上可用的 doc_id，方便模型在多段來源中對齊。
    """
    if not isinstance(meta, dict):
        return ""

    tags: List[str] = []

    domain = meta.get("domain")
    if domain:
        tags.append(f"domain={domain}")

    if "company_name" in meta and meta["company_name"]:
        tags.append(f"company={meta['company_name']}")
    if "court_name" in meta and meta["court_name"]:
        tags.append(f"court={meta['court_name']}")
    if "hospital_patient_name" in meta and meta["hospital_patient_name"]:
        tags.append(f"patient={meta['hospital_patient_name']}")

    if "doc_id" in meta:
        tags.append(f"id={meta['doc_id']}")

    if not tags:
        return ""

    # 例如：[company=XXX | id=39]
    return " [" + " | ".join(tags) + "]"


def _build_context_block(
    query: str,
    context_chunks: List[Dict[str, Any]],
    language: str,
) -> str:
    """
    先用 LLM 粗過濾 chunk，再把每個 chunk 加上清楚的來源標題。
    最後組成一個長字串給 LLM 當作 CONTEXT。
    """
    usable_chunks = _filter_chunks_by_relevance(query, context_chunks, language)

    sections = []
    for i, ch in enumerate(usable_chunks, start=1):
        meta = ch.get("metadata", {}) or {}
        label = _format_metadata(meta)
        text = (ch.get("page_content") or "").strip()
        if not text:
            continue

        header = f"### Source {i}{label}"
        sections.append(f"{header}\n{text}")

    return "\n\n".join(sections)

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

问题：
{query}

回答（请用简体中文）：
"""

def generate_answer(
    query: str,
    context_chunks: List[Dict[str, Any]],
    language: str = "en",
) -> str:
    """
    主流程：
    1. 用 LLM 粗略過濾不相關的 chunk
    2. 把剩下的 chunk + metadata 組成帶標註的 context
    3. 依語言產生 prompt 丟給 Ollama
    """
    context_block = _build_context_block(query, context_chunks, language)

    if language == "zh":
        prompt = _create_prompt_zh(query, context_block)
    else:
        prompt = _create_prompt_en(query, context_block)

    try:
        cfg = load_ollama_config()
        client = Client(host=cfg["host"])
        response = client.generate(
            model=cfg["model"],
            prompt=prompt,
            stream=False,
            options={
                # 低溫度：減少幻覺、盡量貼原文
                "temperature": 0.1,
                "num_ctx": 8192,
            },
        )
        return (response.get("response", "") or "").strip()
    except Exception as e:
        return f"Error using Ollama client: {e}"
