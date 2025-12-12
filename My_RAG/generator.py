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
    英文任務說明：
    - 只用 context
    - 注意時間、實體歸屬、數值單位
    - 無法回答時固定用 "Unable to answer."
    """
    return f"""You are an assistant answering questions over short documents such as
corporate reports, court judgments, and hospital records.

The sources may include multiple documents. They are labeled like "Source 1 [company=... | id=...]", 
"Source 2 [court=...]", or "Source 3 [patient=...]".
Use these labels to keep different companies, courts, and patients separate and avoid mixing them up.

Before answering, infer the type of the question from its wording:

- If the question asks “when”, “what time”, “which date”, treat it as a *temporal factual question*, requiring extraction of an exact date from the context.
- If the question asks “who”, “which”, “what value”, “how much”, treat it as a *value or entity factual question*, requiring one specific fact.
- If the question asks “why” or “how”, or requires connecting more than one event, treat it as a *multi-hop reasoning question*, requiring integration across multiple sources.
- If the question asks for an overview (e.g., “summarize”, “outline the events”, “give a summary”), treat it as a *summary question* and produce a brief structured answer.
- If the question refers to something never mentioned in the context, treat it as an *unanswerable question* and answer exactly “Unable to answer.”

Additional guidance for matching the right source:

- If the question mentions a specific company, court, or person (for example, "Green Fields Agriculture Ltd."), first look for sources whose text or label contains the same name, and answer based on those sources.
- If that name appears in multiple sources, make sure you only use events that clearly belong to that same entity, and do not mix it with other entities.
- If the name in the question never appears in any source, treat the question as unanswerable and reply "Unable to answer."

You must follow these rules:

1. Use ONLY the information given in the Context section. Do not use outside knowledge.
2. Pay careful attention to TIME:
   - Events inside one paragraph may have different months or dates.
   - Always use the specific date next to the event, not just a general year or section heading.
   - For questions asking "who earlier" or "which date is earlier/later", compare the exact dates in the context.
3. Keep ENTITIES straight:
   - Do not mix up different companies, courts, or patients.
   - When the question mentions a specific company / court / patient, use the matching label in the sources.
4. For NUMBERS and COMPARISONS:
   - Convert numbers to the same unit before comparing (for example, all in millions or in thousands).
   - For questions like "who has a higher value" or "who has more/less", clearly state which one is higher/lower and support it with the numbers.
5. If the question clearly asks for something that does NOT appear anywhere in the context
   (for example, favourite color, hobbies, or other unrelated personal details),
   answer exactly and only:
   "Unable to answer."
6. For factual questions, answer with a short phrase or 1–2 concise sentences, focusing on the key value, date, or fact.
7. For summary questions, give a brief, coherent summary (2–4 sentences) that covers the key events and conclusions in chronological or logical order.
8. Do not show your reasoning steps; only output the final answer.
9. Do NOT include any information not explicitly found in the retrieved context.
10.If the question is a factual question asking for a date/time/value/name,
output ONLY the minimal answer span (e.g., "January 2021", "$5 million", "3 years"),
without extra explanation.


Context:
{context}

Question:
{query}

Answer in English:"""


def _create_prompt_zh(query: str, context: str) -> str:
    """
    中文任務說明：
    - 只用 context
    - 注意時間點、實體對應、數值單位
    - 無法回答時用「无法回答。」
    （用簡體中文，配合資料集文字）
    """
    return f"""你是一名专门处理财报、法院判决书与住院病历问答的智能助手。你的任务是根据给定的资料回答问题。

上下文可能包含多篇文档，每一段来源会标注为“Source 1 [company=… | id=…]”“Source 2 [court=…]”“Source 3 [patient=…]”等。
请利用这些标签区分不同的公司、法院和患者，避免混淆。

在回答之前，请根据问题的表述自动判断题目类型：

- 若问题包含“何时”“什么时候”“在什么日期”，视为*时间事实题*，需要从上下文提取确切日期。
- 若问题包含“谁”“哪一个”“多少”“什么数值”，视为*数值或实体事实题*，需要给出单一事实。
- 若问题需要解释原因、过程，或需跨段整合多项信息，视为*多跳推理题*，需整合多个来源。
- 若问题包含“总结”“概述”“描述经过”，视为*总结题*，需给出简短的结构化总结。
- 若问题询问的是上下文完全未出现的信息，视为*不可回答题*，回答：“无法回答。”

在匹配实体和来源时，请特别注意：

- 如果问题中点名了某个公司、法院或人物（例如：“Green Fields Agriculture Ltd.”），请优先在上下文中寻找文字或标签中包含该名称的来源，并基于这些来源作答。
- 如果该名称出现在多个来源中，只使用明确属于同一实体的事件，不要把不同实体的内容混在一起。
- 如果问题中的名称在任何来源中都没有出现，则视为不可回答问题，回答：“无法回答。”

请严格遵守以下规则：

1. 回答必须完全依据“上下文资料”，不要使用任何外部常识或臆测。
2. 特别注意时间信息：
   - 同一段文字中可能出现不同月份或日期的子事件。
   - 回答时要依据事件旁边的具体日期，而不是只看段首的年份或大标题。
   - 如果问题在比较“谁更早入院”“哪一个时间更早/更晚”，请直接比较具体日期并给出结论。
3. 核对实体：
   - 不要把一家公司发生的事情说成是另一家公司。
   - 法院名称、患者姓名等也要对应正确，问题提到谁，就用标签里对应的那个人或单位。
4. 进行数值或金额比较时：
   - 先把数字换算成相同单位再比较，例如都换成“万元”或“亿元”。
   - 如果问题在问“谁的数值更高/更低”，请先指出各自的数值，再明确说明谁更高/更低。
5. 如果问题明显询问的是上下文中完全没出现的信息
   （例如：喜欢的颜色、兴趣爱好等），请直接且只回答：
   “无法回答。”
6. 若是简单事实性问题，请用一个短语或 1–2 句简洁的句子回答，集中给出关键数字、日期或结论。
7. 若是总结性或多跳推理问题，请用 2–4 句简要总结关键经过和结论，可以按时间顺序或逻辑顺序组织。
8. 只要上下文中有任何与问题相关的信息，你就应该尽量利用这些信息作答，不要自行发挥。
9. 请使用简体中文作答，直接给出结论与必要说明，不要显示思考过程。
10. 如果问题是要求提供日期/时间/数值/名称的事实性问题，则仅输出最小答案范围（例如，“2021年1月”、“500万美元”、“3年”），无需额外解释。


上下文资料：
{context}

问题：
{query}

回答（请用简体中文）：
"""

def _check_evidence_sufficiency(
    query: str,
    context_chunks: List[Dict[str, Any]],
    language: str,
) -> bool:
    """
    使用 LLM 判斷檢索到的 chunks 是否足夠回答問題。
    
    返回:
        True: 證據充足，可以使用 chunks 生成答案
        False: 證據不足或不相關，應該使用 ToG 推理
    """
    if not context_chunks:
        return False
    
    # 構建 chunks 摘要
    chunks_summary = []
    for i, ch in enumerate(context_chunks[:5], start=1):  # 只看前5個
        text = (ch.get("page_content") or "").strip()
        if text:
            preview = text[:200] + "..." if len(text) > 200 else text
            chunks_summary.append(f"Chunk {i}: {preview}")
    
    context_preview = "\n".join(chunks_summary)
    
    if language == "zh":
        prompt = f"""你是一個問答系統的判斷模組。請評估檢索到的文檔片段是否足夠回答用戶的問題。

問題：
{query}

檢索到的文檔片段（前幾個）：
{context_preview}

請判斷：
- 如果這些片段包含足夠且相關的信息來回答問題，回答 "SUFFICIENT"
- 如果這些片段證據不足、不相關、或無法回答問題，回答 "INSUFFICIENT"

只回答一個詞：SUFFICIENT 或 INSUFFICIENT"""
    else:
        prompt = f"""You are a judgment module for a Q&A system. Evaluate whether the retrieved document chunks are sufficient to answer the user's question.

Question:
{query}

Retrieved document chunks (first few):
{context_preview}

Please judge:
- If these chunks contain sufficient and relevant information to answer the question, respond "SUFFICIENT"
- If these chunks are insufficient, irrelevant, or cannot answer the question, respond "INSUFFICIENT"

Respond with only one word: SUFFICIENT or INSUFFICIENT"""

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
        return "SUFFICIENT" in ans
    except Exception as e:
        print(f"[WARN] Evidence check failed: {e}, defaulting to sufficient")
        return True  # 預設為充足，避免過度使用 ToG


def _generate_answer_from_tog(
    query: str,
    kg_retriever,
    language: str,
) -> str:
    """
    使用 ToG 推理結果生成答案。
    
    流程:
    1. 使用 kg_retriever 進行 ToG 推理
    2. 從推理路徑和證據構建 context
    3. 使用 LLM 生成答案
    """
    if not kg_retriever:
        return "無法使用知識圖譜推理（KG retriever 未初始化）"
    
    try:
        # 進行 ToG 推理（總是使用 multi-hop）
        ranked_docs = kg_retriever.rank_docs(
            query=query,
            top_k=10,
            use_multi_hop=True,
            return_debug=True,
        )
        
        if not ranked_docs:
            return "無法從知識圖譜中找到相關信息來回答此問題。"
        
        # 構建 ToG 推理結果的 context
        tog_context_parts = []
        for i, doc_result in enumerate(ranked_docs[:5], start=1):  # 取前5個
            doc_id = doc_result.get("doc_id")
            score = doc_result.get("score", 0.0)
            paths = doc_result.get("paths", [])
            evidence = doc_result.get("evidence_triples", [])
            
            if language == "zh":
                tog_context_parts.append(f"### 推理結果 {i} (文檔 {doc_id}, 相關度: {score:.2f})")
                
                if paths:
                    tog_context_parts.append("推理路徑：")
                    for path in paths[:3]:  # 最多3條路徑
                        tog_context_parts.append(f"  - {path}")
                
                if evidence:
                    tog_context_parts.append("證據三元組：")
                    for triple in evidence[:3]:  # 最多3個三元組
                        head = triple.get("head", "")
                        relation = triple.get("relation", "")
                        tail = triple.get("tail", "")
                        tog_context_parts.append(f"  - {head} --[{relation}]--> {tail}")
            else:
                tog_context_parts.append(f"### Reasoning Result {i} (Document {doc_id}, Relevance: {score:.2f})")
                
                if paths:
                    tog_context_parts.append("Reasoning Paths:")
                    for path in paths[:3]:  # Max 3 paths
                        tog_context_parts.append(f"  - {path}")
                
                if evidence:
                    tog_context_parts.append("Evidence Triples:")
                    for triple in evidence[:3]:  # Max 3 triples
                        head = triple.get("head", "")
                        relation = triple.get("relation", "")
                        tail = triple.get("tail", "")
                        tog_context_parts.append(f"  - {head} --[{relation}]--> {tail}")
            
            tog_context_parts.append("")
        
        tog_context = "\n".join(tog_context_parts)
        
        # 構建 prompt
        if language == "zh":
            prompt = f"""你是一個基於知識圖譜推理的問答系統。請根據 ToG (Think-on-Graph) 推理結果回答問題。

問題：
{query}

ToG 推理結果（包含推理路徑和證據三元組）：
{tog_context}

請根據這些推理路徑和證據，用簡潔的語言回答問題。如果推理結果不足以回答問題，請說明。

回答（請用簡體中文）："""
        else:
            prompt = f"""You are a Q&A system based on knowledge graph reasoning. Answer the question based on ToG (Think-on-Graph) reasoning results.

Question:
{query}

ToG reasoning results (including reasoning paths and evidence triples):
{tog_context}

Please answer the question concisely based on these reasoning paths and evidence. If the reasoning results are insufficient, please state so.

Answer in English:"""
        
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
        
    except Exception as e:
        return f"使用 ToG 推理時發生錯誤: {e}"


def generate_answer(
    query: str,
    context_chunks: List[Dict[str, Any]],
    language: str = "en",
    kg_retriever=None,
) -> str:
    """
    主流程：
    1. 使用 LLM 判斷檢索到的 chunks 是否足夠
    2. 如果證據充足：使用 chunks 生成答案（原有流程）
    3. 如果證據不足：使用 ToG 推理生成答案
    """
    # 判斷證據是否充足
    evidence_sufficient = _check_evidence_sufficiency(query, context_chunks, language)
    
    if evidence_sufficient:
        # 原有流程：使用 chunks 生成答案
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
    else:
        # 證據不足：使用 ToG 推理
        if kg_retriever:
            return _generate_answer_from_tog(query, kg_retriever, language)
        else:
            # 如果沒有 kg_retriever，fallback 到原有流程
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
                    options={"temperature": 0.1, "num_ctx": 8192},
                )
                return (response.get("response", "") or "").strip()
            except Exception as e:
                return f"Error using Ollama client: {e}"
                