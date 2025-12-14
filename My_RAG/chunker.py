from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from langchain_text_splitters import RecursiveCharacterTextSplitter


# =========================
# 1) Canonical types
# =========================

CANONICAL_TYPES = [
    "FACTUAL",
    "MULTI_DOC_INTEGRATION",
    "MULTI_DOC_COMPARISON",
    "MULTI_DOC_TIMESEQ",
    "MULTIHOP",
    "SUMMARY",
    "IRRELEVANT_UNSOLVABLE",
    "UNKNOWN",
]


@dataclass
class QueryTypeResult:
    canonical_type: str
    dataset_label: str
    scores: Dict[str, float]
    debug_hits: Dict[str, List[str]]


# =========================
# 2) Regex patterns (ZH / EN)
# =========================

_ZH = {
    "SUMMARY": re.compile(r"(总结|概括|归纳|整理|综述|提炼|简述)"),
    "MULTI_DOC_COMPARISON": re.compile(r"(比较|对比|差异|不同|哪家|哪个.*更|谁.*更)"),
    "MULTI_DOC_TIMESEQ": re.compile(r"(更早|更晚|先于|晚于|先后顺序|先发生|后发生|时间序列)"),
    "MULTI_DOC_INTEGRATION": re.compile(r"(分别|根据.*(和|及).*|结合.*(和|及).*|在\d{4}年.*(和|及).*\d{4}年)"),
    "MULTIHOP": re.compile(r"(如何通过|如何.*来|导致|因此|从而|进而|使得|帮助.*(提升|增加|改善)|影响.*(原因|机制|路径))"),
    "FACTUAL": re.compile(r"(是谁|是什么|多少|几|何时|哪天|日期|时间|地点|职业|年龄|成立|设立|金额|比例|总资产|净利润)"),
    "IRRELEVANT_UNSOLVABLE": re.compile(r"(最喜欢的颜色|空气速度|unladen swallow|不相关)"),
}

_EN = {
    "SUMMARY": re.compile(r"\b(summarize|summary|summarise)\b", re.I),
    "MULTI_DOC_COMPARISON": re.compile(
        r"\b(compare|comparison|difference|vs\.?|versus|which .* (more|higher|larger|older|longer))\b", re.I
    ),
    "MULTI_DOC_TIMESEQ": re.compile(r"\b(earlier|later|before|after|time sequence|timeline)\b", re.I),
    "MULTI_DOC_INTEGRATION": re.compile(
        r"\b(according to .* and .*|respectively|both .* and .*|in \d{4} and \d{4})\b", re.I
    ),
    "MULTIHOP": re.compile(r"\b(how did .* lead to|how did .* result in|how .* help(ed)?|because|therefore|thus)\b", re.I),
    "FACTUAL": re.compile(r"\b(who|what|when|how many|how much|age|established|founded|date of)\b", re.I),
    "IRRELEVANT_UNSOLVABLE": re.compile(r"\b(airspeed velocity of an unladen swallow|favorite color of .* chief physician)\b", re.I),
}


# =========================
# 3) Label mapping (to dataset labels)
# =========================

def _choose_dataset_label(language: str, canonical_type: str) -> str:
    if language.startswith("zh"):
        return {
            "FACTUAL": "事实性问题",
            "MULTI_DOC_INTEGRATION": "多文档信息整合问题",
            "MULTI_DOC_COMPARISON": "多文档对比问题",
            "MULTI_DOC_TIMESEQ": "多文档时间序列问题",
            "MULTIHOP": "多跳推理问题",
            "SUMMARY": "总结性问题",
            "IRRELEVANT_UNSOLVABLE": "无关无解问",
            "UNKNOWN": "无关无解问",
        }[canonical_type]

    return {
        "FACTUAL": "Factual Question",
        "MULTI_DOC_INTEGRATION": "Multi-document Information Integration Question",
        "MULTI_DOC_COMPARISON": "Multi-document Comparison Question",
        "MULTI_DOC_TIMESEQ": "Multi-document Time Sequence Question",
        "MULTIHOP": "Multi-hop Reasoning Question",
        "SUMMARY": "Summary Question",  # 如果你要再拆 Summarization，可在 infer 裡加規則
        "IRRELEVANT_UNSOLVABLE": "Irrelevant Unsolvable Question",
        "UNKNOWN": "Irrelevant Unsolvable Question",
    }[canonical_type]


# =========================
# 4) Query type inference
# =========================

def infer_query_type_from_query(
    query_text: str,
    language: str,
    query_context: Optional[str] = None,
) -> QueryTypeResult:
    q = (query_text or "").strip()
    ctx = (query_context or "").strip()
    text = f"{q}\n{ctx}".strip()

    patterns = _ZH if language.startswith("zh") else _EN

    scores: Dict[str, float] = {t: 0.0 for t in CANONICAL_TYPES}
    hits: Dict[str, List[str]] = {t: [] for t in CANONICAL_TYPES}

    # (1) regex-based scoring
    for t, pat in patterns.items():
        m = pat.findall(text)
        if m:
            scores[t] += min(3.0, 1.0 + 0.5 * (len(m) - 1))
            hits[t].extend([str(x) for x in m[:5]])

    # (2) extra heuristics for multi-doc cues
    if language.startswith("zh"):
        if ("根据" in q and ("和" in q or "及" in q)) or ("分别" in q):
            scores["MULTI_DOC_INTEGRATION"] += 0.6
        if ("比较" in q or "对比" in q) and ("更" in q or "哪家" in q or "谁" in q):
            scores["MULTI_DOC_COMPARISON"] += 0.6
        if any(k in q for k in ["更早", "更晚", "先于", "晚于"]):
            scores["MULTI_DOC_TIMESEQ"] += 0.8
    else:
        if re.search(r"according to .* and .*", q, re.I) or re.search(r"\brespectively\b", q, re.I):
            scores["MULTI_DOC_INTEGRATION"] += 0.6
        if re.search(r"\bcompare\b", q, re.I) and re.search(r"\bwhich\b", q, re.I):
            scores["MULTI_DOC_COMPARISON"] += 0.6
        if re.search(r"\b(earlier|later)\b", q, re.I):
            scores["MULTI_DOC_TIMESEQ"] += 0.8

    # (3) priority boosts (more specific types > generic factoid)
    priority_boost = {
        "MULTI_DOC_TIMESEQ": 0.2,
        "MULTI_DOC_COMPARISON": 0.15,
        "MULTI_DOC_INTEGRATION": 0.1,
        "SUMMARY": 0.1,
        "MULTIHOP": 0.1,
    }
    for k, v in priority_boost.items():
        if scores[k] > 0:
            scores[k] += v

    best = max(scores.items(), key=lambda kv: kv[1])[0]
    if scores[best] == 0.0:
        best = "UNKNOWN"

    return QueryTypeResult(
        canonical_type=best,
        dataset_label=_choose_dataset_label(language, best),
        scores=scores,
        debug_hits=hits,
    )


# =========================
# 5) Chunk policy (query-aware)
# =========================

def _chunk_policy_by_type(language: str, canonical_type: str, base_chunk_size: int, base_overlap: int) -> Tuple[int, int]:
    zh = language.startswith("zh")

    if canonical_type == "FACTUAL":
        return (max(220, base_chunk_size - 140), max(32, base_overlap - 24)) if zh else (90, 18)

    if canonical_type == "MULTI_DOC_TIMESEQ":
        return (max(260, base_chunk_size - 100), max(48, base_overlap - 8)) if zh else (110, 22)

    if canonical_type == "MULTI_DOC_COMPARISON":
        return (base_chunk_size + 96, base_overlap + 32) if zh else (170, 34)

    if canonical_type == "MULTI_DOC_INTEGRATION":
        return (base_chunk_size + 160, base_overlap + 48) if zh else (190, 38)

    if canonical_type == "MULTIHOP":
        return (base_chunk_size + 256, base_overlap + 64) if zh else (230, 46)

    if canonical_type == "SUMMARY":
        return (base_chunk_size + 256, base_overlap + 64) if zh else (240, 48)

    # IRRELEVANT / UNKNOWN → baseline
    return (base_chunk_size, base_overlap) if zh else (140, 28)


# =========================
# 6) Splitter builders (baseline / query-aware)
# =========================

def get_text_splitter(language: str, chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    # ---- zh ----
    if language.startswith("zh"):
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=[
                r"\n{2,}",
                "\n",
                "。", "！", "？",
                "；",
                "，",
                ".", "!", "?",
                ""
            ],
            is_separator_regex=True
        )

    # ---- en (word-based) ----
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=lambda x: len(x.split()),
        separators=[
            r"\n\n+",
            r"\n",
            r"(?<=[.!?])\s+(?=[A-Z])",
            r"\s+",
            ""
        ],
        is_separator_regex=True
    )


def get_query_aware_text_splitter(
    language: str,
    query_text: str,
    base_chunk_size: int = 384,
    base_chunk_overlap: int = 64,
    query_context: Optional[str] = None,
) -> Tuple[RecursiveCharacterTextSplitter, QueryTypeResult]:
    qres = infer_query_type_from_query(query_text=query_text, language=language, query_context=query_context)
    chunk_size, chunk_overlap = _chunk_policy_by_type(language, qres.canonical_type, base_chunk_size, base_chunk_overlap)
    return get_text_splitter(language, chunk_size, chunk_overlap), qres


# =========================
# 7) Chunking runners (baseline / query-aware / auto)
# =========================

def chunk_documents(
    docs,
    language: str,
    chunk_size: int = 384,
    chunk_overlap: int = 64,
):
    """
    Baseline: only language-aware chunking
    """
    chunks = []
    splitter = get_text_splitter(language, chunk_size, chunk_overlap)

    for doc in docs:
        if (
            isinstance(doc, dict)
            and isinstance(doc.get("content"), str)
            and doc.get("language") == language
        ):
            text = doc["content"]
            base_metadata = {k: v for k, v in doc.items() if k != "content"}

            split_docs = splitter.create_documents(texts=[text], metadatas=[base_metadata])
            for i, sd in enumerate(split_docs):
                sd.metadata["chunk_index"] = i
                chunks.append({"page_content": sd.page_content, "metadata": sd.metadata})

    return chunks


def chunk_documents_query_aware(
    docs,
    language: str,
    query_text: str,
    base_chunk_size: int = 384,
    base_chunk_overlap: int = 64,
    query_context: Optional[str] = None,
):
    """
    Query-aware chunking:
    - infer query type
    - apply chunk policy
    - write policy + predicted type to metadata
    """
    chunks = []
    splitter, qres = get_query_aware_text_splitter(
        language=language,
        query_text=query_text,
        base_chunk_size=base_chunk_size,
        base_chunk_overlap=base_chunk_overlap,
        query_context=query_context
    )

    policy_meta = {
        "query_type_pred": qres.dataset_label,
        "query_type_canonical": qres.canonical_type,
        "chunk_size_used": getattr(splitter, "_chunk_size", None),
        "chunk_overlap_used": getattr(splitter, "_chunk_overlap", None),
    }

    for doc in docs:
        if (
            isinstance(doc, dict)
            and isinstance(doc.get("content"), str)
            and doc.get("language") == language
        ):
            text = doc["content"]
            base_metadata = {k: v for k, v in doc.items() if k != "content"}
            base_metadata.update(policy_meta)

            split_docs = splitter.create_documents(texts=[text], metadatas=[base_metadata])
            for i, sd in enumerate(split_docs):
                sd.metadata["chunk_index"] = i
                chunks.append({"page_content": sd.page_content, "metadata": sd.metadata})

    return chunks, qres


def chunk_documents_auto(
    docs,
    language: str,
    query_text: Optional[str] = None,
    chunk_size: int = 384,
    chunk_overlap: int = 64,
    query_context: Optional[str] = None,
):
    """
    ✅ 統一入口：
    - 有 query_text → query-aware
    - 沒有 query_text → baseline
    """
    if query_text and str(query_text).strip():
        return chunk_documents_query_aware(
            docs=docs,
            language=language,
            query_text=query_text,
            base_chunk_size=chunk_size,
            base_chunk_overlap=chunk_overlap,
            query_context=query_context
        )
    return chunk_documents(docs=docs, language=language, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
