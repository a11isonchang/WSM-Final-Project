import json
import argparse
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from tqdm import tqdm
from openai import OpenAI
from openai import APIError, RateLimitError, APIConnectionError

from config import load_config
from utils import load_jsonl


# =========================
# OpenAI 初始化
# =========================

CLIENT: Optional[OpenAI] = None
MODEL: Optional[str] = None


def init_openai():
    """
    初始化 OpenAI Client 與模型名稱：
    - model 來源：config.yaml 的 openai.model 或環境變數 OPENAI_MODEL，預設 gpt-4o-mini
    """
    global CLIENT, MODEL
    cfg = load_config() or {}

    MODEL = (
        cfg.get("openai", {}).get("model")
        or cfg.get("openai_model")
        or "gpt-4o-mini"
    )

    CLIENT = OpenAI()
    print(f"[OpenAI] Using model: {MODEL}")


# =========================
# LLM Prompt：乾淨 KG Schema
# =========================

KG_PROMPT = r"""
You are an information extraction model for a Retrieval-Augmented Generation (RAG) system.
Your goal is to build a **clean, retrieval-friendly knowledge graph** from ONE document.

Return ONLY valid JSON with keys:
- "entities": list[entity]
- "triples": list[triple]

====================================================
GLOBAL CRITICAL RULES (MUST FOLLOW)
====================================================
1) NO numeric-only entities:
   - Do NOT create entities that are purely numeric or are just money/percent/date strings
     (e.g., "10亿元", "82%", "100,000,000元", "2017年度" as an Event).
   - Money/percent/date/time should be stored in triple.properties.

2) No over-semanticized relations:
   - DO NOT use relations like "achieved revenue of", "has employee satisfaction of", etc.
   - Use the canonical relations below.

3) Entity uniqueness & type consistency:
   - If two entities have the SAME "name", they MUST have the SAME "type".
   - NEVER create the same name as both "Time" and "Event" (e.g., "2017年度" cannot be both).

4) Canonical time handling:
   - "Time" entities are allowed ONLY when they represent a specific time period used as a join key:
       Examples: "2017年", "2017年4月", "2017-03-15", "2021年"
   - "年度" time like "2017年度" is allowed ONLY as type "Time", not as "Event".
   - For event triples (dividend/acquisition/incident/etc.), put time into properties.time.
     Avoid using Time as the tail for event relations unless specified below.

5) Events must be informative:
   - Event tails MUST be descriptive and specific, not generic category names.
     BAD tail: "道德与诚信事件"
     GOOD tail: "2017年4月商业贿赂调查事件" / "2017年4月合规违规事件"
   - Always include properties.description (1 sentence) for incidents and governance changes.

====================================================
ENTITY SCHEMA
====================================================
Each entity MUST be:
{
  "name": string,
  "type": string,
  "properties": object
}

Allowed entity types (choose closest):
- Organization
- Person
- Subsidiary
- Event
- ESGAction
- GovernanceChange
- MedicalRecord
- Patient
- Location
- Time

Entity creation guidance:
- Create entities for real-world nouns: companies, persons, subsidiaries, named projects, named incidents.
- Do NOT create entities for amounts/percentages.
- Do NOT create an "Event" entity whose name is only a time label (e.g., "2017年度").

====================================================
TRIPLE SCHEMA
====================================================
Each triple MUST be:
{
  "head": string,
  "head_type": string,
  "relation": string,
  "tail": string,
  "tail_type": string,
  "doc_id": integer,
  "properties": object
}

- head/tail must match entity names.
- relation must be snake_case and chosen from CANONICAL RELATIONS below unless absolutely necessary.

====================================================
CANONICAL RELATIONS (PREFERRED)
====================================================

(A) Organization basics
- was_founded_in: (Organization -> Time)
  STRICT:
  - Tail must be a founding time (year or year-month or exact date).
  - It MUST NOT be a reporting period like "2017年度" unless the document explicitly states the company was founded then.
  - If founding time is not present, do NOT output this triple.

- is_located_in: (Organization -> Location)

(B) Subsidiary / structure
- established_subsidiary: (Organization -> Subsidiary)
  properties: { "time": "...", "location": "...", "ownership_ratio": "..." }

(C) Investments & acquisitions
- asset_acquisition: (Organization -> Organization OR Event)
  properties REQUIRED when available:
  - "time": string  (MUST extract if present in text)
  - "amount": string (if present)
  - "equity_ratio": string (if present, e.g. "70%")
  - "asset_type": string (e.g. "equity acquisition", "property acquisition")
  RULE:
  - If time is mentioned in the document for the acquisition, you MUST include properties.time.

- invested_in_project: (Organization -> Event)
  properties: { "time": "...", "amount": "...", "description": "..." }

(D) Financial performance (IMPORTANT)
- financials: (Organization -> Time)
  RULES:
  - This relation is ONLY for numeric performance metrics.
  - Tail MUST be a Time entity like "2017年" or "2017年度" (Time type).
  - Put all numbers in properties (do NOT create numeric entities).
  Example properties:
    {
      "revenue": "...",
      "operating_income": "...",
      "net_profit": "...",
      "total_assets": "...",
      "total_liabilities": "...",
      "shareholders_equity": "...",
      "cash_flow": "..."
    }

(E) Dividends
- dividend_event: (Organization -> Event)
  STRICT:
  - Tail MUST be a descriptive event name (NOT a pure time label like "2017年度").
    GOOD tail: "2017年10月现金股利分配"
    BAD tail: "2017年度"
  properties REQUIRED:
  - "time": "...", "amount": "...", "type": "cash/stock/..." (if present)

(F) Incidents / ethics / compliance / accidents
- had_incident: (Organization -> Event)
  STRICT:
  - Tail MUST be specific event name, not generic like "道德与诚信事件".
  properties REQUIRED:
  - "time": string
  - "incident_type": one of ["environmental_accident", "ethics_compliance", "safety_accident", "legal_case", "other"]
  - "severity": string (optional)
  - "description": string (1 sentence, REQUIRED)

(G) ESG actions
- implemented_esg_action: (Organization -> ESGAction)
  properties REQUIRED:
  - "time": "..."
  - "category": one of ["carbon_offset", "energy_saving", "green_product", "community_investment", "other"]
  - "description": 1 sentence

(H) Management & governance
- management_change: (Organization -> Person)
  properties REQUIRED when available:
  - "time": "..."
  - "change_type": ["appointment", "resignation", "termination", "promotion"]
  - "position": "CEO/CFO/COO/..."
  - "reason": "..." (optional)

- governance_change: (Organization -> GovernanceChange)
  STRICT:
  - Tail SHOULD be a GovernanceChange entity (not Time).
  properties REQUIRED:
  - "time": "..."
  - "description": 1 sentence (REQUIRED)

(I) Medical (if document is clinical)
- has_medical_record: (Patient -> MedicalRecord)
  properties: { "hospital": "...", "admission_date": "...", "description": "..." }

====================================================
OUTPUT COMPACTNESS
====================================================
- Prefer a compact graph: extract only important and query-relevant facts.
- Do NOT generate more than ~20 triples per document unless the document is very long and information-dense.

====================================================
INPUT YOU WILL RECEIVE
====================================================
- doc_id (integer)
- domain (string)
- language (string)
- document text

You MUST set triple.doc_id accordingly.
Also include domain/language in triple.properties if useful.

Return ONLY JSON. No markdown, no explanation.
"""


# =========================
# 工具函式：解析 LLM 回傳 JSON
# =========================

def _safe_extract_json(text: str) -> Dict[str, Any]:
    """
    將 LLM 回傳的文字盡量轉成 JSON。
    支援有 ```json ... ``` code block 的情況。
    """
    if not text:
        raise ValueError("Empty response from LLM")

    # 去掉 code block 標記
    stripped = text.strip()
    if stripped.startswith("```"):
        # 去掉前三個反引號與可能的語言標記
        stripped = stripped.lstrip("`")
        # 再從第一個 { 開始截
    # 從第一個 '{' 到最後一個 '}' 之間嘗試 parse
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"Cannot find JSON object in response: {text[:200]!r}")

    candidate = stripped[start : end + 1]

    return json.loads(candidate)


# =========================
# 對單一文件做 KG 抽取
# =========================

def _get_doc_text(doc: Dict[str, Any]) -> Optional[str]:
    """
    嘗試從 doc 中拿出正文內容。
    預設優先順序：text > content > body
    """
    for key in ("text", "content", "body"):
        if key in doc and isinstance(doc[key], str) and doc[key].strip():
            return doc[key]
    return None


def extract_kg_from_doc(
    doc: Dict[str, Any],
    doc_id: int,
    domain: str,
    language: str,
    max_retries: int = 5,
) -> Optional[Dict[str, Any]]:
    """
    呼叫 OpenAI, 針對單一文件抽取 KG（entities + triples）。
    回傳格式：
    {
      "doc_id": ...,
      "domain": ...,
      "language": ...,
      "entities": [...],
      "triples": [...]
    }
    """
    assert CLIENT is not None and MODEL is not None, "OpenAI client not initialized"

    text = _get_doc_text(doc)
    if not text:
        print(f"[WARN] doc {doc_id} has no text field, skipping.")
        return None

    user_content = (
        KG_PROMPT
        + f"\n\nDocument metadata:\n"
        + f"- doc_id: {doc_id}\n"
        + f"- domain: {domain}\n"
        + f"- language: {language}\n\n"
        + "Document:\n"
        + text
        + "\n\nRemember: return ONLY JSON.\n"
    )

    for attempt in range(max_retries):
        try:
            response = CLIENT.chat.completions.create(
                model=MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "You extract compact, clean knowledge graphs for a RAG system.",
                    },
                    {"role": "user", "content": user_content},
                ],
                temperature=0,
            )

            content = response.choices[0].message.content
            parsed = _safe_extract_json(content)

            # 保證基本欄位存在
            entities = parsed.get("entities", [])
            triples = parsed.get("triples", [])

            # 填上 doc_id / domain / language
            for t in triples:
                t.setdefault("doc_id", doc_id)
                props = t.get("properties") or {}
                if "domain" not in props and domain:
                    props["domain"] = domain
                if "language" not in props and language:
                    props["language"] = language
                t["properties"] = props

            return {
                "doc_id": doc_id,
                "domain": domain,
                "language": language,
                "entities": entities,
                "triples": triples,
            }

        except (RateLimitError, APIError, APIConnectionError) as e:
            wait_time = 2 ** attempt
            print(
                f"\n[Retryable Error] doc {doc_id}, attempt {attempt + 1}/{max_retries}, "
                f"waiting {wait_time}s. Error: {e}"
            )
            time.sleep(wait_time)

        except Exception as e:
            # 其他錯誤也做 retry，但最後一次失敗就放棄
            print(f"\n[Unexpected Error] doc {doc_id}, attempt {attempt + 1}/{max_retries}: {e}")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
            else:
                print(f"❌ Failed to extract KG for doc {doc_id} after {max_retries} attempts.")
                return None

    return None


# =========================
# 合併多文件的 KG
# =========================

def merge_knowledge_graphs(kg_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    將每篇文件的 KG 統一 merge 成一張大圖。
    輸出格式：
    {
      "entities": [...],
      "relations": [...],   # relation 類型的 summary
      "triples": [...],
      "doc_mapping": {...},
      "doc_domains": {...},
      "statistics": {...}
    }
    """
    entity_index: Dict[Tuple[str, str], str] = {}  # (type, name) -> id
    merged_entities: List[Dict[str, Any]] = []
    merged_triples: List[Dict[str, Any]] = []

    relation_stats: Dict[str, Dict[str, Any]] = {}
    doc_mapping: Dict[str, Dict[str, int]] = {}
    doc_domains: Dict[str, str] = {}

    next_id = 1
    triple_counter = 0

    for kg_doc in kg_list:
        if not kg_doc:
            continue

        doc_id = kg_doc.get("doc_id")
        domain = kg_doc.get("domain", "")
        doc_id_str = str(doc_id)

        doc_domains[doc_id_str] = domain
        doc_start = triple_counter

        entities = kg_doc.get("entities", []) or []
        triples = kg_doc.get("triples", []) or []

        # 先把本文件的 entities 建立成 local map：name+type -> 確定 merged id
        # 但實際上我們只在處理 triples 時才需要建立
        for t in triples:
            head_name = t.get("head")
            tail_name = t.get("tail")
            head_type = t.get("head_type") or "Unknown"
            tail_type = t.get("tail_type") or "Unknown"

            if not head_name or not tail_name:
                continue

            head_key = (head_type, head_name)
            tail_key = (tail_type, tail_name)

            # head entity
            if head_key not in entity_index:
                eid = f"e{next_id}"
                next_id += 1
                entity_index[head_key] = eid
                merged_entities.append(
                    {"id": eid, "name": head_name, "type": head_type, "properties": {}}
                )

            # tail entity
            if tail_key not in entity_index:
                eid = f"e{next_id}"
                next_id += 1
                entity_index[tail_key] = eid
                merged_entities.append(
                    {"id": eid, "name": tail_name, "type": tail_type, "properties": {}}
                )

            head_id = entity_index[head_key]
            tail_id = entity_index[tail_key]

            # 標準 triple 格式：保留 properties（time/amount/...）
            relation = t.get("relation", "")
            props = t.get("properties") or {}

            merged_triples.append(
                {
                    "head": head_name,
                    "head_id": head_id,
                    "head_type": head_type,
                    "relation": relation,
                    "tail": tail_name,
                    "tail_id": tail_id,
                    "tail_type": tail_type,
                    "doc_id": doc_id,
                    "properties": props,
                }
            )
            triple_counter += 1

            # 更新 relation 統計
            if relation:
                stat = relation_stats.setdefault(
                    relation,
                    {
                        "type": relation,
                        "count": 0,
                        "example_head_type": head_type,
                        "example_tail_type": tail_type,
                    },
                )
                stat["count"] += 1

        doc_end = triple_counter - 1
        if doc_id is not None and doc_start <= doc_end:
            doc_mapping[doc_id_str] = {
                "start_triple_index": doc_start,
                "end_triple_index": doc_end,
            }

    relations_summary = list(relation_stats.values())

    statistics = {
        "total_entities": len(merged_entities),
        "total_triples": len(merged_triples),
        "total_relation_types": len(relations_summary),
    }

    return {
        "entities": merged_entities,
        "relations": relations_summary,
        "triples": merged_triples,
        "doc_mapping": doc_mapping,
        "doc_domains": doc_domains,
        "statistics": statistics,
    }


# =========================
# Checkpoint 機制
# =========================

def load_checkpoint(checkpoint_file: str) -> Tuple[List[Dict[str, Any]], int]:
    """Load checkpoint if it exists, return (kg_results, start_index)"""
    path = Path(checkpoint_file)
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            checkpoint = json.load(f)
        kg_results = checkpoint.get("kg_results", [])
        last_idx = checkpoint.get("last_processed_index", -1)
        return kg_results, last_idx + 1
    return [], 0


def save_checkpoint(checkpoint_file: str, kg_results: List[Dict[str, Any]], last_index: int):
    """Save checkpoint"""
    checkpoint = {
        "kg_results": kg_results,
        "last_processed_index": last_index,
    }
    with Path(checkpoint_file).open("w", encoding="utf-8") as f:
        json.dump(checkpoint, f, ensure_ascii=False, indent=2)


# =========================
# Main
# =========================

def main():
    parser = argparse.ArgumentParser(description="Build knowledge graph from documents")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of documents to process",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="dragonball_dataset/dragonball_docs.jsonl",
        help="Path to input JSONL file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="My_RAG/kg_output.json",
        help="Path to output JSON file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="My_RAG/kg_checkpoint.json",
        help="Path to checkpoint file for resume",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=10,
        help="Save checkpoint every N documents",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint if it exists",
    )
    args = parser.parse_args()

    init_openai()

    docs = load_jsonl(args.input)

    # Apply limit if specified
    if args.limit is not None:
        docs = docs[: args.limit]

    # Load checkpoint if resuming
    start_index = 0
    kg_results: List[Dict[str, Any]] = []
    if args.resume:
        kg_results, start_index = load_checkpoint(args.checkpoint)
        if start_index > 0:
            print(
                f"Resuming from checkpoint: {len(kg_results)} results loaded, "
                f"starting from index {start_index}"
            )

    print(f"Extracting KG from {len(docs)} documents (starting from {start_index})...")

    for i, doc in enumerate(
        tqdm(docs[start_index:], initial=start_index, total=len(docs))
    ):
        doc_id = doc.get("doc_id", start_index + i)
        domain = doc.get("domain", "")
        language = doc.get("language", "")

        kg = extract_kg_from_doc(doc, doc_id, domain, language)

        if kg is None:
            print(f"\n[WARN] Failed to extract KG for doc {doc_id}, storing empty result.")
            kg_results.append({})
        else:
            kg_results.append(kg)

        # Save checkpoint periodically
        current_index = start_index + i
        if (current_index + 1) % args.checkpoint_interval == 0:
            save_checkpoint(args.checkpoint, kg_results, current_index)
            print(
                f"\nCheckpoint saved at document {current_index + 1}/{len(docs)}"
            )

    # Final checkpoint save
    save_checkpoint(args.checkpoint, kg_results, len(docs) - 1)

    print("\nMerging KG...")
    final_kg = merge_knowledge_graphs(kg_results)

    with Path(args.output).open("w", encoding="utf-8") as f:
        json.dump(final_kg, f, ensure_ascii=False, indent=2)

    print(f"KG saved to {args.output}")

    # 清掉 checkpoint
    cp_path = Path(args.checkpoint)
    if cp_path.exists():
        cp_path.unlink()
        print(f"Checkpoint file {args.checkpoint} removed")


if __name__ == "__main__":
    main()
