"""
基于知识图谱的检索器
通过 kg_output.json 中的实体和关系，根据 query 检索相关的文档
"""

import json
import jieba
import re
from typing import List, Dict, Any, Set, Optional
from pathlib import Path


class KGRetriever:
    """基于知识图谱的检索器，用于提升检索精度（含 optional multi-hop）"""

    def __init__(
        self,
        kg_path: str = "My_RAG/kg_output.json",
        language: str = "en",
        max_hops: int = 2,
        docs_path: Optional[str] = None,
    ):
        """
        Args:
            kg_path: 知识图谱文件路径
            language: 语言类型 ("zh" 或 "en")
            max_hops: 最大跳数，用于多跳关系检索（默认 2 跳）
            docs_path: 原始文档 JSONL 路径（可选，用于通过 company_name 直接对齐）
        """
        self.language = language
        self.max_hops = max_hops
        self.kg_data = self._load_kg(kg_path)

        # entity_id -> entity object
        self.entities: Dict[str, Dict[str, Any]] = {
            e.get("id"): e for e in self.kg_data.get("entities", []) if e.get("id")
        }

        # 名称索引 + 文档映射 + 实体图
        self.entity_index = self._build_entity_index()
        self.entity_to_docs = self._build_entity_doc_mapping()
        self.relation_index = self._build_relation_index()
        self.entity_graph = self._build_entity_graph()

        # 透过 docs 的 company_name 建立 name -> doc_id 映射
        self.entity_name_to_docs = self._build_entity_name_doc_mapping(docs_path)

    # =====================
    # 基础构建
    # =====================

    def _load_kg(self, kg_path: str) -> Dict[str, Any]:
        path = Path(kg_path)
        if not path.exists():
            print(f"Warning: KG file not found at {kg_path}")
            return {"entities": [], "relations": []}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _build_entity_index(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        建立实体名称到实体对象的索引（含英文大小写、公司名 base name、中文分词）
        """
        index: Dict[str, List[Dict[str, Any]]] = {}
        entities = self.kg_data.get("entities", [])

        for entity in entities:
            name = (entity.get("name") or "").strip()
            if not name:
                continue

            # 原始名称
            index.setdefault(name, []).append(entity)

            # 英文：小写版本 + 去除常见公司后缀
            if self.language == "en":
                name_lower = name.lower()
                if name_lower != name:
                    index.setdefault(name_lower, []).append(entity)

                # 去掉常见公司后缀
                base_name = re.sub(
                    r"\s+(ltd\.?|inc\.?|corp\.?|llc\.?|co\.?|company|services|group)$",
                    "",
                    name,
                    flags=re.IGNORECASE,
                ).strip()
                if base_name and base_name.lower() != name_lower and len(base_name) >= 3:
                    index.setdefault(base_name.lower(), []).append(entity)

            # 中文：把实体名分词后的 token 也 index起来
            if self.language == "zh":
                for token in jieba.cut(name):
                    token = token.strip()
                    if len(token) >= 2:
                        index.setdefault(token, []).append(entity)

        return index

    def _build_entity_doc_mapping(self) -> Dict[str, Set[int]]:
        """
        建立实体ID到 doc_id 集合的映射（透过 relations 的 doc_id）
        """
        mapping: Dict[str, Set[int]] = {}
        for rel in self.kg_data.get("relations", []):
            doc_id = rel.get("doc_id")
            if doc_id is None:
                continue
            for ent_id in (rel.get("source"), rel.get("target")):
                if not ent_id:
                    continue
                mapping.setdefault(ent_id, set()).add(doc_id)
        return mapping

    def _build_relation_index(self) -> Dict[str, List[Dict[str, Any]]]:
        index: Dict[str, List[Dict[str, Any]]] = {}
        for rel in self.kg_data.get("relations", []):
            t = rel.get("type")
            if not t:
                continue
            index.setdefault(t, []).append(rel)
        return index

    def _build_entity_graph(self) -> Dict[str, Set[str]]:
        """
        构建实体关系图（无向图），用于 multi-hop。
        注意：这里只建 entity_id 之间的连接，不在这里做 doc 过滤。
        """
        graph: Dict[str, Set[str]] = {}
        for rel in self.kg_data.get("relations", []):
            s = rel.get("source")
            t = rel.get("target")
            if not s or not t:
                continue
            graph.setdefault(s, set()).add(t)
            graph.setdefault(t, set()).add(s)
        return graph

    def _build_entity_name_doc_mapping(self, docs_path: Optional[str]) -> Dict[str, Set[int]]:
        """
        透過原始文档（JSONL）里的 company_name 建立 name -> doc_id 映射。
        用来处理「KG 没有明确关系，但文档有公司名」的情况。
        """
        mapping: Dict[str, Set[int]] = {}
        if not docs_path:
            return mapping

        path = Path(docs_path)
        if not path.exists():
            print(f"Warning: docs file for KG mapping not found at {docs_path}")
            return mapping

        try:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        doc = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    doc_id = doc.get("doc_id")
                    if doc_id is None:
                        continue

                    company_name = (
                        doc.get("company_name")
                        or doc.get("company")
                        or doc.get("firm_name")
                    )
                    if not company_name:
                        continue

                    name = company_name.strip()
                    mapping.setdefault(name, set()).add(doc_id)

                    # 英文再加一個 lower 版本
                    if self.language == "en":
                        mapping.setdefault(name.lower(), set()).add(doc_id)
        except Exception as e:
            print(f"Warning: failed to build entity_name_to_docs from {docs_path}: {e}")

        return mapping

    # =====================
    # Query 解析
    # =====================

    def _extract_keywords_from_query(self, query: str) -> Set[str]:
        if self.language == "zh":
            tokens = [t.strip() for t in jieba.cut(query) if t.strip()]
            return {t for t in tokens if len(t) >= 2}
        else:
            words = re.findall(r"\b\w+\b", query.lower())
            return {w for w in words if len(w) >= 3}

    def _extract_entities_from_query(self, query: str) -> List[Dict[str, Any]]:
        """
        从 query 中用「字符串包含 + 分词」匹配实体。
        """
        found: List[Dict[str, Any]] = []
        seen_ids: Set[str] = set()

        if self.language == "en":
            haystack = query.lower()
        else:
            haystack = query

        # 1) 直接用名稱 key 做包含匹配
        for name, entities in self.entity_index.items():
            key = name.lower() if self.language == "en" else name
            if not key or len(key) < 2:
                continue
            if key in haystack:
                for e in entities:
                    eid = e.get("id")
                    if eid and eid not in seen_ids:
                        seen_ids.add(eid)
                        found.append(e)

        # 2) 中文再用 query 分詞反向 lookup 一次
        if self.language == "zh":
            for token in jieba.cut(query):
                token = token.strip()
                if len(token) < 2:
                    continue
                if token in self.entity_index:
                    for e in self.entity_index[token]:
                        eid = e.get("id")
                        if eid and eid not in seen_ids:
                            seen_ids.add(eid)
                            found.append(e)

        return found

    # =====================
    # Multi-hop BFS（帶 doc 過濾）
    # =====================

    def _multi_hop_retrieval(self, start_entity_ids: Set[str]) -> Dict[str, float]:
        """
        从起始实体集合开始，做有限跳数的 BFS。
        只对「與起始實體有共同 doc_id」的實體加分，避免跨文件污染。
        """
        if not start_entity_ids:
            return {}

        # 收集起始實體所屬的 doc_id 集合
        start_docs: Set[int] = set()
        for eid in start_entity_ids:
            start_docs |= self.entity_to_docs.get(eid, set())

        if not start_docs:
            return {}

        visited: Set[str] = set(start_entity_ids)
        queue: List[tuple[str, int]] = [(eid, 0) for eid in start_entity_ids]
        entity_scores: Dict[str, float] = {}

        # 重要：只對這些 type 的實體計分
        score_types: Set[str] = {
            "Company",
            "Project",
            "FinancialMetric",
            "Case",
            "Court",
            "Hospital",
            "Patient",
        }

        while queue:
            current_id, hop = queue.pop(0)
            if hop >= self.max_hops:
                continue

            neighbors = self.entity_graph.get(current_id, set())
            next_hop = hop + 1

            for nid in neighbors:
                if nid in visited:
                    continue
                visited.add(nid)

                # 只考慮 doc 交集不為空的實體
                docs_n = self.entity_to_docs.get(nid, set())
                if not docs_n or not (docs_n & start_docs):
                    # 仍然可以繼續往下走，但不計分
                    queue.append((nid, next_hop))
                    continue

                # 只有特定 type 的實體才計分
                et = self.entities.get(nid, {}).get("type")
                if et not in score_types:
                    queue.append((nid, next_hop))
                    continue

                # hop 越遠分數越低：1-hop=1.0, 2-hop=0.5, ...
                hop_score = max(0.0, 1.0 * (0.5 ** (next_hop - 1)))
                entity_scores[nid] = max(entity_scores.get(nid, 0.0), hop_score)

                queue.append((nid, next_hop))

        return entity_scores

    # =====================
    # 實際 doc 檢索
    # =====================

    def retrieve_doc_ids(self, query: str, top_k: int = 10, use_multi_hop: bool = True) -> List[int]:
        """
        根据 query 检索相关的 doc_id，支援多跳关系（可開關）。
        """
        entities = self._extract_entities_from_query(query)
        start_entity_ids: Set[str] = {e.get("id") for e in entities if e.get("id")}

        doc_id_scores: Dict[int, float] = {}
        keywords = self._extract_keywords_from_query(query)

        # 0) 透過 company_name 直接對齊（權重最高）
        for e in entities:
            name = (e.get("name") or "").strip()
            if not name:
                continue
            for key in {name, name.lower()} if self.language == "en" else {name}:
                if key in self.entity_name_to_docs:
                    for doc_id in self.entity_name_to_docs[key]:
                        doc_id_scores[doc_id] = doc_id_scores.get(doc_id, 0.0) + 10.0

        # 1) 直接以實體 ID -> doc
        for eid in start_entity_ids:
            for doc_id in self.entity_to_docs.get(eid, set()):
                doc_id_scores[doc_id] = doc_id_scores.get(doc_id, 0.0) + 5.0

        # 2) 多跳實體帶來的 doc（權重較低）
        if use_multi_hop and start_entity_ids:
            multi_entities = self._multi_hop_retrieval(start_entity_ids)
            for eid, es in multi_entities.items():
                docs = self.entity_to_docs.get(eid, set())
                for doc_id in docs:
                    # 如果這個 doc 已經有高分（直接 match），就不要被多跳洗掉
                    if doc_id_scores.get(doc_id, 0.0) >= 5.0:
                        continue
                    doc_id_scores[doc_id] = doc_id_scores.get(doc_id, 0.0) + es * 0.5

        # 3) relation type / properties 關鍵字 match
        for rel in self.kg_data.get("relations", []):
            doc_id = rel.get("doc_id")
            if doc_id is None:
                continue

            rel_type = (rel.get("type") or "").lower()
            if any(kw in rel_type for kw in keywords):
                doc_id_scores[doc_id] = doc_id_scores.get(doc_id, 0.0) + 1.0

            props = rel.get("properties", {}) or {}
            for v in props.values():
                if isinstance(v, str) and any(kw in v for kw in keywords):
                    doc_id_scores[doc_id] = doc_id_scores.get(doc_id, 0.0) + 0.5

        # 4) entity properties 關鍵字 match
        for e in entities:
            eid = e.get("id")
            if not eid:
                continue
            props = e.get("properties", {}) or {}
            for v in props.values():
                if isinstance(v, str) and any(kw in v for kw in keywords):
                    for doc_id in self.entity_to_docs.get(eid, set()):
                        doc_id_scores[doc_id] = doc_id_scores.get(doc_id, 0.0) + 0.5

        if not doc_id_scores:
            return []

        sorted_docs = sorted(doc_id_scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_id for doc_id, _ in sorted_docs[:top_k]]

    # =====================
    # Debug / inspect
    # =====================

    def get_entity_info(self, query: str, use_multi_hop: bool = True) -> Dict[str, Any]:
        entities = self._extract_entities_from_query(query)
        start_entity_ids: Set[str] = {e.get("id") for e in entities if e.get("id")}

        multi_hop_entities: Dict[str, float] = {}
        if use_multi_hop and start_entity_ids:
            multi_hop_entities = self._multi_hop_retrieval(start_entity_ids)

        doc_ids = self.retrieve_doc_ids(query, top_k=10, use_multi_hop=use_multi_hop)

        return {
            "entities_found": [
                {
                    "id": e.get("id"),
                    "name": e.get("name"),
                    "type": e.get("type"),
                }
                for e in entities
            ],
            "related_doc_ids": doc_ids,
            "entity_count": len(entities),
            "multi_hop": {
                "enabled": bool(use_multi_hop and start_entity_ids),
                "multi_hop_entities": len(multi_hop_entities),
                "sample_entities": [
                    {"id": eid, "score": score}
                    for eid, score in list(multi_hop_entities.items())[:5]
                ],
            },
        }


def create_kg_retriever(
    kg_path: str = "My_RAG/kg_output.json",
    language: str = "en",
    max_hops: int = 2,
    docs_path: Optional[str] = None,
) -> KGRetriever:
    """工厂函数：保持 retriever.py 的呼叫接口不變。"""
    return KGRetriever(
        kg_path=kg_path,
        language=language,
        max_hops=max_hops,
        docs_path=docs_path,
    )
