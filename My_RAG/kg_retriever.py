# My_RAG/kg_retriever.py

import json
from pathlib import Path
from typing import Dict, List, Any, Tuple, Set
from collections import defaultdict, deque
from functools import lru_cache
import jieba
import re

EN_TOKEN_PATTERN = re.compile(r"[a-z0-9']+", re.IGNORECASE)


class KGRetriever:
    """
    Relation-aware、多跳的 KG 檢索器。

    讀取 build_kg.py 產生的 kg_output.json：
    {
      "entities": [...],
      "relations": [...],
      "triples": [
        {
          "head": "...",
          "head_id": "e1",
          "head_type": "Company",
          "relation": "ACQUIRES",
          "tail": "...",
          "tail_id": "e2",
          "tail_type": "Company",
          "doc_id": 12,
          "properties": {...}
        },
        ...
      ],
      ...
    }

    功能：
    - 依 triples 建 entity graph：entity_id <-> entity_id (with relation, doc_id)
    - 依 query 解析出相關實體 → 多源 BFS（最多 max_hops）
    - 輸出：依 KG 分數排序的 doc_id list
    """

    def __init__(
        self,
        kg_path: str,
        language: str = "zh",
        max_hops: int = 2,
        alpha_entity: float = 1.0,
        beta_cooccur: float = 2.0,
        gamma_path: float = 1.0,
    ):
        self.language = language
        self.max_hops = max_hops
        self.alpha = alpha_entity
        self.beta = beta_cooccur
        self.gamma = gamma_path

        # ---- 從 kg_output.json 載入資料 ----
        path = Path(kg_path)
        if not path.exists():
            raise FileNotFoundError(f"KG file not found: {kg_path}")

        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        self.entities: List[Dict[str, Any]] = data.get("entities", [])
        self.triples: List[Dict[str, Any]] = data.get("triples", [])
        self.relations: List[Dict[str, Any]] = data.get("relations", [])

        if not self.triples:
            print("[KGRetriever] Warning: 'triples' is empty in kg_output.json. "
                  "Make sure you are using the updated build_kg.py that outputs triples.")

        # ---- entity_id -> {name, type} ----
        self.entity_info: Dict[str, Dict[str, Any]] = {}
        for ent in self.entities:
            eid = ent.get("id")
            if not eid:
                continue
            self.entity_info[eid] = {
                "name": ent.get("name", ""),
                "type": ent.get("type", ""),
                "properties": ent.get("properties", {}) or {},
            }

        # ---- entity_name 索引（用來從 query 找實體）----
        # 有些名字可能重複，這裡允許一個 name 對應多個 id
        self.name_to_ids: Dict[str, Set[str]] = defaultdict(set)
        for eid, info in self.entity_info.items():
            name = (info.get("name") or "").strip()
            if name:
                self.name_to_ids[name].add(eid)

        # ---- entity graph + entity_docs ----
        self.graph: Dict[str, List[Tuple[str, str, int]]] = defaultdict(list)
        # entity_id -> set(doc_id)
        self.entity_docs: Dict[str, Set[int]] = defaultdict(set)

        for tri in self.triples:
            h_id = tri.get("head_id")
            t_id = tri.get("tail_id")
            rel_type = tri.get("relation", "")
            doc_id = tri.get("doc_id")

            if not h_id or not t_id:
                continue

            # 無 doc_id 的 triple 對檢索幫助不大，可以跳過
            if doc_id is None:
                continue

            # 當作無向圖：head <-> tail
            self.graph[h_id].append((t_id, rel_type, doc_id))
            self.graph[t_id].append((h_id, rel_type, doc_id))

            self.entity_docs[h_id].add(doc_id)
            self.entity_docs[t_id].add(doc_id)

        print(
            f"[KGRetriever] Loaded {len(self.entities)} entities, "
            f"{len(self.relations)} relations, {len(self.triples)} triples "
            f"from {kg_path}"
        )

    # ---------- Query 解析：從 query 抓出 KG 實體 ----------

    def _tokenize_query(self, query: str) -> Tuple[str, List[str]]:
        if self.language == "zh":
            q_text = (query or "").strip()
            tokens = [t.strip() for t in jieba.cut(q_text) if t.strip()]
        else:
            q_text = (query or "").lower()
            tokens = EN_TOKEN_PATTERN.findall(q_text)
        return q_text, tokens

    def _extract_query_entities(self, query: str) -> List[str]:
        """
        從 query 中找到可能對應到 KG entity 的 entity_id 列表。

        策略：
        - 中文：用 entity name 做 substring 比對（跳過太短的名字）
        - 英文：lowercase 後，用 entity name 的 lowercase 做 substring
        """
        query = (query or "").strip()
        if not query:
            return []

        q_text, tokens = self._tokenize_query(query)
        q_entities: Set[str] = set()

        if self.language == "zh":
            for name, ids in self.name_to_ids.items():
                n = name.strip()
                # 避免單字或一個字的 noise
                if len(n) < 2:
                    continue
                if n in q_text:
                    q_entities.update(ids)
        else:
            q_lower = q_text.lower()
            for name, ids in self.name_to_ids.items():
                n = (name or "").strip()
                if not n:
                    continue
                n_lower = n.lower()
                # 名字本身太短也容易誤中，略過長度 < 3 的
                if len(n_lower) < 3:
                    continue
                if n_lower in q_lower:
                    q_entities.update(ids)

        return list(q_entities)

    # ---------- 多源 BFS：從 query entities 出發找 doc ----------

    def _bfs_docs_from_entities(self, start_entities: List[str]) -> Dict[int, float]:
        """
        從多個實體同時出發做 BFS，最多走 self.max_hops。
        回傳：doc_id -> KG BFS 分數

        距離 d 的 entity / edge 對應的 doc，貢獻 1/(1+d)
        """
        if not start_entities:
            return {}

        doc_scores: Dict[int, float] = defaultdict(float)
        q: deque = deque()
        dist_map: Dict[str, int] = {}

        for eid in start_entities:
            if eid not in self.graph and eid not in self.entity_docs:
                continue
            q.append((eid, 0))
            dist_map[eid] = 0

        while q:
            eid, dist = q.popleft()
            if dist > self.max_hops:
                continue

            # 這個節點本身出現的所有 doc
            for doc_id in self.entity_docs.get(eid, []):
                doc_scores[doc_id] += 1.0 / (1.0 + dist)

            # 往 neighbor 展開
            for neighbor_id, rel_type, doc_id in self.graph.get(eid, []):
                next_dist = dist + 1
                if next_dist > self.max_hops:
                    continue
                if neighbor_id not in dist_map or next_dist < dist_map[neighbor_id]:
                    dist_map[neighbor_id] = next_dist
                    q.append((neighbor_id, next_dist))

        return doc_scores

    # ---------- 主 API：依 KG 分數排序 doc_id ----------

    def retrieve_doc_ids(self, query: str, top_k: int = 10) -> List[int]:
        """
        主入口：給 query，回傳依 KG 重要性排序的 doc_id 清單。

        KG 分數由三部分組成：
        1. entity match：doc 裡包含多少個 query 相關實體 → alpha * count
        2. co-occurrence：若 doc 同時包含 >=2 個 query 實體 → beta * (count-1)
        3. multi-hop BFS：從 query entities 出發，用 1/(1+hop) 累加 → gamma * bfs_score
        """
        query = (query or "").strip()
        if not query:
            return []

        # 1) 找出 query 中的 KG 實體
        q_entities = self._extract_query_entities(query)
        if not q_entities:
            return []

        # 2) entity-level scoring
        doc_entity_count: Dict[int, int] = defaultdict(int)

        for eid in q_entities:
            for doc_id in self.entity_docs.get(eid, []):
                doc_entity_count[doc_id] += 1

        doc_scores: Dict[int, float] = defaultdict(float)
        for doc_id, cnt in doc_entity_count.items():
            # 每個 doc 中命中的 query-related entity 數
            doc_scores[doc_id] += self.alpha * cnt
            if cnt >= 2:
                doc_scores[doc_id] += self.beta * (cnt - 1)

        # 3) BFS Multi-hop scoring
        bfs_scores = self._bfs_docs_from_entities(q_entities)
        for doc_id, s in bfs_scores.items():
            doc_scores[doc_id] += self.gamma * s

        if not doc_scores:
            return []

        # 4) 排序後取前 top_k
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        top_docs = [doc_id for doc_id, _ in sorted_docs[:top_k]]

        return top_docs


@lru_cache()
def create_kg_retriever(kg_path: str, language: str = "zh") -> KGRetriever:
    """
    工廠函式（帶快取），和 retriever.py 中的 import 介面一致：
    from kg_retriever import create_kg_retriever
    """
    return KGRetriever(kg_path=kg_path, language=language)
