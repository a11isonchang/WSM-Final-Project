"""
Knowledge-Graph retriever using Think-on-Graph (ToG) method for JSONL/Python RAG pipelines.

This module implements ToG-style reasoning on knowledge graphs:
- Initialization: find seed entities mentioned in query
- Exploration: beam-search over the triple graph for up to max_hops
- Pruning: LLM-based pruning to keep most relevant paths (ToG method)
- Reasoning: LLM-based reasoning over graph paths to determine relevance
- Evidence: collect doc_id votes from triples on top reasoning paths

Based on: "Think-on-Graph: Deep and Responsible Reasoning of Large Language Model on Knowledge Graph" (ICLR 2024)
"""

from __future__ import annotations

import json
import re
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Iterable, DefaultDict
from collections import defaultdict

import jieba

# Ollama for LLM-based reasoning
try:
    from ollama import Client
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    Client = None


EN_TOKEN_PATTERN = re.compile(r"[a-z0-9']+")


def _safe_lower(x: str) -> str:
    return x.lower() if isinstance(x, str) else ""


def _tokenize_zh(text: str) -> List[str]:
    return [t.strip() for t in jieba.lcut(text) if t.strip()]


def _tokenize_en(text: str) -> List[str]:
    return EN_TOKEN_PATTERN.findall(text.lower())


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union > 0 else 0.0


@dataclass
class Edge:
    """A directed edge (triple) in the KG adjacency list."""
    src_id: str
    rel: str
    dst_id: str
    doc_id: int
    triple: Dict[str, Any]  # original triple (assists explain())


@dataclass
class Beam:
    """A reasoning path on the KG."""
    nodes: List[str]                  # entity ids
    edges: List[Edge]                 # traversed edges
    score: float                      # accumulated score

    @property
    def last(self) -> str:
        return self.nodes[-1] if self.nodes else ""


class KGRetriever:
    """
    ToG-style KG retriever that uses LLM-based reasoning on knowledge graphs.

    Notes:
    - This does NOT retrieve chunks directly. It ranks doc_ids, which your chunk retriever
      can then use to boost or filter chunks.
    - Uses LLM for pruning and reasoning (ToG method) instead of heuristic scoring.
    """

    def __init__(
        self,
        kg_path: str = "My_RAG/kg_output.json",
        language: str = "en",
        max_hops: int = 2,
        beam_width: int = 6,
        top_docs: int = 50,
        docs_path: Optional[str] = None,
        use_llm: bool = True,
        llm_model: Optional[str] = None,
    ):
        self.language = language or "en"
        self.max_hops = max(0, int(max_hops))
        self.beam_width = max(1, int(beam_width))
        self.top_docs = max(1, int(top_docs))
        self.use_llm = use_llm and OLLAMA_AVAILABLE

        self.kg_data = self._load_kg(kg_path)
        self.entities: Dict[str, Dict[str, Any]] = {
            e["id"]: e for e in self.kg_data.get("entities", []) if isinstance(e, dict) and e.get("id")
        }
        self.triples: List[Dict[str, Any]] = [
            t for t in self.kg_data.get("triples", []) if isinstance(t, dict)
        ]

        # Build indexes
        self._name_to_eids: DefaultDict[str, Set[str]] = defaultdict(set)
        for eid, e in self.entities.items():
            name = (e.get("name") or "").strip()
            if name:
                self._name_to_eids[_safe_lower(name)].add(eid)

        self._adj: DefaultDict[str, List[Edge]] = defaultdict(list)
        self._doc_to_triples: DefaultDict[int, List[Dict[str, Any]]] = defaultdict(list)

        for t in self.triples:
            h_id = str(t.get("head_id") or "").strip()
            ta_id = str(t.get("tail_id") or "").strip()
            rel = (t.get("relation") or "").strip()
            doc_id = t.get("doc_id")
            if not h_id or not ta_id or not rel or doc_id is None:
                continue
            try:
                doc_id_int = int(doc_id)
            except Exception:
                continue

            edge_fwd = Edge(src_id=h_id, rel=rel, dst_id=ta_id, doc_id=doc_id_int, triple=t)
            edge_rev = Edge(src_id=ta_id, rel=f"REV::{rel}", dst_id=h_id, doc_id=doc_id_int, triple=t)
            self._adj[h_id].append(edge_fwd)
            self._adj[ta_id].append(edge_rev)

            self._doc_to_triples[doc_id_int].append(t)

        # Optional: document text lookup (for later improvements)
        self.docs_path = docs_path

        # Initialize Ollama client for ToG reasoning
        self.ollama_client: Optional[Client] = None
        self.ollama_host: str = "http://localhost:11434"
        self.llm_model: str = llm_model or "granite4:3b"
        if self.use_llm:
            self._init_ollama()

        # runtime cache (for debug / explain)
        self._last_debug: Optional[Dict[str, Any]] = None

    # --------------------------
    # Ollama initialization
    # --------------------------
    def _init_ollama(self):
        """Initialize Ollama client for LLM-based reasoning."""
        if not OLLAMA_AVAILABLE:
            print("[WARN] Ollama not available, falling back to heuristic pruning")
            self.use_llm = False
            return

        try:
            # Load config for Ollama host and model
            try:
                from config import load_config
                config = load_config()
                ollama_cfg = config.get("ollama", {})
                self.ollama_host = ollama_cfg.get("host", "http://localhost:11434")
                self.llm_model = ollama_cfg.get("model", self.llm_model or "granite4:3b")
            except Exception:
                pass

            self.ollama_client = Client(host=self.ollama_host)
            print(f"[ToG] Initialized Ollama client: host={self.ollama_host}, model={self.llm_model}")
        except Exception as e:
            print(f"[WARN] Failed to initialize Ollama: {e}, falling back to heuristic pruning")
            self.use_llm = False

    # --------------------------
    # Loading & tokenization
    # --------------------------
    def _load_kg(self, kg_path: str) -> Dict[str, Any]:
        path = Path(kg_path)
        if not path.exists():
            raise FileNotFoundError(f"kg_output not found: {kg_path}")
        return json.loads(path.read_text(encoding="utf-8"))

    def _tokenize(self, text: str) -> List[str]:
        if self.language.startswith("zh"):
            return _tokenize_zh(text)
        return _tokenize_en(text)

    # --------------------------
    # Seed entity extraction
    # --------------------------
    def _extract_seed_entities(self, query: str, max_seeds: int = 6) -> List[Tuple[str, float]]:
        """
        Return a list of (entity_id, seed_score) detected from query.
        Priority:
        1) exact substring matches for entity names (case-insensitive)
        2) token overlap similarity fallback
        """
        q = query.strip()
        q_low = _safe_lower(q)
        seeds: List[Tuple[str, float]] = []

        # 1) substring matches for longer entity names first
        candidates: List[Tuple[str, str]] = []
        for name_low, eids in self._name_to_eids.items():
            if not name_low:
                continue
            if name_low in q_low and len(name_low) >= 2:
                for eid in eids:
                    candidates.append((eid, name_low))
        candidates.sort(key=lambda x: len(x[1]), reverse=True)

        seen: Set[str] = set()
        for eid, name_low in candidates:
            if eid in seen:
                continue
            seen.add(eid)
            # seed score grows with name length
            seeds.append((eid, min(1.0, 0.35 + 0.02 * len(name_low))))
            if len(seeds) >= max_seeds:
                return seeds

        # 2) fallback: token overlap between query and entity names
        q_tokens = set(self._tokenize(q))
        if not q_tokens:
            return seeds

        scored: List[Tuple[str, float]] = []
        for eid, e in self.entities.items():
            name = (e.get("name") or "").strip()
            if not name:
                continue
            e_tokens = set(self._tokenize(name))
            sim = _jaccard(q_tokens, e_tokens)
            if sim > 0:
                scored.append((eid, sim))
        scored.sort(key=lambda x: x[1], reverse=True)
        for eid, s in scored[:max_seeds]:
            seeds.append((eid, float(s) * 0.8))  # keep weaker than exact hit
        return seeds

    # --------------------------
    # Beam-search exploration (ToG-inspired)
    # --------------------------
    def _edge_relevance(self, query_tokens: Set[str], edge: Edge) -> float:
        """
        Score an edge against query by looking at:
        - relation tokens (without REV::)
        - destination entity name tokens
        - source entity name tokens (small weight)
        """
        rel = edge.rel.replace("REV::", "")
        rel_tokens = set(self._tokenize(rel))
        dst_name = (self.entities.get(edge.dst_id, {}).get("name") or "")
        src_name = (self.entities.get(edge.src_id, {}).get("name") or "")
        dst_tokens = set(self._tokenize(dst_name))
        src_tokens = set(self._tokenize(src_name))

        s_rel = _jaccard(query_tokens, rel_tokens)
        s_dst = _jaccard(query_tokens, dst_tokens)
        s_src = _jaccard(query_tokens, src_tokens)

        # weights tuned for "doc_id ranking" rather than KGQA correctness
        return 0.45 * s_rel + 0.45 * s_dst + 0.10 * s_src

    def _expand(self, beam: Beam, query_tokens: Set[str], hop: int) -> List[Beam]:
        """
        Expand one beam by one hop.
        Uses hop decay to prevent long noisy chains dominating.
        """
        out: List[Beam] = []
        decay = 0.75 ** max(0, hop - 1)

        for edge in self._adj.get(beam.last, []):
            # avoid immediate cycles (A->B->A)
            if len(beam.nodes) >= 2 and edge.dst_id == beam.nodes[-2]:
                continue
            # avoid repeating node too much
            if edge.dst_id in beam.nodes:
                continue

            edge_score = self._edge_relevance(query_tokens, edge)
            if edge_score <= 0:
                continue

            new_score = beam.score + decay * edge_score
            out.append(
                Beam(
                    nodes=beam.nodes + [edge.dst_id],
                    edges=beam.edges + [edge],
                    score=new_score,
                )
            )
        return out

    def _prune(self, beams: List[Beam], query: str = "") -> List[Beam]:
        """
        ToG-style pruning: use LLM to select most relevant paths if available,
        otherwise fall back to heuristic pruning.
        """
        if not self.use_llm or not self.ollama_client or not query:
            # Fallback to heuristic pruning
            return self._prune_heuristic(beams)

        return self._prune_with_llm(beams, query)

    def _prune_heuristic(self, beams: List[Beam]) -> List[Beam]:
        """Heuristic pruning: keep top beam_width by score, but diversify by last node."""
        beams_sorted = sorted(beams, key=lambda b: b.score, reverse=True)
        picked: List[Beam] = []
        used_last: Set[str] = set()
        for b in beams_sorted:
            if b.last in used_last:
                continue
            used_last.add(b.last)
            picked.append(b)
            if len(picked) >= self.beam_width:
                break
        return picked

    def _prune_with_llm(self, beams: List[Beam], query: str) -> List[Beam]:
        """
        ToG-style LLM-based pruning: ask LLM to select most relevant reasoning paths.
        """
        if len(beams) <= self.beam_width:
            return beams

        # Format paths for LLM
        path_descriptions = []
        for i, beam in enumerate(beams):
            path_str = self._format_path(beam)
            path_descriptions.append(f"Path {i+1}: {path_str}")

        # ToG pruning prompt
        if self.language.startswith("zh"):
            prompt = f"""你是一個知識圖譜推理系統。給定一個問題和一系列推理路徑，請選擇最相關的路徑來回答問題。

問題：{query}

推理路徑：
{chr(10).join(path_descriptions[:min(20, len(path_descriptions))])}

請返回最相關的 {self.beam_width} 個路徑的編號（用逗號分隔），例如：1,3,5
只返回數字，不要其他解釋。"""
        else:
            prompt = f"""You are a knowledge graph reasoning system. Given a question and a set of reasoning paths, select the most relevant paths to answer the question.

Question: {query}

Reasoning paths:
{chr(10).join(path_descriptions[:min(20, len(path_descriptions))])}

Return the numbers of the {self.beam_width} most relevant paths (comma-separated), e.g., 1,3,5
Return only numbers, no explanation."""

        try:
            # Use Ollama generate API
            full_prompt = f"You are a knowledge graph reasoning assistant.\n\n{prompt}"
            response = self.ollama_client.generate(
                model=self.llm_model,
                prompt=full_prompt,
                options={
                    "temperature": 0,
                    "num_predict": 50,  # max tokens
                }
            )

            content = response.get("response", "").strip()
            # Parse response to get path indices
            selected_indices = []
            for part in content.split(","):
                try:
                    idx = int(part.strip()) - 1  # Convert to 0-based
                    if 0 <= idx < len(beams):
                        selected_indices.append(idx)
                except ValueError:
                    continue

            if selected_indices:
                return [beams[i] for i in selected_indices if i < len(beams)][:self.beam_width]

        except Exception as e:
            print(f"[WARN] LLM pruning failed: {e}, falling back to heuristic")

        # Fallback to heuristic
        return self._prune_heuristic(beams)

    # --------------------------
    # Public APIs
    # --------------------------
    def retrieve_doc_ids(
        self,
        query: str,
        top_k: int = 50,
        use_multi_hop: bool = True,
    ) -> List[int]:
        """Backward-compatible API: returns doc_ids only (ranked)."""
        scored = self.rank_docs(query=query, top_k=top_k, use_multi_hop=use_multi_hop)
        return [d["doc_id"] for d in scored]

    def rank_docs(
        self,
        query: str,
        top_k: int = 50,
        use_multi_hop: bool = True,
        return_debug: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Main API:
        Returns a list of dicts:
          {doc_id, score, evidence_triples, paths}

        score is in [0,1] (normalized across returned docs).
        """
        top_k = max(1, int(top_k))
        q_tokens = set(self._tokenize(query))
        seeds = self._extract_seed_entities(query)

        debug: Dict[str, Any] = {
            "query": query,
            "language": self.language,
            "seeds": [
                {"entity_id": eid, "name": self.entities.get(eid, {}).get("name"), "score": s}
                for eid, s in seeds
            ],
            "beams": [],
            "doc_scores_raw": {},
            "doc_scores_norm": {},
        }

        if not seeds:
            self._last_debug = debug if return_debug else None
            return []

        # init beams
        beams: List[Beam] = [Beam(nodes=[eid], edges=[], score=float(s)) for eid, s in seeds]
        beams = self._prune(beams, query)

        # explore
        max_hops = self.max_hops if use_multi_hop else min(self.max_hops, 1)
        all_beams: List[Beam] = list(beams)

        for hop in range(1, max_hops + 1):
            expanded: List[Beam] = []
            for b in beams:
                expanded.extend(self._expand(b, q_tokens, hop=hop))
            if not expanded:
                break
            beams = self._prune(expanded, query)  # Pass query for LLM pruning
            all_beams.extend(beams)

        # ToG reasoning: use LLM to reason over paths and score docs
        if self.use_llm and self.ollama_client:
            doc_scores, doc_evidence, doc_paths = self._reason_with_llm(all_beams, query, q_tokens)
        else:
            doc_scores, doc_evidence, doc_paths = self._reason_heuristic(all_beams, q_tokens)

        # doc_scores, doc_evidence, doc_paths are now set by _reason_with_llm or _reason_heuristic
        if not doc_scores:
            self._last_debug = debug if return_debug else None
            return []

        # normalize
        max_s = max(doc_scores.values()) or 1.0
        min_s = min(doc_scores.values()) if len(doc_scores) > 1 else 0.0

        scored_docs: List[Dict[str, Any]] = []
        for doc_id, s in doc_scores.items():
            if max_s - min_s < 1e-9:
                s_norm = 1.0
            else:
                s_norm = (s - min_s) / (max_s - min_s)
            scored_docs.append(
                {
                    "doc_id": int(doc_id),
                    "score": float(max(0.0, min(1.0, s_norm))),
                    "evidence_triples": doc_evidence[doc_id],
                    "paths": doc_paths.get(doc_id, [])[:5],
                }
            )

        scored_docs.sort(key=lambda x: x["score"], reverse=True)
        scored_docs = scored_docs[: min(top_k, self.top_docs, len(scored_docs))]

        if return_debug:
            debug["beams"] = [
                {"score": b.score, "path": self._format_path(b), "hops": len(b.edges)}
                for b in sorted(all_beams, key=lambda x: x.score, reverse=True)[: min(20, len(all_beams))]
            ]
            debug["doc_scores_raw"] = dict(sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:50])
            debug["doc_scores_norm"] = {d["doc_id"]: d["score"] for d in scored_docs}
            self._last_debug = debug
        else:
            self._last_debug = None

        return scored_docs

    def _reason_heuristic(
        self, 
        all_beams: List[Beam], 
        q_tokens: Set[str]
    ) -> Tuple[DefaultDict[int, float], DefaultDict[int, List[Dict[str, Any]]], DefaultDict[int, List[str]]]:
        """
        Heuristic reasoning: vote docs from beams' edges using lexical similarity.
        """
        doc_scores: DefaultDict[int, float] = defaultdict(float)
        doc_evidence: DefaultDict[int, List[Dict[str, Any]]] = defaultdict(list)
        doc_paths: DefaultDict[int, List[str]] = defaultdict(list)

        for b in all_beams:
            if not b.edges:
                continue
            # path contribution is b.score, but attribute per edge/doc with mild edge factor
            for edge in b.edges:
                contrib = max(0.0, b.score) * 0.6 + self._edge_relevance(q_tokens, edge) * 0.4
                doc_scores[edge.doc_id] += contrib

                # Keep a few evidence triples for this doc
                if len(doc_evidence[edge.doc_id]) < 10:
                    doc_evidence[edge.doc_id].append(edge.triple)

            # Human readable path
            if b.edges:
                path_str = self._format_path(b)
                # assign to last edge doc (most specific)
                doc_paths[b.edges[-1].doc_id].append(path_str)

        return doc_scores, doc_evidence, doc_paths

    def _reason_with_llm(
        self,
        all_beams: List[Beam],
        query: str,
        q_tokens: Set[str]
    ) -> Tuple[DefaultDict[int, float], DefaultDict[int, List[Dict[str, Any]]], DefaultDict[int, List[str]]]:
        """
        ToG-style LLM reasoning: ask LLM to reason over paths and determine document relevance.
        """
        # Group paths by doc_id and format for LLM
        doc_paths_dict: DefaultDict[int, List[str]] = defaultdict(list)
        doc_evidence: DefaultDict[int, List[Dict[str, Any]]] = defaultdict(list)
        
        for b in all_beams[:min(30, len(all_beams))]:  # Limit for token efficiency
            if not b.edges:
                continue
            path_str = self._format_path(b)
            for edge in b.edges:
                doc_id = edge.doc_id
                doc_paths_dict[doc_id].append(path_str)
                if len(doc_evidence[doc_id]) < 10:
                    doc_evidence[doc_id].append(edge.triple)

        # Build reasoning prompt
        doc_summaries = []
        for doc_id, paths in list(doc_paths_dict.items())[:20]:  # Limit to top 20 docs
            unique_paths = list(set(paths))[:5]  # Top 5 unique paths per doc
            doc_summaries.append(f"Document {doc_id}:\n  Paths: {'; '.join(unique_paths)}")

        if self.language.startswith("zh"):
            prompt = f"""你是一個知識圖譜推理系統。給定一個問題和一系列文檔的推理路徑，請評估每個文檔與問題的相關性。

問題：{query}

文檔及其推理路徑：
{chr(10).join(doc_summaries)}

請為每個文檔評分（0-1之間），格式為：文檔編號:分數
例如：
1:0.8
2:0.3
3:0.9

只返回評分，每行一個文檔。"""
        else:
            prompt = f"""You are a knowledge graph reasoning system. Given a question and reasoning paths for various documents, evaluate the relevance of each document to the question.

Question: {query}

Documents and their reasoning paths:
{chr(10).join(doc_summaries)}

Score each document (0-1), format: document_number:score
Example:
1:0.8
2:0.3
3:0.9

Return only scores, one document per line."""

        doc_scores: DefaultDict[int, float] = defaultdict(float)

        try:
            # Use Ollama generate API
            full_prompt = f"You are a knowledge graph reasoning assistant.\n\n{prompt}"
            response = self.ollama_client.generate(
                model=self.llm_model,
                prompt=full_prompt,
                options={
                    "temperature": 0,
                    "num_predict": 500,  # max tokens
                }
            )

            content = response.get("response", "").strip()
            # Parse scores
            for line in content.split("\n"):
                line = line.strip()
                if ":" in line:
                    try:
                        parts = line.split(":", 1)
                        doc_id = int(parts[0].strip().replace("Document", "").strip())
                        score = float(parts[1].strip())
                        if 0 <= score <= 1:
                            doc_scores[doc_id] = score
                    except (ValueError, IndexError):
                        continue

        except Exception as e:
            print(f"[WARN] LLM reasoning failed: {e}, falling back to heuristic")
            return self._reason_heuristic(all_beams, q_tokens)

        # If LLM didn't return enough scores, supplement with heuristic
        if len(doc_scores) < len(doc_paths_dict) * 0.5:
            heuristic_scores, _, _ = self._reason_heuristic(all_beams, q_tokens)
            for doc_id, score in heuristic_scores.items():
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = score * 0.7  # Slightly lower weight for heuristic

        return doc_scores, doc_evidence, doc_paths_dict

    def get_entity_info(self, query: str, use_multi_hop: bool = True) -> Dict[str, Any]:
        """
        Used by retriever.py debug printing.

        Returns:
          {
            entities: [entity_obj...],
            doc_ids: [int...],
            ranked_docs: [{doc_id, score, paths, evidence_triples}...],
            multi_hop: {max_hops, beam_width, ...}
          }
        """
        ranked = self.rank_docs(query=query, top_k=self.top_docs, use_multi_hop=use_multi_hop, return_debug=True)
        doc_ids = [d["doc_id"] for d in ranked]
        seed_ids = [s["entity_id"] for s in (self._last_debug or {}).get("seeds", [])]
        entities = [self.entities.get(eid) for eid in seed_ids if eid in self.entities]

        return {
            "entities": entities,
            "doc_ids": doc_ids,
            "ranked_docs": ranked[:10],
            "multi_hop": {
                "enabled": bool(use_multi_hop),
                "max_hops": self.max_hops if use_multi_hop else min(self.max_hops, 1),
                "beam_width": self.beam_width,
            },
            "debug": self._last_debug,
        }

    def explain_doc(self, doc_id: int, max_triples: int = 10) -> Dict[str, Any]:
        """Return evidence triples for a given doc_id."""
        doc_id = int(doc_id)
        triples = self._doc_to_triples.get(doc_id, [])[:max_triples]
        return {"doc_id": doc_id, "triples": triples}

    # --------------------------
    # Helpers
    # --------------------------
    def _format_path(self, beam: Beam) -> str:
        if not beam.nodes:
            return ""
        parts: List[str] = []
        for i, nid in enumerate(beam.nodes):
            name = self.entities.get(nid, {}).get("name", nid)
            parts.append(str(name))
            if i < len(beam.edges):
                rel = beam.edges[i].rel.replace("REV::", "")
                parts.append(f"-[{rel}]->")
        return " ".join(parts)


def create_kg_retriever(
    kg_path: str = "My_RAG/kg_output.json",
    language: str = "en",
    max_hops: int = 2,
    docs_path: Optional[str] = None,
    use_llm: bool = True,
    llm_model: Optional[str] = None,
) -> KGRetriever:
    """
    Factory function: keep retriever.py call-site unchanged.

    You can tune defaults here without touching retriever.py.
    
    Args:
        use_llm: Enable ToG-style LLM-based reasoning (default: True)
        llm_model: OpenAI model to use (default: from config or gpt-4o-mini)
    """
    return KGRetriever(
        kg_path=kg_path,
        language=language,
        max_hops=max_hops,
        beam_width=6,
        top_docs=50,
        docs_path=docs_path,
        use_llm=use_llm,
        llm_model=llm_model,
    )
