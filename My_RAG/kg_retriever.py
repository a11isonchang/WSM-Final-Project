"""
基于知识图谱的检索器
通过kg_output.json中的实体和关系，根据query检索相关的文档
"""

import json
import jieba
import re
from typing import List, Dict, Any, Set, Optional
from pathlib import Path
from functools import lru_cache


class KGRetriever:
    """基于知识图谱的检索器，用于提升检索精度"""
    
    def __init__(self, kg_path: str = "My_RAG/kg_output.json", language: str = "en"):
        """
        初始化知识图谱检索器
        
        Args:
            kg_path: 知识图谱文件路径
            language: 语言类型 ("zh" 或 "en")
        """
        self.language = language
        self.kg_data = self._load_kg(kg_path)
        self.entity_index = self._build_entity_index()
        self.entity_to_docs = self._build_entity_doc_mapping()
        self.relation_index = self._build_relation_index()
        
    def _load_kg(self, kg_path: str) -> Dict[str, Any]:
        """加载知识图谱JSON文件"""
        path = Path(kg_path)
        if not path.exists():
            print(f"Warning: KG file not found at {kg_path}")
            return {"entities": [], "relations": []}
        
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _build_entity_index(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        建立实体名称到实体对象的索引
        支持模糊匹配（包含关系）
        """
        index = {}
        entities = self.kg_data.get("entities", [])
        
        for entity in entities:
            name = entity.get("name", "").strip()
            if not name:
                continue
            
            # 直接匹配
            if name not in index:
                index[name] = []
            index[name].append(entity)
            
            # 对于中文，也添加分词后的关键词
            if self.language == "zh":
                # 提取关键词（去除常见停用词）
                tokens = list(jieba.cut(name))
                for token in tokens:
                    if len(token) >= 2:  # 至少2个字符
                        if token not in index:
                            index[token] = []
                        if entity not in index[token]:
                            index[token].append(entity)
        
        return index
    
    def _build_entity_doc_mapping(self) -> Dict[str, Set[int]]:
        """
        建立实体ID到doc_id集合的映射
        通过relations找到实体关联的文档
        """
        entity_to_docs = {}
        relations = self.kg_data.get("relations", [])
        
        for relation in relations:
            source_id = relation.get("source")
            target_id = relation.get("target")
            doc_id = relation.get("doc_id")
            
            if doc_id is not None:
                if source_id not in entity_to_docs:
                    entity_to_docs[source_id] = set()
                entity_to_docs[source_id].add(doc_id)
                
                if target_id not in entity_to_docs:
                    entity_to_docs[target_id] = set()
                entity_to_docs[target_id].add(doc_id)
        
        return entity_to_docs
    
    def _build_relation_index(self) -> Dict[str, List[Dict[str, Any]]]:
        """建立关系类型索引，用于更精确的检索"""
        index = {}
        relations = self.kg_data.get("relations", [])
        
        for relation in relations:
            rel_type = relation.get("type", "")
            if rel_type:
                if rel_type not in index:
                    index[rel_type] = []
                index[rel_type].append(relation)
        
        return index
    
    def _extract_entities_from_query(self, query: str) -> List[Dict[str, Any]]:
        """
        从query中提取实体
        使用字符串匹配和关键词匹配
        """
        found_entities = []
        query_lower = query.lower() if self.language == "en" else query
        
        # 1. 精确匹配实体名称
        for entity_name, entities in self.entity_index.items():
            if entity_name in query_lower:
                found_entities.extend(entities)
        
        # 2. 对于中文，使用分词匹配
        if self.language == "zh":
            query_tokens = list(jieba.cut(query))
            for token in query_tokens:
                if len(token) >= 2 and token in self.entity_index:
                    found_entities.extend(self.entity_index[token])
        
        # 去重（基于entity id）
        seen_ids = set()
        unique_entities = []
        for entity in found_entities:
            entity_id = entity.get("id")
            if entity_id and entity_id not in seen_ids:
                seen_ids.add(entity_id)
                unique_entities.append(entity)
        
        return unique_entities
    
    def _extract_keywords_from_query(self, query: str) -> Set[str]:
        """从query中提取关键词，用于匹配实体属性"""
        keywords = set()
        
        if self.language == "zh":
            tokens = list(jieba.cut(query))
            # 提取长度>=2的词
            keywords = {t for t in tokens if len(t) >= 2}
        else:
            # 英文：提取单词
            words = re.findall(r'\b\w+\b', query.lower())
            keywords = {w for w in words if len(w) >= 3}
        
        return keywords
    
    def retrieve_doc_ids(self, query: str, top_k: int = 10) -> List[int]:
        """
        根据query检索相关的doc_id
        
        Args:
            query: 查询文本
            top_k: 返回的doc_id数量上限
            
        Returns:
            相关的doc_id列表，按相关性排序
        """
        # 1. 从query中提取实体
        entities = self._extract_entities_from_query(query)
        
        # 2. 收集所有相关的doc_id
        doc_id_scores = {}  # doc_id -> score
        
        for entity in entities:
            entity_id = entity.get("id")
            if entity_id in self.entity_to_docs:
                # 实体直接匹配：高分
                for doc_id in self.entity_to_docs[entity_id]:
                    doc_id_scores[doc_id] = doc_id_scores.get(doc_id, 0) + 2.0
        
        # 3. 通过关系查找相关文档
        keywords = self._extract_keywords_from_query(query)
        relations = self.kg_data.get("relations", [])
        
        for relation in relations:
            rel_type = relation.get("type", "").lower()
            source_id = relation.get("source")
            target_id = relation.get("target")
            doc_id = relation.get("doc_id")
            
            if doc_id is None:
                continue
            
            # 检查关系类型是否与query相关
            if any(kw in rel_type for kw in keywords):
                doc_id_scores[doc_id] = doc_id_scores.get(doc_id, 0) + 1.5
            
            # 检查关系属性是否与query相关
            rel_props = relation.get("properties", {})
            for prop_key, prop_value in rel_props.items():
                if isinstance(prop_value, str):
                    if any(kw in prop_value for kw in keywords):
                        doc_id_scores[doc_id] = doc_id_scores.get(doc_id, 0) + 1.0
        
        # 4. 检查实体属性是否与query相关
        for entity in entities:
            props = entity.get("properties", {})
            keywords = self._extract_keywords_from_query(query)
            
            for prop_key, prop_value in props.items():
                if isinstance(prop_value, str):
                    if any(kw in prop_value for kw in keywords):
                        entity_id = entity.get("id")
                        if entity_id in self.entity_to_docs:
                            for doc_id in self.entity_to_docs[entity_id]:
                                doc_id_scores[doc_id] = doc_id_scores.get(doc_id, 0) + 0.5
        
        # 5. 按分数排序并返回top_k
        sorted_docs = sorted(doc_id_scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_id for doc_id, score in sorted_docs[:top_k]]
    
    def get_entity_info(self, query: str) -> Dict[str, Any]:
        """
        获取query中提取到的实体信息（用于调试）
        """
        entities = self._extract_entities_from_query(query)
        doc_ids = self.retrieve_doc_ids(query, top_k=10)
        
        return {
            "entities_found": [
                {
                    "id": e.get("id"),
                    "name": e.get("name"),
                    "type": e.get("type")
                }
                for e in entities
            ],
            "related_doc_ids": doc_ids,
            "entity_count": len(entities)
        }


@lru_cache(maxsize=1)
def create_kg_retriever(kg_path: str = "My_RAG/kg_output.json", language: str = "en") -> KGRetriever:
    """
    创建并缓存知识图谱检索器（单例模式）
    """
    return KGRetriever(kg_path, language)

