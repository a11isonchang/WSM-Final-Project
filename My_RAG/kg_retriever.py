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
    
    def __init__(self, kg_path: str = "My_RAG/kg_output.json", language: str = "en", max_hops: int = 2, docs_path: str = None):
        """
        初始化知识图谱检索器
        
        Args:
            kg_path: 知识图谱文件路径
            language: 语言类型 ("zh" 或 "en")
            max_hops: 最大跳数，用于多跳关系检索（默认2跳）
            docs_path: 文档文件路径（可选，用于通过company_name匹配）
        """
        self.language = language
        self.max_hops = max_hops
        self.kg_data = self._load_kg(kg_path)
        self.entity_index = self._build_entity_index()
        self.entity_to_docs = self._build_entity_doc_mapping()
        self.relation_index = self._build_relation_index()
        self.entity_graph = self._build_entity_graph()  # 实体关系图，用于多跳检索
        
        # 建立实体名称到doc_id的直接映射（通过文档的company_name字段）
        self.entity_name_to_docs = self._build_entity_name_doc_mapping(docs_path)
        
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
        改进：英文支持不区分大小写匹配
        """
        index = {}
        entities = self.kg_data.get("entities", [])
        
        for entity in entities:
            name = entity.get("name", "").strip()
            if not name:
                continue
            
            # 直接匹配（原始名称）
            if name not in index:
                index[name] = []
            index[name].append(entity)
            
            # 对于英文，添加小写版本用于不区分大小写匹配
            if self.language == "en":
                name_lower = name.lower()
                if name_lower != name:  # 如果不同，添加小写版本
                    if name_lower not in index:
                        index[name_lower] = []
                    if entity not in index[name_lower]:
                        index[name_lower].append(entity)
                
                # 提取公司名称的关键部分（去掉Ltd., Inc.等后缀）
                import re
                # 去掉常见后缀
                base_name = re.sub(r'\s+(Ltd\.?|Inc\.?|Corp\.?|LLC|Co\.?|Company|Services|Group)$', '', name, flags=re.IGNORECASE)
                if base_name != name and len(base_name) >= 3:
                    base_name_lower = base_name.lower()
                    if base_name_lower not in index:
                        index[base_name_lower] = []
                    if entity not in index[base_name_lower]:
                        index[base_name_lower].append(entity)
            
            # 对于中文，也添加分词后的关键词（但只添加较长的词）
            if self.language == "zh":
                # 提取关键词（去除常见停用词）
                tokens = list(jieba.cut(name))
                for token in tokens:
                    # 只索引长度>=3的词，避免索引太短的词导致误匹配
                    if len(token) >= 3:
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
    
    def _build_entity_name_doc_mapping(self, docs_path: Optional[str] = None) -> Dict[str, Set[int]]:
        """
        建立实体名称到doc_id的映射（通过文档的company_name字段）
        用于处理KG中实体没有关系但文档中有company_name的情况
        """
        entity_name_to_docs = {}
        
        if not docs_path:
            return entity_name_to_docs
        
        try:
            from utils import load_jsonl
            docs = load_jsonl(docs_path)
            
            for doc in docs:
                doc_id = doc.get("doc_id")
                company_name = doc.get("company_name", "").strip()
                
                if doc_id is not None and company_name:
                    # 直接匹配
                    if company_name not in entity_name_to_docs:
                        entity_name_to_docs[company_name] = set()
                    entity_name_to_docs[company_name].add(doc_id)
                    
                    # 对于英文，也添加小写版本
                    if self.language == "en":
                        company_lower = company_name.lower()
                        if company_lower != company_name:
                            if company_lower not in entity_name_to_docs:
                                entity_name_to_docs[company_lower] = set()
                            entity_name_to_docs[company_lower].add(doc_id)
        except Exception as e:
            print(f"Warning: Failed to build entity_name_to_docs mapping: {e}")
        
        return entity_name_to_docs
    
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
    
    def _build_entity_graph(self) -> Dict[str, Set[str]]:
        """
        构建实体关系图（邻接表）
        用于多跳关系检索
        
        Returns:
            Dict[entity_id, Set[connected_entity_ids]]
        """
        graph = {}
        relations = self.kg_data.get("relations", [])
        
        for relation in relations:
            source_id = relation.get("source")
            target_id = relation.get("target")
            
            if source_id and target_id:
                # 双向图（无向图）
                if source_id not in graph:
                    graph[source_id] = set()
                graph[source_id].add(target_id)
                
                if target_id not in graph:
                    graph[target_id] = set()
                graph[target_id].add(source_id)
        
        return graph
    
    def _is_valid_entity_name(self, name: str) -> bool:
        """
        判断是否是有效的实体名称（过滤掉年份、通用词等）
        """
        if not name or len(name) < 2:
            return False
        
        # 过滤纯数字（年份）
        if name.isdigit() and (len(name) == 4 or len(name) == 2):
            return False
        
        # 过滤常见的通用词（英文）
        if self.language == "en":
            common_words = {
                "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
                "of", "with", "by", "from", "as", "is", "was", "are", "were", "be",
                "been", "have", "has", "had", "do", "does", "did", "will", "would",
                "should", "could", "may", "might", "can", "must", "this", "that",
                "these", "those", "what", "which", "who", "when", "where", "why",
                "how", "all", "each", "every", "some", "any", "no", "not", "more",
                "most", "many", "much", "few", "little", "other", "another", "such",
                "only", "just", "also", "even", "still", "already", "yet", "again",
                "here", "there", "where", "now", "then", "today", "yesterday",
                "tomorrow", "year", "years", "month", "months", "day", "days",
                "time", "times", "way", "ways", "thing", "things", "people",
                "person", "persons", "man", "men", "woman", "women", "child",
                "children", "company", "companies", "business", "report", "reports"
            }
            if name.lower() in common_words:
                return False
        
        return True
    
    def _extract_company_names_en(self, query: str) -> List[str]:
        """
        从英文查询中提取公司名称
        识别包含Ltd., Inc., Corp., LLC等后缀的公司名称
        """
        import re
        company_patterns = [
            r'([A-Z][a-zA-Z\s&]+(?:Ltd\.?|Inc\.?|Corp\.?|LLC|Co\.?|Company|Services|Group))',
            r'([A-Z][a-zA-Z\s&]+(?:Housekeeping|Agriculture|Emporium|Hospital|General))',
        ]
        
        company_names = []
        for pattern in company_patterns:
            matches = re.findall(pattern, query)
            for match in matches:
                name = match.strip()
                if len(name) >= 5:  # 至少5个字符
                    company_names.append(name)
                    # 也添加去掉后缀的版本
                    for suffix in [' Ltd.', ' Ltd', ' Inc.', ' Inc', ' Corp.', ' Corp', ' LLC', ' Co.', ' Co']:
                        if name.endswith(suffix):
                            base_name = name[:-len(suffix)].strip()
                            if len(base_name) >= 3:
                                company_names.append(base_name)
        
        return list(set(company_names))
    
    def _extract_company_names_zh(self, query: str) -> List[str]:
        """
        从中文查询中提取公司名称
        识别包含"有限公司"、"股份有限公司"等后缀的公司名称
        """
        import re
        # 匹配中文公司名称模式：X公司、X有限公司、X股份有限公司等
        patterns = [
            r'([\u4e00-\u9fa5]+(?:有限公司|股份有限公司|公司|集团|企业))',
            r'([\u4e00-\u9fa5]{2,}(?:服务|科技|娱乐|环保|农业|购物|家政|传媒))',
        ]
        
        company_names = []
        for pattern in patterns:
            matches = re.findall(pattern, query)
            for match in matches:
                name = match.strip()
                if len(name) >= 4:  # 至少4个字符（中文）
                    company_names.append(name)
        
        return list(set(company_names))
    
    def _extract_entities_from_query(self, query: str) -> List[Dict[str, Any]]:
        """
        从query中提取实体
        改进版本：优先匹配完整实体名称，然后才是部分匹配
        """
        found_entities = []
        entity_scores = {}  # entity_id -> score (用于排序，完整匹配得分更高)
        
        query_lower = query.lower() if self.language == "en" else query
        
        # 1. 优先：提取并匹配公司名称（完整匹配，高优先级）
        if self.language == "en":
            company_names = self._extract_company_names_en(query)
        else:
            company_names = self._extract_company_names_zh(query)
        
        for company_name in company_names:
            # 尝试完整匹配
            if company_name in self.entity_index:
                for entity in self.entity_index[company_name]:
                    entity_id = entity.get("id")
                    if entity_id:
                        entity_scores[entity_id] = entity_scores.get(entity_id, 0) + 10.0  # 完整匹配高分
                        found_entities.append(entity)
            
            # 尝试不区分大小写的匹配（英文）
            if self.language == "en":
                company_lower = company_name.lower()
                for entity_name, entities in self.entity_index.items():
                    if entity_name.lower() == company_lower:
                        for entity in entities:
                            entity_id = entity.get("id")
                            if entity_id:
                                entity_scores[entity_id] = entity_scores.get(entity_id, 0) + 10.0
                                found_entities.append(entity)
        
        # 2. 精确匹配实体名称（完整实体名称在query中）
        for entity_name, entities in self.entity_index.items():
            if not self._is_valid_entity_name(entity_name):
                continue
            
            # 检查实体名称是否在query中
            if self.language == "en":
                if entity_name.lower() in query_lower:
                    score = 5.0  # 完整匹配
                elif any(word in query_lower for word in entity_name.lower().split() if len(word) >= 3):
                    score = 2.0  # 部分匹配
                else:
                    continue
            else:
                if entity_name in query:
                    score = 5.0  # 完整匹配
                else:
                    continue
            
            for entity in entities:
                entity_id = entity.get("id")
                if entity_id:
                    entity_scores[entity_id] = entity_scores.get(entity_id, 0) + score
                    found_entities.append(entity)
        
        # 3. 对于中文，使用分词匹配（但只匹配较长的词，避免误匹配）
        if self.language == "zh":
            query_tokens = list(jieba.cut(query))
            for token in query_tokens:
                # 只匹配长度>=4的词，避免匹配到通用词
                if len(token) >= 4 and token in self.entity_index and self._is_valid_entity_name(token):
                    for entity in self.entity_index[token]:
                        entity_id = entity.get("id")
                        if entity_id:
                            entity_scores[entity_id] = entity_scores.get(entity_id, 0) + 1.0  # 分词匹配低分
                            found_entities.append(entity)
        
        # 去重并按分数排序（保留高分实体）
        seen_ids = set()
        unique_entities = []
        entity_list = []
        
        for entity in found_entities:
            entity_id = entity.get("id")
            if entity_id and entity_id not in seen_ids:
                seen_ids.add(entity_id)
                score = entity_scores.get(entity_id, 0)
                entity_list.append((entity, score))
        
        # 按分数排序，优先返回高分实体
        entity_list.sort(key=lambda x: x[1], reverse=True)
        unique_entities = [e for e, _ in entity_list]
        
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
    
    def _multi_hop_retrieval(self, start_entity_ids: Set[str]) -> Dict[str, float]:
        """
        多跳关系检索：从起始实体开始，通过多跳关系找到相关实体
        
        Args:
            start_entity_ids: 起始实体ID集合
            
        Returns:
            Dict[entity_id, score]: 相关实体及其分数（根据跳数计算）
        """
        if not start_entity_ids:
            return {}
        
        # 使用BFS进行多跳遍历
        entity_scores = {}  # entity_id -> score
        visited = set()  # 已访问的实体
        queue = []  # (entity_id, hop_level)
        
        # 初始化：起始实体（0跳）
        for entity_id in start_entity_ids:
            if entity_id in self.entity_graph:
                queue.append((entity_id, 0))
                visited.add(entity_id)
                entity_scores[entity_id] = 3.0  # 0跳：最高分
        
        # BFS遍历
        while queue:
            current_id, hop_level = queue.pop(0)
            
            if hop_level >= self.max_hops:
                continue
            
            # 获取当前实体的邻居
            neighbors = self.entity_graph.get(current_id, set())
            
            for neighbor_id in neighbors:
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    next_hop = hop_level + 1
                    
                    # 根据跳数计算分数（跳数越多，分数越低）
                    # 1跳：2.0, 2跳：1.0, 3跳：0.5
                    score = 3.0 / (next_hop + 1)
                    entity_scores[neighbor_id] = entity_scores.get(neighbor_id, 0) + score
                    
                    if next_hop < self.max_hops:
                        queue.append((neighbor_id, next_hop))
        
        return entity_scores
    
    def retrieve_doc_ids(self, query: str, top_k: int = 10, use_multi_hop: bool = True) -> List[int]:
        """
        根据query检索相关的doc_id，支持多跳关系检索
        
        Args:
            query: 查询文本
            top_k: 返回的doc_id数量上限
            use_multi_hop: 是否使用多跳关系检索（默认True）
            
        Returns:
            相关的doc_id列表，按相关性排序
        """
        # 1. 从query中提取实体
        entities = self._extract_entities_from_query(query)
        start_entity_ids = {e.get("id") for e in entities if e.get("id")}
        
        # 2. 收集所有相关的doc_id
        doc_id_scores = {}  # doc_id -> score
        
        # 2.0 通过实体名称直接匹配doc_id（通过文档的company_name字段）
        # 这是对KG关系的补充，处理实体没有关系但文档中有company_name的情况
        # 这个匹配的优先级最高，因为它是直接的公司名称匹配
        for entity in entities:
            entity_name = entity.get("name", "").strip()
            if entity_name:
                # 直接匹配（最高优先级，因为这是精确的公司名称匹配）
                if entity_name in self.entity_name_to_docs:
                    for doc_id in self.entity_name_to_docs[entity_name]:
                        doc_id_scores[doc_id] = doc_id_scores.get(doc_id, 0) + 10.0  # 最高优先级
                
                # 不区分大小写匹配（英文）
                if self.language == "en":
                    entity_lower = entity_name.lower()
                    if entity_lower in self.entity_name_to_docs:
                        for doc_id in self.entity_name_to_docs[entity_lower]:
                            doc_id_scores[doc_id] = doc_id_scores.get(doc_id, 0) + 10.0  # 最高优先级
        
        # 2.1 直接匹配的实体（0跳，通过KG关系）
        for entity in entities:
            entity_id = entity.get("id")
            if entity_id in self.entity_to_docs:
                # 实体直接匹配：高分
                for doc_id in self.entity_to_docs[entity_id]:
                    doc_id_scores[doc_id] = doc_id_scores.get(doc_id, 0) + 2.0
        
        # 2.2 多跳关系检索
        if use_multi_hop and start_entity_ids:
            multi_hop_entities = self._multi_hop_retrieval(start_entity_ids)
            
            for entity_id, entity_score in multi_hop_entities.items():
                if entity_id in self.entity_to_docs:
                    # 根据实体分数累加文档分数
                    # 但降低多跳关系的权重，避免稀释直接匹配的分数
                    for doc_id in self.entity_to_docs[entity_id]:
                        # 如果这个doc_id已经通过entity_name匹配获得了高分，不再累加多跳分数
                        # 或者降低多跳分数的权重
                        if doc_id_scores.get(doc_id, 0) < 5.0:  # 如果分数还很低，才累加多跳分数
                            doc_id_scores[doc_id] = doc_id_scores.get(doc_id, 0) + entity_score * 0.5  # 降低权重
                        else:
                            # 如果已经有高分（可能是直接匹配），只少量增加
                            doc_id_scores[doc_id] = doc_id_scores.get(doc_id, 0) + entity_score * 0.1
        
        # 3. 通过关系查找相关文档（关系类型和属性匹配）
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
    
    def get_entity_info(self, query: str, use_multi_hop: bool = True) -> Dict[str, Any]:
        """
        获取query中提取到的实体信息（用于调试），包括多跳关系信息
        
        Args:
            query: 查询文本
            use_multi_hop: 是否使用多跳关系检索
        """
        entities = self._extract_entities_from_query(query)
        start_entity_ids = {e.get("id") for e in entities if e.get("id")}
        
        # 获取多跳关系信息
        multi_hop_info = {}
        if use_multi_hop and start_entity_ids:
            multi_hop_entities = self._multi_hop_retrieval(start_entity_ids)
            multi_hop_info = {
                "multi_hop_entities": len(multi_hop_entities),
                "max_hops": self.max_hops,
                "sample_entities": [
                    {
                        "id": eid,
                        "score": score
                    }
                    for eid, score in list(multi_hop_entities.items())[:5]
                ]
            }
        
        doc_ids = self.retrieve_doc_ids(query, top_k=10, use_multi_hop=use_multi_hop)
        
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
            "entity_count": len(entities),
            "multi_hop": multi_hop_info
        }


# 移除@lru_cache，因为docs_path参数可能导致缓存问题
# 如果需要缓存，可以在调用层面处理
def create_kg_retriever(kg_path: str = "My_RAG/kg_output.json", language: str = "en", max_hops: int = 2, docs_path: str = None) -> KGRetriever:
    """
    创建知识图谱检索器
    
    Args:
        kg_path: 知识图谱文件路径
        language: 语言类型
        max_hops: 最大跳数，用于多跳关系检索
        docs_path: 文档文件路径（可选，用于通过company_name匹配）
    """
    return KGRetriever(kg_path, language, max_hops, docs_path)

