"""
构建知识图谱 (Knowledge Graph) 从 dragonball_docs.jsonl
使用 LLM 从文档中提取实体和关系，构建成知识图谱格式
"""

import json
from typing import List, Dict, Any, Set
from pathlib import Path
from tqdm import tqdm
from ollama import Client
from config import load_config
from utils import load_jsonl, save_jsonl


def load_ollama_config() -> dict:
    """读取 config.yaml 内的 ollama 设定"""
    config = load_config()
    assert "ollama" in config, "Ollama configuration not found in config file."
    assert "host" in config["ollama"], "Ollama host not specified in config file."
    assert "model" in config["ollama"], "Ollama model not specified in config file."
    return config["ollama"]


def extract_entities_and_relations(text: str, company_name: str, doc_id: int, language: str = "zh") -> Dict[str, Any]:
    """
    使用 LLM 从文本中提取实体和关系
    
    返回格式：
    {
        "entities": [
            {"id": "e1", "type": "Company", "name": "华夏娱乐有限公司", "properties": {...}},
            {"id": "e2", "type": "Event", "name": "资产收购", "properties": {...}},
            ...
        ],
        "relations": [
            {"source": "e1", "target": "e2", "type": "进行", "properties": {"时间": "2017年2月"}},
            ...
        ]
    }
    """
    # 智能处理长文档：优先保留重要部分
    max_chars = 8000  # 增加上下文窗口
    if len(text) > max_chars:
        # 尝试保留开头、中间重要部分和结尾
        # 开头通常包含公司基本信息
        prefix_len = 1500
        suffix_len = 1000
        middle_start = (len(text) - prefix_len - suffix_len) // 2
        
        # 优先保留包含关键信息的部分
        prefix = text[:prefix_len]
        suffix = text[-suffix_len:]
        middle = text[middle_start:middle_start + (max_chars - prefix_len - suffix_len)]
        text = prefix + "\n...[中间内容省略]...\n" + middle + "\n...[中间内容省略]...\n" + suffix
        text = text[:max_chars]  # 确保不超过限制
    
    if language == "zh":
        prompt = f"""你是一个知识图谱构建专家。请从以下公司财务报告中精确提取实体和关系，构建知识图谱。

公司名称：{company_name}

报告内容：
{text}

请提取以下类型的实体（注意：每个实体必须有一个唯一且精确的名称）：

1. **公司实体** (Company)：
   - 主公司名称：{company_name}
   - 子公司、被收购公司、合作伙伴公司
   - 例如：草莓文化传媒有限公司、嘉悦传媒有限公司

2. **事件实体** (Event)：
   - 财务事件：资产收购、股权收购、融资活动、投资、债务重组、资产重组、股利分发
   - 治理事件：道德与诚信事件、合规与监管更新、董事会变更、高级管理层变动、股东大会决议、公司治理政策修订
   - 环境责任事件：碳抵消项目、节能减排项目、污染防治设施建设、环保产品开发、环境管理系统实施、生态恢复计划
   - 社会责任事件：慈善活动、社区投资、公共服务项目、员工健康与安全计划、员工职业成长计划
   - 可持续性与社会责任倡议

3. **时间实体** (Time)：
   - 具体时间：年份（如2017年）、月份（如2017年2月）、日期
   - 时间范围：2017年度、2020年至2022年

4. **财务指标实体** (FinancialMetric)：
   - 营业收入、净利润、总资产、总负债、股东权益、现金流量
   - 负债比率、资产负债率、净资产收益率

5. **人物实体** (Person)：
   - 高管职位：CEO、CFO、CTO、董事长
   - 具体人物（如果提及姓名）

6. **项目实体** (Project)：
   - 投资项目：D项目、核电站项目、绿色能源科技园
   - 建设项目、研发项目

7. **地点实体** (Location)：
   - 注册地：上海、北京、云南、美国加利福尼亚州
   - 上市地点：上海证券交易所、纽约证券交易所

8. **金额实体** (Amount)：
   - 具体金额：1.2亿元、5000万元、10亿美元

请提取以下类型的关系（关系类型使用英文大写，用下划线连接）：

- HAPPENED_AT: 事件发生在某个时间（Event -> Time）
- INVOLVES: 事件涉及某个实体（Event -> Company/Person/Project/Location）
- ACQUIRES: 公司收购其他公司或资产（Company -> Company/Asset）
- INVESTS_IN: 公司投资项目或公司（Company -> Project/Company）
- RAISES_FUNDS: 公司进行融资活动（Company -> Event，金额在properties中）
- REORGANIZES: 公司进行重组（Company -> Event，如债务重组、资产重组）
- DISTRIBUTES: 公司分发股利（Company -> Event，金额在properties中）
- HAS_METRIC: 公司拥有某个财务指标值（Company -> FinancialMetric，数值在properties中）
- OCCURS_AT: 事件发生在某个地点（Event -> Location）
- COSTS: 事件/项目涉及金额（Event/Project -> Amount）
- WORKS_AT: 人物在公司任职（Person -> Company）
- LOCATED_IN: 公司位于某个地点（Company -> Location）
- SUBSIDIARY_OF: 子公司关系（Company -> Company）
- PART_OF: 项目属于某个事件或计划（Project -> Event）

重要提示：
- 每个实体必须有唯一且清晰的名称
- 关系必须明确且有意义
- 在properties中记录详细信息，如时间、金额、百分比等
- 只提取文档中明确提及的实体和关系，不要推测

请以 JSON 格式输出，格式如下：
{{
    "entities": [
        {{
            "id": "e1",
            "type": "Company",
            "name": "实体名称（精确且唯一）",
            "properties": {{
                "description": "实体描述（如有）",
                "其他属性": "值"
            }}
        }}
    ],
    "relations": [
        {{
            "source": "e1",
            "target": "e2",
            "type": "关系类型（使用英文大写，如HAPPENED_AT）",
            "properties": {{
                "时间": "2017年2月",
                "金额": "1.2亿元",
                "百分比": "70%",
                "其他属性": "值"
            }}
        }}
    ]
}}

只输出 JSON，不要输出其他文字。"""
    else:
        prompt = f"""You are a knowledge graph construction expert. Extract entities and relations from the following company financial report to build a knowledge graph.

Company Name: {company_name}

Report Content:
{text}

Please extract the following types of entities (each entity must have a unique and precise name):

1. **Company** entities:
   - Main company: {company_name}
   - Subsidiaries, acquired companies, partner companies
   
2. **Event** entities:
   - Financial events: asset acquisition, equity acquisition, financing, investment, debt restructuring, asset restructuring, dividend distribution
   - Governance events: ethics and integrity incidents, compliance and regulatory updates, board changes, senior management changes, shareholder meeting resolutions, governance policy revisions
   - Environmental responsibility events: carbon offset projects, energy-saving projects, pollution prevention facilities, green product development, environmental management systems, ecological restoration plans
   - Social responsibility events: charity activities, community investment, public service projects, employee health and safety programs, employee career development programs
   - Sustainability and social responsibility initiatives

3. **Time** entities:
   - Specific times: years (e.g., 2017), months (e.g., February 2017), dates
   - Time ranges: 2017 fiscal year, 2020-2022

4. **FinancialMetric** entities:
   - Revenue, net profit, total assets, total liabilities, shareholder equity, cash flow
   - Debt ratio, asset-liability ratio, return on equity

5. **Person** entities:
   - Executive positions: CEO, CFO, CTO, Chairman
   - Specific individuals (if names are mentioned)

6. **Project** entities:
   - Investment projects, construction projects, R&D projects

7. **Location** entities:
   - Registered location, listing location (e.g., Shanghai Stock Exchange, New York Stock Exchange)

8. **Amount** entities:
   - Specific amounts: 120 million yuan, 50 million USD, etc.

Please extract the following types of relations (use uppercase English with underscores):

- HAPPENED_AT: events happening at certain times (Event -> Time)
- INVOLVES: events involving certain entities (Event -> Company/Person/Project/Location)
- ACQUIRES: companies acquiring other companies or assets (Company -> Company/Asset)
- INVESTS_IN: companies investing in projects or companies (Company -> Project/Company)
- RAISES_FUNDS: companies raising funds (Company -> Event, amount in properties)
- REORGANIZES: companies reorganizing (Company -> Event, such as debt restructuring, asset restructuring)
- DISTRIBUTES: companies distributing dividends (Company -> Event, amount in properties)
- HAS_METRIC: companies having financial metric values (Company -> FinancialMetric, value in properties)
- OCCURS_AT: events occurring at locations (Event -> Location)
- COSTS: events/projects involving amounts (Event/Project -> Amount)
- WORKS_AT: people working at companies (Person -> Company)
- LOCATED_IN: companies located at places (Company -> Location)
- SUBSIDIARY_OF: subsidiary relationships (Company -> Company)
- PART_OF: projects belonging to events or plans (Project -> Event)

Important notes:
- Each entity must have a unique and clear name
- Relations must be explicit and meaningful
- Record detailed information in properties, such as time, amount, percentage, etc.
- Only extract entities and relations explicitly mentioned in the document, do not infer

Output in JSON format as follows:
{{
    "entities": [
        {{
            "id": "e1",
            "type": "Company",
            "name": "Entity Name (precise and unique)",
            "properties": {{
                "description": "Entity description (if any)",
                "other_property": "value"
            }}
        }}
    ],
    "relations": [
        {{
            "source": "e1",
            "target": "e2",
            "type": "Relation Type (uppercase English, e.g., HAPPENED_AT)",
            "properties": {{
                "time": "February 2017",
                "amount": "120 million yuan",
                "percentage": "70%",
                "other_property": "value"
            }}
        }}
    ]
}}

Output only JSON, no other text."""

    try:
        cfg = load_ollama_config()
        client = Client(host=cfg["host"])
        resp = client.generate(
            model=cfg["model"],
            prompt=prompt,
            stream=False,
            options={
                "temperature": 0.1,  # 使用较低温度以获得更一致的提取结果
                "num_ctx": 16384,  # 增加上下文窗口以处理更长文档
            },
        )
        response_text = resp.get("response", "").strip()
        
        # 尝试从响应中提取 JSON
        # 移除可能的 markdown 代码块标记
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
        
        # 解析 JSON
        result = json.loads(response_text)
        result["doc_id"] = doc_id
        return result
        
    except json.JSONDecodeError as e:
        print(f"Warning: Failed to parse JSON from LLM response for doc_id {doc_id}: {e}")
        print(f"Response text: {response_text[:500]}")
        return {"entities": [], "relations": [], "doc_id": doc_id, "error": str(e)}
    except Exception as e:
        print(f"Error extracting entities and relations for doc_id {doc_id}: {e}")
        return {"entities": [], "relations": [], "doc_id": doc_id, "error": str(e)}


def merge_knowledge_graphs(kg_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    合并多个文档的知识图谱提取结果
    
    返回格式：
    {
        "entities": [...],  # 所有实体，去重
        "relations": [...],  # 所有关系
        "doc_mapping": {...}  # 文档ID到实体/关系的映射
    }
    """
    all_entities = []
    all_relations = []
    entity_map = {}  # 用于实体去重：name -> entity
    entity_id_mapping = {}  # 旧ID -> 新ID
    next_entity_id = 1
    doc_mapping = {}
    
    for kg_result in kg_results:
        doc_id = kg_result.get("doc_id")
        if doc_id is None:
            continue
            
        doc_entity_ids = []
        doc_relation_ids = []
        
        # 处理实体（改进去重逻辑）
        for entity in kg_result.get("entities", []):
            entity_name = entity.get("name", "").strip()
            entity_type = entity.get("type", "")
            original_id = entity.get("id", "")
            
            if not entity_name:
                continue
            
            # 标准化实体名称（去除多余空格）
            normalized_name = " ".join(entity_name.split())
            if entity_type in ["Time", "Amount", "FinancialMetric"]:
                # 对于这些类型，名称通常更精确
                normalized_name = entity_name.strip()
            
            # 创建唯一键：类型+标准化名称
            key = f"{entity_type}:{normalized_name}"
            
            if key not in entity_map:
                # 新实体
                new_id = f"e{next_entity_id}"
                next_entity_id += 1
                entity["id"] = new_id
                entity["name"] = normalized_name  # 使用标准化名称
                entity_map[key] = entity
                all_entities.append(entity)
                # 记录原始ID到新ID的映射
                if original_id:
                    entity_id_mapping[original_id] = new_id
                final_id = new_id
            else:
                # 已存在的实体，合并属性
                existing_entity = entity_map[key]
                existing_props = existing_entity.get("properties", {})
                new_props = entity.get("properties", {})
                # 合并属性（智能策略）
                if "description" in new_props and "description" in existing_props:
                    if new_props["description"] not in existing_props["description"]:
                        existing_props["description"] += f"; {new_props['description']}"
                elif "description" in new_props:
                    existing_props["description"] = new_props["description"]
                
                # 对于数值型属性，保留更精确的值
                for k, v in new_props.items():
                    if k != "description":
                        if k not in existing_props or (isinstance(v, (int, float)) and isinstance(existing_props.get(k), str)):
                            existing_props[k] = v
                
                # 记录原始ID到已存在实体ID的映射
                if original_id:
                    entity_id_mapping[original_id] = existing_entity["id"]
                final_id = existing_entity["id"]
            
            # 记录文档中的实体ID（去重，避免同一实体被重复添加）
            if final_id not in doc_entity_ids:
                doc_entity_ids.append(final_id)
        
        # 处理关系（需要更新源和目标ID）
        for relation in kg_result.get("relations", []):
            source_old = relation.get("source", "")
            target_old = relation.get("target", "")
            
            # 查找新的ID
            source_new = entity_id_mapping.get(source_old, source_old)
            target_new = entity_id_mapping.get(target_old, target_old)
            
            # 创建关系对象
            relation_obj = {
                "source": source_new,
                "target": target_new,
                "type": relation.get("type", ""),
                "properties": relation.get("properties", {}),
                "doc_id": doc_id
            }
            
            # 简单去重：相同的源、目标、类型视为重复
            relation_key = (source_new, target_new, relation_obj["type"])
            if relation_key not in {(r["source"], r["target"], r["type"]) for r in all_relations}:
                all_relations.append(relation_obj)
                doc_relation_ids.append(len(all_relations) - 1)
        
        doc_mapping[doc_id] = {
            "entities": doc_entity_ids,
            "relations": doc_relation_ids
        }
    
    return {
        "entities": all_entities,
        "relations": all_relations,
        "doc_mapping": doc_mapping,
        "statistics": {
            "total_entities": len(all_entities),
            "total_relations": len(all_relations),
            "total_docs": len(doc_mapping)
        }
    }


def build_kg(docs_path: str, output_path: str, language: str = "zh", batch_size: int = 1):
    """
    从文档构建知识图谱
    
    Args:
        docs_path: 输入文档路径 (JSONL)
        output_path: 输出知识图谱路径 (JSON)
        language: 语言 ("zh" 或 "en")
        batch_size: 批处理大小（目前为1，可以后续优化）
    """
    print(f"Loading documents from {docs_path}...")
    docs = load_jsonl(docs_path)
    print(f"Loaded {len(docs)} documents.")
    
    print("Extracting entities and relations from documents...")
    kg_results = []
    
    for doc in tqdm(docs, desc="Processing documents"):
        doc_id = doc.get("doc_id")
        company_name = doc.get("company_name", "")
        content = doc.get("content", "")
        doc_language = doc.get("language", language)
        
        if not content:
            continue
        
        kg_result = extract_entities_and_relations(
            text=content,
            company_name=company_name,
            doc_id=doc_id,
            language=doc_language
        )
        kg_results.append(kg_result)
    
    print("Merging knowledge graphs...")
    merged_kg = merge_knowledge_graphs(kg_results)
    
    print(f"Knowledge graph built successfully!")
    print(f"  - Total entities: {merged_kg['statistics']['total_entities']}")
    print(f"  - Total relations: {merged_kg['statistics']['total_relations']}")
    print(f"  - Total documents: {merged_kg['statistics']['total_docs']}")
    
    # 保存知识图谱
    print(f"Saving knowledge graph to {output_path}...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(merged_kg, f, ensure_ascii=False, indent=2)
    
    print("Done!")
    return merged_kg


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Build Knowledge Graph from dragonball_docs.jsonl")
    parser.add_argument(
        "--input",
        type=str,
        default="dragonball_dataset/dragonball_docs.jsonl",
        help="Input documents path (JSONL)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="My_RAG/kg_output.json",
        help="Output knowledge graph path (JSON)"
    )
    parser.add_argument(
        "--language",
        type=str,
        default="zh",
        choices=["zh", "en"],
        help="Language (zh or en)"
    )
    
    args = parser.parse_args()
    
    # 转换为绝对路径
    input_path = Path(__file__).parent.parent / args.input
    output_path = Path(__file__).parent.parent / args.output
    
    build_kg(
        docs_path=str(input_path),
        output_path=str(output_path),
        language=args.language
    )


if __name__ == "__main__":
    main()

