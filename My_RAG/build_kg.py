"""
构建知识图谱 (Knowledge Graph) 从 dragonball_docs.jsonl
使用 LLM 从文档中提取实体和关系，构建成知识图谱格式

版本說明：
- 原本使用 Ollama，已改為使用 OpenAI API（openai.ChatCompletion）
- 模型建議：gpt-4o-mini（可在 config.yaml 或環境變數設定）
"""

import json
import os
from typing import List, Dict, Any, Set
from pathlib import Path

from tqdm import tqdm
from openai import OpenAI

from config import load_config
from utils import load_jsonl, save_jsonl

# =========================
# OpenAI 設定與 Client
# =========================

_openai_client = None
_openai_model = None


def init_openai_from_config():
    """
    初始化 OpenAI Client 與模型名稱：
    - model 來源：config.yaml 的 openai.model 或環境變數 OPENAI_MODEL，預設 gpt-4o-mini
    - api key 來源：環境變數 OPENAI_API_KEY 或 config.yaml 的 openai.api_key_env
    """
    global _openai_client, _openai_model

    if _openai_client is not None and _openai_model is not None:
        return

    config = load_config()
    openai_cfg = config.get("openai", {})

    model = openai_cfg.get("model", os.environ.get("OPENAI_MODEL", "gpt-4o-mini"))
    api_key_env = openai_cfg.get("api_key_env", "OPENAI_API_KEY")
    api_key = os.environ.get(api_key_env)

    if not api_key:
        raise RuntimeError(
            f"OpenAI API key not found. Please set environment variable '{api_key_env}'."
        )

    _openai_client = OpenAI(api_key=api_key)
    _openai_model = model


def get_openai_client():
    """取得全域單例 OpenAI Client"""
    if _openai_client is None or _openai_model is None:
        init_openai_from_config()
    return _openai_client


def get_openai_model() -> str:
    """取得目前使用的 OpenAI 模型名稱"""
    if _openai_client is None or _openai_model is None:
        init_openai_from_config()
    return _openai_model


# =========================
# JSON 修復與提取輔助函數
# =========================

def _extract_json_from_text(text: str) -> str:
    """
    從文本中提取 JSON，使用多種策略
    """
    # 策略1: 提取 ```json ... ``` 中的內容
    if "```json" in text:
        parts = text.split("```json", 1)
        if len(parts) > 1:
            json_part = parts[1].split("```", 1)[0].strip()
            return json_part
    
    # 策略2: 提取 ``` ... ``` 中的內容
    if "```" in text:
        parts = text.split("```", 1)
        if len(parts) > 1:
            json_part = parts[1].split("```", 1)[0].strip()
            # 如果不是 JSON，繼續嘗試其他策略
            if json_part.startswith("{"):
                return json_part
    
    # 策略3: 查找第一個 { 到最後一個 } 之間的內容
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        return text[first_brace:last_brace + 1]
    
    # 策略4: 返回原始文本
    return text.strip()


from concurrent.futures import ThreadPoolExecutor, as_completed

def _process_single_doc(doc: Dict[str, Any], default_language: str) -> tuple[int | None, Dict[str, Any] | None, str | None]:
    """
    供 ThreadPoolExecutor 使用的 worker。
    不改任何抽取邏輯，只是把原本 for 迴圈裡的 per-doc 處理包成一個函式。

    回傳:
        (doc_id, kg_result or None, error_message or None)
    """
    doc_id = doc.get("doc_id")
    if doc_id is None:
        return None, None, "doc_id is None"

    domain = doc.get("domain", "Finance")  # default: Finance
    content = doc.get("content", "")
    doc_language = doc.get("language", default_language)

    if not content:
        return doc_id, None, "empty content"

    # 根據 domain 取名稱欄位
    if domain == "Finance":
        domain_name = doc.get("company_name", "")
    elif domain == "Law":
        domain_name = doc.get("court_name", "")
    elif domain == "Medical":
        domain_name = doc.get("hospital_patient_name", "")
    else:
        domain_name = doc.get(
            "company_name",
            doc.get("court_name", doc.get("hospital_patient_name", "")),
        )

    try:
        kg_result = extract_entities_and_relations(
            text=content,
            domain=domain,
            domain_name=domain_name,
            doc_id=doc_id,
            language=doc_language,
        )
        return doc_id, kg_result, None
    except Exception as e:
        return doc_id, None, str(e)


def _fix_json_string(json_str: str) -> str:
    """
    嘗試修復常見的 JSON 格式問題，包括未終止的字符串
    """
    if not json_str:
        return json_str
    
    # 修復未終止的字符串：從後往前查找，找到未閉合的字符串並關閉它
    result = []
    i = 0
    in_string = False
    escape_next = False
    
    while i < len(json_str):
        char = json_str[i]
        
        if escape_next:
            result.append(char)
            escape_next = False
            i += 1
            continue
        
        if char == '\\':
            result.append(char)
            escape_next = True
            i += 1
            continue
        
        if char == '"':
            result.append(char)
            in_string = not in_string
            i += 1
            continue
        
        result.append(char)
        i += 1
    
    # 如果字符串未終止，關閉它
    if in_string:
        result.append('"')
    
    fixed = ''.join(result)
    
    # 移除可能的尾隨逗號（在最後一個元素後）
    lines = fixed.split('\n')
    fixed_lines = []
    for i, line in enumerate(lines):
        stripped = line.rstrip()
        # 移除行尾的逗號（如果下一行是 } 或 ]）
        if i < len(lines) - 1:
            next_stripped = lines[i + 1].strip()
            if stripped.endswith(',') and (next_stripped.startswith('}') or next_stripped.startswith(']')):
                # 檢查是否在字符串內（簡單檢查）
                quote_count = stripped.count('"') - stripped.count('\\"')
                if quote_count % 2 == 0:  # 不在字符串內
                    stripped = stripped[:-1]
        fixed_lines.append(stripped)
    
    return '\n'.join(fixed_lines)


def _try_parse_json(json_str: str, max_attempts: int = 3) -> Dict[str, Any]:
    """
    嘗試解析 JSON，如果失敗則嘗試修復
    """
    # 第一次嘗試：直接解析
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        pass
    
    # 第二次嘗試：修復後解析
    try:
        fixed = _fix_json_string(json_str)
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass
    
    # 第三次嘗試：查找並提取完整的 JSON 對象，處理截斷情況
    try:
        # 嘗試找到完整的 entities 和 relations 結構
        if '"entities"' in json_str and '"relations"' in json_str:
            # 提取 entities 部分
            entities_start = json_str.find('"entities"')
            relations_start = json_str.find('"relations"')
            
            if entities_start != -1 and relations_start != -1:
                # 構建一個最小可用的 JSON
                entities_part = json_str[entities_start:relations_start]
                relations_part = json_str[relations_start:]
                
                # 嘗試提取完整的數組
                entities_bracket = entities_part.find('[')
                relations_bracket = relations_part.find('[')
                
                if entities_bracket != -1 and relations_bracket != -1:
                    entities_match = entities_part[entities_bracket:]
                    relations_match = relations_part[relations_bracket:]
                    
                    # 找到最後一個完整的 ]，如果沒有則嘗試關閉它
                    entities_end = entities_match.rfind(']')
                    relations_end = relations_match.rfind(']')
                    
                    # 如果 entities 數組未閉合，嘗試關閉它
                    if entities_end == -1:
                        # 找到最後一個完整的對象
                        last_comma = entities_match.rfind(',')
                        if last_comma != -1:
                            # 移除最後的逗號並關閉數組
                            entities_str = entities_match[:last_comma] + ']'
                        else:
                            # 如果沒有逗號，直接關閉
                            entities_str = entities_match.rstrip().rstrip(',') + ']'
                    else:
                        entities_str = entities_match[:entities_end + 1]
                    
                    # 如果 relations 數組未閉合，嘗試關閉它
                    if relations_end == -1:
                        # 找到最後一個完整的對象
                        last_comma = relations_match.rfind(',')
                        if last_comma != -1:
                            # 移除最後的逗號並關閉數組
                            relations_str = relations_match[:last_comma] + ']'
                        else:
                            # 如果沒有逗號，直接關閉
                            relations_str = relations_match.rstrip().rstrip(',') + ']'
                    else:
                        relations_str = relations_match[:relations_end + 1]
                    
                    # 構建完整的 JSON
                    fixed_json = f'{{"entities": {entities_str}, "relations": {relations_str}}}'
                    return json.loads(fixed_json)
        elif '"entities"' in json_str:
            # 只有 entities，沒有 relations（可能被截斷）
            entities_start = json_str.find('"entities"')
            if entities_start != -1:
                entities_part = json_str[entities_start:]
                entities_bracket = entities_part.find('[')
                if entities_bracket != -1:
                    entities_match = entities_part[entities_bracket:]
                    entities_end = entities_match.rfind(']')
                    if entities_end != -1:
                        entities_str = entities_match[:entities_end + 1]
                        fixed_json = f'{{"entities": {entities_str}, "relations": []}}'
                        return json.loads(fixed_json)
    except (json.JSONDecodeError, ValueError):
        pass
    
    # 如果所有嘗試都失敗，拋出異常
    raise json.JSONDecodeError("Failed to parse JSON after multiple attempts", json_str, 0)


# =========================
# LLM 抽取實體與關係
# =========================
def extract_entities_and_relations(
    text: str,
    domain: str,
    domain_name: str,
    doc_id: int,
    language: str = "zh",
) -> Dict[str, Any]:
    """
    使用 LLM 从文本中提取实体和关系（OpenAI v1 SDK 版本）

    Args:
        text: 文档内容
        domain: 领域类型 ("Finance", "Law", "Medical")
        domain_name: 领域相关名称（公司名/法院名/医院患者名）
        doc_id: 文档ID
        language: 语言 ("zh" 或 "en")
    """
    # ========= 1. 文本長度控制 =========
    max_chars = 4000  # 比你原本 8000 小一點，比較省 token
    if len(text) > max_chars:
        prefix_len = 1200
        suffix_len = 800
        middle_start = (len(text) - prefix_len - suffix_len) // 2

        prefix = text[:prefix_len]
        suffix = text[-suffix_len:]
        middle = text[middle_start:middle_start + (max_chars - prefix_len - suffix_len)]
        text = (
            prefix
            + "\n...[中间内容省略]...\n"
            + middle
            + "\n...[中间内容省略]...\n"
            + suffix
        )
        text = text[:max_chars]

    # ========= 2. 根據 domain 選 prompt =========
    if domain == "Finance":
        prompt = _generate_finance_prompt(text, domain_name, language)
    elif domain == "Law":
        prompt = _generate_law_prompt(text, domain_name, language)
    elif domain == "Medical":
        prompt = _generate_medical_prompt(text, domain_name, language)
    else:
        raise ValueError(f"Unsupported domain: {domain}")

    response_text = ""
    try:
        # ========= 3. 建立新版 OpenAI Client =========
        client = get_openai_client()
        model = get_openai_model()

        system_msg_zh = "你是一個專門從長文本中抽取實體與關係，並輸出乾淨 JSON 結構的知識圖譜構建助手。"
        system_msg_en = "You are an expert assistant for extracting entities and relations from long documents and outputting clean JSON for a knowledge graph."
        system_content = system_msg_zh if language == "zh" else system_msg_en

        # ✅ 新版：client.chat.completions.create(...)
        # 增加 max_tokens 以避免 JSON 被截斷
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=4000,  # 從 1500 增加到 4000，避免 JSON 被截斷
        )

        response_text = (resp.choices[0].message.content or "").strip()

        # ========= 4. 從回應中抽 JSON =========
        json_str = _extract_json_from_text(response_text)
        
        # ========= 5. 解析 JSON（帶重試和修復）=========
        result = _try_parse_json(json_str)
        result["doc_id"] = doc_id
        return result

    except json.JSONDecodeError as e:
        print(f"Warning: Failed to parse JSON from LLM response for doc_id {doc_id}: {e}")
        if response_text:
            # 打印更多信息以便調試
            print(f"Response text length: {len(response_text)}")
            print(f"Response text (first 1000 chars): {response_text[:1000]!r}")
            if len(response_text) > 1000:
                print(f"Response text (last 500 chars): {response_text[-500:]!r}")
        return {"entities": [], "relations": [], "doc_id": doc_id, "error": str(e)}
    except Exception as e:
        print(f"Error extracting entities and relations for doc_id {doc_id}: {e}")
        import traceback
        traceback.print_exc()
        return {"entities": [], "relations": [], "doc_id": doc_id, "error": str(e)}


# =========================
# Prompt 生成（Finance / Law / Medical）
# =========================

def _generate_finance_prompt(text: str, company_name: str, language: str) -> str:
    """生成 Finance 领域的 prompt"""
    if language == "zh":
        return f"""你是一个知识图谱构建专家。请从以下公司财务报告中精确提取实体和关系，构建知识图谱。

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
        return f"""You are a knowledge graph construction expert. Extract entities and relations from the following company financial report to build a knowledge graph.

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


def _generate_law_prompt(text: str, court_name: str, language: str) -> str:
    """生成 Law 领域的 prompt"""
    if language == "zh":
        return f"""你是一个知识图谱构建专家。请从以下法律判决书中精确提取实体和关系，构建知识图谱。

法院名称：{court_name}

判决书内容：
{text}

请提取以下类型的实体（注意：每个实体必须有一个唯一且精确的名称）：

1. **法院实体** (Court)：
   - 法院名称：{court_name}
   - 其他相关法院、检察院

2. **案件实体** (Case)：
   - 案件编号、案件类型（如交通肇事罪、盗窃罪等）

3. **人物实体** (Person)：
   - 被告人、原告、辩护人、检察官、法官、证人
   - 具体姓名和身份

4. **指控罪名实体** (Charge)：
   - 具体罪名：交通肇事罪、盗窃罪、诈骗罪等

5. **事件实体** (Event)：
   - 案件发生事件：交通事故、犯罪行为、调查程序、庭审程序
   - 程序性事件：立案、拘留、逮捕、起诉、开庭、判决

6. **时间实体** (Time)：
   - 具体时间：年份、月份、日期
   - 时间范围：2023年4月20日、2023年5月等

7. **地点实体** (Location)：
   - 案发地点：街道、路段、具体地址
   - 相关地点：看守所、法院、检察院

8. **证据实体** (Evidence)：
   - 监控录像、证人证言、法医鉴定、现场照片、交警报告、医院诊断、通话记录等

9. **判决结果实体** (Verdict)：
   - 判决内容：有期徒刑、赔偿金额等

10. **金额实体** (Amount)：
    - 赔偿金额、罚款金额等

请提取以下类型的关系（关系类型使用英文大写，用下划线连接）：

- HAPPENED_AT: 事件发生在某个时间（Event -> Time）
- OCCURS_AT: 事件发生在某个地点（Event -> Location）
- INVOLVES: 事件涉及人物或实体（Event -> Person/Evidence/Location）
- CHARGED_WITH: 人物被指控罪名（Person -> Charge）
- DEFENDED_BY: 被告人由辩护人辩护（Person -> Person）
- PROSECUTED_BY: 案件由检察院起诉（Case -> Court/Person）
- JUDGED_BY: 案件由法院判决（Case -> Court）
- PROVIDES_EVIDENCE: 证据支持案件（Evidence -> Case）
- TESTIFIES: 证人作证（Person -> Case/Event）
- RESULTS_IN: 案件导致判决结果（Case -> Verdict）
- AWARDS: 判决结果包含赔偿金额（Verdict -> Amount）
- LOCATED_IN: 人物或实体位于某个地点（Person/Location -> Location）
- PART_OF: 事件属于案件的一部分（Event -> Case）

重要提示：
- 每个实体必须有唯一且清晰的名称
- 关系必须明确且有意义
- 在properties中记录详细信息，如时间、地点、金额等
- 只提取文档中明确提及的实体和关系，不要推测

请以 JSON 格式输出，格式如下：
{{
    "entities": [
        {{
            "id": "e1",
            "type": "Court",
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
                "时间": "2023年4月20日",
                "地点": "苹果市中心街道",
                "金额": "50万元",
                "其他属性": "值"
            }}
        }}
    ]
}}

只输出 JSON，不要输出其他文字。"""
    else:
        return f"""You are a knowledge graph construction expert. Extract entities and relations from the following legal judgment document to build a knowledge graph.

Court Name: {court_name}

Judgment Content:
{text}

Please extract the following types of entities (each entity must have a unique and precise name):

1. **Court** entities:
   - Court name: {court_name}
   - Other related courts, procuratorates

2. **Case** entities:
   - Case number, case type (e.g., traffic accident crime, theft crime, etc.)

3. **Person** entities:
   - Defendant, plaintiff, defense attorney, prosecutor, judge, witness
   - Specific names and identities

4. **Charge** entities:
   - Specific charges: traffic accident crime, theft crime, fraud crime, etc.

5. **Event** entities:
   - Case events: traffic accidents, criminal acts, investigation procedures, trial procedures
   - Procedural events: case filing, detention, arrest, prosecution, trial, judgment

6. **Time** entities:
   - Specific times: years, months, dates
   - Time ranges: April 20, 2023, May 2023, etc.

7. **Location** entities:
   - Crime scene: streets, road sections, specific addresses
   - Related locations: detention center, court, procuratorate

8. **Evidence** entities:
   - Surveillance video, witness testimony, forensic identification, scene photos, traffic police reports, hospital diagnosis, call records, etc.

9. **Verdict** entities:
   - Judgment content: fixed-term imprisonment, compensation amount, etc.

10. **Amount** entities:
    - Compensation amount, fine amount, etc.

Please extract the following types of relations (use uppercase English with underscores):

- HAPPENED_AT: events happening at certain times (Event -> Time)
- OCCURS_AT: events occurring at locations (Event -> Location)
- INVOLVES: events involving persons or entities (Event -> Person/Evidence/Location)
- CHARGED_WITH: persons charged with crimes (Person -> Charge)
- DEFENDED_BY: defendant defended by attorney (Person -> Person)
- PROSECUTED_BY: case prosecuted by procuratorate (Case -> Court/Person)
- JUDGED_BY: case judged by court (Case -> Court)
- PROVIDES_EVIDENCE: evidence supporting case (Evidence -> Case)
- TESTIFIES: witness testifying (Person -> Case/Event)
- RESULTS_IN: case resulting in verdict (Case -> Verdict)
- AWARDS: verdict awarding compensation (Verdict -> Amount)
- LOCATED_IN: persons or entities located at places (Person/Location -> Location)
- PART_OF: events being part of case (Event -> Case)

Important notes:
- Each entity must have a unique and clear name
- Relations must be explicit and meaningful
- Record detailed information in properties, such as time, location, amount, etc.
- Only extract entities and relations explicitly mentioned in the document, do not infer

Output in JSON format as follows:
{{
    "entities": [
        {{
            "id": "e1",
            "type": "Court",
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
                "time": "April 20, 2023",
                "location": "City Center Street",
                "amount": "500,000 yuan",
                "other_property": "value"
            }}
        }}
    ]
}}

Output only JSON, no other text."""


def _generate_medical_prompt(text: str, hospital_patient_name: str, language: str) -> str:
    """生成 Medical 领域的 prompt"""
    if language == "zh":
        return f"""你是一个知识图谱构建专家。请从以下医疗病历中精确提取实体和关系，构建知识图谱。

医院患者名称：{hospital_patient_name}

病历内容：
{text}

请提取以下类型的实体（注意：每个实体必须有一个唯一且精确的名称）：

1. **医院实体** (Hospital)：
   - 医院名称（从hospital_patient_name中提取）

2. **患者实体** (Patient)：
   - 患者姓名（从hospital_patient_name中提取）
   - 患者基本信息：性别、年龄、职业等

3. **医生实体** (Doctor)：
   - 医生姓名、职称

4. **疾病实体** (Disease)：
   - 诊断疾病：地中海贫血、高血压、糖尿病等

5. **症状实体** (Symptom)：
   - 主诉症状：贫血、体力下降、腹部突出等
   - 其他症状：头晕、心悸、乏力等

6. **检查实体** (Examination)：
   - 检查类型：血常规、腹部超声、血铁蛋白测定、血生化检查等

7. **检查结果实体** (ExaminationResult)：
   - 检查结果数值：血红蛋白80g/L、红细胞计数3.2x10^12/L等

8. **治疗实体** (Treatment)：
   - 治疗方法：铁螯合剂治疗、药物治疗等

9. **药物实体** (Medication)：
   - 药物名称：铁螯合剂等

10. **时间实体** (Time)：
    - 入院时间、记录时间、检查时间等

11. **地点实体** (Location)：
    - 患者住址、医院地址等

12. **诊断实体** (Diagnosis)：
    - 初步诊断、鉴别诊断等

请提取以下类型的关系（关系类型使用英文大写，用下划线连接）：

- ADMITTED_TO: 患者入院（Patient -> Hospital）
- TREATED_BY: 患者由医生治疗（Patient -> Doctor）
- HAS_SYMPTOM: 患者有症状（Patient -> Symptom）
- DIAGNOSED_WITH: 患者被诊断为疾病（Patient -> Disease）
- UNDERGOES: 患者接受检查（Patient -> Examination）
- SHOWS_RESULT: 检查显示结果（Examination -> ExaminationResult）
- INDICATES: 检查结果指示疾病（ExaminationResult -> Disease）
- RECEIVES_TREATMENT: 患者接受治疗（Patient -> Treatment）
- USES_MEDICATION: 治疗使用药物（Treatment -> Medication）
- PRESCRIBED_BY: 药物由医生开具（Medication -> Doctor）
- HAPPENED_AT: 事件发生在某个时间（Event -> Time）
- LOCATED_IN: 患者或医院位于某个地点（Patient/Hospital -> Location）
- PART_OF: 症状或检查属于诊断的一部分（Symptom/Examination -> Diagnosis）

重要提示：
- 每个实体必须有唯一且清晰的名称
- 关系必须明确且有意义
- 在properties中记录详细信息，如时间、数值、单位等
- 只提取文档中明确提及的实体和关系，不要推测

请以 JSON 格式输出，格式如下：
{{
    "entities": [
        {{
            "id": "e1",
            "type": "Patient",
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
            "type": "关系类型（使用英文大写，如HAS_SYMPTOM）",
            "properties": {{
                "时间": "2月4日",
                "数值": "80g/L",
                "其他属性": "值"
            }}
        }}
    ]
}}

只输出 JSON，不要输出其他文字。"""
    else:
        return f"""You are a knowledge graph construction expert. Extract entities and relations from the following medical record to build a knowledge graph.

Hospital Patient Name: {hospital_patient_name}

Medical Record Content:
{text}

Please extract the following types of entities (each entity must have a unique and precise name):

1. **Hospital** entities:
   - Hospital name (extracted from hospital_patient_name)

2. **Patient** entities:
   - Patient name (extracted from hospital_patient_name)
   - Patient basic information: gender, age, occupation, etc.

3. **Doctor** entities:
   - Doctor name, title

4. **Disease** entities:
   - Diagnosed diseases: thalassemia, hypertension, diabetes, etc.

5. **Symptom** entities:
   - Chief complaints: anemia, physical decline, abdominal protrusion, etc.
   - Other symptoms: dizziness, palpitations, fatigue, etc.

6. **Examination** entities:
   - Examination types: blood routine, abdominal ultrasound, ferritin test, blood biochemistry, etc.

7. **ExaminationResult** entities:
   - Examination result values: hemoglobin 80g/L, red blood cell count 3.2x10^12/L, etc.

8. **Treatment** entities:
   - Treatment methods: iron chelator therapy, medication, etc.

9. **Medication** entities:
   - Medication names: iron chelator, etc.

10. **Time** entities:
    - Admission time, record time, examination time, etc.

11. **Location** entities:
    - Patient address, hospital address, etc.

12. **Diagnosis** entities:
    - Preliminary diagnosis, differential diagnosis, etc.

Please extract the following types of relations (use uppercase English with underscores):

- ADMITTED_TO: patient admitted to hospital (Patient -> Hospital)
- TREATED_BY: patient treated by doctor (Patient -> Doctor)
- HAS_SYMPTOM: patient has symptom (Patient -> Symptom)
- DIAGNOSED_WITH: patient diagnosed with disease (Patient -> Disease)
- UNDERGOES: patient undergoes examination (Patient -> Examination)
- SHOWS_RESULT: examination shows result (Examination -> ExaminationResult)
- INDICATES: examination result indicates disease (ExaminationResult -> Disease)
- RECEIVES_TREATMENT: patient receives treatment (Patient -> Treatment)
- USES_MEDICATION: treatment uses medication (Treatment -> Medication)
- PRESCRIBED_BY: medication prescribed by doctor (Medication -> Doctor)
- HAPPENED_AT: events happening at certain times (Event -> Time)
- LOCATED_IN: patients or hospitals located at places (Patient/Hospital -> Location)
- PART_OF: symptoms or examinations being part of diagnosis (Symptom/Examination -> Diagnosis)

Important notes:
- Each entity must have a unique and clear name
- Relations must be explicit and meaningful
- Record detailed information in properties, such as time, values, units, etc.
- Only extract entities and relations explicitly mentioned in the document, do not infer

Output in JSON format as follows:
{{
    "entities": [
        {{
            "id": "e1",
            "type": "Patient",
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
            "type": "Relation Type (uppercase English, e.g., HAS_SYMPTOM)",
            "properties": {{
                "time": "February 4",
                "value": "80g/L",
                "other_property": "value"
            }}
        }}
    ]
}}

Output only JSON, no other text."""


# =========================
# 合併多文件的 KG 結果
# =========================
def merge_knowledge_graphs(kg_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    合并多个文档的知识图谱提取结果

    返回格式：
    {
        "entities": [...],   # 去重後的實體
        "relations": [...],  # 已更新 source/target 為全局 id，並帶 doc_id
        "triples": [...],    # ⭐ 方便檢索的三元組列表 (head, relation, tail, doc_id, ...)
        "doc_mapping": {...},
        "statistics": {...}
    }
    """
    all_entities: List[Dict[str, Any]] = []
    all_relations: List[Dict[str, Any]] = []

    # key: "type:name" -> entity（用來做去重）
    entity_map: Dict[str, Dict[str, Any]] = {}
    # 原始 per-doc entity id -> 全局 entity id（e1, e2, ...）
    entity_id_mapping: Dict[str, str] = {}
    next_entity_id = 1

    # doc_id -> 該 doc 牽涉到的全局 entity ids / relation indices
    doc_mapping: Dict[int, Dict[str, Any]] = {}

    for kg_result in kg_results:
        doc_id = kg_result.get("doc_id")
        if doc_id is None:
            continue

        doc_entity_ids: List[str] = []
        doc_relation_ids: List[int] = []

        # === 處理實體（含去重與屬性合併） ===
        for entity in kg_result.get("entities", []):
            entity_name = (entity.get("name") or "").strip()
            entity_type = entity.get("type", "")
            original_id = entity.get("id", "")

            if not entity_name:
                continue

            # 標準化名稱
            normalized_name = " ".join(entity_name.split())
            if entity_type in ["Time", "Amount", "FinancialMetric"]:
                # 對時間、金額等，不特別壓空白，避免失真
                normalized_name = entity_name.strip()

            key = f"{entity_type}:{normalized_name}"

            if key not in entity_map:
                # 新實體
                new_id = f"e{next_entity_id}"
                next_entity_id += 1
                entity["id"] = new_id
                entity["name"] = normalized_name
                entity_map[key] = entity
                all_entities.append(entity)

                if original_id:
                    entity_id_mapping[original_id] = new_id
                final_id = new_id
            else:
                # 已存在，合併屬性
                existing_entity = entity_map[key]
                existing_props = existing_entity.get("properties", {}) or {}
                new_props = entity.get("properties", {}) or {}

                # description 特別處理：盡量拼起來，不要覆蓋
                if "description" in new_props and "description" in existing_props:
                    if new_props["description"] not in existing_props["description"]:
                        existing_props["description"] += f"; {new_props['description']}"
                elif "description" in new_props:
                    existing_props["description"] = new_props["description"]

                # 其他屬性：簡單 merge（新值補到舊值裡）
                for k, v in new_props.items():
                    if k == "description":
                        continue
                    if k not in existing_props:
                        existing_props[k] = v

                existing_entity["properties"] = existing_props

                if original_id:
                    entity_id_mapping[original_id] = existing_entity["id"]
                final_id = existing_entity["id"]

            if final_id not in doc_entity_ids:
                doc_entity_ids.append(final_id)

        # === 處理關係（更新 source/target ID，並簡單去重） ===
        # 注意：existing_relation_keys 必須在「所有 doc 共用」，所以放在外面
        # 但這裡要先確保它已經存在於外層 scope
        # 我們可以在這裡判斷，如果 all_relations 是空的才初始化，避免每次 loop 重建
        if not hasattr(merge_knowledge_graphs, "_relation_keys"):
            merge_knowledge_graphs._relation_keys = set()
        existing_relation_keys = merge_knowledge_graphs._relation_keys  # type: ignore

        for relation in kg_result.get("relations", []):
            source_old = relation.get("source", "")
            target_old = relation.get("target", "")

            # 把原始 per-doc entity id 映射到全局 id
            source_new = entity_id_mapping.get(source_old, source_old)
            target_new = entity_id_mapping.get(target_old, target_old)

            relation_obj = {
                "source": source_new,
                "target": target_new,
                "type": relation.get("type", ""),
                "properties": relation.get("properties", {}) or {},
                "doc_id": doc_id,
            }

            relation_key = (relation_obj["source"], relation_obj["target"], relation_obj["type"], doc_id)
            if relation_key not in existing_relation_keys:
                all_relations.append(relation_obj)
                existing_relation_keys.add(relation_key)
                doc_relation_ids.append(len(all_relations) - 1)

        doc_mapping[doc_id] = {
            "entities": doc_entity_ids,
            "relations": doc_relation_ids,
        }

    # === 生成三元組 triple list (head, relation, tail, doc_id, ...) ===
    id_to_entity = {e["id"]: e for e in all_entities if "id" in e}
    triples: List[Dict[str, Any]] = []

    for rel in all_relations:
        src_id = rel.get("source")
        tgt_id = rel.get("target")
        rel_type = rel.get("type", "")
        doc_id = rel.get("doc_id")

        head_ent = id_to_entity.get(src_id, {})
        tail_ent = id_to_entity.get(tgt_id, {})

        triples.append(
            {
                "head": head_ent.get("name", src_id),
                "head_id": src_id,
                "head_type": head_ent.get("type", ""),
                "relation": rel_type,
                "tail": tail_ent.get("name", tgt_id),
                "tail_id": tgt_id,
                "tail_type": tail_ent.get("type", ""),
                "doc_id": doc_id,
                "properties": rel.get("properties", {}) or {},
            }
        )

    return {
        "entities": all_entities,
        "relations": all_relations,
        "triples": triples,  # ⭐ 給 retriever 用的 head–relation–tail–doc_id 三元組
        "doc_mapping": doc_mapping,
        "statistics": {
            "total_entities": len(all_entities),
            "total_relations": len(all_relations),
            "total_triples": len(triples),
            "total_docs": len(doc_mapping),
        },
    }

# =========================
# Checkpoint 讀寫
# =========================

def load_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
    """加载检查点文件"""
    if Path(checkpoint_path).exists():
        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if "processed_doc_ids" in data and isinstance(data["processed_doc_ids"], list):
                    data["processed_doc_ids"] = set(data["processed_doc_ids"])
                return data
        except Exception as e:
            print(f"Warning: Failed to load checkpoint: {e}")
            return {"processed_doc_ids": set(), "kg_results": []}
    return {"processed_doc_ids": set(), "kg_results": []}


def save_checkpoint(checkpoint_path: str, processed_doc_ids: Set[int], kg_results: List[Dict[str, Any]]):
    """保存检查点文件"""
    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    checkpoint_data = {
        "processed_doc_ids": list(processed_doc_ids),
        "kg_results": kg_results,
    }
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)


def save_partial_kg(output_path: str, kg_results: List[Dict[str, Any]]):
    """保存部分知识图谱"""
    if not kg_results:
        return

    merged_kg = merge_knowledge_graphs(kg_results)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(merged_kg, f, ensure_ascii=False, indent=2)

    print(
        f"💾 Saved partial KG: {len(kg_results)} docs, "
        f"{merged_kg['statistics']['total_entities']} entities, "
        f"{merged_kg['statistics']['total_relations']} relations"
    )


# =========================
# 主流程：從文檔建 KG
# =========================
def build_kg(
    docs_path: str,
    output_path: str,
    language: str = "zh",
    batch_size: int = 1,      # 先保留，不特別用
    limit: int = None,
    save_interval: int = 20,  # 建議略微調大，避免太頻繁 I/O
    resume: bool = True,
    max_workers: int = 4,     # ⭐ 併發數：可以先用 3~5，之後再調
):
    """
    从文档构建知识图谱（支援增量保存、断点续传、併发處理）

    Args:
        docs_path: 输入文档路径 (JSONL)
        output_path: 输出知识图谱路径 (JSON)
        language: 语言 ("zh" 或 "en")
        batch_size: 保留參數，目前未使用
        limit: 限制处理的文档数量
        save_interval: 每处理多少个文档保存一次
        resume: 是否从检查点恢复
        max_workers: 併發 worker 數量（建議 3~5，避免 API 被打爆）
    """
    print(f"Loading documents from {docs_path}...")
    docs = load_jsonl(docs_path)
    print(f"Loaded {len(docs)} documents.")

    if limit is not None and limit > 0:
        docs = docs[:limit]
        print(f"⚠️  TEST MODE: Processing only first {len(docs)} documents.")

    checkpoint_path = str(Path(output_path).with_suffix(".checkpoint.json"))
    kg_results: List[Dict[str, Any]] = []
    processed_doc_ids: Set[int] = set()

    # ==== 1. 從 checkpoint 恢復 ====
    if resume and Path(checkpoint_path).exists():
        print(f"📂 Found checkpoint file: {checkpoint_path}")
        checkpoint = load_checkpoint(checkpoint_path)
        processed_doc_ids_raw = checkpoint.get("processed_doc_ids", set())
        if isinstance(processed_doc_ids_raw, list):
            processed_doc_ids = set(processed_doc_ids_raw)
        else:
            processed_doc_ids = processed_doc_ids_raw
        kg_results = checkpoint.get("kg_results", [])
        print(f"   Resuming from checkpoint: {len(processed_doc_ids)} docs already processed")

        if kg_results:
            save_partial_kg(output_path, kg_results)

    # 把還沒處理過的 doc 抽出來
    docs_to_process = [doc for doc in docs if doc.get("doc_id") not in processed_doc_ids]
    total_to_process = len(docs_to_process)
    print(f"Total docs to process this run: {total_to_process}")

    if total_to_process == 0:
        print("No new documents to process. Merging existing KG results...")
        merged_kg = merge_knowledge_graphs(kg_results)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(merged_kg, f, ensure_ascii=False, indent=2)
        print("Done (no new docs).")
        return merged_kg

    print(f"Extracting entities and relations with up to {max_workers} parallel workers...")
    print(f"   Save interval: every {save_interval} newly processed documents")

    newly_processed_count = 0

    # ==== 2. 併發執行每篇文件的抽取 ====
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 建立 future -> doc 對應，方便 debug
        future_to_docid = {}
        for doc in docs_to_process:
            future = executor.submit(_process_single_doc, doc, language)
            future_to_docid[future] = doc.get("doc_id")

        # 用 tqdm 包裝 as_completed，顯示進度
        for future in tqdm(as_completed(future_to_docid), total=total_to_process, desc="Processing docs (parallel)"):
            doc_id, kg_result, error_msg = future.result()

            # 有些 doc 可能 doc_id 是 None 或 content 空
            if doc_id is None:
                continue

            if error_msg is not None:
                print(f"❌ Error processing doc_id {doc_id}: {error_msg}")
                # 即使錯誤也記錄為已處理，避免無限重試
                processed_doc_ids.add(doc_id)
                newly_processed_count += 1
            else:
                if kg_result is not None:
                    kg_results.append(kg_result)
                processed_doc_ids.add(doc_id)
                newly_processed_count += 1

            # 定期保存 partial KG + checkpoint
            if newly_processed_count > 0 and newly_processed_count % save_interval == 0:
                save_partial_kg(output_path, kg_results)
                save_checkpoint(checkpoint_path, processed_doc_ids, kg_results)
                print(
                    f"   Progress: {newly_processed_count}/{total_to_process} "
                    f"new docs processed in this run (total processed: {len(processed_doc_ids)})"
                )

    # ==== 3. 最終合併與保存 ====
    print("Merging knowledge graphs...")
    merged_kg = merge_knowledge_graphs(kg_results)

    print("Knowledge graph built successfully!")
    print(f"  - Total entities: {merged_kg['statistics']['total_entities']}")
    print(f"  - Total relations: {merged_kg['statistics']['total_relations']}")
    print(f"  - Total documents: {merged_kg['statistics']['total_docs']}")

    print(f"Saving final knowledge graph to {output_path}...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(merged_kg, f, ensure_ascii=False, indent=2)

    if Path(checkpoint_path).exists():
        Path(checkpoint_path).unlink()
        print("✅ Checkpoint file removed (processing complete)")

    print("Done!")
    return merged_kg

# =========================
# CLI 入口
# =========================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Build Knowledge Graph from dragonball_docs.jsonl")
    parser.add_argument(
        "--input",
        type=str,
        default="dragonball_dataset/dragonball_docs.jsonl",
        help="Input documents path (JSONL)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="My_RAG/kg_output.json",
        help="Output knowledge graph path (JSON)",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="zh",
        choices=["zh", "en"],
        help="Language (zh or en)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of documents to process (for testing, e.g., --limit 5)",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=10,
        help="Save progress every N documents (default: 10)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Do not resume from checkpoint (start from scratch)",
    )

    args = parser.parse_args()

    # 轉成專案根目錄的相對路徑
    root_dir = Path(__file__).parent.parent
    input_path = root_dir / args.input
    output_path = root_dir / args.output

    build_kg(
        docs_path=str(input_path),
        output_path=str(output_path),
        language=args.language,
        limit=args.limit,
        save_interval=args.save_interval,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()
