"""
检查KG是否被正确使用的脚本
"""

import sys
from pathlib import Path
from config import load_config
from kg_retriever import create_kg_retriever

def check_kg():
    print("=" * 60)
    print("检查KG配置和使用情况")
    print("=" * 60)
    
    # 1. 检查配置
    config = load_config()
    retrieval_config = config.get("retrieval", {})
    kg_boost = retrieval_config.get("kg_boost", 0.0)
    kg_path = retrieval_config.get("kg_path", "My_RAG/kg_output.json")
    kg_max_hops = retrieval_config.get("kg_max_hops", 2)
    
    print(f"\n1. 配置检查:")
    print(f"   kg_boost: {kg_boost}")
    print(f"   kg_path: {kg_path}")
    print(f"   kg_max_hops: {kg_max_hops}")
    
    if kg_boost <= 0:
        print("\n❌ KG未启用！kg_boost <= 0")
        return False
    
    # 2. 检查KG文件是否存在
    print(f"\n2. KG文件检查:")
    kg_file = Path(kg_path)
    if not kg_file.exists():
        print(f"   ❌ KG文件不存在: {kg_path}")
        return False
    else:
        file_size = kg_file.stat().st_size / (1024 * 1024)  # MB
        print(f"   ✓ KG文件存在: {kg_path} ({file_size:.2f} MB)")
    
    # 3. 尝试加载KG
    print(f"\n3. KG加载测试:")
    try:
        kg_retriever = create_kg_retriever(kg_path, "zh", kg_max_hops)
        print(f"   ✓ KG检索器创建成功")
        
        # 检查实体和关系数量
        entities = kg_retriever.kg_data.get("entities", [])
        relations = kg_retriever.kg_data.get("relations", [])
        print(f"   - 实体数量: {len(entities)}")
        print(f"   - 关系数量: {len(relations)}")
        print(f"   - 实体图大小: {len(kg_retriever.entity_graph)}")
        
    except Exception as e:
        print(f"   ❌ KG加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 4. 测试查询
    print(f"\n4. 测试查询:")
    test_query = "绿源环保有限公司在2017年4月发生了什么事故？"
    print(f"   查询: {test_query}")
    
    try:
        doc_ids = kg_retriever.retrieve_doc_ids(test_query, top_k=5, use_multi_hop=True)
        print(f"   ✓ 检索成功，找到 {len(doc_ids)} 个相关文档")
        print(f"   相关doc_ids: {doc_ids[:5]}")
        
        entity_info = kg_retriever.get_entity_info(test_query, use_multi_hop=True)
        entities_found = entity_info.get("entities_found", [])
        print(f"   找到 {len(entities_found)} 个实体")
        if entities_found:
            print(f"   实体示例: {entities_found[0].get('name')}")
        
        multi_hop = entity_info.get("multi_hop", {})
        if multi_hop:
            print(f"   多跳实体数量: {multi_hop.get('multi_hop_entities', 0)}")
        
    except Exception as e:
        print(f"   ❌ 查询测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print(f"\n{'=' * 60}")
    print("✓ 所有检查通过！KG应该可以正常使用")
    print(f"{'=' * 60}")
    return True

if __name__ == "__main__":
    success = check_kg()
    sys.exit(0 if success else 1)

