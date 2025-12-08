from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict, Any

def chunk_documents(docs: List[Dict[str, Any]], language: str, 
                    chunk_size_en: int = 1000, chunk_overlap_en: int = 200,
                    chunk_size_zh: int = 384, chunk_overlap_zh: int = 64) -> List[Dict[str, Any]]:
    
    chunks = []
    
    # --- Language-Specific Recursive Character Splitters ---
    SEPARATORS_EN = ["\n\n", "\n", ".", "?", "!", " ", ""]
    SEPARATORS_ZH = ["\n\n", "\n", "。", "！", "？", "；", "：", "，", "、", " "]

    splitter_en = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size_en,
        chunk_overlap=chunk_overlap_en,
        separators=SEPARATORS_EN,
        keep_separator="end",
        strip_whitespace=True
    )
    
    splitter_zh = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size_zh,
        chunk_overlap=chunk_overlap_zh,
        separators=SEPARATORS_ZH,
        keep_separator="end",
        strip_whitespace=True
    )

    for doc_index, doc in enumerate(docs):
        if 'content' in doc and isinstance(doc['content'], str) and 'language' in doc:
            text = doc['content']
            lang = doc['language']
            
            # Select appropriate splitter based on the document's language
            if lang == 'zh':
                splitter = splitter_zh
            else: # Default to English splitter for other languages or if language not explicitly 'zh'
                splitter = splitter_en
            
            if lang == language: # Only process documents matching the target language for the run
                doc_chunks = splitter.split_text(text)
                for i, chunk_text in enumerate(doc_chunks):
                    chunk_metadata = doc.copy()
                    chunk_metadata.pop('content', None)
                    chunk_metadata['chunk_index'] = i
                    chunk_metadata['doc_idx'] = doc_index
                    chunks.append({'page_content': chunk_text, 'metadata': chunk_metadata})
                    
    return chunks