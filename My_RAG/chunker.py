def chunk_documents(docs, language, chunk_size=1000, chunk_overlap=200):
    # Validate parameters to prevent infinite loop
    if chunk_overlap >= chunk_size:
        raise ValueError(f"chunk_overlap ({chunk_overlap}) must be < chunk_size ({chunk_size})")
    
    chunks = []
    for doc_index, doc in enumerate(docs):
        if 'content' in doc and isinstance(doc['content'], str) and 'language' in doc:
            text = doc['content']
            text_len = len(text)
            lang = doc['language']
            
            # Skip documents of different language
            if lang != language:
                continue
            
            # Skip empty documents
            if not text.strip():
                continue
            
            start_index = 0
            chunk_count = 0
            step = chunk_size - chunk_overlap
            
            while start_index < text_len:
                end_index = min(start_index + chunk_size, text_len)
                chunk_text = text[start_index:end_index].strip()
                
                # Only add non-empty chunks
                if chunk_text:
                    chunk_metadata = doc.copy()
                    chunk_metadata.pop('content', None)
                    chunk_metadata['chunk_index'] = chunk_count
                    chunk = {
                        'page_content': chunk_text,
                        'metadata': chunk_metadata
                    }
                    chunks.append(chunk)
                    chunk_count += 1
                
                start_index += step
    return chunks
