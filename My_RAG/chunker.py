import re

def split_sentences(text, language):
    if language == 'en':
        # Split by . ! ? followed by whitespace
        # We use a lookbehind to keep the punctuation
        sentences = re.split(r'(?<=[.!?])\s+', text)
    elif language == 'zh':
        # Split by Chinese punctuation
        sentences = re.split(r'(?<=[。！？])', text)
    else:
        sentences = [text]
    return [s.strip() for s in sentences if s.strip()]

def chunk_documents(docs, language, chunk_size=1000, chunk_overlap=200):
    chunks = []
    for doc in docs:
        if 'content' not in doc or not isinstance(doc['content'], str) or 'language' not in doc:
            continue
        
        doc_lang = doc['language']
        if doc_lang != language:
            continue
            
        text = doc['content']
        sentences = split_sentences(text, language)
        
        i = 0
        chunk_count = 0
        while i < len(sentences):
            current_chunk_sentences = []
            current_length = 0
            
            # 1. Forward pass: Add sentences until chunk_size is reached
            j = i
            while j < len(sentences):
                sent = sentences[j]
                sent_len = len(sent)
                
                # Separator length (space for en, empty for zh)
                sep_len = 1 if language == 'en' and current_length > 0 else 0
                
                if current_length + sep_len + sent_len > chunk_size and current_length > 0:
                    break
                
                current_chunk_sentences.append(sent)
                current_length += sep_len + sent_len
                j += 1
            
            # If a single sentence is larger than chunk_size, we have to add it anyway (or split it further, but let's keep it simple)
            if not current_chunk_sentences and i < len(sentences):
                current_chunk_sentences.append(sentences[i])
                j = i + 1
            
            # Join sentences
            sep = " " if language == 'en' else ""
            chunk_content = sep.join(current_chunk_sentences)
            
            # Create chunk object
            chunk_metadata = doc.copy()
            chunk_metadata.pop('content', None)
            chunk_metadata['chunk_index'] = chunk_count
            chunk = {
                'page_content': chunk_content,
                'metadata': chunk_metadata
            }
            chunks.append(chunk)
            chunk_count += 1
            
            # 2. Calculate next start index (i) based on overlap
            if j >= len(sentences):
                break
                
            # We want to start the next chunk such that it includes the last ~chunk_overlap characters of the current chunk
            overlap_accumulated = 0
            next_i = j 
            
            # Traverse backwards from the end of current chunk (j-1)
            # to find how many sentences fit in chunk_overlap
            k = j - 1
            while k > i:
                sent_len = len(sentences[k])
                sep_len = 1 if language == 'en' and overlap_accumulated > 0 else 0
                
                if overlap_accumulated + sep_len + sent_len > chunk_overlap:
                    break
                
                overlap_accumulated += sep_len + sent_len
                next_i = k
                k -= 1
            
            # Ensure we always advance at least one sentence to avoid infinite loops
            if next_i <= i:
                next_i = i + 1
                
            i = next_i

    return chunks
