from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_documents(docs, language, chunk_size=1000, chunk_overlap=200):
    """
    Chunk documents using LangChain's RecursiveCharacterTextSplitter.
    """
    chunks = []
    
    # Define separators based on language
    if language == 'zh':
        separators = [
            "\n\n",
            "\n",
            "。\n",  # Chinese period + newline
            "！\n",  # Chinese exclamation + newline
            "？\n",  # Chinese question + newline
            "。",    # Chinese period
            "！",    # Chinese exclamation
            "？",    # Chinese question
            "；",    # Chinese semicolon
            "；",    # Chinese semicolon
            " ",
            ""
        ]
        # Note: LangChain length_function defaults to len(), which is fine for characters
    else:
        # Default separators for English (paragraphs, sentences, words)
        separators = ["\n\n", "\n", " ", ""]

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=separators,
        is_separator_regex=False 
    )

    for doc in docs:
        if 'content' not in doc or not isinstance(doc['content'], str) or 'language' not in doc:
            continue
        
        doc_lang = doc['language']
        if doc_lang != language:
            continue
            
        text = doc['content']
        
        # Use LangChain to split text
        split_texts = text_splitter.split_text(text)
        
        for i, chunk_content in enumerate(split_texts):
            # Create chunk object preserving metadata
            chunk_metadata = doc.copy()
            chunk_metadata.pop('content', None)
            chunk_metadata['chunk_index'] = i
            
            chunk = {
                'page_content': chunk_content,
                'metadata': chunk_metadata
            }
            chunks.append(chunk)

    return chunks
