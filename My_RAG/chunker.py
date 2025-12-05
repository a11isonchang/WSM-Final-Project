import re
import nltk
from typing import List, Dict, Any

# Download required NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)


def _split_sentences(text: str, language: str) -> List[str]:
    """
    Split text into sentences based on language.
    """
    if language == "zh":
        # Chinese sentence splitting based on punctuation
        sentences = re.split(r'([。！？\n]+)', text)
        # Combine punctuation with preceding text
        result = []
        for i in range(0, len(sentences) - 1, 2):
            if i + 1 < len(sentences):
                result.append(sentences[i] + sentences[i + 1])
            else:
                result.append(sentences[i])
        if len(sentences) % 2 == 1:
            result.append(sentences[-1])
        return [s.strip() for s in result if s.strip()]
    else:
        # English sentence splitting using NLTK
        try:
            return nltk.sent_tokenize(text)
        except Exception:
            # Fallback to simple splitting
            return re.split(r'(?<=[.!?])\s+', text)


def _create_semantic_chunks(
    sentences: List[str], 
    chunk_size: int, 
    chunk_overlap: int
) -> List[str]:
    """
    Create chunks from sentences, respecting semantic boundaries.
    Ensures chunks are close to chunk_size but don't break sentences.
    """
    if not sentences:
        return []
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    i = 0
    while i < len(sentences):
        sentence = sentences[i]
        sentence_len = len(sentence)
        
        # If single sentence exceeds chunk_size, add it as its own chunk
        if sentence_len > chunk_size:
            if current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_length = 0
            chunks.append(sentence)
            i += 1
            continue
        
        # If adding this sentence would exceed chunk_size, finalize current chunk
        if current_length + sentence_len > chunk_size and current_chunk:
            chunks.append(' '.join(current_chunk))
            
            # Create overlap by going back
            overlap_chunk = []
            overlap_length = 0
            j = len(current_chunk) - 1
            
            # Collect sentences for overlap from the end
            while j >= 0 and overlap_length < chunk_overlap:
                sent = current_chunk[j]
                overlap_length += len(sent)
                overlap_chunk.insert(0, sent)
                j -= 1
            
            current_chunk = overlap_chunk
            current_length = overlap_length
        
        # Add sentence to current chunk
        current_chunk.append(sentence)
        current_length += sentence_len + 1  # +1 for space
        i += 1
    
    # Add final chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks


def chunk_documents(docs, language, chunk_size=1000, chunk_overlap=200):
    """
    Intelligent chunking strategy that respects sentence boundaries.
    
    Args:
        docs: List of documents with 'content' and 'language' fields
        language: Target language to filter ('zh' or 'en')
        chunk_size: Target size for each chunk (characters)
        chunk_overlap: Overlap between consecutive chunks (characters)
    
    Returns:
        List of chunks with 'page_content' and 'metadata'
    """
    chunks = []
    
    for doc_index, doc in enumerate(docs):
        if 'content' not in doc or not isinstance(doc['content'], str):
            continue
        if 'language' not in doc or doc['language'] != language:
            continue
        
        text = doc['content'].strip()
        if not text:
            continue
        
        # Split into sentences
        sentences = _split_sentences(text, language)
        
        if not sentences:
            continue
        
        # Create semantic chunks
        chunk_texts = _create_semantic_chunks(sentences, chunk_size, chunk_overlap)
        
        # Create chunk objects with metadata
        for chunk_index, chunk_text in enumerate(chunk_texts):
            chunk_metadata = doc.copy()
            chunk_metadata.pop('content', None)
            chunk_metadata['chunk_index'] = chunk_index
            chunk_metadata['total_chunks'] = len(chunk_texts)
            
            chunk = {
                'page_content': chunk_text,
                'metadata': chunk_metadata
            }
            chunks.append(chunk)
    
    return chunks
