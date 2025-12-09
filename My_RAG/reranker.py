from sentence_transformers import CrossEncoder
from typing import List, Dict, Any

class Reranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initializes the Reranker with a specified Cross-Encoder model.
        """
        try:
            self.model = CrossEncoder(model_name)
            print(f"Loaded Cross-Encoder model: {model_name}")
        except Exception as e:
            print(f"Failed to load Cross-Encoder model {model_name}: {e}")
            self.model = None

    def rerank(self, query: str, chunks: List[Dict[str, Any]], top_n: int = 5) -> List[Dict[str, Any]]:
        """
        Re-ranks a list of chunks based on their relevance to the query using a Cross-Encoder.

        Args:
            query (str): The search query.
            chunks (List[Dict[str, Any]]): A list of chunk dictionaries, each containing 'page_content'.
            top_n (int): The number of top chunks to return after re-ranking.

        Returns:
            List[Dict[str, Any]]: The re-ranked list of chunk dictionaries.
        """
        if not self.model or not chunks:
            return chunks # Return original chunks if model not loaded or no chunks

        # Prepare sentence pairs for the Cross-Encoder
        sentence_pairs = [[query, chunk["page_content"]] for chunk in chunks]

        # Predict scores
        # scores = self.model.predict(sentence_pairs, show_progress_bar=False)
        # For some reason, running with show_progress_bar=False fails, so I'll leave it as default.
        scores = self.model.predict(sentence_pairs)

        # Pair chunks with their scores and sort
        chunk_scores = []
        for i, chunk in enumerate(chunks):
            chunk_copy = chunk.copy()
            chunk_copy["rerank_score"] = float(scores[i]) # Ensure score is a plain float
            chunk_scores.append(chunk_copy)
        
        chunk_scores.sort(key=lambda x: x["rerank_score"], reverse=True)

        return chunk_scores[:top_n]

if __name__ == '__main__':
    # Example Usage:
    reranker = Reranker()
    query = "What is the capital of France?"
    documents = [
        {"page_content": "Paris is the capital of France.", "metadata": {"source": "wiki"}},
        {"page_content": "The Eiffel Tower is in Paris.", "metadata": {"source": "travel"}},
        {"page_content": "France is a country in Europe.", "metadata": {"source": "geo"}},
        {"page_content": "The capital of Germany is Berlin.", "metadata": {"source": "wiki"}},
    ]

    reranked_docs = reranker.rerank(query, documents, top_n=2)
    for doc in reranked_docs:
        print(f"Content: {doc['page_content']}, Rerank Score: {doc['rerank_score']:.4f}")

