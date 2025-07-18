import math
from typing import List, Dict
from models.state import State
from langchain.schema import Document
from config import RERANK_THRESHOLD
from . import shared_reranker

class RerankNode:
    def __init__(self):
        self.reranker = shared_reranker

    def _calculate_top_k(self, num_tasks: int, num_retrieved: int) -> int:
        """Dynamic top-k calculation based on number of tasks and retrieved documents."""
        min_k = num_tasks
        max_k = min(num_retrieved, 15)

        task_component = math.pow(num_tasks, 1.25)
        retrieved_component = math.pow(num_retrieved, 0.35)

        raw_k = int(round(task_component + retrieved_component))
        
        top_k = max(min_k, min(raw_k, max_k))

        return top_k

    def _batch_rerank(self, query: str, documents: List[Document], batch_size: int = 32) -> List[float]:
        """
        Rerank documents in batches using FastEmbed cross-encoder.
        """
        if not documents:
            return []
        
        # Extract document texts
        doc_texts = [doc.page_content for doc in documents]
        
        # Process in batches for better performance
        all_scores = []
        for i in range(0, len(doc_texts), batch_size):
            batch_texts = doc_texts[i:i + batch_size]
            
            # Use cross-encoder to rerank the batch
            batch_scores = list(self.reranker.rerank(query, batch_texts))
            all_scores.extend(batch_scores)
        
        return all_scores

    def invoke(self, state: State) -> Dict[str, List[Document]]:
        print("---RE-RANKING DOCUMENTS (FastEmbed)---")
        query = state["user_query"]
        fused_docs = state.get("fused_docs", [])

        if not fused_docs:
            return {"retrieved_docs": []}

        # Batch rerank for better performance
        scores = self._batch_rerank(query, fused_docs)
        
        # Pair documents with scores
        doc_score_pairs = list(zip(fused_docs, scores))
        
        # Filter by score threshold before sorting
        if RERANK_THRESHOLD > 0:
            doc_score_pairs = [pair for pair in doc_score_pairs if pair[1] > RERANK_THRESHOLD]
            print(f"Filtered docs by score threshold > {RERANK_THRESHOLD}. Kept {len(doc_score_pairs)} docs.")

        # Sort by score (descending)
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
        
        reranked_docs = [doc for doc, score in doc_score_pairs]

        # Dynamic Top-K Calculation
        num_tasks = len(state.get("tasks", []))
        num_reranked = len(reranked_docs)
        top_k = self._calculate_top_k(num_tasks, num_reranked)

        print(f"Dynamically selected Top {top_k} documents out of {num_reranked} for {num_tasks} tasks.")
        
        return {"retrieved_docs": reranked_docs[:top_k]}