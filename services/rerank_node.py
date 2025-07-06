import torch
import math # Import the math library
from typing import List, Dict
from models.state import State
from langchain.schema import Document
from sentence_transformers.cross_encoder import CrossEncoder
from config import RERANK_THRESHOLD

class RerankNode:
    def __init__(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device=device)
        print("✅ Initialized RerankNode with Cross-Encoder")

    def _calculate_top_k(self, num_tasks: int, num_retrieved: int) -> int:
        min_k = num_tasks
        max_k = min(num_retrieved, 15)

        task_component = math.pow(num_tasks, 1.25)
        retrieved_component = math.pow(num_retrieved, 0.35)

        raw_k = int(round(task_component + retrieved_component))
        
        top_k = max(min_k, min(raw_k, max_k))

        return top_k


    def invoke(self, state: State) -> Dict[str, List[Document]]:
        print("---RE-RANKING DOCUMENTS---")
        query = state["user_query"]
        fused_docs = state.get("fused_docs", [])

        if not fused_docs:
            return {"retrieved_docs": []}

        pairs = [[query, doc.page_content] for doc in fused_docs]
        scores = self.encoder.predict(pairs)
        
        doc_score_pairs = list(zip(fused_docs, scores))
        
        # Filter by score threshold before sorting
        if RERANK_THRESHOLD > 0:
            doc_score_pairs = [pair for pair in doc_score_pairs if pair[1] > RERANK_THRESHOLD]
            print(f"Filtered docs by score threshold > {RERANK_THRESHOLD}. Kept {len(doc_score_pairs)} docs.")

        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
        
        reranked_docs = [doc for doc, score in doc_score_pairs]

        # Dynamic Top-K Calculation using the new formula
        num_tasks = len(state.get("tasks", []))
        num_reranked = len(reranked_docs)
        top_k = self._calculate_top_k(num_tasks, num_reranked)

        print(f"Dynamically selected Top {top_k} documents out of {num_reranked} for {num_tasks} tasks.")
        
        return {"retrieved_docs": reranked_docs[:top_k]}