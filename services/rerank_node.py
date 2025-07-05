import torch
from typing import List, Dict
from models.state import State
from langchain.schema import Document
from sentence_transformers.cross_encoder import CrossEncoder

class RerankNode:
    def __init__(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # This cross-encoder is specifically trained for re-ranking tasks.
        self.encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device=device)
        print("✅ Initialized RerankNode with Cross-Encoder")

    def invoke(self, state: State) -> Dict[str, List[Document]]:
        """
        Re-ranks the fused document list for maximum relevance to the original query.
        """
        print("---RE-RANKING DOCUMENTS---")
        query = state["user_query"]
        fused_docs = state.get("fused_docs", [])

        if not fused_docs:
            return {"retrieved_docs": []}

        # Create pairs of [query, document_content] for the model
        pairs = [[query, doc.page_content] for doc in fused_docs]
        
        # Get scores and re-sort the documents
        scores = self.encoder.predict(pairs)
        
        # Combine docs with their scores and sort
        doc_score_pairs = list(zip(fused_docs, scores))
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # Unpack the sorted documents
        reranked_docs = [doc for doc, score in doc_score_pairs]

        print(f"Re-ranked {len(fused_docs)} documents. Top result has score {max(scores):.2f}.")
        # Return only the top K most relevant documents for the final context
        return {"retrieved_docs": reranked_docs}