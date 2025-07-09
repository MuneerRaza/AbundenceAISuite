from typing import Dict, Any
from models.state import State
from fastembed import TextEmbedding
import numpy as np

class IntentDetectionNode:
    def __init__(self):
        # Using FastEmbed for consistency and performance
        self.embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5", max_length=384)
        
        self.retrieval_prototypes = [
            "look in the attached document", "read the file", "what does the pdf say",
            "analyze the provided text", "refer to the attachment", "search the document",
            "find in the file", "what's in the document", "check the attachment"
        ]
        self.search_prototypes = [
            "what is the latest", "search the web for", "find recent information about",
            "current news about", "latest updates on", "recent developments in",
            "what's happening with", "current status of"
        ]
        print("✅ Initialized IntentDetectionNode with FastEmbed")

    def _compute_similarity(self, query: str, prototypes: list) -> float:
        """Compute maximum similarity between query and prototypes."""
        if not prototypes:
            return 0.0
        
        # Get query embedding
        query_embedding = list(self.embedder.embed([query]))[0]
        
        # Get prototype embeddings
        prototype_embeddings = list(self.embedder.embed(prototypes))
        
        # Compute cosine similarities
        similarities = []
        for proto_emb in prototype_embeddings:
            similarity = np.dot(query_embedding, proto_emb) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(proto_emb)
            )
            similarities.append(float(similarity))
        
        return max(similarities)

    def invoke(self, state: State) -> Dict[str, Any]:
        print("---DETECTING INTENT (FastEmbed)---")
        user_query = state["user_query"]
        do_retrieval = state.get("do_retrieval", False)
        do_search = state.get("do_search", False)

        if do_retrieval and do_search:
            print("🔄 Both retrieval and search are already enabled. Skipping intent detection.")
            return {
                "do_retrieval": do_retrieval,
                "do_search": do_search,
            }
        
        if not do_retrieval:
            retrieval_score = self._compute_similarity(user_query, self.retrieval_prototypes)
            
            if retrieval_score > 0.7:
                print(f"Query implies retrieval. Overriding boolean (max score: {retrieval_score:.2f}).")
                do_retrieval = True

        if not do_search:
            search_score = self._compute_similarity(user_query, self.search_prototypes)
            
            if search_score > 0.6:
                print(f"Query implies web search. Overriding boolean (max score: {search_score:.2f}).")
                do_search = True

        return {
            "do_retrieval": do_retrieval,
            "do_search": do_search,
        }