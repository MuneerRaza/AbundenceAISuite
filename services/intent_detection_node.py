import os
import pickle
from typing import Dict, Any
from models.state import State
from . import shared_reranker

class IntentDetectionNode:
    def __init__(self):
        self.reranker = shared_reranker

        # Improved and more descriptive prototype examples
        self.retrieval_prototypes = [
            "look in the attached document", "read the file", "what does the pdf say", 
            "refer to the attachment", "search the document", "find in the file", 
            "what's in the document", "check the attachment", "see document content",
            "extract info from the document", "look up in the uploaded file"
        ]
        
        self.search_prototypes = [
            "what is the latest", "search the web for", "find recent information about", 
            "latest updates on", "recent developments in",
            "current status of", "search online", "look up on the internet",
            "web search for", "fetch from web"
        ]

    def _compute_similarity(self, query: str, prototypes: list) -> float:

        scores = list(self.reranker.rerank(query, prototypes))
        score = max(scores) if scores else 0.0

        return score

    def invoke(self, state: State) -> Dict[str, Any]:
        print("---DETECTING INTENT (Reranker)---")
        user_query = state["user_query"]
        do_retrieval = state.get("do_retrieval", False)
        do_search = state.get("do_search", False)

        if do_retrieval and do_search:
            print("Both retrieval and search are already enabled. Skipping intent detection.")
            return {
                "do_retrieval": do_retrieval,
                "do_search": do_search,
            }

        if not do_retrieval:
            retrieval_score = self._compute_similarity(user_query, self.retrieval_prototypes)
            print(f"[IntentDetectionNode] Retrieval Score: {retrieval_score:.2f}")
            if retrieval_score > 0.3:
                print(f"Query implies retrieval with score {retrieval_score:.2f}. Overriding boolean.")
                do_retrieval = True

        if not do_search:
            search_score = self._compute_similarity(user_query, self.search_prototypes)
            print(f"[IntentDetectionNode] Search Score: {search_score:.2f}")
            if search_score > 0.2:
                print(f"Query implies web search with score {search_score:.2f}. Overriding boolean.")
                do_search = True

        return {
            "do_retrieval": do_retrieval,
            "do_search": do_search,
        }
