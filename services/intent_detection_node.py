import torch
from typing import Dict, Any
from models.state import State
from sentence_transformers.cross_encoder import CrossEncoder

class IntentDetectionNode:
    def __init__(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device=device)
        
        # Prototypes representing the core intent
        self.retrieval_prototypes = [
            "look in the attached document", "read the file", "what does the pdf say",
            "analyze the provided text", "refer to the attachment"
        ]
        self.search_prototypes = [
            "what is the latest news on", "search the web for", "find recent information about",
            "who is the current CEO of", "what is the weather today"
        ]
        print("✅ Initialized IntentDetectionNode with Cross-Encoder")

    def invoke(self, state: State) -> Dict[str, Any]:
        """
        Determines user intent using a cross-encoder for high accuracy.
        """
        print("---DETECTING INTENT (CROSS-ENCODER)---")
        user_query = state["user_query"]
        do_retrieval = state.get("do_retrieval", False)
        do_search = state.get("do_search", False)

        # If both are already true, return as is
        if do_retrieval and do_search:
            print("🔄 Both retrieval and search are already enabled. Skipping intent detection.")
            return {
                "do_retrieval": do_retrieval,
                "do_search": do_search,
            }

        # Only process retrieval scoring if do_retrieval is not already true
        if not do_retrieval:
            retrieval_pairs = [[user_query, proto] for proto in self.retrieval_prototypes]
            retrieval_scores = self.encoder.predict(retrieval_pairs)
            
            if max(retrieval_scores) > 0.5: # 0.5 is a good threshold
                print(f"💡 Query implies retrieval. Overriding boolean (max score: {max(retrieval_scores):.2f}).")
                do_retrieval = True

        # Only process search scoring if do_search is not already true
        if not do_search:
            search_pairs = [[user_query, proto] for proto in self.search_prototypes]
            search_scores = self.encoder.predict(search_pairs)
            
            if max(search_scores) > 0.1: # Lower threshold for search
                print(f"💡 Query implies web search. Overriding boolean (max score: {max(search_scores):.2f}).")
                do_search = True

        return {
            "do_retrieval": do_retrieval,
            "do_search": do_search,
        }