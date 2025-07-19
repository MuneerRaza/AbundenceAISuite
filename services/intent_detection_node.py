from typing import Dict, Any
from models.state import State

class IntentDetectionNode:
    def __init__(self):

        # Improved and more descriptive prototype examples
        self.retrieval_keywords = ["pdf", "document", "doc", "file", "attachment"]
        
        self.search_keywords = ["search", "latest", "updates", "recent", "web", "internet"]

    def invoke(self, state: State) -> Dict[str, Any]:
        print("---DETECTING INTENT (Keyword)---")
        user_query = state["user_query"].lower() # Use lowercase for matching
        do_retrieval = state.get("do_retrieval", False)
        do_search = state.get("do_search", False)

        # Check if any keyword for retrieval is in the user's query
        if not do_retrieval and any(keyword in user_query for keyword in self.retrieval_keywords):
            print("Query contains a retrieval keyword. Enabling retrieval.")
            do_retrieval = True

        # Check if any keyword for web search is in the user's query
        if not do_search and any(keyword in user_query for keyword in self.search_keywords):
            print("Query contains a search keyword. Enabling web search.")
            do_search = True

        return {
            "do_retrieval": do_retrieval,
            "do_search": do_search,
        }
