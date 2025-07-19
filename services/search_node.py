import re
from typing import Dict, List
from models.state import State
from langchain_tavily import TavilyExtract, TavilySearch
from dotenv import load_dotenv
load_dotenv()


class SearchNode:
    def __init__(self):
        self.content_search = TavilySearch(
            max_results=3,
            include_image_descriptions=False,
            search_depth="basic",
        )
        self.url_search = TavilyExtract(search_depth="basic")
        # self.url_pattern = r'\b((?:https?://|www\.)[^\s,<>"]+)'

    def invoke(self, state: State):
        tasks = state.get("tasks", [])
        if not tasks:
            return {"web_search_results": []}

        web_search_results: List[Dict] = []
        for task in tasks:
            web_search_results.append(self._content_search(task))
        
        return { "web_search_results": web_search_results }


    def _content_search(self, query: str) -> dict:
        """Perform a content search and return results."""
        print("---SEARCHING THE WEB---")
        try:
            results = self.content_search.invoke(query)
            formatted_results = self._format_results(results.get("results", []))
            if formatted_results:
                return formatted_results
            return {'URL': 'N/A', 'content': 'No content found.'}
        except Exception as e:
            print(f"Error during Tavily search: {e}")
            return {'URL': 'N/A', 'content': 'Search failed.'}

    def _format_results(self, results: list) -> dict:
        """Format the results into a readable string."""
        formatted_results = {}
        for result in results:
            url = result.get("url", "")
            content = result.get("content", "").strip()
            if url and content:
                formatted_results['URL'] = url
                formatted_results['content'] = content
        return formatted_results if formatted_results else {}