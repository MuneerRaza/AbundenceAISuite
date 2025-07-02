import re
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
        user_query = state.get("user_query", "")
        if not user_query:
            return {"web_search_results": ""}

        return self._content_search(user_query)

    # def _url_search(self, urls: list) -> dict:
    #     print("---SEARCHING THE URLS---")
    #     try:
    #         results = self.url_search.invoke({"urls": urls})
    #         print(f"Results: {results}")
    #         formatted_results = self._format_results(results.get("results", []))
    #         return {"web_search_results": formatted_results}
    #     except Exception as e:
    #         print(f"Error during Tavily URL extraction: {e}")
    #         return {"web_search_results": "Error extracting URLs."}

    def _content_search(self, query: str) -> dict:
        """Perform a content search and return results."""
        print("---SEARCHING THE WEB---")
        try:
            results = self.content_search.invoke(query)
            formatted_results = self._format_results(results.get("results", []))
            if formatted_results:
                return {"web_search_results": formatted_results}
            return {"web_search_results": "No relevant web search results found."}
        except Exception as e:
            print(f"Error during Tavily search: {e}")
            return {"web_search_results": "Error performing web search."}

    def _format_results(self, results: list) -> str:
        """Format the results into a readable string."""
        formatted_results = ""
        for result in results:
            url = result.get("url", "")
            content = result.get("content", "")
            formatted_results += f"\n\nURL: {url}\nSearch Result: {content}\n\n"
        return formatted_results.strip()