import re
import asyncio
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

    async def _content_search(self, query: str) -> dict:
        """Perform a content search and return results."""
        print("---SEARCHING THE WEB---")
        try:
            # Run the search in a thread pool since TavilySearch is sync
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(None, self.content_search.invoke, query)
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

    async def invoke(self, state: State):
        tasks = state.get("tasks", [])
        if not tasks:
            return {"web_search_results": []}

        # Create async tasks for parallel search
        search_tasks = [self._content_search(task) for task in tasks]
        
        # Execute all search tasks in parallel
        web_search_results = await asyncio.gather(*search_tasks, return_exceptions=True)
        
        # Handle any exceptions
        processed_results = []
        for i, result in enumerate(web_search_results):
            if isinstance(result, Exception):
                print(f"Search task {i} failed: {result}")
                processed_results.append({'URL': 'N/A', 'content': 'Search failed.'})
            else:
                processed_results.append(result)
        
        return {"web_search_results": processed_results}

    # Sync wrapper for backward compatibility
    def invoke_sync(self, state: State):
        return asyncio.run(self.invoke(state))