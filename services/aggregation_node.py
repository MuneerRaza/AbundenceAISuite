from typing import Dict
from models.state import State
from langchain_core.messages import HumanMessage

class AggregationNode:

    def _aggregate(self, retrieved_docs, web_search_results):

        print("---AGGREGATING CONTEXT---")
        
        final_context = ""

        if retrieved_docs:
            formatted_docs = "\n\n".join(
                [f"Source: {doc.metadata.get('path', 'N/A')}\nContent: {doc.page_content}" for doc in retrieved_docs]
            )
            final_context += f"---Information from Documents---\n{formatted_docs}\n\n"
        
        if web_search_results:
            final_context += f"---Information from Web Search---\n{web_search_results}\n"
            
        if not final_context:
            final_context = ""

        return final_context.strip()

        
    
    def invoke(self, state: State):
        retrieved_docs = state.get("retrieved_docs", [])
        web_search_results = state.get("web_search_results", "")
        final_context = self._aggregate(retrieved_docs, web_search_results)
        summary = state.get("conversation_summary", "")
        recent = state.get("recent_messages", [])
        user_query = state.get("user_query", "")


        prompt_messages = []

        if summary:
            prompt_messages.append(HumanMessage(
                content=f"CONVERSATION SUMMARY:\n{summary}"
            ))

        if len(recent) > 1:
            prompt_messages.extend(recent[:-1])

        if final_context:
            final_message = f"User query: {user_query}\n\nContext:\n{final_context}"
        else:
            final_message = f"User query: {user_query}"
        prompt_messages.append(HumanMessage(content=final_message))
            
        return {"prompt_messages": prompt_messages}