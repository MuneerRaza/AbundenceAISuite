from models.state import State
from langchain_core.messages import HumanMessage
from collections import defaultdict

class AggregationNode:
    def invoke(self, state: State) -> dict:
        print("---AGGREGATING ALL CONTEXT---")
        retrieved_docs = state.get("retrieved_docs", [])
        web_search_results = state.get("web_search_results", "")
        summary = state.get("conversation_summary", "")
        user_query = state.get("user_query", "")

        final_context = ""
        if summary:
            final_context += f"CONVERSATION SUMMARY:\n{summary}\n\n"

        if retrieved_docs:
            final_context += "---EVIDENCE FROM DOCUMENTS---\n"
            docs_by_task = defaultdict(list)
            for doc in retrieved_docs:
                task = doc.metadata.get("source_task", "General")
                docs_by_task[task].append(f"Source: {doc.metadata.get('path', 'N/A')}\nContent: {doc.page_content}")
            
            for task, contents in docs_by_task.items():
                final_context += f"\nRelevant to sub-task: '{task}':\n"
                final_context += "\n\n".join(contents)

        if web_search_results:
            final_context += f"\n\n---EVIDENCE FROM WEB SEARCH---\n{web_search_results}\n"

        prompt_messages = [
            HumanMessage(content=f"User Query: {user_query}\n\n{final_context.strip()}")
        ]
        
        return {"prompt_messages": prompt_messages}