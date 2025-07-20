from typing import Annotated, TypedDict, List, Dict
from langchain.schema import Document
from langgraph.graph.message import add_messages

class State(TypedDict):
    recent_messages: Annotated[list, add_messages]
    user_query: str
    conversation_summary: str
    
    # User and thread identification
    user_id: str
    thread_id: str
    
    do_retrieval: bool
    do_search: bool
    
    tasks: List[str]
    
    retrieved_docs: List[Document]
    web_search_results: List[Dict]
    final_context: str