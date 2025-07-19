from typing import Annotated, TypedDict, List
from langchain.schema import Document
from langgraph.graph.message import add_messages

class State(TypedDict):
    recent_messages: Annotated[list, add_messages]
    user_query: str
    conversation_summary: str
    
    do_retrieval: bool
    do_search: bool
    
    tasks: List[str]
    
    retrieved_docs: List[Document]
    web_search_results: str
    final_context: str