from typing import TypedDict, List, Optional, Any
from langchain_core.messages import BaseMessage
from langchain.schema import Document

class State(TypedDict):
    recent_messages: List[BaseMessage]
    user_query: str
    conversation_summary: str
    
    do_retrieval: bool
    do_search: bool
    
    tasks: List[str]
    
    fused_docs: List[Document]
    retrieved_docs: List[Document]
    
    web_search_results: str
    
    prompt_messages: List[BaseMessage]
    final_context: Optional[str]
