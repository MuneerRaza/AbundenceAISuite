from typing import TypedDict, Annotated, Sequence, List
from langchain_core.messages import BaseMessage
from langchain.schema import Document
from langgraph.graph.message import add_messages 

class State(TypedDict):
    recent_messages: Annotated[Sequence[BaseMessage], add_messages]
    user_query: str
    conversation_summary: str

    do_retrieval: bool
    do_search: bool
    tasks: List[str]
    task_plans: List

    fused_docs: List[Document]
    web_search_results: str
    retrieved_docs: List[Document]
    
    prompt_messages: List