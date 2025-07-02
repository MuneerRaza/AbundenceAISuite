import re
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from services.summarizer import SummarizerNode
# from services.rewriter import RewriterNode
from services.query_router import QueryRouterNode
from services.decompose_node import DecomposeNode
from services.retrieval_node import RetrievalNode
from services.search_node import SearchNode
from services.aggregation_node import AggregationNode
from services.call_model import CallModelNode
from models.state import State
from utils.checkpointer import checkpointer
from config import SUMMARY_THRESHOLD, MODEL_ID, UTILS_MODEL_ID, THREAD_ID

from dotenv import load_dotenv
load_dotenv()

# TODO: setup proper config

from langchain_core.runnables import RunnableConfig

config = RunnableConfig(configurable={"thread_id": THREAD_ID})

def summary_router(state):
    if len(state.get("recent_messages", [])) > SUMMARY_THRESHOLD:
        return "summarize"
    else:
        return "continue"

def search_router(state: State) -> str:
    if not state["recent_messages"]:
        return "no"
    # TODO: Check boolean flag in state for web search

    content = state["user_query"] 

    url_pattern = r'\b((?:https?://|www\.)[^\s,<>"]+)'
    if re.search(url_pattern, content):
        return "yes"

    return "no"
    

model = ChatGroq(model=MODEL_ID)
utils_model = ChatGroq(model=UTILS_MODEL_ID)
summarizer = SummarizerNode(llm=utils_model)
# rewriter = RewriterNode(llm=utils_model)
query_router = QueryRouterNode(llm=utils_model)
decomposer = DecomposeNode(llm=utils_model)
retriever = RetrievalNode(config=config)
searcher = SearchNode()
aggregator = AggregationNode()
call_model = CallModelNode(model=model)

def build_workflow():

    workflow = StateGraph(State)
    workflow.add_node("summarizer", summarizer.run)
    # workflow.add_node("rewrite_query", rewriter.invoke)
    workflow.add_node("decompose", decomposer.invoke)
    workflow.add_node("retrieve", retriever.invoke)
    workflow.add_node("search", searcher.invoke)
    workflow.add_node("aggregate", aggregator.invoke)
    workflow.add_node("call_model", call_model.invoke)


    workflow.add_conditional_edges(
        START,
        summary_router,
        {
            "summarize": "summarizer",
            "continue": "aggregate",
        },
    )
    workflow.add_conditional_edges(
        START,
        query_router.invoke,
        {
            "informational": "decompose",
            "conversational": "call_model",
        },
    )
    workflow.add_conditional_edges(
        START,
        search_router,
        {
            "yes": "search",
            "no": "aggregate",
        },
    )
    workflow.add_edge("decompose", "retrieve")
    workflow.add_edge("summarizer", "aggregate")
    workflow.add_edge("retrieve", "aggregate")
    workflow.add_edge("search", "aggregate")
    workflow.add_edge("aggregate", "call_model")
    workflow.add_edge("call_model", END)

    graph = workflow.compile(checkpointer=checkpointer)
    
    # try:
    #     png_bytes = graph.get_graph().draw_mermaid_png()
    #     output_file_path = r"D:\Projects\AbundenceAISuite\workflow_diagram.png"

    #     with open(output_file_path, "wb") as f:
    #         f.write(png_bytes)
        
    #     print(f"✅ Graph diagram successfully saved to: {output_file_path}")
    # except Exception:
    #     pass
    return graph