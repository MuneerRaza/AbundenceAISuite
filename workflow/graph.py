import asyncio
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableParallel
from langchain_groq import ChatGroq

from services.intent_detection_node import IntentDetectionNode
from services.decompose_node import DecomposeNode
from services.retrieval_node import RetrievalNode
from services.search_node import SearchNode
from services.evaluater_node import EvaluatorNode
from services.aggregation_node import AggregationNode
from services.call_model import CallModelNode

from models.state import State
from utils.checkpointer import get_checkpointer
from config import MODEL_ID, UTILS_MODEL_ID


utils_model = ChatGroq(model=UTILS_MODEL_ID)
main_model = ChatGroq(model=MODEL_ID)

intent_detector = IntentDetectionNode()
decomposer = DecomposeNode(llm=utils_model)
retriever = RetrievalNode()
evaluator = EvaluatorNode(llm=utils_model)
searcher = SearchNode()
aggregator = AggregationNode()
call_model = CallModelNode(model=main_model)

def route_after_intent_detection(state: State) -> str:
    do_retrieval = state.get("do_retrieval")
    do_search = state.get("do_search")
    
    if do_retrieval or do_search:
        return "decompose"
    
    return "direct_to_llm"

def route_after_decomposition(state: State) -> str:
    do_retrieval = state.get("do_retrieval")
    do_search = state.get("do_search")
    if do_retrieval and do_search:
        return "parallel_evidence"
    if do_retrieval:
        return "retrieve_only"
    if do_search:
        return "search_only"
    return "direct_to_llm"


async def parallel_node(state):
    """Execute retrieval and search in parallel using async operations."""
    # Create async tasks for parallel execution
    retrieval_task = retriever.invoke(state)
    search_task = searcher.invoke(state)
    
    # Execute both tasks in parallel
    retrieval_result, search_result = await asyncio.gather(
        retrieval_task, search_task, return_exceptions=True
    )
    
    # Handle exceptions and extract results
    if isinstance(retrieval_result, Exception):
        print(f"Retrieval failed: {retrieval_result}")
        retrieved_docs = []
    else:
        # Type assertion to help linter understand this is a dict
        retrieval_dict = retrieval_result if isinstance(retrieval_result, dict) else {}
        retrieved_docs = retrieval_dict.get('retrieved_docs', [])
    
    if isinstance(search_result, Exception):
        print(f"Search failed: {search_result}")
        web_search_results = []
    else:
        # Type assertion to help linter understand this is a dict
        search_dict = search_result if isinstance(search_result, dict) else {}
        web_search_results = search_dict.get('web_search_results', [])
    
    return {
        "retrieved_docs": retrieved_docs,
        "web_search_results": web_search_results
    }

def build_workflow():
    workflow = StateGraph(State)

    workflow.add_node("intent_detector", intent_detector.invoke)
    workflow.add_node("decompose", decomposer.invoke)
    workflow.add_node("parallel_evidence", parallel_node)
    workflow.add_node("retrieve_only", retriever.invoke)
    workflow.add_node("search_only", searcher.invoke)
    workflow.add_node("evaluate", evaluator.invoke)
    workflow.add_node("aggregate", aggregator.invoke)
    workflow.add_node("call_model", call_model.invoke)

    workflow.set_entry_point("intent_detector")

    workflow.add_conditional_edges(
        "intent_detector",
        route_after_intent_detection,
        {
            "decompose": "decompose",
            "direct_to_llm": "aggregate"
        }
    )

    workflow.add_conditional_edges(
        "decompose",
        route_after_decomposition,
        {
            "retrieve_only": "retrieve_only",
            "search_only": "search_only",
            "direct_to_llm": "aggregate",
            "parallel_evidence": "parallel_evidence"
        }
    )

    workflow.add_edge("parallel_evidence", "evaluate")
    workflow.add_edge("retrieve_only", "evaluate")
    workflow.add_edge("search_only", "aggregate")
    workflow.add_edge("evaluate", "aggregate")

    workflow.add_edge("aggregate", "call_model")
    workflow.add_edge("call_model", END)

    graph = workflow.compile(checkpointer=get_checkpointer())
    # try:
    #     png_bytes = graph.get_graph().draw_mermaid_png()
    #     output_file_path = "workflow_diagram.png"
    #     with open(output_file_path, "wb") as f:
    #         f.write(png_bytes)
    #     print(f"✅ Graph diagram successfully saved to: {output_file_path}")
    # except Exception as e:
    #     print(f"⚠️  Could not generate workflow diagram: {e}")
    return graph