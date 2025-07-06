from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableParallel
from langchain_groq import ChatGroq

# Import all nodes (OrchestratorNode is removed)
from services.intent_detection_node import IntentDetectionNode
from services.decompose_node import DecomposeNode
from services.retrieval_node import RetrievalNode
from services.rerank_node import RerankNode
from services.search_node import SearchNode
from services.aggregation_node import AggregationNode
from services.call_model import CallModelNode

from models.state import State
from utils.checkpointer import checkpointer
from config import MODEL_ID, UTILS_MODEL_ID

# --- Node Initialization ---
utils_model = ChatGroq(model=UTILS_MODEL_ID)
main_model = ChatGroq(model=MODEL_ID)

intent_detector = IntentDetectionNode()
decomposer = DecomposeNode(llm=utils_model)
# The orchestrator is no longer initialized
retriever = RetrievalNode()
reranker = RerankNode()
searcher = SearchNode()
aggregator = AggregationNode()
call_model = CallModelNode(model=main_model)

# --- Routing and Parallel Logic ---
def route_after_decomposition(state: State) -> str:
    """Routes to the correct evidence gathering path after decomposition."""
    do_retrieval = state.get("do_retrieval")
    do_search = state.get("do_search")

    if do_retrieval and do_search:
        return "parallel_evidence"
    if do_retrieval:
        return "retrieve_only"
    if do_search:
        return "search_only"
    return "direct_to_llm"


def build_workflow():
    workflow = StateGraph(State)

    # Add all nodes to the graph
    workflow.add_node("intent_detector", intent_detector.invoke)
    workflow.add_node("decompose", decomposer.invoke)

    # This parallel node runs retrieval and search concurrently
    def parallel_node(state):
        parallel_evidence_gatherer = RunnableParallel(retrieval=retriever.invoke, search=searcher.invoke)
        results = parallel_evidence_gatherer.invoke(state)
        return {
            "fused_docs": results.get('retrieval', {}).get('fused_docs', []),
            "web_search_results": results.get('search', {}).get('web_search_results', '')
        }
    workflow.add_node("parallel_evidence", parallel_node)

    workflow.add_node("retrieve_only", retriever.invoke)
    workflow.add_node("search_only", searcher.invoke)
    workflow.add_node("rerank", reranker.invoke)
    workflow.add_node("aggregate", aggregator.invoke)
    workflow.add_node("call_model", call_model.invoke)

    # --- Graph Wiring ---
    workflow.set_entry_point("intent_detector")

    # 1. After intent, decompose the task
    workflow.add_edge("intent_detector", "decompose")

    # 2. After decomposition, route to the correct evidence gathering path
    workflow.add_conditional_edges(
        "decompose",
        route_after_decomposition,
        {
            "parallel_evidence": "parallel_evidence",
            "retrieve_only": "retrieve_only",
            "search_only": "aggregate", # Search doesn't need reranking
            "direct_to_llm": "aggregate"
        }
    )

    # 3. Evidence gathering paths converge on reranking or aggregation
    workflow.add_edge("parallel_evidence", "rerank")
    workflow.add_edge("retrieve_only", "rerank")
    workflow.add_edge("rerank", "aggregate")

    # 4. Aggregation to Final Call
    workflow.add_edge("aggregate", "call_model")
    workflow.add_edge("call_model", END)

    graph = workflow.compile(checkpointer=checkpointer)
    try:
        # Saving the graph diagram is optional but helpful for visualization
        png_bytes = graph.get_graph().draw_mermaid_png()
        output_file_path = "workflow_diagram.png"
        with open(output_file_path, "wb") as f:
            f.write(png_bytes)
        print(f"✅ Graph diagram successfully saved to: {output_file_path}")
    except Exception as e:
        print(f"⚠️  Could not generate workflow diagram: {e}")
    return graph