from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableParallel
from langchain_groq import ChatGroq

# Import all nodes
from services.intent_detection_node import IntentDetectionNode
from services.decompose_node import DecomposeNode
from services.orchestration_node import OrchestratorNode
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
orchestrator = OrchestratorNode(llm=utils_model)
retriever = RetrievalNode()
reranker = RerankNode()
searcher = SearchNode()
aggregator = AggregationNode()
call_model = CallModelNode(model=main_model)

# --- Routing and Parallel Logic ---
def route_after_intent(state: State) -> str:
    """Routes to the correct path based on the detected intent."""
    if state.get("do_retrieval"):
        # If retrieval is needed, we ALWAYS go down the decomposition path first
        # to ensure tasks are planned correctly.
        return "decompose"
    if state.get("do_search"):
        # If ONLY search is needed, we can go straight to the search node.
        return "search_only"

    # If neither is needed, it's a simple conversational turn.
    return "direct_to_llm"

def route_after_orchestration(state: State) -> str:
    """Decides whether to run search in parallel with retrieval."""
    if state.get("do_search"):
        # This check happens AFTER planning, ensuring we only do parallel
        # search if the intent was also to retrieve.
        return "parallel_evidence"
    else:
        return "retrieve_only"


def build_workflow():
    workflow = StateGraph(State)

    # Add all nodes to the graph (Summarizer is removed from this core graph)
    workflow.add_node("intent_detector", intent_detector.invoke)
    workflow.add_node("decompose", decomposer.invoke)
    workflow.add_node("orchestrate", orchestrator.invoke)

    def parallel_node(state):
        # Using RunnableParallel to execute retrieval and search concurrently
        parallel_evidence_gatherer = RunnableParallel(retrieval=retriever.invoke, search=searcher.invoke)
        results = parallel_evidence_gatherer.invoke(state)
        # The retrieval node must be updated to return a dict with 'fused_docs'
        return {
            "fused_docs": results.get('retrieval', {}).get('fused_docs', []),
            "web_search_results": results.get('search', {}).get('web_search_results', '')
        }
    workflow.add_node("parallel_evidence", parallel_node)

    workflow.add_node("retrieve_only", retriever.invoke) # This node now outputs to 'fused_docs'
    workflow.add_node("search_only", searcher.invoke)
    workflow.add_node("rerank", reranker.invoke)
    workflow.add_node("aggregate", aggregator.invoke)
    workflow.add_node("call_model", call_model.invoke)

    # --- Graph Wiring ---
    workflow.set_entry_point("intent_detector")

    # 1. Routing from intent detection
    workflow.add_conditional_edges(
        "intent_detector",
        route_after_intent,
        {"decompose": "decompose", "search_only": "search_only", "direct_to_llm": "call_model"}
    )
    # 2. Retrieval Path
    workflow.add_edge("decompose", "orchestrate")
    workflow.add_conditional_edges("orchestrate", route_after_orchestration, {
        "parallel_evidence": "parallel_evidence", "retrieve_only": "retrieve_only"
    })
    workflow.add_edge("parallel_evidence", "rerank")
    workflow.add_edge("retrieve_only", "rerank")
    workflow.add_edge("rerank", "aggregate")
    # 3. Search-Only Path
    workflow.add_edge("search_only", "aggregate")
    # 4. Aggregation to Final Call
    workflow.add_edge("aggregate", "call_model")

    workflow.add_edge("call_model", END)

    graph = workflow.compile(checkpointer=checkpointer)
    try:
        png_bytes = graph.get_graph().draw_mermaid_png()
        output_file_path = r"D:\Projects\AbundenceAISuite\workflow_diagram.png"

        with open(output_file_path, "wb") as f:
            f.write(png_bytes)
        
        print(f"✅ Graph diagram successfully saved to: {output_file_path}")
    except Exception as e:
        print(f"⚠️  Could not generate workflow diagram: {e}")
    return graph
