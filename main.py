import asyncio
import threading
from workflow.graph import build_workflow
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from config import THREAD_ID, USER_ID, SUMMARY_THRESHOLD, UTILS_MODEL_ID
from utils.checkpointer import checkpointer, delete_thread_sync
from services.summarizer import SummarizerNode
from langchain_groq import ChatGroq
from models.state import State
import os

os.environ["ANONYMIZED_TELEMETRY"] = "false"
os.environ['FASTEMBED_CACHE_PATH'] = 'cache'

utils_model = ChatGroq(model=UTILS_MODEL_ID)
summarizer = SummarizerNode(llm=utils_model)

# MODIFICATION: The function now accepts the 'graph' object
async def run_background_summarization(config, graph):
    print("\n---Starting background summarization...---")
    try:
        state = graph.get_state(config)
        updates = await summarizer.run(state)
        if updates:
            graph.update_state(config, updates)
            print("---Background summarization complete and state updated.---")
        else:
            print("---No summarization was needed.---")
    except Exception as e:
        print(f"---Error during background summarization: {e}---")

def run_background_summarization_sync(config, graph):
    """Sync wrapper for background summarization."""
    asyncio.run(run_background_summarization(config, graph))

async def run_workflow_async():
    graph = build_workflow()
    config = RunnableConfig(configurable={"thread_id": THREAD_ID})

    print("Welcome to the Chatbot! Type 'q' to quit or 'd' to delete the thread.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "q":
            break
        if user_input.lower() == "d":
            delete_thread_sync(USER_ID, THREAD_ID)
            print(f"Thread '{THREAD_ID}' for user '{USER_ID}' deleted.")
            continue

        # Simulate front-end booleans for intent
        use_attachments = "pdf" in user_input.lower()
        use_internet = "search" in user_input.lower()

        initial_state = State({
            "recent_messages": [HumanMessage(content=user_input)],
            "user_query": user_input,
            "conversation_summary": "",
            "do_retrieval": use_attachments,
            "do_search": use_internet,
            "tasks": [],
            "retrieved_docs": [],
            "web_search_results": [],
            "final_context": "",
        })

        response = await graph.ainvoke(initial_state, config=config)
        final_message = response['recent_messages'][-1].content
        print(f"Bot: {final_message}")
        current_state = graph.get_state(config)

        if len(current_state.values.get("recent_messages", [])) > SUMMARY_THRESHOLD:
            # Run background summarization in a separate thread
            summary_thread = threading.Thread(
                target=run_background_summarization_sync,
                args=(config, graph)
            )
            summary_thread.start()

def run_workflow():
    """Sync wrapper for the async workflow."""
    asyncio.run(run_workflow_async())


if __name__ == "__main__":
    run_workflow()