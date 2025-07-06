import threading
from workflow.graph import build_workflow
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from config import THREAD_ID, SUMMARY_THRESHOLD, UTILS_MODEL_ID
from utils.checkpointer import checkpointer, delete_thread
from services.summarizer import SummarizerNode
from langchain_groq import ChatGroq
from models.state import State
import os

os.environ["ANONYMIZED_TELEMETRY"] = "false"

# --- Initialize Summarizer separately for background tasks ---
utils_model = ChatGroq(model=UTILS_MODEL_ID)
summarizer = SummarizerNode(llm=utils_model)

def run_background_summarization(config):
    """Function to run the summarizer in a separate, non-blocking thread."""
    print("\n---Starting background summarization...---")
    try:
        # We use the checkpointer to get the latest state of the thread
        thread_state = checkpointer.get(config)
        if thread_state:
            thread_state = thread_state.get('channel_values', {})
            if not thread_state:
                return
            state: State = {
                "recent_messages": thread_state.get("recent_messages", []),
                "user_query": thread_state.get("user_query", ""),
                "conversation_summary": thread_state.get("conversation_summary", ""),
                "do_retrieval": thread_state.get("do_retrieval", False),
                "do_search": thread_state.get("do_search", False),
                "tasks": thread_state.get("tasks", []),
                "task_plans": thread_state.get("task_plans", []),
                "fused_docs": thread_state.get("fused_docs", []),
                "web_search_results": thread_state.get("web_search_results", ""),
                "retrieved_docs": thread_state.get("retrieved_docs", []),
                "prompt_messages": thread_state.get("prompt_messages", []),
            }
            summarizer.run(state)

            print("---Background summarization complete.---")
        else:
            print("---Could not find thread state for summarization.---")
    except Exception as e:
        print(f"---Error during background summarization: {e}---")


def run_workflow():
    graph = build_workflow()
    config = RunnableConfig(configurable={"thread_id": THREAD_ID})

    print("Welcome to the Chatbot! Type 'q' to quit or 'd' to delete the thread.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "q":
            break
        if user_input.lower() == "d":
            delete_thread(THREAD_ID)
            print(f"Thread '{THREAD_ID}' deleted.")
            break

        # Simulate front-end booleans for intent
        use_attachments = "pdf" in user_input.lower() or "document" in user_input.lower()
        use_internet = "latest" in user_input.lower() or "search for" in user_input.lower()

        initial_state = {
            "recent_messages": [HumanMessage(content=user_input)],
            "user_query": user_input,
            "do_retrieval": use_attachments,
            "do_search": use_internet,
        }

        # --- This call is now non-blocking for summarization ---
        response = graph.invoke(initial_state, config=config)
        final_message = response['recent_messages'][-1].content

        # --- Immediately print the response to the user ---
        print(f"Bot: {final_message}")

        # # --- BACKGROUND TASK LOGIC ---
        # After responding, check if summarization is needed.
        current_state = checkpointer.get(config)
        if current_state and len(current_state.get("values", {}).get("recent_messages", [])) > SUMMARY_THRESHOLD:
            # Run summarization in a separate thread to not block the next input prompt
            summary_thread = threading.Thread(
                target=run_background_summarization,
                args=(config,)
            )
            summary_thread.start()


if __name__ == "__main__":
    run_workflow()
    # run_background_summarization(RunnableConfig(configurable={"thread_id": THREAD_ID}))