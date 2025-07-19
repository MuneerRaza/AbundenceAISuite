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
os.environ['FASTEMBED_CACHE_PATH'] = 'cache'

utils_model = ChatGroq(model=UTILS_MODEL_ID)
summarizer = SummarizerNode(llm=utils_model)

# MODIFICATION: The function now accepts the 'graph' object
def run_background_summarization(config, graph):
    print("\n---Starting background summarization...---")
    try:
        state = graph.get_state(config)
        updates = summarizer.run(state)
        if updates:
            graph.update_state(config, updates)
            print("---Background summarization complete and state updated.---")
        else:
            print("---No summarization was needed.---")
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
            continue

        # Simulate front-end booleans for intent
        use_attachments = "pdf" in user_input.lower()
        use_internet = "search" in user_input.lower()

        initial_state = {
            "recent_messages": [HumanMessage(content=user_input)],
            "user_query": user_input,
            "do_retrieval": use_attachments,
            "do_search": use_internet,
        }

        response = graph.invoke(initial_state, config=config)
        final_message = response['recent_messages'][-1].content
        print(f"Bot: {final_message}")
        current_state = graph.get_state(config)

        if len(current_state.values.get("recent_messages", [])) > SUMMARY_THRESHOLD:
            summary_thread = threading.Thread(
                target=run_background_summarization,
                args=(config, graph)
            )
            summary_thread.start()


if __name__ == "__main__":
    run_workflow()