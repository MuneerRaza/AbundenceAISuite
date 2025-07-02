from workflow.graph import build_workflow
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from config import THREAD_ID
from dotenv import load_dotenv
load_dotenv()

import os
os.environ["ANONYMIZED_TELEMETRY"] = "false"


config = RunnableConfig(configurable={"thread_id": THREAD_ID})
def run_workflow():

    graph = build_workflow()

    print("Welcome to the Chatbot! Type 'q' to quit and 'd' to delete the thread.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "q":
            break
        if user_input.lower() == "d":
            from utils.checkpointer import delete_thread
            delete_thread(THREAD_ID)
            break
        
        input_msg = HumanMessage(content=user_input)
        response = graph.invoke(
            {"recent_messages": [input_msg],
             "user_query": user_input},
            config=config
        )
        print(f"Bot: {response['recent_messages'][-1].content}")

if __name__ == "__main__":
    run_workflow()