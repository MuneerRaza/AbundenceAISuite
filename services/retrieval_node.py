from models.state import State
from services.vectordb import advanced_retrieve
from langchain_core.runnables import RunnableConfig
import concurrent.futures

class RetrievalNode:
    def __init__(self, config: RunnableConfig):
        self.config = config

    def _retrieve_for_task(self, task: str, thread_id: str):
        """Helper function to retrieve documents for a single task."""
        print(f"---STARTING RETRIEVAL FOR TASK: '{task}'---")
        retrieved_docs = advanced_retrieve(query=task, thread_id=thread_id)
        print(f"---COMPLETED RETRIEVAL FOR TASK: '{task}'---")
        return retrieved_docs

    def invoke(self, state: State):
        """
        Retrieves documents for each task in parallel and aggregates them.
        """
        tasks = state.get("tasks", [])
        thread_id = self.config.get("configurable", {}).get("thread_id")
        
        if not tasks or not thread_id:
            return {"retrieved_docs": []}

        print(f"---RETRIEVING IN PARALLEL FOR {len(tasks)} TASKS---")
        all_docs = []
        
        # Use a ThreadPoolExecutor to run retrieval for all tasks concurrently
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Create a future for each task
            future_to_task = {
                executor.submit(self._retrieve_for_task, task, thread_id): task for task in tasks
            }
            # As each future completes, gather its results
            for future in concurrent.futures.as_completed(future_to_task):
                try:
                    all_docs.extend(future.result())
                except Exception as e:
                    task_name = future_to_task[future]
                    print(f"Task '{task_name}' failed during retrieval: {e}")

        # Simple de-duplication based on page content
        unique_docs = {doc.page_content: doc for doc in all_docs}.values()
        
        print("---AGGREGATED ALL PARALLEL RESULTS---")
        return {"retrieved_docs": list(unique_docs)}