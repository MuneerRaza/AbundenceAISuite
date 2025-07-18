import concurrent.futures
from typing import Dict, List
from models.state import State
from vector_db.vector_service import VectorService
from langchain.schema import Document
from config import USER_ID, THREAD_ID

class RetrievalNode:
    def _retrieve_for_task(self, task: str, user_id: str, thread_id: str) -> List[Document]:
        """Helper function to retrieve documents for a single task using a hybrid strategy."""
        print(f"---STARTING RETRIEVAL FOR TASK: '{task}'---")
        retrieved_docs = VectorService.retrieve_documents(query=task, user_id=user_id, thread_id=thread_id)
        for doc in retrieved_docs:
            doc.metadata["source_task"] = task
        return retrieved_docs

    def invoke(self, state: State) -> Dict[str, List[Document]]:
        """
        Retrieves documents for each task in parallel.
        """
        do_retrieval = state.get("do_retrieval", False)
        if not do_retrieval:
            print("Skipping retrieval as per graph routing.")
            return {"fused_docs": []}

        tasks = state.get("tasks", [])
        
        if not tasks or not THREAD_ID:
            return {"fused_docs": []}

        print(f"---RETRIEVING IN PARALLEL FOR {len(tasks)} TASKS---")
        all_docs = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # The node now uses the list of task strings directly
            future_to_task = {
                executor.submit(self._retrieve_for_task, task, USER_ID,THREAD_ID): task for task in tasks
            }
            for future in concurrent.futures.as_completed(future_to_task):
                try:
                    all_docs.extend(future.result())
                except Exception as e:
                    task_name = future_to_task[future]
                    print(f"Task '{task_name}' failed during retrieval: {e}")

        unique_docs = {doc.page_content: doc for doc in all_docs}.values()
        print(f"---AGGREGATED {len(unique_docs)} UNIQUE DOCUMENTS FROM ALL TASKS---")
        return {"fused_docs": list(unique_docs)}