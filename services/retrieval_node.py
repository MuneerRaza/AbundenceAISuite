import asyncio
from typing import Dict, List
from models.state import State
from vector_db.vector_service import VectorService
from langchain.schema import Document

class RetrievalNode:
    async def _retrieve_for_task(self, task: str, user_id: str, thread_id: str) -> List[Document]:
        """Helper function to retrieve documents for a single task using a hybrid strategy."""
        print(f"---STARTING RETRIEVAL FOR TASK: '{task}'---")
        try:
            retrieved_docs = await VectorService.retrieve_documents(query=task, user_id=user_id, thread_id=thread_id)
            for doc in retrieved_docs:
                doc.metadata["source_task"] = task
            return retrieved_docs[:5]
        except Exception as e:
            print(f"Error retrieving documents for task '{task}': {e}")
            return []

    async def invoke(self, state: State) -> Dict[str, List[Document]]:
        """
        Retrieves documents for each task in parallel using async operations.
        """
        do_retrieval = state.get("do_retrieval", False)
        if not do_retrieval:
            print("Skipping retrieval as per graph routing.")
            return {"retrieved_docs": []}

        tasks = state.get("tasks", [])
        user_id = state.get("user_id", "default_user")
        thread_id = state.get("thread_id", "default_thread")
        
        if not tasks:
            return {"retrieved_docs": []}

        print(f"---RETRIEVING IN PARALLEL FOR {len(tasks)} TASKS---")
        
        # Create async tasks for parallel retrieval
        retrieval_tasks = [
            self._retrieve_for_task(task, user_id, thread_id) for task in tasks
        ]
        
        # Execute all retrieval tasks in parallel
        all_doc_lists = await asyncio.gather(*retrieval_tasks, return_exceptions=True)
        
        all_docs: List[Document] = []
        for i, result in enumerate(all_doc_lists):
            if isinstance(result, Exception):
                task_name = tasks[i]
                print(f"Task '{task_name}' failed during retrieval: {result}")
            elif isinstance(result, list):
                # Safe to extend since we've confirmed it's a list
                all_docs.extend(result)

        unique_docs = {doc.page_content: doc for doc in all_docs}.values()
        print(f"---RETRIEVED {len(unique_docs)} UNIQUE DOCUMENTS FROM ALL TASKS---")
        return {"retrieved_docs": list(unique_docs)}

    # Sync wrapper for backward compatibility
    def invoke_sync(self, state: State) -> Dict[str, List[Document]]:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're in an async context, create a task
                return loop.run_until_complete(self.invoke(state))
            else:
                # If no event loop is running, we can use asyncio.run
                return asyncio.run(self.invoke(state))
        except RuntimeError:
            # Fallback: create a new event loop
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                return new_loop.run_until_complete(self.invoke(state))
            finally:
                new_loop.close()