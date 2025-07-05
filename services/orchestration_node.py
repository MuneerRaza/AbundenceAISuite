from typing import List, Literal, Dict
from pydantic import BaseModel, Field
from models.state import State
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

class TaskPlan(BaseModel):
    """A plan for how to gather evidence for a single task."""
    task: str = Field(..., description="The original task.")
    retrieval_strategy: Literal["vector_search", "keyword_search", "hybrid"] = Field(
        ...,
        description="The optimal retrieval strategy for this specific task. Use 'vector_search' for conceptual queries, 'keyword_search' for specific terms or codes, and 'hybrid' for most general queries."
    )

class OrchestratorNode:
    def __init__(self, llm: BaseChatModel):
        self.llm = llm.with_structured_output(TaskPlan)
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an expert retrieval strategist. For the given user task, determine the best retrieval strategy. "
                    "Your goal is to choose the most efficient method to find the correct information. "
                    "A 'hybrid' search is generally best. Choose 'vector_search' only if the query is purely conceptual with no specific keywords. "
                    "Choose 'keyword_search' only if the query contains very specific, unique identifiers, codes, or exact phrases that must be matched."
                ),
                ("human", "Analyze this task: '{task}'"),
            ]
        )
        self.chain = self.prompt | self.llm
        print("✅ Initialized OrchestratorNode")


    def invoke(self, state: State) -> Dict[str, List[TaskPlan]]:
        """
        Creates a detailed evidence-gathering plan for each task.
        """
        print("---ORCHESTRATING TASKS---")
        tasks = state["tasks"]
        do_retrieval = state.get("do_retrieval", False)
        
        if not do_retrieval:
            print("No retrieval required. Skipping orchestration.")
            # Even if we don't retrieve, we need a plan structure for the aggregator
            task_plans = [TaskPlan(task=task, retrieval_strategy="hybrid") for task in tasks]
            return {"task_plans": task_plans}

        task_plans = self.chain.batch([{"task": task} for task in tasks])
        plans = []
        for task_plan in task_plans:
            if isinstance(task_plan, TaskPlan):
                plans.append(task_plan)
            elif isinstance(task_plan, dict):
                # Handle case where the output is a dict with a single key
                if len(task_plan) == 1:
                    plans.append(TaskPlan(**task_plan))
                else:
                    raise ValueError(f"Unexpected output format: {task_plan}")
            else:
                raise ValueError(f"Unexpected output type: {type(task_plan)}. Expected TaskPlan or dict.")
        if not plans:
            print("No valid task plans generated. Returning empty list.")
            return {"task_plans": []}
    
        return {"task_plans": plans}