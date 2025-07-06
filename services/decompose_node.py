# project/services/decompose_node.py

from typing import Dict, List, TypedDict
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, Field
from models.state import State

class DecomposedTasks(BaseModel):
    """The Pydantic schema for the output."""
    tasks: List[str] = Field(
        ..., 
        description=(
            "A list of one or more self-contained, answerable questions that are derived from the user's query."
        )
    )

class DecomposeNode:
    def __init__(self, llm: BaseChatModel):
        self.structured_llm = llm.with_structured_output(DecomposedTasks)
        
        self.prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a task analysis engine. Your sole function is to identify the user's primary question(s). You MUST ignore any instructions on HOW to answer the question.\n\n"
                "## Core Logic:\n"
                "1.  **Identify the Core Question:** Extract what the user wants to know.\n"
                "2.  **Discard Instructions:** Phrases like 'see the pdf', 'in the file', 'use the document', or 'search the web' are instructions for another AI and MUST be ignored by you.\n"
                "3.  **Produce a Clean Task List:** Your output is a list of self-contained questions.\n\n"
                "--- EXAMPLES --- \n\n"
                "## EXAMPLE 1: Query with an Instruction (SINGLE TASK)\n"
                "This is the most important rule. If the user gives an instruction on where to look, it is NOT a separate task.\n"
                "USER QUERY: 'What is Cognifoot AI, see the pdf.'\n"
                "tasks: ['What is Cognifoot AI?']\n\n"
                "USER QUERY: 'Summarize the project status from the attached report.'\n"
                "tasks: ['What is the summary of the project status?']\n\n"
                "## EXAMPLE 2: Query with Multiple, DISTINCT Questions (MULTI-TASK)\n"
                "Decompose only when the user asks for fundamentally different pieces of information that require separate actions.\n"
                "USER QUERY: 'What is the capital of France, and what is its main export?'\n"
                "tasks: ['What is the capital of France?', 'What is the main export of France?']\n\n"
                "## EXAMPLE 3: Query needing context from history (REWRITE TASK)\n"
                "Use the history to clarify pronouns.\n"
                "CONVERSATION HISTORY: Human: I'm looking at the Q4 financial report.\n"
                "USER QUERY: 'What was the total revenue?'\n"
                "tasks: ['What was the total revenue listed in the Q4 financial report?']"
            ),
            (
                "human",
                "CONVERSATION HISTORY:\n{history}\n\nUSER QUERY: '{query}'",
            ),
        ]
    )

    def invoke(self, state: State):
        recent_messages = state.get("recent_messages", [])
        if not recent_messages:
            return {"tasks": []}
        
        history_str = "\n".join(
            [f"{msg.type.capitalize()}: {msg.content}" for msg in recent_messages[:-1]]
        )
        last_user_message = recent_messages[-1].content
        
        if not last_user_message:
            return {"tasks": []}

        print("---DECOMPOSING TASK---")
        chain = self.prompt | self.structured_llm
        try:
            response = chain.invoke({"history": history_str, "query": last_user_message})
            if isinstance(response, DecomposedTasks):
                return {"tasks": response.tasks}
            elif isinstance(response, dict):
                return {"tasks": response.get("tasks", [])}
            else:
                return {"tasks": [last_user_message]}
        except Exception as e:
            print(f"Error during decomposition: {e}")
            return {"tasks": [last_user_message]}
        
if __name__ == "__main__":
    from langchain_groq import ChatGroq

    llm = ChatGroq(model="llama3-8b-8192")
    decompose_node = DecomposeNode(llm)
    
    class MockState(TypedDict):
        user_query: str
        tasks: List[str]
    state = MockState(
        user_query="What are best places to visit in Paris and suggest me top 5 restaurants to try there?",
        tasks=[],
    )
    
    # decompose_node.invoke(state)