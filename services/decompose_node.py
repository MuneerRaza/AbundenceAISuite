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
        
        # --- NEW, MORE POWERFUL PROMPT ---
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an expert at analyzing user queries and breaking them down into one or more clear, self-contained, and answerable questions. Your goal is to create the **minimum number of tasks** required to fully cover the user's intent.\n\n"
                    "Respond with only the classification in this JSON format.\n\n"
                    "## Your Process:\n"
                    "1. **Analyze History:** First, review the 'CONVERSATION HISTORY' (if available) to understand context and resolve any pronouns or ambiguities in the 'USER QUERY'. Do not assume or invent information.\n"
                    "2. **Identify Core Tasks:** Determine the fundamental questions the user is asking.\n"
                    "3. **Decomposition Rule:** Decompose only if the query contains multiple distinct questions that must be addressed separately. If it's a single, cohesive question, rewrite it clearly as one self-contained task.\n\n"
                    "## EXAMPLE 1 (Decomposition):\n"
                    "CONVERSATION HISTORY: Human: I'm planning a trip.\n"
                    "USER QUERY: 'Tell me about Paris and also the top 3 restaurants in Rome.'\n"
                    "tasks: ['What are some key facts about Paris for a tourist?', 'What are the top 3 rated restaurants in Rome?']\n\n"
                    "## EXAMPLE 2 (No Decomposition; Rewriting Needed):\n"
                    "CONVERSATION HISTORY: Human: What is the capital of France?\nAI: Paris is the capital of France.\n"
                    "USER QUERY: 'What about its population?'\n"
                    "tasks: ['What is the population of Paris, France?']\n\n"
                    "## EXAMPLE 3 (No Decomposition; Simple Query):\n"
                    "CONVERSATION HISTORY: [] (empty)\n"
                    "USER QUERY: 'How does a VAE work?'\n"
                    "tasks: ['How does a Variational Autoencoder (VAE) work?']"

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
    # Example usage
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