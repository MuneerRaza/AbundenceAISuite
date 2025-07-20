import json
import asyncio
from typing import List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import JsonOutputParser
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
        self.llm = llm
        self.output_parser = JsonOutputParser(pydantic_object=DecomposedTasks)

        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a task analysis engine. Your sole function is to identify the user's primary question(s) and respond ONLY in JSON format. You MUST ignore any instructions on HOW to answer the question.\n\n"
                    "## Core Logic:\n"
                    "1.  **Identify the Core Question:** Extract what the user wants to know.\n"
                    "2.  **Discard Instructions:** Phrases like 'see the pdf', 'in the file', 'use the document', or 'search the web' are instructions for another AI and MUST be ignored by you.\n"
                    "3.  **Produce a Clean JSON:** Your output must be a JSON object with a single key 'tasks', which contains a list of self-contained questions.\n"
                    "4.  **No Extra Commentary:** Do NOT add explanations, notes, or any extra text outside the JSON structure.\n\n"
                    "--- EXAMPLES --- \n\n"
                    "## EXAMPLE 1: Query with an Instruction (SINGLE TASK)\n"
                    "This is the most important rule. If the user gives an instruction on where to look, it is NOT a separate task.\n"
                    "USER QUERY: 'What is Cognifoot AI, see the pdf.'\n"
                    '{{ "tasks": ["What is Cognifoot AI?"] }}\n\n'
                    "USER QUERY: 'Summarize the project status from the attached report.'\n"
                    '{{ "tasks": ["What is the summary of the project status?"] }}\n\n'
                    "## EXAMPLE 2: Query with Multiple, DISTINCT Questions (MULTI-TASK)\n"
                    "Decompose only when the user asks for fundamentally different pieces of information that require separate actions.\n"
                    "USER QUERY: 'What is the capital of France, and what is its main export?'\n"
                    '{{ "tasks": ["What is the capital of France?", "What is the main export of France?"] }}\n\n'
                    "## EXAMPLE 3: Query needing context from history (REWRITE TASK)\n"
                    "Use the history to clarify pronouns.\n"
                    "CONVERSATION HISTORY: Human: I'm looking at the Q4 financial report.\n"
                    "USER QUERY: 'What was the total revenue?'\n"
                    '{{ "tasks": ["What was the total revenue listed in the Q4 financial report?"] }}\n\n'
                    "## Your output MUST be ONLY the JSON object."
                ),
                (
                    "human",
                    "CONVERSATION HISTORY:\n{history}\n\nUSER QUERY: '{query}'",
                ),
            ]
        )

    async def invoke(self, state: State):
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
        chain = self.prompt | self.llm | self.output_parser
        try:
            # Run the chain in a thread pool since it's sync
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, chain.invoke, {"history": history_str, "query": last_user_message})
            # The JsonOutputParser will return a dictionary
            return {"tasks": response.get("tasks", [last_user_message])}
        except json.JSONDecodeError as e:
            print(f"Error during JSON parsing: {e}. Falling back to original query.")
            # Fallback for cases where the LLM output is not valid JSON
            return {"tasks": [last_user_message]}
        except Exception as e:
            print(f"An unexpected error occurred during decomposition: {e}")
            return {"tasks": [last_user_message]}

    # Sync wrapper for backward compatibility
    def invoke_sync(self, state: State):
        return asyncio.run(self.invoke(state))

if __name__ == "__main__":
    from langchain_groq import ChatGroq

    # It's recommended to specify the response format if the provider supports it
    llm = ChatGroq(model="llama3-8b-8192",
                   model_kwargs={"response_format": {"type": "json_object"}})
    
    decompose_node = DecomposeNode(llm=llm)
    state: State = {
        "recent_messages": [
            BaseMessage(type="human", content="Can you tell me about supervisors of cognifoot ai? and also tell me what should I tell when they ask that why I have low grade in my last degree? see pdf")
        ],
        "user_query": "Can you tell me about supervisors of cognifoot ai? and also tell me what should I tell when they ask that why I have low grade in my last degree? see pdf",
        "conversation_summary": "",
        "do_retrieval": False,
        "do_search": False,
        "tasks": [],
        "web_search_results": [],
        "retrieved_docs": [],
        "final_context": "",
    }
    result = asyncio.run(decompose_node.invoke(state))
    print(result)