import asyncio
from config import (
    PROMPT_NO_SUMMARY_NO_CONTENT,
    PROMPT_SUMMARY_ONLY,
    PROMPT_CONTENT_ONLY,
    PROMPT_SUMMARY_AND_CONTENT,
)
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.language_models.chat_models import BaseChatModel
from models.state import State

def get_prompt_template(state: State) -> str:
    has_summary = bool(state.get("conversation_summary"))
    has_content = bool(state.get("final_context"))

    if has_summary and has_content:
        return PROMPT_SUMMARY_AND_CONTENT
    elif not has_summary and has_content:
        return PROMPT_CONTENT_ONLY
    elif has_summary and not has_content:
        return PROMPT_SUMMARY_ONLY
    else:  
        return PROMPT_NO_SUMMARY_NO_CONTENT

class CallModelNode:
    def __init__(self, model: BaseChatModel):
        self.model = model
    
    async def invoke(self, state: State):
        user_query = state.get("user_query", "")
        final_context = state.get("final_context", "")
        recent_messages = state.get("recent_messages", [])
        # remove last message if it's a user query
        if recent_messages and isinstance(recent_messages[-1], HumanMessage):
            recent_messages = recent_messages[:-1]
            
        if not user_query:
            return {"recent_messages": [HumanMessage(content="No user query provided.")]}
        recent_messages.append(
            HumanMessage(content=f"User Query: {user_query}\n\n{final_context.strip()}")
        )
        
        # Run the model call in a thread pool since it's sync
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None, 
            self.model.invoke, 
            [SystemMessage(content=get_prompt_template(state))] + recent_messages
        )
        return {"recent_messages": [response], "tasks": [], "web_search_results": [], "retrieved_docs": []}

    async def invoke_stream(self, state: State):
        """Stream the model response token by token."""
        user_query = state.get("user_query", "")
        final_context = state.get("final_context", "")
        recent_messages = state.get("recent_messages", [])
        
        # remove last message if it's a user query
        if recent_messages and isinstance(recent_messages[-1], HumanMessage):
            recent_messages = recent_messages[:-1]
            
        if not user_query:
            yield "No user query provided."
            return
            
        recent_messages.append(
            HumanMessage(content=f"User Query: {user_query}\n\n{final_context.strip()}")
        )
        
        # Create messages for streaming
        messages = [SystemMessage(content=get_prompt_template(state))] + recent_messages
        
        # Stream the response
        try:
            # Use the model's stream method if available
            if hasattr(self.model, 'astream'):
                async for chunk in self.model.astream(messages):
                    if hasattr(chunk, 'content') and chunk.content:
                        yield chunk.content
            else:
                # Fallback to regular invoke
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(None, self.model.invoke, messages)
                yield response.content
        except Exception as e:
            yield f"Error generating response: {str(e)}"

    # Sync wrapper for backward compatibility
    def invoke_sync(self, state: State):
        return asyncio.run(self.invoke(state))