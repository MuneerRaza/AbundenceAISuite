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
    
    def invoke(self, state: State):
        prompt_with_context = state.get("prompt_messages", [])
        response = self.model.invoke([SystemMessage(content=get_prompt_template(state))] + prompt_with_context)
        return {"recent_messages": [response]}