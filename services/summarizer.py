from typing import List, Sequence
from config import MESSAGES_TO_RETAIN
from models.state import State
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage, RemoveMessage
from langchain_core.language_models.chat_models import BaseChatModel


class SummarizerNode:

    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self._initial_summary_prompt = (
            "**Role:** You are a Knowledge Architect AI that converts dialogues into structured, categorized fact sheets.\n\n"
            "**Instructions:**\n"
            "- Analyze the provided conversation chunk.\n"
            "- Derive relevant category headings from the dialogue's context.\n"
            "- Under each heading, list key facts as short, declarative bullet points. Attribute facts to specific people or goals where applicable.\n"
            "- **Factuality is paramount:** Only include explicitly stated facts. Do not add assumptions or outside information.\n"
            "- **Signal over Noise:** Ignore conversational filler, greetings, and obvious statements. Extract only meaningful, non-redundant information.\n"
            "- **Obvious Facts:** Do not include facts that are already well-known or obvious.\n\n"
            "**Output Format:**\n"
            "- The output MUST be only the Markdown fact sheet. Do not include any other text.\n"
            "- Follow this structure:\n"
            "### [Derived Category 1]\n"
            "- [Extracted detail]\n"
            "...\n"
            "### [Derived Category 2]\n"
            "-  [Another detail]\n"
            "..."
        )
        self._update_summary_prompt = (
            "**Role:** You are a Knowledge Architect AI responsible for updating an existing fact sheet with new information from a dialogue.\n\n"
            "**Task:** Your goal is to seamlessly integrate new information to produce a single, updated, concise, and coherent knowledge base.\n\n"
            "**Critical Update Rules:**\n"
            "- **Analyze & Integrate:** Carefully analyze the existing fact sheet and the new conversation messages.\n"
            "- **Integrate & Categorize:** Add new facts under the most relevant existing heading, or create a new one if necessary.\n"
            "- **Revise, Don't Contradict:** If new information clarifies or corrects an old fact, you MUST modify the existing fact instead of adding a new one.\n"
            "- **No Duplicates:** You MUST NOT add facts that are already present or redundant.\n"
            "- **Maintain Factuality:** Do not assume or invent information.\n\n"
            "**Output Format:**\n"
            "- The output MUST be the complete, updated fact sheet formatted in Markdown. Do not add any other text."
        )


    def run(self, state) -> dict:
        summary = state.values.get("conversation_summary", "")
        all_recent_messages = state.values.get("recent_messages", [])
        messages_to_summarize = all_recent_messages[:-MESSAGES_TO_RETAIN]

        if not messages_to_summarize:
            return {}

        messages = self._build_prompt_messages(summary, messages_to_summarize)
    
        response = self.llm.invoke(messages)
        new_summary = response.content

        messages_to_delete = [
            RemoveMessage(id=m.id) for m in messages_to_summarize if hasattr(m, 'id') and m.id is not None
        ]
        print(f"SummarizerNode: {new_summary}")

        return {
            "conversation_summary": new_summary,
            "recent_messages": messages_to_delete,
        }

    def _build_prompt_messages(self, summary: str, messages_to_summarize: Sequence[BaseMessage]) -> List[BaseMessage]:

        conversation_chunk = "\n".join(f"{msg.type}: {msg.content}" for msg in messages_to_summarize)

        if not summary:
            system_prompt = self._initial_summary_prompt
            human_content = (
                f"**Conversation to Analyze:**\n{conversation_chunk}\n\n"
                "**Structured Fact Sheet Output:**"
            )
        else:
            system_prompt = self._update_summary_prompt
            human_content = (
                f"**Existing Fact Sheet:**\n{summary}\n\n"
                f"**New Conversation Messages:**\n{conversation_chunk}\n\n"
                "**Direct Updated Fact Sheet Output:**"
            )
            
        return [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_content)
        ]