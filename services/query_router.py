from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, Field
from models.state import State
import logging

class QueryType(BaseModel):
    query_type: Literal["informational", "conversational"] = Field(
        ...,
        description=(
            "Classify the user's query. Use 'informational' if the user is asking a factual question that "
            "could be answered by searching through documents. Use 'conversational' for all other types of "
            "queries, such as greetings, small talk, or creative requests."
        ),
    )

class QueryRouterNode:
    def __init__(self, llm: BaseChatModel):
        self.structured_llm = llm.with_structured_output(QueryType)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are an expert at classifying user queries. Your task is to determine if a query is "
             "'informational' (seeking facts from a knowledge base) or 'conversational' (all other questions). "
             "Classify 'conversational' only if you are very confident. If you are unsure, default to 'informational'. "
             "Respond with only the classification in the specified JSON format."),
            ("human",
             "Classify the following user query: '{query}'")
        ])
        self.chain = self.prompt | self.structured_llm

    def invoke(self, state: State) -> str:
        user_query = state.get("user_query", "")
        if not user_query:
            logging.warning("No rewritten query found, defaulting to conversational.")
            return "conversational"

        logging.info(f"---CLASSIFYING QUERY: '{user_query}'---")
        try:
            result = self.chain.invoke({"query": user_query})
            if isinstance(result, QueryType):
                return result.query_type
            else:
                return "informational"  
        except Exception as e:
            logging.error(f"Error during query classification: {e}. Defaulting to informational path.")
            return "informational"