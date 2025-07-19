from models.state import State
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers.string import StrOutputParser
from langchain.schema import Document
from collections import defaultdict

class EvaluatorNode:
    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an expert at evaluating and consolidating content. Your task is to analyze a list of retrieved document chunks based on a specific task and rewrite them into a single, concise, and relevant document. "
                    "You MUST follow these rules:\n"
                    "1. **Analyze all Chunks:** Carefully read all the provided document chunks.\n"
                    "2. **Evaluate Relevance:** Compare the content of each chunk with the given task and identify the parts that are directly relevant.\n"
                    "3. **Synthesize and Rewrite:** Combine the relevant information from all chunks into a single, coherent, and concise document. Remove all irrelevant details, redundant information, and conversational filler.\n"
                    "4. **No Extra Commentary:** Do NOT add any extra commentary, explanations, or apologies. Your output should only be the rewritten document content.\n"
                    "5. **No Made-up Information:** Do NOT make up any information. Your rewritten content must be based solely on the provided context."
                ),
                (
                    "human",
                    "**Task:**\n{task}\n\n"
                    "**Retrieved Document Chunks:**\n"
                    "-----\n"
                    "{context}\n"
                    "-----\n\n"
                ),
            ]
        )
        self.chain = self.prompt | self.llm | StrOutputParser()

    def invoke(self, state: State):
        retrieved_docs = state.get("retrieved_docs", [])
        tasks = state.get("tasks", [])

        if not retrieved_docs or not tasks:
            return {"retrieved_docs": []}

        # Group documents by task
        docs_by_task = defaultdict(list)
        for doc in retrieved_docs:
            source_task = doc.metadata.get("source_task")
            if source_task in tasks:
                docs_by_task[source_task].append(doc)

        rewritten_docs = []
        for task, docs in docs_by_task.items():
            # Combine the content of all docs for the task
            combined_context = "\n-----\n".join([doc.page_content for doc in docs])

            # Invoke the LLM once per task
            rewritten_content = self.chain.invoke({
                "task": task,
                "context": combined_context
            })

            # Create a new Document with the rewritten content
            # and combined metadata from the first document in the group
            if docs:
                new_doc = Document(page_content=rewritten_content, metadata=docs[0].metadata)
                rewritten_docs.append(new_doc)

        return {"retrieved_docs": rewritten_docs}