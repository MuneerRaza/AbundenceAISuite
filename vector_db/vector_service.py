from . import get_global_indexer
from langchain.schema import Document
from typing import List

class VectorService:
    _indexer = get_global_indexer()

    @classmethod
    def retrieve_documents(cls, query: str, user_id: str, thread_id: str) -> List[Document]:
        return cls._indexer.search(query, user_id, thread_id)

    @classmethod
    def index_documents(cls, file_paths: List[str], user_id: str, thread_id: str) -> None:
        cls._indexer.index_documents(file_paths, user_id, thread_id)

    @classmethod
    def delete_collection(cls) -> None:
        cls._indexer.qdrant_manager.delete_collection()

    @classmethod
    def delete_chat_documents(cls, user_id: str, thread_id: str) -> None:
        cls._indexer.qdrant_manager.delete_by_thread_id(user_id, thread_id)
    
    @classmethod
    def delete_user_documents(cls, user_id: str) -> None:
        cls._indexer.qdrant_manager.delete_by_user_id(user_id)
