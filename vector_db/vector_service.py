import asyncio
from . import get_global_indexer
from langchain.schema import Document
from typing import List, Dict, Any

class VectorService:
    _indexer = None

    @classmethod
    def _get_indexer(cls):
        """Get the indexer instance, creating it if necessary."""
        if cls._indexer is None:
            cls._indexer = get_global_indexer()
        return cls._indexer

    @classmethod
    async def retrieve_documents(cls, query: str, user_id: str, thread_id: str) -> List[Document]:
        indexer = cls._get_indexer()
        # Ensure vector store is initialized before searching
        if not indexer._initialized:
            await indexer._initialize_vector_store()
        return await indexer.search(query, user_id, thread_id)

    @classmethod
    async def index_documents(cls, file_paths: List[str], user_id: str, thread_id: str) -> Dict[str, Any]:
        indexer = cls._get_indexer()
        return await indexer.index_documents(file_paths, user_id, thread_id)

    @classmethod
    async def delete_collection(cls) -> None:
        indexer = cls._get_indexer()
        await indexer.qdrant_manager.delete_collection()

    @classmethod
    async def delete_chat_documents(cls, user_id: str, thread_id: str) -> None:
        indexer = cls._get_indexer()
        await indexer.qdrant_manager.delete_by_thread_id(user_id, thread_id)
    
    @classmethod
    async def delete_user_documents(cls, user_id: str) -> None:
        indexer = cls._get_indexer()
        await indexer.qdrant_manager.delete_by_user_id(user_id)

    # Sync wrappers for backward compatibility
    @classmethod
    def retrieve_documents_sync(cls, query: str, user_id: str, thread_id: str) -> List[Document]:
        try:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(cls.retrieve_documents(query, user_id, thread_id))
        except RuntimeError:
            # No event loop running, create one
            return asyncio.run(cls.retrieve_documents(query, user_id, thread_id))

    @classmethod
    def index_documents_sync(cls, file_paths: List[str], user_id: str, thread_id: str) -> Dict[str, Any]:
        try:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(cls.index_documents(file_paths, user_id, thread_id))
        except RuntimeError:
            # No event loop running, create one
            return asyncio.run(cls.index_documents(file_paths, user_id, thread_id))

    @classmethod
    def delete_collection_sync(cls) -> None:
        try:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(cls.delete_collection())
        except RuntimeError:
            # No event loop running, create one
            return asyncio.run(cls.delete_collection())

    @classmethod
    def delete_chat_documents_sync(cls, user_id: str, thread_id: str) -> None:
        try:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(cls.delete_chat_documents(user_id, thread_id))
        except RuntimeError:
            # No event loop running, create one
            return asyncio.run(cls.delete_chat_documents(user_id, thread_id))
    
    @classmethod
    def delete_user_documents_sync(cls, user_id: str) -> None:
        try:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(cls.delete_user_documents(user_id))
        except RuntimeError:
            # No event loop running, create one
            return asyncio.run(cls.delete_user_documents(user_id))
