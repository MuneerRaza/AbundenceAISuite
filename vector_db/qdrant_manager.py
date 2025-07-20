import logging
import asyncio
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams
from qdrant_client.http import models

class QdrantManager:
    """Async Qdrant database manager."""
    
    def __init__(self, host: str = "localhost", port: int = 6333, collection_name: str = "documents"):
        self.collection_name = collection_name
        self.host = host
        self.port = port
        
        # Priority: Try gRPC first, fallback to HTTP
        try:
            # Try gRPC connection first (port 6334)
            self.client = AsyncQdrantClient(host=host, grpc_port=6334, prefer_grpc=True)
            logging.info(f"Connected to Qdrant using gRPC at {host}:6334")
        except Exception as e:
            logging.warning(f"gRPC connection failed: {e}, falling back to HTTP")
            try:
                # Fallback to HTTP connection (port 6333)
                self.client = AsyncQdrantClient(host=host, port=6333)
                logging.info(f"Connected to Qdrant using HTTP at {host}:6333")
            except Exception as e2:
                logging.error(f"Both gRPC and HTTP connections failed: {e2}")
                raise

    async def create_collection(self, vector_size: int, force_recreate: bool = False):
        """Create collection for storing vectors."""
        if force_recreate:
            await self.delete_collection()

        try:
            # Create collection
            await self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE
                )
            )
            
            # Create indexes for efficient filtering
            await self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="metadata.thread_id",
                field_schema=models.PayloadSchemaType.KEYWORD
            )
            
            await self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="metadata.file_hash", 
                field_schema=models.PayloadSchemaType.KEYWORD
            )
            
            await self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="metadata.user_id",
                field_schema=models.PayloadSchemaType.KEYWORD
            )

        except Exception as e:
            if "already exists" not in str(e):
                logging.error(f"Error creating collection: {e}")
                raise
    
    async def collection_exists(self) -> bool:
        """Check if collection exists."""
        try:
            collections = await self.client.get_collections()
            exists = any(col.name == self.collection_name for col in collections.collections)
            return exists
        except Exception as e:
            logging.error(f"Collection existence check failed: {e}")
            return False
    
    async def document_exists_globally(self, file_hash: str) -> bool:
        """Check if document with given hash exists in any thread (for debugging)."""
        try:
            if not await self.collection_exists():
                return False
            
            result = await self.client.count(
                collection_name=self.collection_name,
                count_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="metadata.file_hash", 
                            match=models.MatchValue(value=file_hash)
                        )
                    ]
                ),
                exact=True
            )
            exists = result.count > 0
            logging.debug(f"Global document exists check ({file_hash[:8]}...): {exists} ({result.count} docs across all threads)")
            return exists
        except Exception as e:
            logging.error(f"Global document existence check failed: {e}")
            return False

    async def document_exists(self, file_hash: str, user_id: str, thread_id: str) -> bool:
        """Check if document with given hash already exists in the specific thread."""
        try:
            if not await self.collection_exists():
                return False
            
            result = await self.client.count(
                collection_name=self.collection_name,
                count_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="metadata.file_hash", 
                            match=models.MatchValue(value=file_hash)
                        ),
                        models.FieldCondition(
                            key="metadata.user_id",
                            match=models.MatchValue(value=user_id)
                        ),
                        models.FieldCondition(
                            key="metadata.thread_id",
                            match=models.MatchValue(value=thread_id)
                        )
                    ]
                ),
                exact=True
            )
            exists = result.count > 0
            logging.debug(f"Document exists check ({file_hash[:8]}...) for user '{user_id}' thread '{thread_id}': {exists} ({result.count} docs)")
            return exists
        except Exception as e:
            logging.error(f"Document existence check failed: {e}")
            return False
    
    async def delete_collection(self):
        """Delete entire collection."""
        try:
            await self.client.delete_collection(self.collection_name)
            logging.info(f"Deleted collection '{self.collection_name}'")
        except Exception as e:
            logging.error(f"Error deleting collection: {e}")

    async def delete_by_thread_id(self, user_id: str, thread_id: str) -> None:
        """Delete all documents for a specific thread."""
        try:
            await self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="metadata.user_id",
                            match=models.MatchValue(value=user_id)
                        ),
                        models.FieldCondition(
                            key="metadata.thread_id",
                            match=models.MatchValue(value=thread_id)
                        )
                    ]
                )
            )
            logging.info(f"Deleted documents for user '{user_id}' thread '{thread_id}'")
        except Exception as e:
            logging.error(f"Error deleting documents for thread: {e}")
            # Don't raise the exception, just log it

    async def delete_by_user_id(self, user_id: str) -> None:
        """Delete all documents for a specific user."""
        try:
            await self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="metadata.user_id",
                            match=models.MatchValue(value=user_id)
                        )
                    ]
                )
            )
            logging.info(f"Deleted documents for user '{user_id}'")
        except Exception as e:
            logging.error(f"Error deleting documents for user: {e}")
            # Don't raise the exception, just log it

    # Sync wrappers for backward compatibility
    def create_collection_sync(self, vector_size: int, force_recreate: bool = False):
        return asyncio.run(self.create_collection(vector_size, force_recreate))
    
    def collection_exists_sync(self) -> bool:
        return asyncio.run(self.collection_exists())
    
    def document_exists_sync(self, file_hash: str, user_id: str, thread_id: str) -> bool:
        return asyncio.run(self.document_exists(file_hash, user_id, thread_id))
    
    def document_exists_globally_sync(self, file_hash: str) -> bool:
        return asyncio.run(self.document_exists_globally(file_hash))
    
    def delete_collection_sync(self):
        return asyncio.run(self.delete_collection())
    
    def delete_by_thread_id_sync(self, user_id: str, thread_id: str):
        return asyncio.run(self.delete_by_thread_id(user_id, thread_id))
    
    def delete_by_user_id_sync(self, user_id: str):
        return asyncio.run(self.delete_by_user_id(user_id))

