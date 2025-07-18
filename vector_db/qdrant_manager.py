import logging
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from qdrant_client.http import models

class QdrantManager:
    """Simple Qdrant database manager."""
    
    def __init__(self, host: str = "localhost", port: int = 6333, collection_name: str = "documents"):
        self.collection_name = collection_name
        
        # Try to connect with appropriate protocol based on port
        if port == 6334:
            try:
                # Try gRPC connection first for port 6334
                self.client = QdrantClient(host=host, grpc_port=port, prefer_grpc=True)
                logging.info(f"Connected to Qdrant using gRPC at {host}:{port}")
            except Exception as e:
                logging.warning(f"gRPC connection failed: {e}, trying HTTP on port 6333")
                self.client = QdrantClient(host=host, port=6333)
                logging.info(f"Connected to Qdrant using HTTP at {host}:6333")
        else:
            # Use HTTP connection for other ports
            self.client = QdrantClient(host=host, port=port)
            logging.info(f"Connected to Qdrant using HTTP at {host}:{port}")

    def create_collection(self, vector_size: int, force_recreate: bool = False):
        """Create collection for storing vectors."""
        if force_recreate:
            self.delete_collection()

        try:
            # Create collection
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE
                )
            )
            
            # Create indexes for efficient filtering
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="metadata.thread_id",
                field_schema=models.PayloadSchemaType.KEYWORD
            )
            
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="metadata.file_hash", 
                field_schema=models.PayloadSchemaType.KEYWORD
            )
            
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="metadata.user_id",
                field_schema=models.PayloadSchemaType.KEYWORD
            )

        except Exception as e:
            if "already exists" not in str(e):
                logging.error(f"Error creating collection: {e}")
                raise
    
    def collection_exists(self) -> bool:
        """Check if collection exists."""
        try:
            collections = self.client.get_collections()
            exists = any(col.name == self.collection_name for col in collections.collections)
            return exists
        except Exception as e:
            logging.error(f"Collection existence check failed: {e}")
            return False
    
    def document_exists(self, file_hash: str) -> bool:
        """Check if document with given hash already exists."""
        try:
            if not self.collection_exists():
                return False
            
            result = self.client.count(
                collection_name=self.collection_name,
                count_filter=models.Filter(
                    must=[models.FieldCondition(
                        key="metadata.file_hash", 
                        match=models.MatchValue(value=file_hash)
                    )]
                ),
                exact=True
            )
            exists = result.count > 0
            logging.debug(f"Document exists check ({file_hash[:8]}...): {exists} ({result.count} docs)")
            return exists
        except Exception as e:
            logging.error(f"Document existence check failed: {e}")
            return False
    
    def delete_collection(self):
        """Delete entire collection."""
        try:
            self.client.delete_collection(self.collection_name)
            logging.info(f"Deleted collection '{self.collection_name}'")
        except Exception as e:
            logging.error(f"Error deleting collection: {e}")

    def delete_by_thread_id(self, user_id: str, thread_id: str):
        """Delete all documents with specific thread_id."""
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
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
            )
            logging.info(f"Deleted documents for thread_id '{thread_id}'")
        except Exception as e:
            logging.error(f"Error deleting by thread_id: {e}")

    def delete_by_user_id(self, user_id: str):
        """Delete all documents with specific user_id."""
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[models.FieldCondition(
                            key="metadata.user_id", 
                            match=models.MatchValue(value=user_id)
                        )]
                    )
                )
            )
            logging.info(f"Deleted documents for user_id '{user_id}'")
        except Exception as e:
            logging.error(f"Error deleting by user_id: {e}")

