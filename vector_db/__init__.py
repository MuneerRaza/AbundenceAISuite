"""
Simple Vector Database Package

This package provides a clean, simple interface for document indexing and retrieval
using Qdrant vector database and FastEmbed embeddings.

Components:
- EmbeddingManager: Handles text embeddings
- DocumentProcessor: Processes PDF documents 
- QdrantManager: Manages Qdrant database operations
- DocumentIndexer: Main interface combining all components
"""

from .embedding_manager import EmbeddingManager
from .document_processor import DocumentProcessor  
from .qdrant_manager import QdrantManager
from .document_indexer import DocumentIndexer
from config import QDRANT_HOST, QDRANT_PORT, QDRANT_COLLECTION

# Global indexer instance - initialized once and shared across the application
_global_indexer = None

def get_global_indexer():
    """Get the global DocumentIndexer instance, creating it if necessary."""
    global _global_indexer
    if _global_indexer is None:
        _global_indexer = DocumentIndexer(
            qdrant_host=QDRANT_HOST,
            qdrant_port=QDRANT_PORT,
            collection_name=QDRANT_COLLECTION,
            distance_threshold=0.7,  # Restored to proper value
            embedding_provider="fastembed",  # fastembed or deepinfra
            force_recreate=False  # Set to True to recreate collection
        )
    return _global_indexer

__all__ = [
    'EmbeddingManager',
    'DocumentProcessor', 
    'QdrantManager',
    'DocumentIndexer',
    'get_global_indexer'
]
