import logging
from typing import List
from langchain_community.embeddings import FastEmbedEmbeddings
from config import EMBEDDING_MODEL_ID, CACHE_DIR

class EmbeddingManager:
    """Simple embedding manager using FastEmbed."""
    
    def __init__(self):
        self.embeddings = FastEmbedEmbeddings(
            model_name=EMBEDDING_MODEL_ID,
            cache_dir=CACHE_DIR,
            doc_embed_type="passage"
        )
        logging.info(f"Initialized embedding manager with model: {EMBEDDING_MODEL_ID}")
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embedding model."""
        model_dimensions = {
            "BAAI/bge-small-en-v1.5": 384,
            "jinaai/jina-embeddings-v2-small-en": 512,
            "BAAI/bge-base-en-v1.5": 768,
            "BAAI/bge-m3": 1024
        }
        return model_dimensions.get(EMBEDDING_MODEL_ID, 384)
    
    def embed_query(self, query: str) -> List[float]:
        """Embed a single query."""
        return self.embeddings.embed_query(query)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        if not texts:
            return []
        
        logging.info(f"Embedding {len(texts)} documents...")
        embeddings = self.embeddings.embed_documents(texts)
        logging.info(f"Successfully embedded {len(embeddings)} documents")
        return embeddings
