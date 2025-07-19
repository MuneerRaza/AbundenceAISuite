import logging
import os
from typing import List, Literal, Dict
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_deepinfra import DeepInfraEmbeddings
from config import EMBEDDING_MODEL_ID, CACHE_DIR

class EmbeddingManager:
    """Embedding manager supporting FastEmbed and DeepInfra providers."""
    
    def __init__(self, embedding_provider: Literal["fastembed", "deepinfra"] = "fastembed"):
        self.provider = embedding_provider
        
        if embedding_provider == "fastembed":
            self.embeddings = FastEmbedEmbeddings(
                model_name=EMBEDDING_MODEL_ID,
                cache_dir=CACHE_DIR,
                doc_embed_type="passage"
            )
            logging.info(f"Initialized FastEmbed embedding manager with model: {EMBEDDING_MODEL_ID}")
            
        elif embedding_provider == "deepinfra":
            api_key = os.getenv("DEEPINFRA_API_TOKEN")
            if not api_key:
                raise ValueError(
                    "DEEPINFRA_API_TOKEN environment variable is required for DeepInfra embeddings. "
                    "Please set it in your environment or .env file."
                )
            
            self.embeddings = DeepInfraEmbeddings(
                model=EMBEDDING_MODEL_ID
            )
            logging.info(f"Initialized DeepInfra embedding manager with model: {EMBEDDING_MODEL_ID}")
            
        else:
            raise ValueError(f"Unsupported embedding provider: {embedding_provider}")
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embedding model."""
        model_dimensions = {
            "BAAI/bge-small-en-v1.5": 384,
            "jinaai/jina-embeddings-v2-small-en": 512,
            "BAAI/bge-base-en-v1.5": 768,
            "jinaai/jina-embeddings-v2-base-en": 768,
            "BAAI/bge-m3": 1024
        }
        return model_dimensions.get(EMBEDDING_MODEL_ID, 384)
    
    def embed_query(self, query: str) -> List[float]:
        """Embed a single query using consistent embedding type."""
        return self.embeddings.embed_query(query)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        if not texts:
            return []
        
        logging.info(f"Embedding {len(texts)} documents...")
        embeddings = self.embeddings.embed_documents(texts)
        logging.info(f"Successfully embedded {len(embeddings)} documents")
        return embeddings
