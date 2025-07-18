from fastembed.rerank.cross_encoder import TextCrossEncoder
from config import RERANK_MODEL

class ModelManager:
    """Singleton class to manage shared model instances."""
    _instance = None
    _reranker = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def get_reranker(self) -> TextCrossEncoder:
        """Get the shared reranker instance, creating it if necessary."""
        if self._reranker is None:
            print(f"Loading shared reranker model: {RERANK_MODEL}")
            self._reranker = TextCrossEncoder(model_name=RERANK_MODEL)
            print(f"Shared reranker model loaded successfully")
        return self._reranker

# Global instance
model_manager = ModelManager()
