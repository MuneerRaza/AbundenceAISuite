"""
Services package initialization with shared model instances.
This ensures models are loaded only once and shared across all service nodes.
"""
from fastembed.rerank.cross_encoder.onnx_text_cross_encoder import OnnxTextCrossEncoder
from config import RERANK_MODEL, CACHE_DIR

shared_reranker = OnnxTextCrossEncoder(model_name=RERANK_MODEL, cache_dir=CACHE_DIR)
print("Reranker model loaded successfully")

__all__ = ['shared_reranker']
