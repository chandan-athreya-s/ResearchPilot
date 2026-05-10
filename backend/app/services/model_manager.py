# app/services/model_manager.py
"""
Global model manager for shared ML models to prevent repeated loading.
"""

from typing import Optional

# Global model instances
_embedding_model = None
_reranker_model = None

def get_embedding_model():
    """Get or create the shared embedding model."""
    global _embedding_model
    if _embedding_model is None:
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            _embedding_model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
        except ImportError as e:
            raise RuntimeError(
                "Missing optional dependency langchain_huggingface. "
                "Install it to use embedding models."
            ) from e
    return _embedding_model

def get_reranker_model():
    """Get or create the shared reranker model."""
    global _reranker_model
    if _reranker_model is None:
        try:
            from sentence_transformers import CrossEncoder
            _reranker_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        except ImportError:
            _reranker_model = None
    return _reranker_model