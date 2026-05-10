from langchain_community.vectorstores import FAISS
from pathlib import Path

from .model_manager import get_embedding_model

INDEX_DIR = Path(__file__).parent.parent.parent.parent / "data" / "embeddings" / "faiss_index"


def create_vector_store(chunks, existing_vector_store=None):
    """
    Create or update a vector store with new chunks.
    If existing_vector_store is provided, add chunks to it incrementally.
    Otherwise, create a fresh index.
    """
    embeddings = get_embedding_model()

    if existing_vector_store is not None:
        # Add new chunks to existing index
        existing_vector_store.add_documents(chunks)
        vector_store = existing_vector_store
    else:
        # Create fresh index
        vector_store = FAISS.from_documents(chunks, embeddings)
    
    # Save the index
    vector_store.save_local(str(INDEX_DIR))
    return vector_store

