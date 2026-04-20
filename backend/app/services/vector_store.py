from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from pathlib import Path

INDEX_DIR = Path(__file__).parent.parent.parent.parent / "data" / "embeddings" / "faiss_index"

def create_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Always create fresh index to avoid phantom chunks from previous runs
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    # Save the index
    vector_store.save_local(str(INDEX_DIR))
    return vector_store

