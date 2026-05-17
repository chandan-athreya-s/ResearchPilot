from pathlib import Path

INDEX_DIR = Path(__file__).parent.parent.parent.parent / "data" / "embeddings" / "faiss_index"

def create_vector_store(chunks):
    try:
        from langchain_community.vectorstores import FAISS
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError as error:
        raise ImportError(
            "langchain_community and langchain_huggingface are required to create a vector store. "
            "Install the required packages or mock create_vector_store in tests."
        ) from error

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Always create fresh index to avoid phantom chunks from previous runs
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    # Save the index
    vector_store.save_local(str(INDEX_DIR))
    return vector_store

