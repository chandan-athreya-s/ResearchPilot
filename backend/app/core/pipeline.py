from app.services.openalex_client import fetch_papers
from app.services.text_processor import process_documents
from app.services.vector_store import create_vector_store
from app.services.retriever import retrieve_chunks
from app.services.llm_service import generate_answer


def run_pipeline(query):
    print("Fetching papers...")
    papers = fetch_papers(query)

    print("Processing text...")
    chunks = process_documents(papers)
    print(f"Number of chunks created: {len(chunks)}")

    if not chunks:
        return "No relevant documents found for the query."

    print("Creating vector store...")
    vector_store = create_vector_store(chunks)

    print("Retrieving relevant chunks...")
    docs = retrieve_chunks(vector_store, query)
    print(f"Number of docs retrieved: {len(docs)}")

    print("Generating answer...")
    answer = generate_answer(query, docs)

    return answer