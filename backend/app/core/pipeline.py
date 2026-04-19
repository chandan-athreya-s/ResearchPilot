from app.services.openalex_client import fetch_papers
from app.services.pdf_downloader import download_pdf_with_fallbacks
from app.services.pdf_extractor import extract_text_from_pdf
from app.services.text_processor import process_documents
from app.services.vector_store import create_vector_store
from app.services.retriever import retrieve_chunks
from app.services.llm_service import generate_answer


def run_pipeline(query):
    print("Fetching papers...")
    papers = fetch_papers(query)
    
    # Store full OpenAlex metadata keyed by paper_id BEFORE any modifications
    metadata_store = {p["paper_id"]: p.copy() for p in papers}

    print("Downloading and extracting PDFs...")
    papers_with_extracted_text = set()
    for paper in papers:
        pdf_path = download_pdf_with_fallbacks(paper)
        if pdf_path:
            full_text = extract_text_from_pdf(pdf_path)
            if len(full_text) > 500:  # Substantial text
                paper["full_text"] = full_text
                papers_with_extracted_text.add(paper["paper_id"])
                print(f"✓ Extracted {len(full_text)} chars from {paper['paper_id']}")
            else:
                print(f"✗ Extracted text too short for {paper['paper_id']}")
        else:
            print(f"✗ No PDF/text available for {paper['paper_id']}")

    # Filter papers to only those with extracted text
    filtered_papers = [p for p in papers if p["paper_id"] in papers_with_extracted_text]
    
    if len(filtered_papers) < 2:
        print(f"⚠ Warning: Only {len(filtered_papers)} papers have extractable text. Results may be limited.")
    
    print("Processing text...")
    chunks = process_documents(filtered_papers)
    print(f"Number of chunks created: {len(chunks)}")

    if not chunks:
        return "No relevant documents found for the query."

    print("Creating vector store...")
    vector_store = create_vector_store(chunks)

    print("Retrieving relevant chunks...")
    docs = retrieve_chunks(vector_store, query)
    print(f"Number of docs retrieved: {len(docs)}")

    print("Generating answer...")
    answer = generate_answer(query, docs, filtered_papers, metadata_store, papers_with_extracted_text)

    return answer