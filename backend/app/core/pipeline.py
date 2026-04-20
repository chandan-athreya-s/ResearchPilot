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
    
    # Bug 1 fix: Store IMMUTABLE snapshot of full OpenAlex metadata keyed by paper_id BEFORE any modifications
    metadata_store = {p["paper_id"]: p.copy() for p in papers}

    print("Downloading and extracting PDFs...")
    papers_with_extracted_text = set()
    for paper in papers:
        pdf_path = download_pdf_with_fallbacks(paper)
        if pdf_path:
            # Bug 1 fix: Verify downloaded paper's arXiv ID matches the OpenAlex record
            downloaded_arxiv_id = paper.get("arxiv_id")
            recorded_arxiv_id = metadata_store[paper["paper_id"]].get("arxiv_id")
            if downloaded_arxiv_id and recorded_arxiv_id and downloaded_arxiv_id != recorded_arxiv_id:
                print(f"⚠ arXiv ID mismatch for {paper['paper_id']}: downloaded {downloaded_arxiv_id} != recorded {recorded_arxiv_id}. Skipping.")
                continue
            
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
    chunks = process_documents(papers)
    print(f"Number of chunks created: {len(chunks)}")

    # Debug: Print metadata of first chunk to identify correct key
    if chunks:
        print(f"First chunk metadata: {chunks[0].metadata}")

    def extract_chunk_paper_id(chunk):
        for key in ["paper_id", "url", "source", "id"]:
            if chunk.metadata.get(key):
                return chunk.metadata.get(key)
        return None

    # Fix 1: Filter chunks to only include those whose paper ID is in papers_with_extracted_text
    filtered_chunks = [chunk for chunk in chunks if extract_chunk_paper_id(chunk) in papers_with_extracted_text]
    print(f"Number of chunks after phantom filtering: {len(chunks)} → {len(filtered_chunks)}")

    # Assertion and detailed debugging
    assert len(filtered_chunks) <= len(chunks), "Filter ran"
    removed_count = len(chunks) - len(filtered_chunks)
    print(f"Removed {removed_count} phantom chunks")

    if removed_count == 0 and chunks:
        print("DEBUG: No chunks removed. Checking metadata keys...")
        print(f"papers_with_extracted_text: {papers_with_extracted_text}")
        for i, chunk in enumerate(chunks[:3]):
            print(f"Chunk {i} metadata: {chunk.metadata}")
            print(f"Chunk {i} paper id candidates: {{'paper_id': chunk.metadata.get('paper_id'), 'url': chunk.metadata.get('url'), 'source': chunk.metadata.get('source'), 'id': chunk.metadata.get('id')}}")

    # Fix 2: Cap chunks per paper at 300 during indexing (increased from 200)
    from collections import defaultdict
    chunks_by_paper = defaultdict(list)
    for chunk in filtered_chunks:
        paper_id = extract_chunk_paper_id(chunk)
        if paper_id:
            chunks_by_paper[paper_id].append(chunk)

    capped_chunks = []
    for paper_id, paper_chunks in chunks_by_paper.items():
        capped_chunks.extend(paper_chunks[:300])  # Keep only first 300 chunks per paper

    print(f"Number of chunks after per-paper capping: {len(capped_chunks)}")

    if not capped_chunks:
        return "No relevant documents found for the query."

    print("Creating vector store...")
    vector_store = create_vector_store(capped_chunks)

    print("Retrieving relevant chunks...")
    docs = retrieve_chunks(vector_store, query)
    print(f"Number of docs retrieved: {len(docs)}")

    print("Generating answer...")
    answer = generate_answer(query, docs, filtered_papers, metadata_store, papers_with_extracted_text)

    return answer