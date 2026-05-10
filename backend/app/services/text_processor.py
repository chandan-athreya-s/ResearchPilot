from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def process_documents(papers):
    """Process papers into Document objects with metadata preserved during chunking."""
    valid_papers = []
    for paper in papers:
        if paper.get("full_text"):
            valid_papers.append(paper)
        elif paper.get("abstract"):
            valid_papers.append(paper)
    
    if not valid_papers:
        return []  # Return empty list if no valid papers

    # Create Document objects with full content and metadata
    documents = []
    for paper in valid_papers:
        title = paper.get("title", "Unknown")
        full_text = paper.get("full_text", "")
        abstract = paper.get("abstract", "")
        url = paper.get("url", "Unknown")
        
        # Use full_text if available, else abstract
        content = full_text if full_text else abstract
        
        # Include title in page_content for better context
        page_content = f"Title: {title}\n\n{content}"
        
        metadata = {
            "title": title,
            "url": url,
            "paper_id": paper.get("paper_id"),
            "authors": paper.get("authors", []),
            "year": paper.get("year"),
            "venue": paper.get("venue"),
            "doi": paper.get("doi")
        }
        if paper.get("document_type") == "abstract_only":
            metadata["abstract_fallback"] = True

        doc = Document(
            page_content=page_content,
            metadata=metadata
        )
        documents.append(doc)

    # Chunk documents while preserving metadata
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = splitter.split_documents(documents)
    
    # Add chunk_index to metadata
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
    
    return chunks