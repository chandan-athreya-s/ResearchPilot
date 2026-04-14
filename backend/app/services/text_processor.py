from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def process_documents(papers):
    """Process papers into Document objects with metadata preserved during chunking."""
    valid_papers = [paper for paper in papers if paper.get("abstract")]
    
    if not valid_papers:
        return []  # Return empty list if no valid papers

    # Create Document objects with full content and metadata
    documents = []
    for paper in valid_papers:
        title = paper.get("title", "Unknown")
        abstract = paper.get("abstract", "")
        url = paper.get("url", "Unknown")
        
        # Include title in page_content for better context
        page_content = f"Title: {title}\n\nAbstract: {abstract}"
        
        doc = Document(
            page_content=page_content,
            metadata={"title": title, "url": url}
        )
        documents.append(doc)

    # Chunk documents while preserving metadata
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = splitter.split_documents(documents)
    return chunks