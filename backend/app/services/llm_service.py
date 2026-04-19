from openai import OpenAI
import os
from dotenv import load_dotenv

#Load environment variables from .env
load_dotenv()

#Fetch API key
api_key = os.getenv("OPENAI_API_KEY")

# (Optional safety check)
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

#Initialize client
client = OpenAI(api_key=api_key)


def generate_answer(query, docs, papers, metadata_store, papers_with_extracted_text):
    """Generate an answer based on retrieved documents with proper source tracking.
    
    Args:
        query: The research query
        docs: Retrieved chunks with metadata
        papers: Filtered papers with extracted text
        metadata_store: Original OpenAlex metadata keyed by paper_id (source of truth)
        papers_with_extracted_text: Set of paper_ids that have extracted text
    """
    context_parts = []
    source_references = {}  # Track unique sources
    source_to_label = {}   # Map paper_id to labels
    label_counter = 1
    
    # Process documents and create context with labels
    for doc in docs:
        paper_id = doc.metadata.get("paper_id")
        chunk_index = doc.metadata.get("chunk_index", 0)
        content = doc.page_content
        
        # Only include papers that have extracted text (Bug 1 fix: filter at reference level)
        if paper_id not in papers_with_extracted_text:
            print(f"⚠ Skipping phantom reference for {paper_id}")
            continue
        
        # Assign label if not already assigned
        if paper_id not in source_to_label:
            label = f"Source {label_counter}"
            source_to_label[paper_id] = label
            # Bug 2 fix: Always use full OpenAlex metadata as source of truth
            original_meta = metadata_store.get(paper_id, {})
            source_references[label] = original_meta
            label_counter += 1
        
        label = source_to_label[paper_id]
        context_parts.append(f"[{label}, Chunk {chunk_index}]\n{content}")
    
    context = "\n\n".join(context_parts)
    
    # Debug: Print source diversity
    source_ids = sorted(set(source_to_label.values()))
    print(f"✓ Retrieved chunks from {len(source_ids)} sources: {', '.join(source_ids)}")
    print(f"✓ References will include only papers with extracted text: {len(source_references)} sources")
    
    # Build reference guide for the prompt
    references_guide = "\n".join(
        [f"[{label}]: {info.get('title', 'Unknown')} ({info.get('url', 'Unknown')})" 
         for label, info in source_references.items()]
    )

    prompt = f"""
    Answer the query using STRICTLY the provided context chunks. Do not introduce any information not present in the retrieved chunks. If the chunks do not support a claim, omit it.

    Structure your response in the following format:

    1. Introduction: Provide a brief overview of the topic based on the context.
    2. Key Findings: Summarize the main points and insights from the sources. You have been given chunks from multiple different sources. You MUST cite at least one chunk from each available source number in your Key Findings. If you have Sources 1, 2, and 3, each must appear at least once in the body. When making a claim, add an inline citation tag immediately after the sentence, e.g. [Source 3, Chunk 2]. Never cite the same [Source N, Chunk X] pair more than once.
    3. Conclusion: Offer a concise conclusion based on the findings.

    Keep the response concise. Base every claim strictly on the context chunks provided.

    If the context does not contain enough information to answer the query, say:
    "I don't have enough information from the retrieved papers."

    Query:
    {query}

    Source Reference Guide:
    {references_guide}

    Context:
    {context}
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    answer = response.choices[0].message.content
    
    # Post-process: verify citations
    verified_answer = post_process_citations(answer, source_references)
    
    # Append proper References section with rich metadata
    # Only include references from papers that have extracted text (Bug 1 fix)
    references_section = "\n\n4. References:\n"
    for label, info in source_references.items():
        # Ensure this paper is in papers_with_extracted_text
        if info.get("paper_id") not in papers_with_extracted_text:
            continue
            
        authors = info.get("authors", [])
        year = info.get("year")
        venue = info.get("venue")
        doi = info.get("doi")
        url = info.get("url")
        
        # Format: [Source N] AuthorLastName et al. (Year). Title. Venue. DOI or URL
        if authors:
            author_str = authors[0].split()[-1] + " et al." if len(authors) > 1 else authors[0]
        else:
            author_str = "Unknown"
        
        ref = f"   [{label}] {author_str}"
        if year:
            ref += f" ({year})"
        ref += f". {info.get('title', 'Unknown Title')}."
        if venue:
            ref += f" {venue}."
        if doi:
            ref += f" https://doi.org/{doi}"
        elif url:
            ref += f" {url}"
        references_section += ref + "\n"
    
    return verified_answer + references_section


def post_process_citations(answer, source_references):
    """Verify that all [Source N, Chunk M] citations in the body correspond to available sources."""
    lines = answer.split('\n')
    valid_labels = set(f"Source {i}" for i in range(1, len(source_references) + 1))
    
    processed_lines = []
    for line in lines:
        # Find all [Source X, Chunk Y] in the line
        import re
        citations = re.findall(r'\[Source \d+, Chunk \d+\]', line)
        invalid_citations = []
        for cit in citations:
            source_part = cit.split(',')[0]  # [Source X
            source_label = source_part[1:]  # Source X
            if source_label not in valid_labels:
                invalid_citations.append(cit)
        
        if invalid_citations:
            # Remove invalid citations
            for invalid in invalid_citations:
                line = line.replace(invalid, '[Citation removed - source not available]')
        
        processed_lines.append(line)
    
    return '\n'.join(processed_lines)

