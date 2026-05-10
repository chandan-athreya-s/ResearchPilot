import json
import logging
import os

import httpx
from dotenv import load_dotenv
from langchain_ollama import OllamaLLM

# Load environment variables from .env
load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_TIMEOUT = httpx.Timeout(connect=10.0, read=120.0, write=10.0, pool=5.0)
OLLAMA_CLIENT = OllamaLLM(
    model="qwen2.5:7b",
    base_url=OLLAMA_BASE_URL,
    client_kwargs={"timeout": OLLAMA_TIMEOUT},
)

logger = logging.getLogger(__name__)


def ollama_generate_text(prompt: str) -> str:
    """Generate text from Ollama with a hard timeout and graceful fallback."""
    try:
        response = OLLAMA_CLIENT.generate([prompt])
        if not response or not getattr(response, "generations", None):
            return ""

        generations = response.generations
        if not generations or not generations[0]:
            return ""

        return generations[0][0].text or ""
    except httpx.TimeoutException:
        logger.error("Ollama generate timed out after 120s. Returning partial/empty result.")
        return ""
    except Exception as e:
        logger.error(f"Ollama generate failed: {e}")
        return ""


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
    
    # Bug 2 fix: Deduplicate paper IDs that should be skipped to prevent duplicate guard logs
    phantom_refs_logged = set()
    
    # Process documents and create context with labels
    for doc in docs:
        paper_id = doc.metadata.get("paper_id", "unknown")  # Safe access with fallback
        chunk_index = doc.metadata.get("chunk_index", 0)
        content = doc.page_content
        
        # Only include papers that have extracted text (Bug 2 fix: log phantom refs only once)
        if paper_id not in papers_with_extracted_text:
            if paper_id not in phantom_refs_logged:
                print(f"⚠ Skipping phantom reference for {paper_id}")
                phantom_refs_logged.add(paper_id)
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
    
    formatted_chunks = "\n\n".join(context_parts)
    
    # Debug: Print source diversity
    source_ids = sorted(set(source_to_label.values()))
    print(f"✓ Retrieved chunks from {len(source_ids)} sources: {', '.join(source_ids)}")
    print(f"✓ References will include only papers with extracted text: {len(source_references)} sources")

    prompt = f"""You are a research assistant writing a report.

RETRIEVED CONTEXT:
{formatted_chunks}

RESEARCH QUERY: {query}

Instructions:
- Write an Introduction of 2-3 sentences about the topic.
- Write exactly 3 Key Findings. Each finding must:
    * Be a specific claim drawn from the context above
    * End with the source number and chunk number in parentheses
      like this: (Source 1, Chunk 23)
- Write a Conclusion of 2-3 sentences.
- Do not repeat findings in the conclusion.
- Do not write "Thank you" or any conversational phrases.
- Do not output any instruction text — only output the report.

Begin the report now:

Introduction:"""

    # Use the installed OllamaLLM API directly with a prompt list
    response = OLLAMA_CLIENT.generate([prompt])
    answer = response.generations[0][0].text
    
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

