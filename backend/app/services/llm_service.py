import os
from dotenv import load_dotenv
from langchain_ollama import OllamaLLM

# Load environment variables from .env
load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
client = OllamaLLM(model="qwen2.5:7b", base_url=OLLAMA_BASE_URL)


def normalize_doi(doi_value):
    """Normalize DOI value to remove duplicate URL prefixes.
    
    Args:
        doi_value: Raw DOI string (may be raw DOI or already formatted URL)
    
    Returns:
        Clean DOI string ready for URL formatting, or None if invalid
    """
    if not doi_value:
        return None
    
    doi_value = str(doi_value).strip()
    
    # Remove any existing URL prefixes
    doi_value = doi_value.replace("https://doi.org/", "")
    doi_value = doi_value.replace("http://doi.org/", "")
    doi_value = doi_value.replace("doi.org/", "")
    
    # Ensure we have a valid DOI (typically starts with 10.)
    if doi_value and (doi_value.startswith("10.") or doi_value.startswith("http")):
        return doi_value.strip()
    
    return None


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

    prompt = f"""You are an expert research analyst synthesizing findings from academic papers.

RETRIEVED CONTEXT:
{formatted_chunks}

RESEARCH QUERY: {query}

TASK: Generate a structured research synthesis report (not a summary). Synthesize across papers, identify patterns, and provide critical analysis.

STRICT RULES FOR CITATIONS:
- EVERY technical claim, finding, or result must cite its source: [Source N, Chunk M]
- When synthesizing across multiple papers, cite each relevant source
- Do not make unsupported claims
- Do not cite sources not present in the context above

REPORT STRUCTURE:

1. INTRODUCTION
- 2-3 sentences establishing the research context and significance
- Briefly outline the scope of this analysis
- Do NOT include citations in introduction

2. METHODS & APPROACHES
- Identify distinct methodological approaches mentioned in the papers
- For each approach: describe its key components and technical details [Source X, Chunk Y]
- When 2+ approaches exist: compare them - highlight similarities, differences, trade-offs [Source X, Chunk Y] [Source Z, Chunk W]
- Focus on technical depth: algorithms, architectures, key parameters, novel techniques
- Avoid generic descriptions; extract specific technical concepts

3. IMPORTANT FINDINGS
- Present 4-6 key findings that synthesize across papers (NOT just one-per-paper)
- Each finding should:
  * Combine insights from related chunks and papers
  * Include specific results, metrics, or observations [Source X, Chunk Y]
  * When findings differ across sources, highlight the differences [Source A] vs [Source B]
  * Extract actionable insights, not just facts
- Group related findings thematically rather than by source

4. LIMITATIONS & CHALLENGES
- Identify technical limitations acknowledged in the papers [Source X, Chunk Y]
- Note open problems or gaps in the research [Source X, Chunk Y]
- Discuss trade-offs between different approaches when mentioned [Source X, Chunk Y]
- Include both acknowledged limitations and inferred gaps from the context

5. FUTURE DIRECTIONS
- Extract stated future work from the papers [Source X, Chunk Y]
- Identify promising research directions based on current findings [Source X, Chunk Y]
- Suggest logical next steps given the current state of methods/findings

6. CONCLUSION
- 2-3 sentences synthesizing the overall state of research on this topic
- Do NOT repeat findings; instead provide meta-analysis
- Do NOT include citations in conclusion

7. REFERENCES
[Will be auto-generated from sources]

OUTPUT ONLY THE REPORT. No meta-commentary, explanations, or instruction acknowledgments.
Begin now:

1. INTRODUCTION"""

    # Use the installed OllamaLLM API directly with a prompt list
    response = client.generate([prompt])
    answer = response.generations[0][0].text
    
    # Sort references by source number for consistent output
    sorted_refs = sorted(source_references.items(), 
                        key=lambda x: int(x[0].split()[-1]))
    
    # Build mapping from old source labels (Source N) to new sequential ref numbers [1], [2], etc.
    old_to_new_ref_num = {}
    for new_num, (old_label, _) in enumerate(sorted_refs, 1):
        old_source_num = int(old_label.split()[-1])
        old_to_new_ref_num[old_source_num] = new_num
    
    # Post-process: verify and replace internal citations with clean academic format
    verified_answer = post_process_citations(answer, source_references, old_to_new_ref_num)
    
    # Build structured References section with new sequential numbering and clean DOIs
    references_section = "\n\n7. REFERENCES\n"
    
    for new_num, (old_label, info) in enumerate(sorted_refs, 1):
        # Ensure this paper is in papers_with_extracted_text
        if info.get("paper_id") not in papers_with_extracted_text:
            continue
            
        authors = info.get("authors", [])
        year = info.get("year")
        venue = info.get("venue")
        doi = info.get("doi")
        url = info.get("url")
        title = info.get("title", "Unknown Title")
        
        # Format: [N] AuthorLastName et al. (Year). Title. Venue. DOI/URL
        if authors:
            author_str = authors[0].split()[-1] + " et al." if len(authors) > 1 else authors[0]
        else:
            author_str = "Unknown"
        
        ref = f"[{new_num}] {author_str}"
        if year:
            ref += f" ({year})"
        ref += f". {title}."
        if venue:
            ref += f" {venue}."
        
        # Fix 1: Normalize DOI to prevent duplication
        clean_doi = normalize_doi(doi)
        if clean_doi:
            ref += f" https://doi.org/{clean_doi}"
        elif url:
            ref += f" {url}"
        
        references_section += ref + "\n"
    
    return verified_answer + references_section


def post_process_citations(answer, source_references, old_to_new_ref_num):
    """Replace internal citations with clean academic format and validate grounding.
    
    Converts [Source X, Chunk Y] and (Source X, Chunk Y) citations to clean [N] format.
    Validates that all citations correspond to available sources.
    
    Args:
        answer: The raw answer text with internal citations
        source_references: Dict mapping old labels (Source N) to paper metadata
        old_to_new_ref_num: Dict mapping old source numbers to new sequential [N] numbers
    
    Returns:
        Processed answer text with clean academic citation format [N]
    """
    import re
    
    lines = answer.split('\n')
    processed_lines = []
    citation_stats = {'replaced': 0, 'invalid': 0, 'removed': 0}
    
    # Extract valid source numbers
    valid_source_nums = set(old_to_new_ref_num.keys())
    
    for line in lines:
        # Fix 3a: Replace [Source X, Chunk Y] citations with clean [N] format
        def replace_bracket_citation(match):
            source_part = match.group(1)
            source_num = int(source_part)
            if source_num in valid_source_nums:
                new_num = old_to_new_ref_num[source_num]
                citation_stats['replaced'] += 1
                return f"[{new_num}]"
            else:
                citation_stats['invalid'] += 1
                citation_stats['removed'] += 1
                return ""
        
        # Replace [Source X, Chunk Y] pattern
        line = re.sub(r'\[Source (\d+), Chunk \d+\]', replace_bracket_citation, line)
        
        # Fix 3b: Replace (Source X, Chunk Y) pattern with clean [N] format
        def replace_paren_citation(match):
            source_part = match.group(1)
            source_num = int(source_part)
            if source_num in valid_source_nums:
                new_num = old_to_new_ref_num[source_num]
                citation_stats['replaced'] += 1
                return f"[{new_num}]"
            else:
                citation_stats['invalid'] += 1
                citation_stats['removed'] += 1
                return ""
        
        # Replace (Source X, Chunk Y) pattern
        line = re.sub(r'\(Source (\d+), Chunk \d+\)', replace_paren_citation, line)
        
        # Clean up extra spaces caused by removed citations
        line = re.sub(r'\s+', ' ', line).strip()
        
        # Only add non-empty lines
        if line.strip():
            processed_lines.append(line)
    
    # Log citation transformation statistics
    total_replaced = citation_stats['replaced']
    total_removed = citation_stats['removed']
    
    if total_replaced > 0:
        print(f"✓ Citation transformation: {total_replaced} internal citations replaced with clean [N] format")
    if citation_stats['invalid'] > 0:
        print(f"⚠ Citation cleanup: {total_removed} invalid citations removed")
    
    return '\n'.join(processed_lines)

