import os
import re
from typing import Dict, List

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv():
        return None

# Load environment variables from .env
load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


def _get_ollama_client():
    try:
        from langchain_ollama import OllamaLLM
    except ImportError as error:
        raise ImportError(
            "langchain_ollama is required for LLM generation. "
            "Install the package or mock generate_answer in tests."
        ) from error
    return OllamaLLM(model="qwen2.5:7b", base_url=OLLAMA_BASE_URL)


def classify_chunk_aspects(chunk_content: str) -> dict:
    """Classify what research aspects a chunk covers.
    
    Returns dict with boolean flags for: methods, datasets, metrics, tradeoffs, 
    limitations, applications, computational_efficiency, architectural_details
    """
    content_lower = chunk_content.lower()
    
    aspects = {
        "methods": bool(re.search(r"\b(method|algorithm|approach|technique|architecture|framework|strategy|procedure|implementation)\b", content_lower)),
        "datasets": bool(re.search(r"\b(dataset|bench|corpus|data set|imagenet|coco|benchmark|training\s+set|test\s+set)\b", content_lower)),
        "metrics": bool(re.search(r"\b(accuracy|precision|recall|f1|auc|score|performance|metric|evaluation|measured|compared|baseline|result)\b", content_lower)),
        "tradeoffs": bool(re.search(r"\b(trade.off|tradeoff|trade.off|versus|versus|trade\s+between|balance|compromise|efficiency|speed|memory)\b", content_lower)),
        "limitations": bool(re.search(r"\b(limitation|limitation|challenge|problem|difficult|weakness|drawback|constraint|scalability|overhead|fail|failure|error)\b", content_lower)),
        "applications": bool(re.search(r"\b(application|use\s+case|domain|real.world|practical|deploy|production|task|problem|system)\b", content_lower)),
        "computational": bool(re.search(r"\b(complexity|efficient|memory|runtime|computation|speed|fast|slow|cost|expensive|cheap|o\(|linear|quadratic)\b", content_lower)),
        "architecture": bool(re.search(r"\b(architecture|architecture|layer|component|module|network|structure|pipeline|system|design)\b", content_lower)),
    }
    
    return aspects


def scaffold_evidence_for_comparison(docs, query_intent: dict = None):
    """Organize retrieved chunks by research aspect for comparison queries.
    
    Returns dict grouping chunks by aspect: methods, datasets, metrics, tradeoffs, 
    limitations, applications, computational, architecture.
    
    This helps the LLM reason about comparisons more systematically.
    """
    if query_intent is None or query_intent.get("query_type") != "comparison":
        return {}
    
    aspect_groups = {
        "methods": [],
        "datasets": [],
        "metrics": [],
        "tradeoffs": [],
        "limitations": [],
        "applications": [],
        "computational": [],
        "architecture": [],
    }
    
    aspect_to_label = {}
    
    for doc in docs:
        paper_id = doc.metadata.get("paper_id", "unknown")
        chunk_index = doc.metadata.get("chunk_index", 0)
        content = doc.page_content
        
        aspects = classify_chunk_aspects(content)
        
        # Group by dominant aspects
        for aspect, is_present in aspects.items():
            if is_present:
                label = f"[Source {paper_id[:8] if isinstance(paper_id, str) else paper_id}, Chunk {chunk_index}]"
                aspect_groups[aspect].append((label, content, doc))
    
    return aspect_groups


def generate_comparison_aware_prompt(query: str, docs, aspect_groups: dict = None, comparison_pairs: list = None) -> str:
    """Generate a concise comparison prompt for evidence-grounded synthesis."""
    if aspect_groups is None:
        aspect_groups = {}
    if comparison_pairs is None:
        comparison_pairs = []

    context_parts = ["EVIDENCE CORPUS:"]
    for label, header in [
        ("methods", "METHODOLOGIES & APPROACHES"),
        ("datasets", "DATASETS & BENCHMARKS"),
        ("metrics", "PERFORMANCE METRICS"),
        ("tradeoffs", "TRADEOFFS & LIMITATIONS"),
        ("computational", "COMPUTATIONAL EFFICIENCY"),
        ("architecture", "ARCHITECTURE & DESIGN"),
        ("limitations", "LIMITATIONS"),
        ("applications", "APPLICATIONS"),
    ]:
        items = aspect_groups.get(label)
        if items:
            context_parts.append(f"{header}:")
            for source_label, content, _ in items:
                context_parts.append(f"{source_label}\n{content}")
            context_parts.append("")

    comparison_pairs_str = "\n".join([f"- {a} vs {b}" for a, b in comparison_pairs[:3]])

    prompt = f"""You are a research analyst specializing in comparative technical evaluation.

{'\n'.join(context_parts)}
QUERY: {query}

COMPARISON FOCUS:
{comparison_pairs_str or '- No explicit comparison pair found'}

TASK: Compare the listed approaches using only the evidence above. Cite every technical claim as [N]. Avoid unsupported claims.

STRUCTURE:
1. EXECUTIVE SUMMARY: Key technical differences and most significant distinction.
2. Methodological Approach: Compare algorithmic/design choices.
3. PERFORMANCE ANALYSIS: Compare metrics, results, and benchmarks.
4. COMPUTATIONAL TRADEOFFS: Compare efficiency, memory, and scalability.
5. USE CASE SUITABILITY: When each approach is most appropriate.
6. EXPLICITLY IDENTIFIED TRADEOFFS: Explicit tradeoffs from the papers.
7. GAPS: What remains unresolved or under-addressed.
8. CONCLUSION: Grounded synthesis, no new speculation.

Do not use internal labels such as Source N or Chunk M in the final report. Do not generate a References section; it will be appended after your answer."""
    return prompt


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


def strip_generated_reference_section(text: str) -> str:
    """Remove any existing References section produced by the model."""
    pattern = re.compile(r'(?is)(?:\r?\n|^)\s*(?:\d+\.\s*)?references\s*[:\-]?\s*\r?\n.*$')
    return re.sub(pattern, '', text).strip()


def replace_internal_source_citations(answer: str, old_to_new_ref_num: dict) -> str:
    """Replace internal source labels with clean numeric citations."""
    def replace_source(match):
        source_num = int(match.group(1))
        if source_num in old_to_new_ref_num:
            return f"[{old_to_new_ref_num[source_num]}]"
        return ""

    answer = re.sub(r'\[Source\s+(\d+)(?:,\s*Chunk\s*\d+)?\]', replace_source, answer)
    answer = re.sub(r'\(Source\s+(\d+)(?:,\s*Chunk\s*\d+)?\)', replace_source, answer)
    answer = re.sub(r'\bSource\s+(\d+)\b', replace_source, answer)
    answer = re.sub(r'\bChunk\s+\d+\b', '', answer)
    answer = re.sub(r'\s+', ' ', answer)
    answer = re.sub(r'\[\s*(\d+)\s*\]', r'[\1]', answer)
    answer = re.sub(r'\(\s*\)', '', answer)
    answer = re.sub(r'\s+([.,;:])', r'\1', answer)
    answer = re.sub(r'\b(Source|Chunk)\b', '', answer)
    return answer.strip()


def validate_citations(answer: str, valid_ids: List[int]) -> str:
    """Remove citations that do not map to existing reference IDs."""
    if not answer:
        return answer

    valid_set = set(valid_ids)

    def replace(match):
        source_num = int(match.group(1))
        return f"[{source_num}]" if source_num in valid_set else ""

    answer = re.sub(r'\[\s*(\d+)\s*\]', replace, answer)
    answer = re.sub(r'\[\s*\]', '', answer)
    answer = re.sub(r'\s+([.,;:])', r'\1', answer)
    answer = re.sub(r'\s{2,}', ' ', answer)
    answer = re.sub(r' +\n', '\n', answer)
    return answer.strip()


def validate_inline_author_year_mentions(answer: str, source_references: dict) -> str:
    """Remove unsupported author/year mentions that are not backed by valid sources."""
    if not answer:
        return answer

    valid_surnames = set()
    valid_years = set()
    for source_info in source_references.values():
        authors = source_info.get("authors", []) or []
        for author in authors:
            if isinstance(author, str):
                surname = author.strip().split()[-1]
                if surname:
                    valid_surnames.add(surname)
        year = source_info.get("year")
        if year:
            valid_years.add(str(year))

    def replace_author_year(match):
        surname = match.group(1)
        year = match.group(2)
        if surname not in valid_surnames or (year and year not in valid_years):
            return ""
        return match.group(0)

    answer = re.sub(r'\b([A-Z][a-z]+) et al\.\s*\(?([0-9]{4})\)?', replace_author_year, answer)
    answer = re.sub(r'\b([A-Z][a-z]+)\s*\(\s*([0-9]{4})\s*\)', replace_author_year, answer)
    answer = re.sub(r'\s+([.,;:])', r'\1', answer)
    answer = re.sub(r'\s{2,}', ' ', answer)
    return answer.strip()


def remove_placeholder_citations(answer: str) -> tuple[str, int]:
    """Remove placeholder citations like [N123] or (N123) and count removals."""
    if not answer:
        return answer, 0
    placeholders = re.findall(r'\[\s*N\d+\s*\]|\(\s*N\d+\s*\)', answer, flags=re.I)
    cleaned = re.sub(r'\[\s*N\d+\s*\]|\(\s*N\d+\s*\)', '', answer, flags=re.I)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned, len(placeholders)


def clean_final_answer(answer: str, old_to_new_ref_num: dict) -> str:
    """Clean the generated text before appending a single canonical References section."""
    answer = strip_generated_reference_section(answer)
    answer = replace_internal_source_citations(answer, old_to_new_ref_num)
    valid_ids = sorted(set(old_to_new_ref_num.values()))
    answer = validate_citations(answer, valid_ids)
    return answer


def compress_context_chunks(docs, max_chars: int = 1200):
    """Compress document content to reduce prompt bloat while preserving meaning."""
    for doc in docs:
        content = doc.page_content
        content = re.sub(r'\s+', ' ', content).strip()
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        seen = set()
        deduped_lines = []
        for line in lines:
            if line in seen:
                continue
            seen.add(line)
            deduped_lines.append(line)

        compressed = '\n'.join(deduped_lines)
        if len(compressed) > max_chars:
            compressed = compressed[:max_chars].rstrip()
            if not compressed.endswith('.'):
                compressed = compressed.rsplit(' ', 1)[0].rstrip()
            compressed += ' ...'

        doc.page_content = compressed
    return docs


def generate_answer(query, docs, papers, metadata_store, papers_with_extracted_text, query_intent: dict = None, diagnostics: dict = None):
    """Generate an answer based on retrieved documents with proper source tracking.
    
    Args:
        query: The research query
        docs: Retrieved chunks with metadata
        papers: Filtered papers with extracted text
        metadata_store: Original OpenAlex metadata keyed by paper_id (source of truth)
        papers_with_extracted_text: Set of paper_ids that have extracted text
        query_intent: Optional dict with query_type, focus_terms, comparison_pairs
    """
    if query_intent is None:
        query_intent = {"query_type": "general", "focus_terms": [], "comparison_pairs": []}
    
    # Detect if this is a comparison query
    is_comparison_query = query_intent.get("query_type") == "comparison"
    
    # For comparison queries, use specialized evidence scaffolding
    if is_comparison_query:
        docs = compress_context_chunks(docs)
        aspect_groups = scaffold_evidence_for_comparison(docs, query_intent)
        comparison_pairs = query_intent.get("comparison_pairs", [])
        prompt = generate_comparison_aware_prompt(query, docs, aspect_groups, comparison_pairs)
    else:
        # Original logic for non-comparison queries
        context_parts = []
        source_references = {}  # Track unique sources
        source_to_label = {}   # Map paper_id to labels
        label_counter = 1
        
        # Bug 2 fix: Deduplicate paper IDs that should be skipped to prevent duplicate guard logs
        phantom_refs_logged = set()
        
        # Process documents and create context with labels
        docs = compress_context_chunks(docs)
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

        prompt = f"""You are a research analyst who writes concise, evidence-grounded technical syntheses.

RETRIEVED EVIDENCE:
{formatted_chunks}

RESEARCH QUERY: {query}

TASK: Synthesize the evidence into a structured research report. Cite every claim with numeric brackets [N]. Avoid unsupported language.

STRUCTURE:
1. INTRODUCTION: 2-3 sentences about scope and significance.
2. METHODS & APPROACHES: Compare technical approaches and architectures.
3. FINDINGS: List the strongest evidence-backed insights.
4. LIMITATIONS & CHALLENGES: Note weaknesses or open problems.
5. FUTURE DIRECTIONS: Extract stated future work or likely next steps.
6. CONCLUSION: Provide a brief, grounded summary.

Do not use internal labels such as Source N or Chunk M in the final answer. Do not generate a References section; it will be appended after your answer.

OUTPUT ONLY THE REPORT. Begin now:
1. INTRODUCTION"""
        
        # Use the installed OllamaLLM API directly with a prompt list
        client = _get_ollama_client()
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
        verified_answer, citation_stats = post_process_citations(answer, source_references, old_to_new_ref_num)
        verified_answer = clean_final_answer(verified_answer, old_to_new_ref_num)
        if diagnostics is not None:
            diagnostics["citation_cleanup_count"] = citation_stats.get("placeholder_removed", 0) + citation_stats.get("invalid", 0)
            diagnostics["grounding_validation"] = {
                "replaced": citation_stats.get("replaced", 0),
                "invalid": citation_stats.get("invalid", 0),
                "placeholder_removed": citation_stats.get("placeholder_removed", 0),
            }
        
        # Build structured References section with new sequential numbering and clean DOIs
        references_section = "\n\nREFERENCES\n"
        
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
    
    # Handle comparison queries with specialized synthesis
    if is_comparison_query:
        # Use the comparison-aware prompt with client
        client = _get_ollama_client()
        response = client.generate([prompt])
        answer = response.generations[0][0].text
        
        # Build references from documents
        source_references = {}
        source_to_label = {}
        label_counter = 1
        phantom_refs_logged = set()
        
        for doc in docs:
            paper_id = doc.metadata.get("paper_id", "unknown")
            
            if paper_id not in papers_with_extracted_text:
                if paper_id not in phantom_refs_logged:
                    print(f"⚠ Skipping phantom reference for {paper_id}")
                    phantom_refs_logged.add(paper_id)
                continue
            
            if paper_id not in source_to_label:
                label = f"Source {label_counter}"
                source_to_label[paper_id] = label
                original_meta = metadata_store.get(paper_id, {})
                source_references[label] = original_meta
                label_counter += 1
        
        # Sort and build references
        sorted_refs = sorted(source_references.items(), 
                            key=lambda x: int(x[0].split()[-1]))
        
        old_to_new_ref_num = {}
        for new_num, (old_label, _) in enumerate(sorted_refs, 1):
            old_source_num = int(old_label.split()[-1])
            old_to_new_ref_num[old_source_num] = new_num
        
        verified_answer, citation_stats = post_process_citations(answer, source_references, old_to_new_ref_num)
        verified_answer = clean_final_answer(verified_answer, old_to_new_ref_num)
        if diagnostics is not None:
            diagnostics["citation_cleanup_count"] = citation_stats.get("placeholder_removed", 0) + citation_stats.get("invalid", 0)
            diagnostics["grounding_validation"] = {
                "replaced": citation_stats.get("replaced", 0),
                "invalid": citation_stats.get("invalid", 0),
                "placeholder_removed": citation_stats.get("placeholder_removed", 0),
            }
        
        references_section = "\n\nREFERENCES\n"
        for new_num, (old_label, info) in enumerate(sorted_refs, 1):
            if info.get("paper_id") not in papers_with_extracted_text:
                continue
            
            authors = info.get("authors", [])
            year = info.get("year")
            venue = info.get("venue")
            doi = info.get("doi")
            url = info.get("url")
            title = info.get("title", "Unknown Title")
            
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
            
            clean_doi = normalize_doi(doi)
            if clean_doi:
                ref += f" https://doi.org/{clean_doi}"
            elif url:
                ref += f" {url}"
            
            references_section += ref + "\n"
        
        return verified_answer + references_section
    
    # Should not reach here, but return empty string as fallback
    return ""


def post_process_citations(answer, source_references, old_to_new_ref_num):
    """Replace internal citations with clean academic format and validate grounding.

    Converts [Source X, Chunk Y], (Source X, Chunk Y), and inline Source X citations
    to clean [N] format. Removes placeholders and invalid citations.

    Args:
        answer: The raw answer text with internal citations
        source_references: Dict mapping old labels (Source N) to paper metadata
        old_to_new_ref_num: Dict mapping old source numbers to new sequential [N] numbers

    Returns:
        Tuple of processed answer text and citation stats
    """
    import re

    answer, placeholder_removed = remove_placeholder_citations(answer)
    lines = answer.split('\n')
    processed_lines = []
    citation_stats = {'replaced': 0, 'invalid': 0, 'removed': 0, 'placeholder_removed': placeholder_removed}
    
    # Extract valid source numbers
    valid_source_nums = set(old_to_new_ref_num.keys())
    
    def replace_source_citation(match):
        source_part = match.group(1)
        source_num = int(source_part)
        if source_num in valid_source_nums:
            new_num = old_to_new_ref_num[source_num]
            citation_stats['replaced'] += 1
            return f"[{new_num}]"
        citation_stats['invalid'] += 1
        citation_stats['removed'] += 1
        return ""
    
    for line in lines:
        # Replace [Source X, Chunk Y] and [Source X] citations
        line = re.sub(r'\[Source\s+(\d+)(?:,\s*Chunk\s*\d+)?\]', replace_source_citation, line)
        # Replace (Source X, Chunk Y) and (Source X) citations
        line = re.sub(r'\(Source\s+(\d+)(?:,\s*Chunk\s*\d+)?\)', replace_source_citation, line)
        # Replace plain inline Source X labels if they appear
        line = re.sub(r'\bSource\s+(\d+)\b', replace_source_citation, line)
        # Remove leftover chunk references after citation cleanup
        line = re.sub(r'\bChunk\s+\d+\b', '', line)
        
        # Normalize spacing and punctuation from removed labels
        line = re.sub(r'\s+', ' ', line).strip()
        line = re.sub(r'\[\s*(\d+)\s*\]', r'[\1]', line)
        line = re.sub(r'\(\s*\)', '', line)
        line = re.sub(r'\s+([.,;:])', r'\1', line)
        line = re.sub(r'\b(Source|Chunk)\b', '', line)
        
        if line.strip():
            processed_lines.append(line)
    
    total_replaced = citation_stats['replaced']
    total_removed = citation_stats['removed']
    
    if total_replaced > 0:
        print(f"✓ Citation transformation: {total_replaced} internal citations replaced with clean [N] format")
    if citation_stats['invalid'] > 0:
        print(f"⚠ Citation cleanup: {total_removed} invalid citations removed")
    
    cleaned_answer = '\n'.join(processed_lines)
    cleaned_answer = validate_inline_author_year_mentions(cleaned_answer, source_references)
    return cleaned_answer, citation_stats

