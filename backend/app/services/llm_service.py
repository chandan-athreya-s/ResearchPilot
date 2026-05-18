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


def _format_evidence_objects_for_prompt(evidence_objects, query_intent: dict = None) -> str:
    if not evidence_objects:
        return ""

    lines = ["STRUCTURED EVIDENCE OBJECTS:"]
    for idx, evidence in enumerate(evidence_objects, start=1):
        lines.append(f"Evidence {idx}:")
        lines.append(f"Paper: {evidence.paper_title} ({evidence.year if evidence.year else 'unknown'})")
        if evidence.authors:
            lines.append(f"Authors: {', '.join(evidence.authors)}")
        source_ref = evidence.source_reference.get("url") or evidence.source_reference.get("doi") or evidence.paper_id
        lines.append(f"Source reference: {source_ref}")
        if evidence.rank_score:
            lines.append(f"Rank score: {round(evidence.rank_score, 2)}")
        if evidence.method:
            lines.append(f"Methodology: {evidence.method}")
        if evidence.dataset_name:
            lines.append(f"Dataset: {evidence.dataset_name}")
        if evidence.benchmark_name:
            lines.append(f"Benchmark: {evidence.benchmark_name}")
        if evidence.metrics:
            lines.append(f"Metrics: {'; '.join(evidence.metrics)}")
        if evidence.metric_names and evidence.metric_values:
            metric_pairs = [f"{name}: {value}" for name, value in zip(evidence.metric_names, evidence.metric_values)]
            lines.append(f"Quantitative metrics: {'; '.join(metric_pairs)}")
        if evidence.findings:
            lines.append(f"Findings: {'; '.join(evidence.findings)}")
        if evidence.quantitative_findings:
            lines.append(f"Quantitative findings: {'; '.join(evidence.quantitative_findings)}")
        if evidence.latency:
            lines.append(f"Latency: {evidence.latency}")
        if evidence.memory_cost:
            lines.append(f"Memory / cost: {evidence.memory_cost}")
        if evidence.computational_cost:
            lines.append(f"Compute / cost: {evidence.computational_cost}")
        if evidence.scalability_notes:
            lines.append(f"Scalability notes: {evidence.scalability_notes}")
        if evidence.tradeoffs:
            lines.append(f"Tradeoffs: {'; '.join(evidence.tradeoffs)}")
        if evidence.limitations:
            lines.append(f"Limitations: {'; '.join(evidence.limitations)}")
        if evidence.relevance_to_query:
            lines.append(f"Relevance: {evidence.relevance_to_query}")
        if evidence.extracted_text:
            lines.append(f"Extracted text: {evidence.extracted_text}")
        lines.append("")
    return "\n".join(lines)


def _build_query_structure(query_type: str) -> str:
    if query_type == "comparison":
        return (
            "1. EXECUTIVE SUMMARY: Key technical distinctions and strongest evidence-backed difference.\n"
            "2. METHODOLOGIES: Compare algorithmic and architectural choices.\n"
            "3. PERFORMANCE & BENCHMARK ANALYSIS: Compare metrics, dataset results, and benchmark behavior.\n"
            "4. TRADEOFFS & SCALABILITY: Compare computational cost, memory, latency, and operational tradeoffs.\n"
            "5. LIMITATIONS & GAPS: Identify weaknesses, bottlenecks, and unresolved issues.\n"
            "6. COVERAGE: Ensure all major query entities are represented.\n"
            "7. CONCLUSION: Evidence-grounded recommendation with no unsupported claims."
        )
    if query_type == "survey":
        return (
            "1. EXECUTIVE SUMMARY: What the evidence says about the topic.\n"
            "2. METHODS & TRENDS: Summarize methodologies and emerging patterns.\n"
            "3. DATASETS & BENCHMARKS: Highlight the most common evaluation settings.\n"
            "4. FINDINGS: Key results supported by extracted metrics.\n"
            "5. LIMITATIONS: Main constraints and research gaps.\n"
            "6. CONCLUSION: Evidence-grounded survey synopsis."
        )
    if query_type == "challenges":
        return (
            "1. EXECUTIVE SUMMARY: Main challenge areas and evidence-backed concerns.\n"
            "2. LIMITATIONS & BOTTLENECKS: List observed weaknesses and performance obstacles.\n"
            "3. SCALABILITY & COMPUTATIONAL CONCERNS: Highlight costs, overhead, and failure modes.\n"
            "4. TRADEOFFS: Identify key compromises and risks.\n"
            "5. OPEN PROBLEMS: Where evidence reports remaining gaps.\n"
            "6. CONCLUSION: Grounded risk assessment and future directions."
        )
    return (
        "1. EXECUTIVE SUMMARY: Scope and significance of the evidence.\n"
        "2. METHODS & APPROACHES: Technical design and implementation observations.\n"
        "3. FINDINGS: Strong evidence-backed outcomes.\n"
        "4. LIMITATIONS: Noted weaknesses and caveats.\n"
        "5. FUTURE DIRECTIONS: Likely next steps indicated by the evidence.\n"
        "6. CONCLUSION: Summary grounded in extracted evidence."
    )


def _build_source_references(evidence_objects, metadata_store, papers_with_extracted_text):
    source_references = {}
    paper_order = []
    for evidence in evidence_objects:
        paper_id = evidence.paper_id
        if paper_id in source_references or paper_id not in papers_with_extracted_text:
            continue
        source_references[paper_id] = metadata_store.get(paper_id, {"paper_id": paper_id})
        paper_order.append(paper_id)
    return source_references, paper_order


def generate_structured_evidence_prompt(query, evidence_objects, query_intent: dict = None):
    if query_intent is None:
        query_intent = {"query_type": "general", "focus_terms": [], "comparison_pairs": []}

    evidence_text = _format_evidence_objects_for_prompt(evidence_objects, query_intent)
    structure = _build_query_structure(query_intent.get("query_type", "general"))
    comparison_pairs = query_intent.get("comparison_pairs", [])
    comparison_text = "\n".join([f"- {a} vs {b}" for a, b in comparison_pairs])
    if query_intent.get("query_type") == "comparison" and not comparison_text:
        comparison_text = "- No explicit comparison pair found"

    prompt = f"""You are a research analyst who writes concise, evidence-grounded technical syntheses.

{evidence_text}
QUERY: {query}
FOCUS TERMS: {', '.join(query_intent.get('focus_terms', []))}
COMPARISON PAIRS: {comparison_text}

TASK: Use only the structured evidence above to answer the query. Rely on extracted findings, metrics, tradeoffs, limitations, and methodological details. Cite every technical claim with numeric brackets [N]. Avoid unsupported claims and do not invent comparisons that are not directly supported.

STRUCTURE:
{structure}

Do not use internal evidence labels in the final answer. Do not generate a References section; it will be appended after your answer.

OUTPUT ONLY THE REPORT. Begin now."""
    return prompt


def generate_answer(query, evidence_objects, papers, metadata_store, papers_with_extracted_text, query_intent: dict = None, diagnostics: dict = None):
    """Generate an answer based on structured evidence objects with proper source tracking."""
    if query_intent is None:
        query_intent = {"query_type": "general", "focus_terms": [], "comparison_pairs": []}

    if evidence_objects is None:
        evidence_objects = []

    if not evidence_objects:
        return "No structured evidence extracted to support the answer."

    prompt = generate_structured_evidence_prompt(query, evidence_objects, query_intent)
    client = _get_ollama_client()
    response = client.generate([prompt])
    answer = response.generations[0][0].text

    source_references, paper_order = _build_source_references(evidence_objects, metadata_store, papers_with_extracted_text)
    source_labels = {paper_id: f"Source {idx+1}" for idx, paper_id in enumerate(paper_order)}
    old_to_new_ref_num = {idx + 1: idx + 1 for idx in range(len(paper_order))}

    verified_answer, citation_stats = post_process_citations(answer, {label: source_references[paper_id] for paper_id, label in zip(paper_order, source_labels.values())}, old_to_new_ref_num)
    verified_answer = clean_final_answer(verified_answer, old_to_new_ref_num)

    if diagnostics is not None:
        diagnostics["citation_cleanup_count"] = citation_stats.get("placeholder_removed", 0) + citation_stats.get("invalid", 0)
        diagnostics["grounding_validation"] = {
            "replaced": citation_stats.get("replaced", 0),
            "invalid": citation_stats.get("invalid", 0),
            "placeholder_removed": citation_stats.get("placeholder_removed", 0),
        }

    references_section = "\n\nREFERENCES\n"
    for idx, paper_id in enumerate(paper_order, start=1):
        info = source_references.get(paper_id, {})
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

        ref = f"[{idx}] {author_str}"
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

