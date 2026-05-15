import os
import re
from dotenv import load_dotenv
from langchain_ollama import OllamaLLM

# Load environment variables from .env
load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
client = OllamaLLM(model="qwen2.5:7b", base_url=OLLAMA_BASE_URL)


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
    """Generate a comparison-specific prompt for evidence-grounded synthesis.
    
    Creates structured prompt that guides comparative analysis with evidence scaffolding.
    """
    if aspect_groups is None:
        aspect_groups = {}
    if comparison_pairs is None:
        comparison_pairs = []
    
    # Build context with explicit aspect organization
    context_parts = []
    
    # Organize by aspect for comparison clarity
    if aspect_groups and any(aspect_groups.values()):
        context_parts.append("EVIDENCE ORGANIZED BY RESEARCH ASPECT:\n")
        
        if aspect_groups.get("methods"):
            context_parts.append("METHODOLOGIES & APPROACHES:")
            for label, content, _ in aspect_groups["methods"]:
                context_parts.append(f"{label}\n{content}")
            context_parts.append("")
        
        if aspect_groups.get("datasets"):
            context_parts.append("DATASETS & EVALUATION BENCHMARKS:")
            for label, content, _ in aspect_groups["datasets"]:
                context_parts.append(f"{label}\n{content}")
            context_parts.append("")
        
        if aspect_groups.get("metrics"):
            context_parts.append("PERFORMANCE METRICS & RESULTS:")
            for label, content, _ in aspect_groups["metrics"]:
                context_parts.append(f"{label}\n{content}")
            context_parts.append("")
        
        if aspect_groups.get("tradeoffs"):
            context_parts.append("EXPLICIT TRADEOFFS & COMPARISONS:")
            for label, content, _ in aspect_groups["tradeoffs"]:
                context_parts.append(f"{label}\n{content}")
            context_parts.append("")
        
        if aspect_groups.get("computational"):
            context_parts.append("COMPUTATIONAL & EFFICIENCY CONSIDERATIONS:")
            for label, content, _ in aspect_groups["computational"]:
                context_parts.append(f"{label}\n{content}")
            context_parts.append("")
        
        if aspect_groups.get("architecture"):
            context_parts.append("TECHNICAL ARCHITECTURE & DESIGN:")
            for label, content, _ in aspect_groups["architecture"]:
                context_parts.append(f"{label}\n{content}")
            context_parts.append("")
        
        if aspect_groups.get("limitations"):
            context_parts.append("LIMITATIONS & CHALLENGES:")
            for label, content, _ in aspect_groups["limitations"]:
                context_parts.append(f"{label}\n{content}")
            context_parts.append("")
        
        if aspect_groups.get("applications"):
            context_parts.append("APPLICATIONS & USE CASES:")
            for label, content, _ in aspect_groups["applications"]:
                context_parts.append(f"{label}\n{content}")
            context_parts.append("")
    
    formatted_chunks = "\n".join(context_parts)
    
    # Build comparison-specific prompt
    comparison_instruction = ""
    if comparison_pairs:
        comparison_pairs_str = "\n".join([f"  - {a} vs. {b}" for a, b in comparison_pairs[:3]])
        comparison_instruction = f"""
COMPARISON FOCUS:
Provide explicit comparative analysis of:
{comparison_pairs_str}

For each pair, structure your analysis around:
1. **Methodological Approach**: How do they differ technically? [cite evidence]
2. **Performance & Metrics**: Which performs better on what metrics? [cite specific results]
3. **Computational Efficiency**: What are the computational tradeoffs? [cite evidence]
4. **Use Case Suitability**: When is each preferred? What domains? [cite examples]
5. **Architectural Differences**: Key technical distinctions? [cite details]
6. **Acknowledged Tradeoffs**: What do papers explicitly identify as tradeoffs?
7. **Research Gaps**: Where do they diverge in addressing challenges?
"""
    
    prompt = f"""You are an expert research analyst specializing in comparative technical analysis.

EVIDENCE CORPUS (organized by research aspect):
{formatted_chunks}

RESEARCH QUERY: {query}
{comparison_instruction}

TASK: Generate a structured comparative research synthesis that:
1. Extracts concrete technical differences (not generic statements)
2. Grounds every comparative claim in the evidence above
3. Identifies and explains tradeoffs explicitly
4. Maps use-case suitability based on technical characteristics
5. Avoids vague language ("may", "could", "likely") - use only what evidence supports
6. Synthesizes across papers rather than treating each separately

STRICT EVIDENCE GROUNDING:
- EVERY technical claim, comparison, or finding must cite its source using numeric bracket references only, e.g. [1], [2]
- Do not use internal labels such as Source N or Chunk M in the final report
- When comparing approaches: cite specifics from each approach's evidence
- When discussing tradeoffs: cite which paper identifies each tradeoff
- Do not make unsupported generalizations
- If evidence is insufficient for a claim, omit it
- Do not generate a References section; a single canonical REFERENCES section will be appended after your answer

REPORT STRUCTURE:

1. EXECUTIVE SUMMARY
- 2-3 sentences: Key technical differences between compared approaches
- Lead with most significant methodological distinction [cite evidence]

2. COMPARATIVE METHODOLOGIES  
Detailed side-by-side technical analysis:
- Approach A:
  * Algorithm/Architecture: [cite specific details]
  * Key Parameters/Design Choices: [cite evidence]
  * Technical Innovation: [cite distinguishing features]
- Approach B:
  * Algorithm/Architecture: [cite specific details]
  * Key Parameters/Design Choices: [cite evidence]
  * Technical Innovation: [cite distinguishing features]
- Methodological Synthesis: Explicit comparison of design philosophy, scalability assumptions, optimization targets [cite evidence]

3. PERFORMANCE ANALYSIS
For each relevant metric:
- Metric: [name]
- Approach A Result: [value] [cite source, dataset]
- Approach B Result: [value] [cite source, dataset]
- Interpretation: Which is superior for what scenario? [cite evidence supporting interpretation]

4. COMPUTATIONAL TRADEOFFS
Explicit analysis of efficiency considerations:
- Memory Requirements: A vs. B [cite evidence]
- Runtime Complexity: A vs. B [cite evidence]
- Scalability: A vs. B [cite evidence]
- Which is preferred for resource-constrained scenarios? [cite supporting evidence]

5. USE CASE SUITABILITY
Based on technical characteristics, when is each approach appropriate?
- Approach A suited for: [specific use cases/domains] because [cite technical evidence]
- Approach B suited for: [specific use cases/domains] because [cite technical evidence]
- Domain-specific considerations: [cite examples from papers]

6. EXPLICITLY IDENTIFIED TRADEOFFS
What do the papers themselves identify as core tradeoffs?
- Tradeoff 1: [Description] [cite paper that identifies this]
- Tradeoff 2: [Description] [cite paper that identifies this]
- Synthesis: How do researchers characterize the fundamental tension?

7. RESEARCH GAPS & OPEN QUESTIONS
- What does approach A not address that B does? [cite evidence]
- What limitations does approach B have that A avoids? [cite evidence]
- What remains unsolved in both approaches? [cite evidence]

8. CONCLUSION
- Synthesize the state of comparative research on this topic (not a summary)
- What are the key takeaways for practitioners/researchers? [cite evidence]
- No vague future speculations - only grounded observations

REFERENCES
[Will be auto-generated from sources]

OUTPUT ONLY THE REPORT. Begin now:

1. EXECUTIVE SUMMARY"""
    
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


def clean_final_answer(answer: str, old_to_new_ref_num: dict) -> str:
    """Clean the generated text before appending a single canonical References section."""
    answer = strip_generated_reference_section(answer)
    answer = replace_internal_source_citations(answer, old_to_new_ref_num)
    return answer


def generate_answer(query, docs, papers, metadata_store, papers_with_extracted_text, query_intent: dict = None):
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
- EVERY technical claim, finding, or result must cite its source using numeric bracket references only, e.g. [1], [2]
- Do not use internal labels such as Source N, Chunk M, or any debug identifiers in the final report
- Do not make unsupported claims
- Do not cite sources not present in the context above
- Do not generate a References section; a single canonical REFERENCES section will be appended after your answer

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

REFERENCES
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
        verified_answer = clean_final_answer(verified_answer, old_to_new_ref_num)
        
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
        
        verified_answer = post_process_citations(answer, source_references, old_to_new_ref_num)
        verified_answer = clean_final_answer(verified_answer, old_to_new_ref_num)
        
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
    to clean [N] format. Removes any invalid or unsupported internal labels.
    
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
    
    return '\n'.join(processed_lines)

