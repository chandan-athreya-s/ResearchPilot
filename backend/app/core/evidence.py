from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class EvidenceObject:
    paper_id: str
    paper_title: str
    authors: List[str] = field(default_factory=list)
    year: Optional[int] = None
    method: str = ""
    dataset: str = ""
    dataset_name: str = ""
    benchmark: str = ""
    benchmark_name: str = ""
    metrics: List[str] = field(default_factory=list)
    metric_names: List[str] = field(default_factory=list)
    metric_values: List[str] = field(default_factory=list)
    metric_name: str = ""
    metric_value: str = ""
    latency: str = ""
    memory_cost: str = ""
    computational_cost: str = ""
    scalability_notes: str = ""
    findings: List[str] = field(default_factory=list)
    quantitative_findings: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    tradeoffs: List[str] = field(default_factory=list)
    relevance_to_query: str = ""
    source_reference: Dict[str, Any] = field(default_factory=dict)
    extracted_text: str = ""
    rank_score: float = 0.0


def _split_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text.strip())
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def _find_section(text: str, headings: List[str]) -> str:
    pattern = r"(?im)^(?:" + r"|".join(re.escape(h) for h in headings) + r")[\s:\-]*$"
    lines = text.splitlines()
    collecting = False
    collected: List[str] = []
    for line in lines:
        if re.match(pattern, line.strip()):
            collecting = True
            continue
        if collecting:
            if not line.strip():
                break
            collected.append(line.strip())
    return " ".join(collected).strip()


def _match_lines(text: str, patterns: List[str]) -> List[str]:
    results: List[str] = []
    for sentence in _split_sentences(text):
        for pattern in patterns:
            if re.search(pattern, sentence, re.I):
                if sentence not in results:
                    results.append(sentence)
                break
    return results


def normalize_metric_value(value: str) -> str:
    if not value:
        return ""
    normalized = value.strip().replace("–", "-").replace("—", "-")
    normalized = re.sub(r"\s*([%s])\s*" % re.escape("%"), r"\1", normalized)
    normalized = re.sub(r"\bsecs?\b", "s", normalized, flags=re.I)
    normalized = re.sub(r"\bmilliseconds\b", "ms", normalized, flags=re.I)
    normalized = re.sub(r"\bhours\b", "h", normalized, flags=re.I)
    normalized = re.sub(r"\bminutes\b", "m", normalized, flags=re.I)
    normalized = re.sub(r"\b(GB|MB|TB|kB|B)\b", lambda m: m.group(1), normalized, flags=re.I)
    return normalized.strip()


def extract_numerical_metrics(text: str, query_intent: dict = None) -> List[Dict[str, str]]:
    if not text:
        return []

    metric_terms = [
        "accuracy", "precision", "recall", "f1", "f1 score", "auc", "auroc",
        "map", "BLEU", "ROUGE", "latency", "throughput", "speed", "memory",
        "runtime", "cost", "inference cost", "training cost", "compute", "flops",
        "parameters", "fps", "error rate", "loss"
    ]
    metric_regex = re.compile(
        r"(?P<metric>" + r"|".join(re.escape(term) for term in metric_terms) + r")\b.*?(?P<value>\d+(?:\.\d+)?\s*(?:%|ms|s|sec|seconds|minutes|hours|h|m|GB|MB|TB|kB|B|x|flops)?)",
        re.I,
    )
    trailing_regex = re.compile(
        r"(?P<value>\d+(?:\.\d+)?\s*(?:%|ms|s|sec|seconds|minutes|hours|h|m|GB|MB|TB|kB|B|x|flops)?)\s*(?:for|on|in|of|with)?\s*(?P<metric>" + r"|".join(re.escape(term) for term in metric_terms) + r")\b",
        re.I,
    )

    extracted: List[Dict[str, str]] = []
    for sentence in _split_sentences(text):
        for pattern in (metric_regex, trailing_regex):
            for match in pattern.finditer(sentence):
                name = match.group("metric").strip().lower()
                value = normalize_metric_value(match.group("value"))
                if name and value:
                    extracted.append({"name": name, "value": value, "sentence": sentence})
    unique: List[Dict[str, str]] = []
    seen = set()
    for item in extracted:
        key = (item["name"], item["value"])
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


def extract_benchmarks(text: str, query_intent: dict = None) -> str:
    if not text:
        return ""
    benchmark_patterns = [
        r"\b(ImageNet|COCO|SQuAD|GLUE|SuperGLUE|MNIST|CIFAR|WikiText|WebText|MIMIC|OpenWebText|C4|WMT|SST|MS COCO|PASCAL VOC|Cityscapes)\b",
        r"\b(?:benchmark|benchmark dataset|standard benchmark|evaluation set|leaderboard|testbed|suite)\b",
    ]
    lines = _match_lines(text, benchmark_patterns)
    return lines[0] if lines else ""


def extract_datasets(text: str, query_intent: dict = None) -> str:
    if not text:
        return ""
    dataset_patterns = [
        r"\b(ImageNet|COCO|SQuAD|GLUE|SuperGLUE|MNIST|CIFAR|WikiText|WebText|MIMIC|OpenWebText|C4|WMT|SST|MS COCO|PASCAL VOC|Cityscapes)\b",
        r"\b(?:dataset|corpus|training set|test set|validation set|data set|corpus)\b",
    ]
    lines = _match_lines(text, dataset_patterns)
    return lines[0] if lines else ""


def extract_performance_comparisons(text: str, query_intent: dict = None) -> List[str]:
    if not text:
        return []
    patterns = [
        r"\b(outperform|better than|worse than|lower latency|higher latency|faster than|slower than|tradeoff|trade off|versus|vs\.?|compared to|compared with)\b",
    ]
    return _match_lines(text, patterns)


def extract_latency(text: str, query_intent: dict = None) -> str:
    if not text:
        return ""
    patterns = [r"\b\d+(?:\.\d+)?\s*(?:ms|s|sec|seconds|milliseconds)\b", r"\blatency\b"]
    lines = _match_lines(text, patterns)
    return lines[0] if lines else ""


def extract_memory_cost(text: str, query_intent: dict = None) -> str:
    if not text:
        return ""
    patterns = [r"\b\d+(?:\.\d+)?\s*(?:GB|MB|TB|kB|B)\b", r"\bmemory\b", r"\bfootprint\b"]
    lines = _match_lines(text, patterns)
    return lines[0] if lines else ""


def extract_computational_cost(text: str, query_intent: dict = None) -> str:
    if not text:
        return ""
    patterns = [
        r"\b(?:compute|computational|training cost|inference cost|flops|FLOPs|throughput|runtime|latency)\b",
    ]
    lines = _match_lines(text, patterns)
    return lines[0] if lines else ""


def extract_scalability_notes(text: str, query_intent: dict = None) -> str:
    if not text:
        return ""
    patterns = [
        r"\b(scalab|scale|scaling|scalable|throughput|parallel|distributed|large-scale|billions|millions)\b",
    ]
    lines = _match_lines(text, patterns)
    return lines[0] if lines else ""


def extract_quantitative_findings(text: str, query_intent: dict = None) -> List[str]:
    if not text:
        return []
    patterns = [
        r"\b(outperform|better than|achieve|achieves|achieved|accuracy|precision|recall|f1|auc|throughput|latency|memory|runtime|cost|flops)\b",
    ]
    sentences = _match_lines(text, patterns)
    return [s for s in sentences if re.search(r"\d", s)]


def extract_methodology(text: str, query_intent: dict = None) -> str:
    if not text:
        return ""
    sections = ["method", "methodology", "approach", "architecture", "implementation", "experimental setup", "system design"]
    section_text = _find_section(text, sections)
    if section_text:
        return section_text
    sentences = _match_lines(text, [r"\b(method|methodology|approach|architecture|framework|algorithm|technique|pipeline|implementation)\b"])
    return sentences[0] if sentences else ""


def extract_metrics(text: str, query_intent: dict = None) -> List[str]:
    if not text:
        return []

    metric_patterns = [
        r"\b(?:accuracy|precision|recall|f1|auc|auroc|mAP|BLEU|ROUGE|latency|throughput|speed|memory|runtime|cost|error|loss|score|performance)\b",
        r"\b\d+(?:\.\d+)?%\b",
        r"\b\d+(?:\.\d+)?\s*(?:points|pp|ms|s|seconds|minutes|hours|GB|MB|kB|TB)\b",
    ]
    metrics = []
    for sentence in _split_sentences(text):
        if any(re.search(pattern, sentence, re.I) for pattern in metric_patterns):
            metrics.append(sentence)
    return metrics


def extract_findings(text: str, query_intent: dict = None) -> List[str]:
    if not text:
        return []
    keywords = [
        r"\b(improve|improves|improved|outperform|outperforms|outperformed|achieve|achieves|achieved|demonstrate|demonstrates|demonstrated|show|shows|showed|indicate|indicates|indicated|yield|yields|reaches|reach|exceed|exceeds|better than)\b",
    ]
    findings = []
    for sentence in _split_sentences(text):
        if any(re.search(pattern, sentence, re.I) for pattern in keywords):
            findings.append(sentence)
    if not findings:
        findings = _match_lines(text, [r"\b(result|finding|show|demonstrate|indicate|improve|outperform)\b"])
    return findings


def extract_limitations(text: str, query_intent: dict = None) -> List[str]:
    if not text:
        return []
    patterns = [
        r"\b(limitation|limiting|challenge|difficult|drawback|weakness|constraint|failure|overhead|bottleneck|scalability|fragile|error|risk|cost)\b",
    ]
    return _match_lines(text, patterns)


def extract_tradeoffs(text: str, query_intent: dict = None) -> List[str]:
    if not text:
        return []
    patterns = [
        r"\b(trade[- ]?off|tradeoff|tradeoffs|trading off|versus|vs\.?|balance|compromise|more than|less than|compared to|compared with|instead of|rather than|while also|at the expense of|cost .* performance|efficiency .* accuracy|accuracy .* efficiency|memory .* speed|speed .* memory|memory overhead|higher memory|lower latency|training cost|computational cost|overhead)\b",
    ]
    return _match_lines(text, patterns)


def extract_dataset(text: str, query_intent: dict = None) -> str:
    if not text:
        return ""
    patterns = [
        r"\b(?:dataset|benchmark|corpus|training set|test set|validation set|data set|corpus)\b",
        r"\b(?:ImageNet|COCO|SQuAD|GLUE|SuperGLUE|MNIST|CIFAR|WikiText|WebText|MIMIC|OpenWebText|C4)\b",
    ]
    lines = _match_lines(text, patterns)
    return lines[0] if lines else ""


def extract_benchmark(text: str, query_intent: dict = None) -> str:
    if not text:
        return ""
    benchmark_patterns = [
        r"\b(ImageNet|COCO|SQuAD|GLUE|SuperGLUE|MNIST|CIFAR|WikiText|WebText|MIMIC|OpenWebText|C4)\b",
        r"\b(?:benchmark|benchmark dataset|standard benchmark|evaluation set|leaderboard|testbed|suite)\b",
    ]
    lines = _match_lines(text, benchmark_patterns)
    return lines[0] if lines else ""


def _compute_relevance(text: str, query_intent: dict = None) -> str:
    if not text:
        return ""
    focus_terms = query_intent.get("focus_terms", []) if query_intent else []
    found = [term for term in focus_terms if re.search(re.escape(term), text, re.I)]
    if found:
        return f"Matches focus terms: {', '.join(found)}"
    return "Evidence is relevant to the query by describing methods, benchmarks, metrics, and results."


def build_evidence_object(chunk: Any, metadata: Dict[str, Any], query_intent: dict = None) -> EvidenceObject:
    extracted_text = getattr(chunk, "page_content", "") or ""
    paper_id = metadata.get("paper_id", "unknown")
    paper_title = metadata.get("title") or metadata.get("paper_title") or "Unknown Title"
    authors = metadata.get("authors") or []
    year = metadata.get("year")
    source_reference = {
        "paper_id": paper_id,
        "title": paper_title,
        "authors": authors,
        "year": year,
        "venue": metadata.get("venue"),
        "doi": metadata.get("doi"),
        "url": metadata.get("url"),
    }

    method = extract_methodology(extracted_text, query_intent)
    dataset_name = extract_dataset(extracted_text, query_intent)
    benchmark_name = extract_benchmarks(extracted_text, query_intent)
    metrics = extract_metrics(extracted_text, query_intent)
    numerical_metrics = extract_numerical_metrics(extracted_text, query_intent)
    metric_names = [m["name"] for m in numerical_metrics]
    metric_values = [m["value"] for m in numerical_metrics]
    metric_name = metric_names[0] if metric_names else ""
    metric_value = metric_values[0] if metric_values else ""
    findings = extract_findings(extracted_text, query_intent)
    quantitative_findings = extract_quantitative_findings(extracted_text, query_intent)
    limitations = extract_limitations(extracted_text, query_intent)
    tradeoffs = extract_tradeoffs(extracted_text, query_intent)
    latency = extract_latency(extracted_text, query_intent)
    memory_cost = extract_memory_cost(extracted_text, query_intent)
    computational_cost = extract_computational_cost(extracted_text, query_intent)
    scalability_notes = extract_scalability_notes(extracted_text, query_intent)
    relevance_to_query = _compute_relevance(extracted_text, query_intent)

    return EvidenceObject(
        paper_id=paper_id,
        paper_title=paper_title,
        authors=authors,
        year=year,
        method=method,
        dataset=dataset_name,
        dataset_name=dataset_name,
        benchmark=benchmark_name,
        benchmark_name=benchmark_name,
        metrics=metrics,
        metric_names=metric_names,
        metric_values=metric_values,
        metric_name=metric_name,
        metric_value=metric_value,
        latency=latency,
        memory_cost=memory_cost,
        computational_cost=computational_cost,
        scalability_notes=scalability_notes,
        findings=findings,
        quantitative_findings=quantitative_findings,
        limitations=limitations,
        tradeoffs=tradeoffs,
        relevance_to_query=relevance_to_query,
        source_reference=source_reference,
        extracted_text=extracted_text,
    )
