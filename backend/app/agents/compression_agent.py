from __future__ import annotations

import re
from collections import defaultdict
from difflib import SequenceMatcher
from typing import Any, Dict, List, Tuple

from app.agents.base_agent import BaseAgent
from app.core.state import ResearchState

MAX_TARGET_TOKENS = 650
MIN_TARGET_TOKENS = 300
KEYWORD_PRIORITY = [
    "method", "algorithm", "approach", "architecture", "model", "training", "evaluation",
    "metric", "performance", "result", "accuracy", "precision", "recall", "loss", "tradeoff",
    "limitation", "challenge", "constraint", "efficiency", "memory", "speed", "experiment",
    "benchmark", "dataset", "finding", "conclusion", "analysis", "contribution", "architecture",
    "implementation", "deployment", "evaluation", "benchmark", "dataset",
]
LOW_VALUE_PATTERNS = [
    r"\bthis paper\b",
    r"\bthis work\b",
    r"\bin summary\b",
    r"\bin this paper\b",
    r"\bmay\b",
    r"\bcould\b",
    r"\bwe\b",
    r"\bthe rest of the paper\b",
    r"\bintroduction\b",
    r"\brelated work\b",
    r"\bprevious work\b",
    r"\bsection\b",
    r"\brelated studies\b",
    r"\border of this paper\b",
    r"\bpaper is organized\b",
    r"\bthis article\b",
    r"\bwe propose\b",
    r"\bwe present\b",
]


def estimate_token_count(text: str) -> int:
    """Estimate rough token count using whitespace tokenization."""
    if not text:
        return 0
    return len(re.findall(r"\S+", text))


def _normalize_text(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def _split_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return [text]
    return sentences


def _sentence_score(sentence: str) -> int:
    score = 0
    normalized = sentence.lower()
    keyword_hits = sum(1 for keyword in KEYWORD_PRIORITY if keyword in normalized)
    score += keyword_hits * 2
    if re.search(r"\d+\%|\b\d+(\.\d+)?\b", normalized):
        score += 2
    if re.search(r"\[\d+\]", sentence):
        score += 2
    if re.search(r"\b(e\.g\.|i\.e\.|for example|such as|including|especially)\b", normalized):
        score += 1
    if re.search(r"\b(our|we|this|the)\b.*\bmethod\b", normalized):
        score += 1
    if keyword_hits > 3:
        score += 2
    return score


def _is_low_value_sentence(sentence: str) -> bool:
    normalized = sentence.lower()
    if len(normalized.split()) < 6 and not re.search(r"\d", normalized):
        return True
    for pattern in LOW_VALUE_PATTERNS:
        if re.search(pattern, normalized):
            return True
    return False


def deduplicate_chunks(chunks: List[Any], threshold: float = 0.92) -> List[Any]:
    """Remove near duplicate chunks by normalized content similarity."""
    unique = []
    seen = []
    for chunk in chunks:
        content = _normalize_text(chunk.page_content)
        keep = True
        for existing_content in seen:
            similarity = SequenceMatcher(None, content, existing_content).ratio()
            if similarity >= threshold:
                keep = False
                break
        if keep:
            unique.append(chunk)
            seen.append(content)
    return unique


def compress_chunk(chunk: Any, target_tokens: int = MAX_TARGET_TOKENS) -> Any:
    """Compress a single chunk with heuristic sentence selection."""
    if not getattr(chunk, "page_content", None):
        return chunk

    content = re.sub(r"\s+", " ", chunk.page_content).strip()
    if not content:
        chunk.page_content = ""
        return chunk

    sentences = _split_sentences(content)
    seen_sentences = set()
    deduped = []
    for sentence in sentences:
        normalized = sentence.strip()
        key = _normalize_text(normalized)
        if key in seen_sentences:
            continue
        seen_sentences.add(key)
        deduped.append(normalized)

    scored = []
    for idx, sentence in enumerate(deduped):
        score = _sentence_score(sentence)
        scored.append((sentence, score, idx))

    scored.sort(key=lambda item: (item[1], -item[2]), reverse=True)
    selected = []
    total_tokens = 0

    # Choose best sentences until target budget is reached.
    for sentence, score, original_index in scored:
        if score < 1 and total_tokens > target_tokens // 2:
            continue
        tokens = estimate_token_count(sentence)
        if tokens == 0:
            continue
        if total_tokens + tokens > target_tokens:
            continue
        selected.append((original_index, sentence))
        total_tokens += tokens

    if not selected and deduped:
        selected = [(0, deduped[0])]
        total_tokens = estimate_token_count(deduped[0])

    selected.sort(key=lambda item: item[0])
    compressed = " ".join(item[1] for item in selected)

    if not compressed and deduped:
        compressed = deduped[0]

    if estimate_token_count(compressed) > target_tokens:
        tokens = compressed.split()
        compressed = " ".join(tokens[:target_tokens]).rstrip()
        if not compressed.endswith("."):
            compressed = compressed.rsplit(" ", 1)[0].rstrip()
        compressed = compressed.strip()
        if compressed:
            compressed += " ..."

    chunk.page_content = compressed
    return chunk


def compress_context(chunks: List[Any], target_tokens: int = MAX_TARGET_TOKENS) -> Tuple[List[Any], Dict[str, Any]]:
    """Compress and deduplicate retrieved chunks before reasoning."""
    if not chunks:
        return [], {
            "compression_ratio": 1.0,
            "compressed_chunk_count": 0,
            "estimated_prompt_tokens": 0,
            "deduplicated_chunks": 0,
            "removed_redundancies": 0,
        }

    total_original_tokens = 0
    total_compressed_tokens = 0
    sentence_dedup_count = 0

    compressed_chunks = []
    for chunk in chunks:
        original_tokens = estimate_token_count(chunk.page_content)
        total_original_tokens += original_tokens
        compressed = compress_chunk(chunk, target_tokens=target_tokens)
        compressed_tokens = estimate_token_count(compressed.page_content)
        total_compressed_tokens += compressed_tokens
        compressed_chunks.append(compressed)
        if original_tokens > compressed_tokens:
            sentence_dedup_count += 1

    unique_chunks = deduplicate_chunks(compressed_chunks, threshold=0.94)
    removed_duplicates = len(compressed_chunks) - len(unique_chunks)
    prompt_tokens = sum(estimate_token_count(chunk.page_content) for chunk in unique_chunks) + 50

    stats = {
        "compression_ratio": round((total_compressed_tokens / total_original_tokens) if total_original_tokens else 1.0, 3),
        "compressed_chunk_count": len(unique_chunks),
        "estimated_prompt_tokens": prompt_tokens,
        "deduplicated_chunks": removed_duplicates,
        "removed_redundancies": sentence_dedup_count + removed_duplicates,
    }

    return unique_chunks, stats


class CompressionAgent(BaseAgent):
    """Compress retrieved evidence before final reasoning."""

    name = "CompressionAgent"

    def run(self, state: ResearchState) -> ResearchState:
        try:
            if not state.retrieved_chunks:
                self._log("No retrieved chunks available for compression.")
                return state

            compressed_chunks, stats = compress_context(state.retrieved_chunks)
            state.retrieved_chunks = compressed_chunks
            state.diagnostics["compression_ratio"] = stats["compression_ratio"]
            state.diagnostics["compressed_chunk_count"] = stats["compressed_chunk_count"]
            state.diagnostics["estimated_prompt_tokens"] = stats["estimated_prompt_tokens"]
            state.diagnostics["deduplicated_chunks"] = stats["deduplicated_chunks"]
            state.diagnostics["removed_redundancies"] = stats["removed_redundancies"]
            self._log(
                f"Compressed {len(state.retrieved_chunks)} chunks; prompt tokens ~{stats['estimated_prompt_tokens']}, ratio {stats['compression_ratio']}"
            )
        except Exception as error:
            error_message = f"Compression failed: {error}"
            self._log(error_message)
            state.errors.append(error_message)
        return state
