from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Dict, List, Tuple

from app.agents.base_agent import BaseAgent
from app.core.evidence import EvidenceObject, build_evidence_object
from app.core.state import ResearchState


class EvidenceExtractorAgent(BaseAgent):
    """Extract structured evidence objects from compressed retrieval chunks."""

    name = "EvidenceExtractorAgent"

    def run(self, state: ResearchState) -> ResearchState:
        try:
            if not state.retrieved_chunks:
                self._log("No retrieved chunks available for evidence extraction.")
                return state

            evidence_objects: List[EvidenceObject] = []
            unique_papers = set()
            method_signatures = set()
            dataset_signatures = set()
            benchmark_signatures = set()

            for chunk in state.retrieved_chunks:
                metadata = getattr(chunk, "metadata", {}) or {}
                paper_id = metadata.get("paper_id") or "unknown"
                paper_meta = state.metadata_store.get(paper_id, {})
                entry_metadata = {**paper_meta, **metadata}

                evidence = build_evidence_object(chunk, entry_metadata, state.query_intent)
                evidence.rank_score = self._score_evidence(evidence)
                evidence_objects.append(evidence)

                unique_papers.add(paper_id)
                if evidence.method:
                    method_signatures.add(evidence.method)
                if evidence.dataset_name:
                    dataset_signatures.add(evidence.dataset_name)
                if evidence.benchmark_name:
                    benchmark_signatures.add(evidence.benchmark_name)

            evidence_objects = self._rank_evidence_objects(evidence_objects)
            evidence_objects = self.enforce_evidence_coverage(evidence_objects, state)

            state.evidence_objects = evidence_objects
            state.diagnostics["evidence_objects_created"] = len(evidence_objects)
            state.diagnostics["extracted_metrics_count"] = sum(len(e.metric_values) for e in evidence_objects)
            state.diagnostics["quantitative_metrics_extracted"] = len([m for e in evidence_objects for m in e.metric_values])
            state.diagnostics["benchmark_count"] = len(benchmark_signatures)
            state.diagnostics["dataset_count"] = len(dataset_signatures)
            state.diagnostics["extracted_findings_count"] = sum(len(e.findings) for e in evidence_objects)
            state.diagnostics["extracted_tradeoffs_count"] = sum(len(e.tradeoffs) for e in evidence_objects)
            state.diagnostics["evidence_coverage"] = self._build_evidence_coverage(evidence_objects, state.query_intent)
            state.diagnostics["evidence_diversity"] = {
                "unique_papers": len(unique_papers),
                "unique_methodologies": len(method_signatures),
                "unique_datasets": len(dataset_signatures),
                "unique_benchmarks": len(benchmark_signatures),
            }
            state.diagnostics["evidence_ranking_scores"] = [round(e.rank_score, 2) for e in evidence_objects]

            self._log(
                f"Extracted {len(evidence_objects)} evidence objects from {len(state.retrieved_chunks)} chunks across {len(unique_papers)} papers."
            )
        except Exception as error:
            error_message = f"Evidence extraction failed: {error}"
            self._log(error_message)
            state.errors.append(error_message)
        return state

    def _score_evidence(self, evidence: EvidenceObject) -> float:
        score = 0.0
        score += len(evidence.metric_values) * 4
        score += len(evidence.quantitative_findings) * 2
        score += len(evidence.tradeoffs) * 2
        score += len(evidence.findings)
        score += len(evidence.limitations) * 0.5
        score += 3 if evidence.benchmark_name else 0
        score += 2 if evidence.dataset_name else 0
        if evidence.latency:
            score += 1.5
        if evidence.memory_cost:
            score += 1.5
        if evidence.computational_cost:
            score += 1.5
        if evidence.scalability_notes:
            score += 1.5
        if re.search(r"\b(enterprise|production|deploy|real world|system|deployment|business|industry)\b", evidence.extracted_text, re.I):
            score += 1.0
        if evidence.relevance_to_query:
            score += 0.5
        return score

    def _rank_evidence_objects(self, evidence_objects: List[EvidenceObject]) -> List[EvidenceObject]:
        return sorted(evidence_objects, key=lambda e: e.rank_score, reverse=True)

    def enforce_evidence_coverage(self, evidence_objects: List[EvidenceObject], state: ResearchState) -> List[EvidenceObject]:
        query_intent = state.query_intent or {}
        query_type = query_intent.get("query_type")
        if query_type != "comparison":
            return evidence_objects

        comparison_pairs = query_intent.get("comparison_pairs", [])
        entities = [entity.lower() for pair in comparison_pairs for entity in pair]
        entities = list(dict.fromkeys(entities))
        if not entities:
            entities = [term.lower() for term in query_intent.get("focus_terms", [])][:2]

        coverage_counts = self.detect_underrepresented_entities(evidence_objects, entities)
        state.diagnostics["evidence_coverage_balance"] = coverage_counts

        if any(count < 1 for count in coverage_counts.values()):
            self.retrieve_additional_evidence_if_needed(state, coverage_counts)

        balanced = self._balance_evidence_order(evidence_objects, entities)
        return balanced

    def detect_underrepresented_entities(self, evidence_objects: List[EvidenceObject], entities: List[str]) -> Dict[str, int]:
        counts = {entity: 0 for entity in entities}
        for entity in entities:
            pattern = re.compile(re.escape(entity), re.I)
            for evidence in evidence_objects:
                findings_text = " ".join(evidence.findings)
                if (
                    pattern.search(evidence.extracted_text)
                    or pattern.search(evidence.method)
                    or pattern.search(findings_text)
                    or pattern.search(evidence.dataset_name)
                    or pattern.search(evidence.benchmark_name)
                ):
                    counts[entity] += 1
        return counts

    def retrieve_additional_evidence_if_needed(self, state: ResearchState, coverage_counts: Dict[str, int]) -> None:
        missing = [entity for entity, count in coverage_counts.items() if count < 1]
        if missing:
            state.diagnostics["coverage_enforcement"] = {
                "underrepresented_entities": missing,
                "recommendation": "Consider expanding retrieval or boosting focus terms for missing entities.",
            }
            self._log(f"Underrepresented comparison entities detected: {missing}")

    def _balance_evidence_order(self, evidence_objects: List[EvidenceObject], entities: List[str]) -> List[EvidenceObject]:
        if not entities:
            return evidence_objects
        groups: Dict[str, List[EvidenceObject]] = {entity: [] for entity in entities}
        others: List[EvidenceObject] = []
        for evidence in evidence_objects:
            assigned = False
            text = " ".join([evidence.method, evidence.dataset_name, evidence.benchmark_name, " ".join(evidence.findings), evidence.extracted_text]).lower()
            for entity in entities:
                if entity in text:
                    groups[entity].append(evidence)
                    assigned = True
                    break
            if not assigned:
                others.append(evidence)

        balanced: List[EvidenceObject] = []
        pointers = {entity: 0 for entity in entities}
        while True:
            added = False
            for entity in entities:
                items = groups.get(entity, [])
                if pointers[entity] < len(items):
                    balanced.append(items[pointers[entity]])
                    pointers[entity] += 1
                    added = True
            if not added:
                break
        balanced.extend([e for e in evidence_objects if e not in balanced])
        return balanced

    def _build_evidence_coverage(self, evidence_objects: List[EvidenceObject], query_intent: Dict[str, Any]) -> Dict[str, int]:
        coverage: Dict[str, int] = {}
        focus_terms = query_intent.get("focus_terms", []) if query_intent else []
        for term in focus_terms:
            term_pattern = re.compile(re.escape(term), re.I)
            coverage[term] = sum(
                1
                for evidence in evidence_objects
                if term_pattern.search(evidence.extracted_text)
                or term_pattern.search(evidence.method)
                or term_pattern.search(evidence.dataset_name)
                or term_pattern.search(evidence.benchmark_name)
                or any(term_pattern.search(item) for item in evidence.metrics)
                or any(term_pattern.search(item) for item in evidence.metric_values)
                or any(term_pattern.search(item) for item in evidence.findings)
                or any(term_pattern.search(item) for item in evidence.tradeoffs)
                or any(term_pattern.search(item) for item in evidence.limitations)
            )
        return coverage
