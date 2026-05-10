import asyncio
import json
import logging
import os
import uuid
from typing import Dict, Optional, List, Set
from collections import defaultdict

from app.agents.query_agent import process_query
from app.agents.retrieval_agent import retrieve_papers
from app.agents.citation_agent import augment_papers
from app.agents.report_agent import generate_report
from app.services.retriever import retrieve_chunks
from app.services.text_processor import process_documents
from app.services.vector_store import create_vector_store
from storage.faiss_store import save_index, load_index

ORCHESTRATION_TIMEOUT_SECONDS = 600  # 10 minutes max
SESSION_DIR = "./sessions"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _draft_report_path(session_id: str) -> str:
    return os.path.join(SESSION_DIR, session_id, "draft_report.json")


def _load_draft_report(session_id: str) -> Optional[Dict]:
    path = _draft_report_path(session_id)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load draft report for session {session_id}: {e}")
        return None


def _save_draft_report(session_id: str, draft_data: Dict) -> None:
    path = _draft_report_path(session_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(draft_data, f)
    logger.info(f"Draft report saved to {path}")


def _build_partial_result(session_id: str, session_state: Dict, draft_report: str | None = None) -> Dict:
    draft = draft_report or session_state.get("draft_report") or ""
    paper_sources = []
    if session_state.get("paper_ids_seen"):
        paper_sources = [{"paper_id": pid} for pid in sorted(session_state["paper_ids_seen"])]
    return {
        "answer": draft,
        "sources": paper_sources,
        "papers_used": len(session_state.get("paper_ids_seen", [])),
        "session_id": session_id,
        "iteration_count": session_state.get("iteration", 0),
        "mode": "partial_timeout",
        "note": "Note: Generation timed out. Results may be incomplete.",
        "timeout": True
    }


def _execute_with_timeout(func, session_id: str, session_state: Dict, *args, **kwargs):
    async def runner():
        return await asyncio.wait_for(asyncio.to_thread(func, *args, **kwargs), ORCHESTRATION_TIMEOUT_SECONDS)

    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(runner())
    except asyncio.TimeoutError:
        logger.warning(f"Orchestration timed out after {ORCHESTRATION_TIMEOUT_SECONDS} seconds")
        return _build_partial_result(session_id, session_state)
    finally:
        loop.close()


def _refine_sub_queries(sub_queries: List[str], keywords: List[str]) -> List[str]:
    """Generate new focused sub-queries using keywords instead of mutating existing ones."""
    refined = []
    for kw in keywords[:3]:  # Use up to 3 keywords
        for sub in sub_queries[:2]:  # Combine with up to 2 existing sub-queries
            new_query = f"{kw} {sub}"
            refined.append(new_query)
    return refined[:len(sub_queries)]  # Keep same number


def _evaluate_retrieval_coverage(papers: List[Dict], sub_queries: List[str], verbose: bool) -> Dict:
    """Evaluate coverage across retrieved papers and requested sub-queries."""
    unique_papers = {paper.get("paper_id") for paper in papers if paper.get("paper_id")}
    covered_subqueries = set()
    for paper in papers:
        for source_query in paper.get("source_sub_queries", []):
            if source_query in sub_queries:
                covered_subqueries.add(source_query)

    covered_count = len(covered_subqueries)
    required_count = len(sub_queries)
    covered_all = covered_count == required_count and unique_papers and len(unique_papers) >= 6

    if verbose:
        logger.info(f"Retrieval coverage: {len(unique_papers)} unique papers, {covered_count}/{required_count} sub-queries covered")
        if not covered_all:
            missing = [q for q in sub_queries if q not in covered_subqueries]
            logger.info(f"Missing sub-query coverage: {missing}")

    return {
        "unique_papers": len(unique_papers),
        "covered_count": covered_count,
        "required_subqueries": required_count,
        "covered_all_sub_queries": covered_all,
        "missing_subqueries": [q for q in sub_queries if q not in covered_subqueries]
    }


def orchestrate(
    query: str,
    session_id: Optional[str] = None,
    follow_up: bool = False,
    verbose: bool = False
) -> Dict:
    """
    Main orchestrator for the modular agent-based RAG system with persistence and iterative retrieval.

    Supports multi-turn conversations and iterative paper retrieval for better coverage.

    Args:
        query: User's research query string
        session_id: Optional session ID for persistence and multi-turn support
        follow_up: Whether this is a follow-up query in an existing session
        verbose: Enable detailed logging for debugging

    Returns:
        Dict with keys: answer, sources, papers_used, session_id, iteration_count
    """
    if session_id is None:
        session_id = str(uuid.uuid4())

    session_state = {
        'session_id': session_id,
        'original_query': query,
        'sub_queries': [],
        'iteration': 0,
        'paper_ids_seen': set(),
        'faiss_index_path': None,
        'metadata_path': None,
        'conversation_history': []
    }

    if verbose:
        logger.info(f"Orchestrator starting: session_id={session_id}, follow_up={follow_up}")
    else:
        logger.info(f"Processing query: {query[:50]}...")

    try:
        if session_id and not follow_up:
            draft_data = _load_draft_report(session_id)
            if draft_data:
                logger.info("Loaded existing draft report; skipping retrieval and initial generation")
                return {
                    "answer": draft_data.get("report", ""),
                    "sources": draft_data.get("sources", []),
                    "papers_used": draft_data.get("papers_used", 0),
                    "session_id": session_id,
                    "iteration_count": draft_data.get("iteration_count", 0),
                    "mode": "resume",
                    "note": "Resumed from saved draft report. Results may be partial.",
                    "timeout": False
                }

        loaded_session = load_index(session_id) if session_id else None
        if loaded_session and follow_up:
            logger.info("AGENT: Multi-turn conversation mode")
            return _handle_multi_turn(query, session_id, loaded_session, session_state, verbose)

        logger.info("AGENT: Full pipeline mode with iterative retrieval")
        return _execute_with_timeout(_handle_full_pipeline, session_id, session_state, query, session_id, session_state, verbose)

    except Exception as e:
        logger.error(f"AGENT: Orchestration failed: {str(e)}", exc_info=verbose)
        return {
            "answer": f"Error during processing: {str(e)}",
            "sources": [],
            "papers_used": 0,
            "session_id": session_id,
            "iteration_count": session_state.get('iteration', 0),
            "error": str(e)
        }


def _handle_multi_turn(
    query: str,
    session_id: str,
    loaded_session,
    session_state: Dict,
    verbose: bool
) -> Dict:
    """Handle multi-turn conversation by loading existing index and retrieving chunks."""
    index, metadata = loaded_session

    if verbose:
        logger.info("Multi-turn mode: loading existing session")

    retrieved_docs = retrieve_chunks(index, query, k=10)

    if verbose:
        logger.info(f"Retrieved {len(retrieved_docs)} chunks for follow-up query")

    augmented_papers = list(metadata.values())

    session_state['conversation_history'].append({
        'query': query,
        'chunks_used': len(retrieved_docs),
        'timestamp': None
    })

    # FIX 1: Replaced undefined generate_answer() call with generate_report(),
    # which is the correct function used throughout the rest of the module.
    report_result = generate_report(
        retrieved_docs,
        metadata,
        query,
        session_state.get('conversation_history', []),
        verbose=verbose
    )
    answer = report_result['report']

    sources = [
        {
            "paper_id": p.get("paper_id"),
            "title": p.get("title"),
            "url": p.get("url"),
            "authors": p.get("authors", []),
            "year": p.get("year"),
            "venue": p.get("venue")
        }
        for p in augmented_papers[:10]
    ]

    if verbose:
        logger.info("✓ Multi-turn response generated")

    return {
        "answer": answer,
        "sources": sources,
        "papers_used": len(augmented_papers),
        "session_id": session_id,
        "iteration_count": 0,
        "mode": "multi_turn"
    }


def _handle_full_pipeline(
    query: str,
    session_id: str,
    session_state: Dict,
    verbose: bool
) -> Dict:
    """Handle full pipeline with iterative retrieval."""
    if verbose:
        logger.info("Step 1: Query decomposition")

    query_result = process_query(query, verbose=verbose)
    primary_query = query_result["primary_query"]
    sub_queries = query_result["sub_queries"]
    keywords = query_result["keywords"]

    # FIX 2: Initialise session_state['sub_queries'] from the query agent result
    # so that iteration 0 and subsequent iterations both read from the same place,
    # removing the (sub_queries vs session_state['sub_queries']) inconsistency.
    session_state['sub_queries'] = sub_queries

    if verbose:
        logger.info(f"  Primary: {primary_query}")
        logger.info(f"  Sub-queries: {sub_queries}")
        logger.info(f"  Keywords: {keywords}")

    all_papers = []
    all_chunks = []
    metadata_store = {}
    papers_with_extracted_text: Set[str] = set()
    vector_store = None
    final_report = None  # FIX 3: Explicit initialisation avoids fragile locals() check later.
    prev_coverage_score = None
    original_sub_queries = list(sub_queries)

    max_iterations = 3

    for iteration in range(max_iterations):
        session_state['iteration'] = iteration

        if verbose:
            logger.info(f"=== ITERATION {iteration + 1}/{max_iterations} ===")

        if verbose:
            logger.info("  Step 2: Retrieval from OpenAlex")

        # FIX 2 (cont.): Always use session_state['sub_queries'] — consistent across iterations.
        retrieval_result = retrieve_papers(
            session_state['sub_queries'],
            keywords,
            primary_query,
            verbose=verbose
        )
        new_papers = retrieval_result.get("papers", [])
        new_chunks = retrieval_result.get("chunks", [])
        new_metadata_store = retrieval_result.get("metadata_store", {})
        new_papers_with_text = retrieval_result.get("papers_with_extracted_text", set())

        # Filter out already-seen papers (and their chunks) to avoid duplicates.
        new_papers = [p for p in new_papers if p['paper_id'] not in session_state['paper_ids_seen']]
        # FIX 4: Filter chunks to match the filtered papers so all_chunks stays consistent.
        new_paper_ids = {p['paper_id'] for p in new_papers}
        new_chunks = [c for c in new_chunks if c.metadata.get('paper_id') in new_paper_ids]

        if verbose:
            logger.info(f"    New papers retrieved: {len(new_papers)}")
            logger.info(f"    New chunks created: {len(new_chunks)}")

        all_papers.extend(new_papers)
        all_chunks.extend(new_chunks)
        session_state['paper_ids_seen'].update(p['paper_id'] for p in new_papers)
        metadata_store.update(new_metadata_store)
        papers_with_extracted_text.update(new_papers_with_text)

        if vector_store is None and retrieval_result.get("vector_store"):
            vector_store = retrieval_result["vector_store"]
        elif retrieval_result.get("vector_store") and all_chunks:
            vector_store = create_vector_store(all_chunks)

        retrieval_coverage = _evaluate_retrieval_coverage(all_papers, session_state['sub_queries'], verbose)

        if verbose:
            logger.info("  Step 3: Initial report generation for citation candidate extraction")

        top_chunks = retrieve_chunks(vector_store, primary_query, k=10) if vector_store else []
        initial_report_result = generate_report(
            top_chunks,
            metadata_store,
            query,
            session_state.get('conversation_history', []),
            verbose=verbose
        )
        draft_report = initial_report_result.get('report', "")
        session_state['draft_report'] = draft_report

        # Persist the draft report immediately to avoid losing work on timeout.
        _save_draft_report(session_id, {
            "report": draft_report,
            "coverage_score": initial_report_result.get('coverage_score', 0),
            "cited_paper_ids": initial_report_result.get('cited_paper_ids', []),
            "papers_used": len(all_papers),
            "sources": [
                {
                    "paper_id": p.get("paper_id"),
                    "title": p.get("title")
                }
                for p in all_papers[:10]
            ],
            "iteration_count": iteration + 1
        })

        cited_paper_ids = initial_report_result.get('cited_paper_ids', [])

        if verbose:
            logger.info(f"    Extracted {len(cited_paper_ids)} cited paper IDs from preliminary report")
            logger.info("  Step 4: Citation augmentation")

        citation_result = augment_papers(
            all_papers,
            paper_ids_seen=session_state['paper_ids_seen'],
            cited_paper_ids=cited_paper_ids,
            primary_query=primary_query,
            verbose=verbose
        )

        new_papers_added = citation_result["new_papers_added"]
        if new_papers_added == 0:
            logger.info("Citation agent added 0 papers — skipping final report regeneration, using draft report as final.")
            final_report = draft_report
            break
        if verbose:
            logger.info(f"    Citation agent added {new_papers_added} new paper(s) via references")

        new_augmented = [p for p in citation_result["augmented_papers"] if p['paper_id'] not in session_state['paper_ids_seen']]
        all_papers.extend(new_augmented)
        session_state['paper_ids_seen'].update(p['paper_id'] for p in new_augmented)

        if verbose:
            logger.info(f"    New papers from citations: {len(new_augmented)}")

        if new_augmented:
            extra_chunks = process_documents(new_augmented)
            all_chunks.extend(extra_chunks)
            if all_chunks:
                vector_store = create_vector_store(all_chunks)

        # FIX 3: Update FAISS index incrementally with new citation papers
        if new_papers_added > 0:
            logger.info(f"Updating FAISS index with {new_papers_added} new citation papers")
            new_chunks = process_documents(new_augmented)
            # Note: create_vector_store already called above, but we need incremental add
            # For now, recreate the full index (FIXME: implement incremental add)
            if all_chunks:
                vector_store = create_vector_store(all_chunks)
            logger.info(f"FAISS index updated: {len(new_chunks)} new chunks added")
        else:
            logger.info("No new citation papers — skipping FAISS update")

        try:
            if new_papers_added == 0:
                report_result = initial_report_result
            else:
                if new_augmented and vector_store:
                    top_chunks = retrieve_chunks(vector_store, primary_query, k=10)
                elif not new_augmented:
                    top_chunks = top_chunks
                else:
                    top_chunks = []

                report_result = generate_report(
                    top_chunks,
                    metadata_store,
                    query,
                    session_state.get('conversation_history', []),
                    verbose=verbose
                )

            coverage_score = report_result.get('coverage_score', 0)
            citation_integrity = report_result.get('citation_integrity_score', 1.0)
            retrieval_covered = retrieval_coverage.get('covered_all_sub_queries', False)

            if verbose:
                logger.info(f"    Coverage score: {coverage_score}")
                logger.info(f"    Retrieval coverage across sub-queries: {retrieval_coverage['covered_count']}/{retrieval_coverage['required_subqueries']}")
                logger.info(f"    Citation integrity score: {citation_integrity:.2f}")

            # FIX 2: Scaled stop threshold and minimum iteration floor
            MIN_COVERAGE_TO_STOP = {1: 999, 2: 5, 3: 8}  # require more coverage for more iterations
            stop_threshold = MIN_COVERAGE_TO_STOP.get(max_iterations, 8)

            if iteration < 2:
                logger.info(f"Minimum iteration floor: running iteration {iteration + 1} regardless of coverage.")
                # skip early-stop check for first 2 iterations
            elif coverage_score >= stop_threshold:
                logger.info(f"Coverage {coverage_score} >= threshold {stop_threshold}. Stopping early.")
                final_report = report_result['report']
                break

            if iteration > 0 and prev_coverage_score is not None and coverage_score < prev_coverage_score:
                logger.warning("Coverage score regressed from previous iteration; refreshing sub-queries with advanced search terms")
                session_state['sub_queries'] = [f"{q} recent advances" for q in original_sub_queries]
                session_state['sub_queries'] += [f"{q} systematic review" for q in original_sub_queries[:2]]
                prev_coverage_score = coverage_score
                continue

            if iteration > 1 and prev_coverage_score is not None and coverage_score == prev_coverage_score:
                if verbose:
                    logger.info("Coverage plateau detected, stopping early")
                final_report = report_result['report']
                break

            prev_coverage_score = coverage_score

            if iteration == max_iterations - 1:
                if verbose:
                    logger.warning("Weak coverage after final iteration, using best available report")
                final_report = report_result['report']
                break

        except Exception as e:
            if verbose:
                logger.warning(f"    Report generation failed for coverage check: {str(e)}")

        if verbose:
            logger.info("    Coverage insufficient, refining sub-queries")
        session_state['sub_queries'] = _refine_sub_queries(session_state['sub_queries'], keywords)

    if vector_store:
        save_index(vector_store, metadata_store, session_id)
        if verbose:
            logger.info("  Saved FAISS index to session")

    if not all_papers:
        logger.warning("No documents available for answer generation")
        return {
            "answer": "Unable to generate answer - no relevant papers found.",
            "sources": [],
            "papers_used": 0,
            "session_id": session_id,
            "iteration_count": session_state['iteration'] + 1
        }

    # FIX 3 (cont.): final_report is None only when every iteration's report generation
    # raised an exception. Fall back to one final attempt rather than crashing.
    if final_report is None:
        if verbose:
            logger.info("Step 4: Fallback report generation (all iterations failed coverage check)")
        top_chunks = retrieve_chunks(vector_store, primary_query, k=10)
        report_result = generate_report(
            top_chunks,
            metadata_store,
            query,
            session_state.get('conversation_history', []),
            verbose=verbose
        )
        final_report = report_result['report']

    # FIX 6: Removed the now-redundant separate Step 4 retrieve_chunks call whose
    # result (retrieved_docs) was never actually used in building the final answer.

    if verbose:
        logger.info(f"✓ Orchestration complete: {len(all_papers)} papers used")
    else:
        logger.info(f"Orchestration complete: generated answer from {len(all_papers)} papers")

    sources = [
        {
            "paper_id": p.get("paper_id"),
            "title": p.get("title"),
            "url": p.get("url"),
            "authors": p.get("authors", []),
            "year": p.get("year"),
            "venue": p.get("venue")
        }
        for p in all_papers[:10]
    ]

    return {
        "answer": final_report,
        "sources": sources,
        "papers_used": len(all_papers),
        "session_id": session_id,
        "iteration_count": session_state['iteration'] + 1,
        "mode": "full_pipeline"
    }