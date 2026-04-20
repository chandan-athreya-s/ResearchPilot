from sentence_transformers import CrossEncoder
from collections import defaultdict

reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def retrieve_chunks(vector_store, query):
    # Step 1 — retrieve a large candidate pool from FAISS:
    candidate_docs = vector_store.similarity_search(query, k=30)

    # Step 2 — group candidates by source paper ID:
    grouped = defaultdict(list)
    for doc in candidate_docs:
        source_id = doc.metadata.get("paper_id", "unknown")  # Use .get() with fallback
        grouped[source_id].append(doc)

    # Step 3 — cap each source at 3 chunks, keeping highest similarity first:
    capped = []
    for source_id, docs in grouped.items():
        capped.extend(docs[:3])

    # Step 4 — rerank the capped pool and take top 5:
    reranked = rerank(query, capped)  # your existing reranker call
    final_docs = reranked[:5]

    # Step 5 — log which sources made it through:
    source_ids = set(d.metadata.get("paper_id", "unknown") for d in final_docs)
    print(f"✓ Final chunks drawn from {len(source_ids)} sources: {source_ids}")

    return final_docs

def rerank(query, docs):
    if not docs:
        return []
    passages = [doc.page_content for doc in docs]
    rerank_scores = reranker.predict([(query, passage) for passage in passages])
    reranked = sorted(zip(docs, rerank_scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in reranked]