from sentence_transformers import CrossEncoder

reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def retrieve_chunks(vector_store, query, k=10):
    docs_and_scores = vector_store.similarity_search_with_score(query, k=k)
    
    # Filter by similarity score
    filtered = [(doc, score) for doc, score in docs_and_scores if score >= 0.35]
    
    if len(filtered) < 3:
        # Relax threshold
        filtered = [(doc, score) for doc, score in docs_and_scores if score >= 0.25]
    
    docs = [doc for doc, score in filtered]
    
    # Rerank with cross-encoder
    if docs:
        passages = [doc.page_content for doc in docs]
        rerank_scores = reranker.predict([(query, passage) for passage in passages])
        # Sort by rerank score descending
        reranked = sorted(zip(docs, rerank_scores), key=lambda x: x[1], reverse=True)
        docs = [doc for doc, score in reranked[:5]]  # Top 5 after reranking
    
    return docs