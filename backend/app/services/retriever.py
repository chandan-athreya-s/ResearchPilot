def retrieve_chunks(vector_store, query, k=5):
    docs = vector_store.similarity_search(query, k=k)
    return docs