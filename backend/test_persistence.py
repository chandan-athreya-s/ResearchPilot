#!/usr/bin/env python3
"""Simple test for the new persistence features."""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from storage.faiss_store import save_index, load_index, list_sessions, delete_session

def test_faiss_store():
    print("Testing FAISS store...")

    # Test list_sessions (should be empty initially)
    sessions = list_sessions()
    print(f"Initial sessions: {sessions}")
    assert len(sessions) == 0, "Should start with no sessions"

    # Test save_index with mock data
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_core.documents import Document

    # Create a small mock index
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    docs = [Document(page_content="test content", metadata={"paper_id": "test"})]
    index = FAISS.from_documents(docs, embeddings)

    metadata = {"test": "data"}

    # Save
    success = save_index(index, metadata, "test_session")
    print(f"Save successful: {success}")
    assert success, "Save should succeed"

    # Test list_sessions again
    sessions = list_sessions()
    print(f"Sessions after save: {sessions}")
    assert "test_session" in sessions, "Session should be listed"

    # Test load_index
    loaded = load_index("test_session")
    print(f"Load successful: {loaded is not None}")
    assert loaded is not None, "Load should succeed"

    loaded_index, loaded_metadata = loaded
    assert loaded_metadata == metadata, "Metadata should match"

    # Test delete_session
    success = delete_session("test_session")
    print(f"Delete successful: {success}")
    assert success, "Delete should succeed"

    # Test list_sessions after delete
    sessions = list_sessions()
    print(f"Sessions after delete: {sessions}")
    assert len(sessions) == 0, "Should be empty after delete"

    print("✓ All FAISS store tests passed!")

if __name__ == "__main__":
    test_faiss_store()