import os
import shutil
import pickle
import logging
from typing import Dict, Any, List, Optional, Tuple
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

SESSIONS_DIR = "./sessions"

def _get_session_dir(session_id: str) -> str:
    """Get the directory path for a session."""
    return os.path.join(SESSIONS_DIR, session_id)

def _ensure_sessions_dir():
    """Ensure the sessions directory exists."""
    os.makedirs(SESSIONS_DIR, exist_ok=True)

def _estimate_faiss_size(index: FAISS) -> float:
    """Estimate the size of FAISS index in MB."""
    # FAISS index size estimation: num_vectors * dimension * 4 bytes per float
    # This is approximate
    try:
        # Get the underlying FAISS index
        faiss_index = index.index
        num_vectors = faiss_index.ntotal
        dimension = faiss_index.d
        # Rough estimate: 4 bytes per float, plus some overhead
        size_bytes = num_vectors * dimension * 4 * 1.2  # 20% overhead
        size_mb = size_bytes / (1024 * 1024)
        return size_mb
    except Exception as e:
        logger.warning(f"Could not estimate FAISS size: {e}")
        return 0.0

def save_index(index: FAISS, metadata: Dict[str, Any], session_id: str) -> bool:
    """
    Save FAISS index and metadata for a session.

    Args:
        index: FAISS vector store
        metadata: Metadata dictionary
        session_id: Session identifier

    Returns:
        True if saved successfully, False otherwise
    """
    try:
        _ensure_sessions_dir()
        session_dir = _get_session_dir(session_id)
        os.makedirs(session_dir, exist_ok=True)

        # Estimate size and warn if large
        size_mb = _estimate_faiss_size(index)
        if size_mb > 500:
            logger.warning(f"FAISS index size estimated at {size_mb:.1f}MB (>500MB limit)")

        # Save FAISS index
        index_path = os.path.join(session_dir, "index")
        index.save_local(index_path)

        # Save metadata
        metadata_path = os.path.join(session_dir, "metadata.pkl")
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)

        logger.info(f"Saved session {session_id} to {session_dir}")
        return True

    except Exception as e:
        logger.error(f"Failed to save index for session {session_id}: {e}")
        return False

def load_index(session_id: str) -> Optional[Tuple[FAISS, Dict[str, Any]]]:
    """
    Load FAISS index and metadata for a session.

    Args:
        session_id: Session identifier

    Returns:
        Tuple of (FAISS index, metadata) if successful, None otherwise
    """
    try:
        session_dir = _get_session_dir(session_id)
        if not os.path.exists(session_dir):
            logger.warning(f"Session directory {session_dir} does not exist")
            return None

        # Load FAISS index
        index_path = os.path.join(session_dir, "index")
        if not os.path.exists(index_path):
            logger.warning(f"FAISS index not found at {index_path}")
            return None

        # Use the same embeddings as in the system
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        index = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)

        # Load metadata
        metadata_path = os.path.join(session_dir, "metadata.pkl")
        if not os.path.exists(metadata_path):
            logger.warning(f"Metadata file not found at {metadata_path}")
            return None

        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)

        logger.info(f"Loaded session {session_id} from {session_dir}")
        return index, metadata

    except Exception as e:
        logger.error(f"Failed to load index for session {session_id}: {e}")
        return None

def list_sessions() -> List[str]:
    """
    List all available session IDs.

    Returns:
        List of session IDs
    """
    try:
        _ensure_sessions_dir()
        if not os.path.exists(SESSIONS_DIR):
            return []

        sessions = []
        for item in os.listdir(SESSIONS_DIR):
            session_dir = os.path.join(SESSIONS_DIR, item)
            if os.path.isdir(session_dir):
                # Check if it has the required files
                index_path = os.path.join(session_dir, "index")
                metadata_path = os.path.join(session_dir, "metadata.pkl")
                if os.path.exists(index_path) and os.path.exists(metadata_path):
                    sessions.append(item)

        return sorted(sessions)

    except Exception as e:
        logger.error(f"Failed to list sessions: {e}")
        return []

def delete_session(session_id: str) -> bool:
    """
    Delete a session and all its data.

    Args:
        session_id: Session identifier

    Returns:
        True if deleted successfully, False otherwise
    """
    try:
        session_dir = _get_session_dir(session_id)
        if not os.path.exists(session_dir):
            logger.warning(f"Session directory {session_dir} does not exist")
            return False

        shutil.rmtree(session_dir)
        logger.info(f"Deleted session {session_id}")
        return True

    except Exception as e:
        logger.error(f"Failed to delete session {session_id}: {e}")
        return False