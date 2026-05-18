from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


def _normalize_query_for_cache(query: str) -> str:
    """Normalize query string for cache key generation."""
    query = query.strip().lower()
    query = re.sub(r"\s+", " ", query)
    return query


def _get_cache_key(query: str) -> str:
    """Generate a cache key from a query string."""
    normalized = _normalize_query_for_cache(query)
    return hashlib.sha256(normalized.encode()).hexdigest()[:16]


class RetrievalCache:
    """In-memory cache for retrieval results with optional JSON persistence."""

    def __init__(self, cache_dir: Optional[str] = None, ttl_seconds: int = 3600):
        """Initialize the retrieval cache.
        
        Args:
            cache_dir: Optional directory for JSON cache persistence
            ttl_seconds: Cache time-to-live in seconds (default 1 hour)
        """
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.ttl_seconds = ttl_seconds
        self.memory_cache: Dict[str, Dict[str, Any]] = {}
        self.access_counts: Dict[str, int] = {}
        
        # Create cache directory if provided
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get(self, query: str, source: str) -> Optional[List[Dict[str, Any]]]:
        """Retrieve cached results for a query/source combination.
        
        Args:
            query: The search query
            source: The retrieval source (openalex, core)
            
        Returns:
            Cached results or None if not found or expired
        """
        cache_key = _get_cache_key(query)
        full_key = f"{cache_key}:{source}"
        
        # Check memory cache first
        if full_key in self.memory_cache:
            entry = self.memory_cache[full_key]
            if time.time() - entry["timestamp"] < self.ttl_seconds:
                self.access_counts[full_key] = self.access_counts.get(full_key, 0) + 1
                return entry["data"]
            else:
                # Expired, remove from cache
                del self.memory_cache[full_key]
                if full_key in self.access_counts:
                    del self.access_counts[full_key]
        
        # Try to load from disk if available
        if self.cache_dir:
            disk_data = self._load_from_disk(cache_key, source)
            if disk_data is not None:
                results, timestamp = disk_data
                if time.time() - timestamp < self.ttl_seconds:
                    # Restore to memory cache and return
                    self.memory_cache[full_key] = {"data": results, "timestamp": timestamp}
                    self.access_counts[full_key] = self.access_counts.get(full_key, 0) + 1
                    return results
        
        return None

    def set(self, query: str, source: str, results: List[Dict[str, Any]]) -> None:
        """Store retrieval results in the cache.
        
        Args:
            query: The search query
            source: The retrieval source
            results: The retrieval results to cache
        """
        cache_key = _get_cache_key(query)
        full_key = f"{cache_key}:{source}"
        timestamp = time.time()
        
        # Store in memory cache
        self.memory_cache[full_key] = {"data": results, "timestamp": timestamp}
        self.access_counts[full_key] = self.access_counts.get(full_key, 0) + 1
        
        # Optionally persist to disk
        if self.cache_dir:
            self._save_to_disk(cache_key, source, results, timestamp)

    def _load_from_disk(self, cache_key: str, source: str) -> Optional[tuple[List[Dict[str, Any]], float]]:
        """Load cache entry from disk."""
        if not self.cache_dir:
            return None
        
        cache_file = self.cache_dir / f"{cache_key}_{source}.json"
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file, "r") as f:
                data = json.load(f)
                return data.get("results", []), data.get("timestamp", 0)
        except Exception:
            return None

    def _save_to_disk(self, cache_key: str, source: str, results: List[Dict[str, Any]], timestamp: float) -> None:
        """Persist cache entry to disk."""
        if not self.cache_dir:
            return
        
        cache_file = self.cache_dir / f"{cache_key}_{source}.json"
        try:
            with open(cache_file, "w") as f:
                json.dump({"results": results, "timestamp": timestamp}, f)
        except Exception as e:
            print(f"[RetrievalCache] Failed to save cache to {cache_file}: {e}")

    def clear(self) -> None:
        """Clear all cached entries."""
        self.memory_cache.clear()
        self.access_counts.clear()
        print("[RetrievalCache] Cache cleared")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_entries = len(self.memory_cache)
        total_accesses = sum(self.access_counts.values())
        
        # Count active vs expired
        active_count = 0
        expired_count = 0
        current_time = time.time()
        
        for entry in self.memory_cache.values():
            if current_time - entry["timestamp"] < self.ttl_seconds:
                active_count += 1
            else:
                expired_count += 1
        
        return {
            "total_entries": total_entries,
            "active_entries": active_count,
            "expired_entries": expired_count,
            "total_accesses": total_accesses,
            "ttl_seconds": self.ttl_seconds,
        }


# Global singleton instance
_retrieval_cache: Optional[RetrievalCache] = None


def get_retrieval_cache(cache_dir: Optional[str] = None, ttl_seconds: int = 3600) -> RetrievalCache:
    """Get the global retrieval cache instance."""
    global _retrieval_cache
    if _retrieval_cache is None:
        _retrieval_cache = RetrievalCache(cache_dir=cache_dir, ttl_seconds=ttl_seconds)
    return _retrieval_cache
