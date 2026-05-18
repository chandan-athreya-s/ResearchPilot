from __future__ import annotations

import time
from typing import Dict, Optional


class RateLimitManager:
    """Manage API rate limits, cooldowns, and exponential backoff."""

    def __init__(self):
        self.source_status: Dict[str, Dict] = {
            "core": {
                "enabled": True,
                "cooldown_until": 0.0,
                "retry_after": None,
                "failure_count": 0,
                "last_failure_time": 0.0,
            },
            "openalex": {
                "enabled": True,
                "cooldown_until": 0.0,
                "retry_after": None,
                "failure_count": 0,
                "last_failure_time": 0.0,
            },
        }
        self.total_rate_limit_events = 0

    def record_rate_limit(self, source: str, retry_after: Optional[int] = None) -> None:
        """Record a rate limit event (429 response) for a source."""
        if source not in self.source_status:
            return

        self.total_rate_limit_events += 1
        status = self.source_status[source]
        status["failure_count"] += 1
        status["last_failure_time"] = time.time()

        # Calculate exponential backoff
        backoff = min(2 ** (status["failure_count"] - 1), 60)  # Max 60s backoff
        
        # Use provided retry-after if available
        if retry_after:
            backoff = max(backoff, retry_after)
        
        status["cooldown_until"] = time.time() + backoff
        status["retry_after"] = retry_after
        status["enabled"] = False
        
        print(f"[RateLimitManager] {source.upper()} rate limited. Cooldown: {backoff}s")

    def record_network_error(self, source: str) -> None:
        """Record a network error for a source."""
        if source not in self.source_status:
            return

        status = self.source_status[source]
        status["failure_count"] += 1
        status["last_failure_time"] = time.time()

        # Slightly aggressive backoff for network errors
        backoff = min(2 ** (status["failure_count"] - 1) * 0.5, 30)
        status["cooldown_until"] = time.time() + backoff
        
        if status["failure_count"] >= 3:
            status["enabled"] = False
            print(f"[RateLimitManager] {source.upper()} disabled after {status['failure_count']} failures. Cooldown: {backoff}s")

    def record_success(self, source: str) -> None:
        """Record a successful request for a source."""
        if source not in self.source_status:
            return

        status = self.source_status[source]
        status["enabled"] = True
        status["failure_count"] = 0
        status["retry_after"] = None
        status["cooldown_until"] = 0.0

    def is_enabled(self, source: str) -> bool:
        """Check if a source is currently enabled and not in cooldown."""
        if source not in self.source_status:
            return True

        status = self.source_status[source]
        
        # Check if cooldown has expired
        if time.time() >= status["cooldown_until"]:
            status["enabled"] = True
            status["cooldown_until"] = 0.0
        
        return status["enabled"]

    def get_time_until_retry(self, source: str) -> float:
        """Get seconds remaining until a source is available again."""
        if source not in self.source_status:
            return 0.0

        status = self.source_status[source]
        remaining = status["cooldown_until"] - time.time()
        return max(0.0, remaining)

    def get_status(self) -> Dict:
        """Get the current status of all sources."""
        status = {}
        for source, info in self.source_status.items():
            status[source] = {
                "enabled": self.is_enabled(source),
                "failure_count": info["failure_count"],
                "cooldown_remaining": self.get_time_until_retry(source),
                "retry_after": info["retry_after"],
            }
        status["total_rate_limit_events"] = self.total_rate_limit_events
        return status

    def reset(self, source: Optional[str] = None) -> None:
        """Reset the status for a source or all sources."""
        if source:
            if source in self.source_status:
                self.source_status[source].update({
                    "enabled": True,
                    "cooldown_until": 0.0,
                    "retry_after": None,
                    "failure_count": 0,
                    "last_failure_time": 0.0,
                })
        else:
            for key in self.source_status:
                self.reset(key)


# Global singleton instance
_rate_limit_manager: Optional[RateLimitManager] = None


def get_rate_limit_manager() -> RateLimitManager:
    """Get the global rate limit manager instance."""
    global _rate_limit_manager
    if _rate_limit_manager is None:
        _rate_limit_manager = RateLimitManager()
    return _rate_limit_manager
