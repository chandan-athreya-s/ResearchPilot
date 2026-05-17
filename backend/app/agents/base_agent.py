from __future__ import annotations

import time
import traceback
from typing import Any

from app.core.state import ResearchState


class BaseAgent:
    """Base abstraction for execution wrappers that mutate ResearchState."""

    name: str = "BaseAgent"

    def run(self, state: ResearchState) -> ResearchState:
        """Execute agent logic against the shared ResearchState."""
        raise NotImplementedError

    def execute(self, state: ResearchState) -> ResearchState:
        """Wrap run() with logging, timing, and error capture."""
        start = time.perf_counter()
        self._log("Starting")
        try:
            return self.run(state)
        except Exception as error:
            error_message = f"{self.name} failed: {error}"
            self._log(error_message)
            state.errors.append(error_message)
            state.errors.append(traceback.format_exc())
            return state
        finally:
            elapsed = time.perf_counter() - start
            metric_key = f"{self.name.replace('Agent', '').lower()}_time"
            state.diagnostics[metric_key] = elapsed
            self._log(f"Completed in {elapsed:.2f}s")

    def _log(self, message: str) -> None:
        print(f"[{self.name}] {message}")
