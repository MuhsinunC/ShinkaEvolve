"""Global LLM request pool with concurrency control.

This module provides a centralized pool for ALL LLM calls with throttling
and queue management via semaphore-based concurrency control.

All LLM requests flow through this pool regardless of source:
- Evolution jobs (main LLM)
- Meta analysis
- Novelty checking
- batch_query() calls

This prevents rate limit issues by ensuring only max_concurrent API calls
can be active at any time.
"""
import threading
import logging
from dataclasses import dataclass, field
from typing import Optional, Callable, Any

logger = logging.getLogger(__name__)


@dataclass
class LLMPoolStats:
    """Statistics for the LLM pool."""
    total_requests: int = 0
    active_requests: int = 0
    total_cost: float = 0.0
    peak_concurrent: int = 0
    total_cache_read_tokens: int = 0
    total_cache_write_tokens: int = 0


class LLMPool:
    """
    Global singleton pool for all LLM calls.

    Features:
    - Max concurrency control via semaphore
    - Global statistics tracking
    - Thread-safe

    Usage:
        pool = get_llm_pool(max_concurrent=60)
        result = pool.submit(query_fn, *args, **kwargs)
    """
    _instance: Optional['LLMPool'] = None
    _lock = threading.Lock()

    def __new__(cls, max_concurrent: int = 60):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, max_concurrent: int = 60):
        if self._initialized:
            return
        self.max_concurrent = max_concurrent
        self._semaphore = threading.Semaphore(max_concurrent)
        self._stats = LLMPoolStats()
        self._stats_lock = threading.Lock()
        self._initialized = True
        logger.info(f"LLMPool initialized with max_concurrent={max_concurrent}")

    def submit(self, query_fn: Callable, *args, **kwargs) -> Any:
        """Submit an LLM request through the pool.

        This method blocks until a slot is available in the semaphore,
        ensuring we never exceed max_concurrent active API calls.

        Args:
            query_fn: The function to call (typically _query_impl)
            *args: Positional arguments for query_fn
            **kwargs: Keyword arguments for query_fn

        Returns:
            The result from query_fn
        """
        # Track active requests before acquiring semaphore
        with self._stats_lock:
            self._stats.total_requests += 1

        # Block until semaphore slot available
        with self._semaphore:
            return self._execute(query_fn, *args, **kwargs)

    def _execute(self, query_fn: Callable, *args, **kwargs) -> Any:
        """Execute the query and track stats."""
        # Track active count
        with self._stats_lock:
            self._stats.active_requests += 1
            if self._stats.active_requests > self._stats.peak_concurrent:
                self._stats.peak_concurrent = self._stats.active_requests

        try:
            result = query_fn(*args, **kwargs)

            # Track cost and cache metrics if available
            with self._stats_lock:
                if hasattr(result, 'cost') and result.cost:
                    self._stats.total_cost += result.cost
                # Debug: Log cache token tracking
                cache_read = getattr(result, 'cache_read_input_tokens', 0) or 0
                cache_write = getattr(result, 'cache_creation_input_tokens', 0) or 0
                if cache_read > 0 or cache_write > 0:
                    logger.info(f"Pool tracking cache: read={cache_read}, write={cache_write}")
                self._stats.total_cache_read_tokens += cache_read
                self._stats.total_cache_write_tokens += cache_write

            return result
        finally:
            with self._stats_lock:
                self._stats.active_requests -= 1

    def get_stats(self) -> dict:
        """Get current pool statistics."""
        with self._stats_lock:
            return {
                "total_requests": self._stats.total_requests,
                "active_requests": self._stats.active_requests,
                "peak_concurrent": self._stats.peak_concurrent,
                "total_cost": self._stats.total_cost,
                "max_concurrent": self.max_concurrent,
                "total_cache_read_tokens": self._stats.total_cache_read_tokens,
                "total_cache_write_tokens": self._stats.total_cache_write_tokens,
            }

    def reconfigure(self, max_concurrent: int) -> None:
        """Reconfigure the pool with a new max_concurrent value.

        Note: This recreates the semaphore. Existing in-flight requests
        are not affected, but new requests will use the new limit.
        """
        with self._lock:
            old_max = self.max_concurrent
            self.max_concurrent = max_concurrent
            self._semaphore = threading.Semaphore(max_concurrent)
            logger.info(
                f"LLMPool reconfigured: max_concurrent {old_max} -> {max_concurrent}"
            )

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton (primarily for testing)."""
        with cls._lock:
            if cls._instance is not None:
                logger.info("LLMPool reset - singleton cleared")
            cls._instance = None


# Module-level convenience functions

_default_max_concurrent = 60


def get_llm_pool(max_concurrent: Optional[int] = None) -> LLMPool:
    """Get the global LLM pool instance.

    On first call, initializes the pool with max_concurrent.
    Subsequent calls return the existing singleton (max_concurrent ignored
    unless you call reconfigure()).

    Args:
        max_concurrent: Maximum concurrent API calls. Only used on first call
                        or after reset(). Defaults to 60.

    Returns:
        The global LLMPool instance
    """
    if max_concurrent is None:
        max_concurrent = _default_max_concurrent
    return LLMPool(max_concurrent)


def configure_pool(max_concurrent: int) -> LLMPool:
    """Configure or reconfigure the global LLM pool.

    Use this at application startup to set the concurrency limit.
    If the pool already exists, it will be reconfigured.

    Args:
        max_concurrent: Maximum concurrent API calls

    Returns:
        The configured LLMPool instance
    """
    pool = get_llm_pool(max_concurrent)
    # If pool was already initialized with different value, reconfigure
    if pool.max_concurrent != max_concurrent:
        pool.reconfigure(max_concurrent)
    return pool


def reset_pool() -> None:
    """Reset the global LLM pool singleton.

    Primarily useful for testing. After reset, the next call to
    get_llm_pool() will create a fresh instance.
    """
    LLMPool.reset()
