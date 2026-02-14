"""Global LLM request pool with concurrency control and circuit breaker.

This module provides a centralized pool for ALL LLM calls with:
- Adaptive concurrency via CUBIC algorithm (reacts to 429 rate limits)
- Circuit breaker for endpoint health (reacts to 500s, timeouts)
- Global statistics tracking

All LLM requests flow through this pool regardless of source:
- Evolution jobs (main LLM)
- Meta analysis
- Novelty checking
- batch_query() calls

Concurrency modes (controlled by max_concurrent):
- max_concurrent=1: Fixed semaphore, no adaptive behavior
- max_concurrent>1: CUBIC adaptive concurrency with max_concurrent as ceiling
- max_concurrent=0: CUBIC adaptive concurrency with no artificial ceiling (auto mode)
"""
import threading
import logging
from dataclasses import dataclass
from typing import Optional, Callable, Any

from shinka.llm.circuit_breaker import CircuitBreaker
from shinka.llm.cubic import CubicConcurrency, is_rate_limit_error

logger = logging.getLogger(__name__)

# Auto mode ceiling — high enough to never be the bottleneck for CUBIC concurrency
AUTO_MODE_CEILING = 10000

# Thread pool cap for auto mode — decoupled from CUBIC ceiling to avoid
# spawning thousands of threads that contend for _db_lock.
AUTO_MODE_THREAD_POOL_SIZE = 200


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
    - Adaptive concurrency control via CUBIC (when max_concurrent > 1)
    - Fixed concurrency via semaphore (when max_concurrent == 1)
    - Circuit breaker for endpoint health (500s, timeouts)
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
        self._adaptive = max_concurrent != 1

        if self._adaptive:
            ceiling = AUTO_MODE_CEILING if max_concurrent == 0 else max_concurrent
            self._cubic = CubicConcurrency(ceiling=ceiling)
            self._semaphore = None
            mode_str = f"CUBIC adaptive (ceiling={ceiling})"
        else:
            self._cubic = None
            self._semaphore = threading.Semaphore(1)
            mode_str = "fixed (sequential)"

        self._stats = LLMPoolStats()
        self._stats_lock = threading.Lock()
        self._breaker = CircuitBreaker(
            name="llm-pool",
            failure_threshold=5,
            recovery_timeout=30.0,
            max_recovery_timeout=120.0,
        )
        self._initialized = True
        logger.info(f"LLMPool initialized: max_concurrent={max_concurrent}, mode={mode_str}")

    def submit(self, query_fn: Callable, *args, **kwargs) -> Any:
        """Submit an LLM request through the pool.

        Flow:
        1. Circuit breaker check — blocks if endpoint is down (500s, timeouts)
        2. Concurrency control — blocks until a slot is available
           (CUBIC adaptive or fixed semaphore depending on mode)
        3. Execute query
        4. Update concurrency controller based on outcome

        Args:
            query_fn: The function to call (typically _query_impl)
            *args: Positional arguments for query_fn
            **kwargs: Keyword arguments for query_fn

        Returns:
            The result from query_fn
        """
        # Circuit breaker check — blocks if endpoint is down
        self._breaker.before_request()

        with self._stats_lock:
            self._stats.total_requests += 1

        if self._adaptive:
            return self._submit_adaptive(query_fn, *args, **kwargs)
        else:
            with self._semaphore:
                return self._execute(query_fn, *args, **kwargs)

    def _submit_adaptive(self, query_fn: Callable, *args, **kwargs) -> Any:
        """Submit through CUBIC adaptive concurrency control."""
        self._cubic.acquire()
        rate_limited = False
        success = False
        try:
            result = self._execute(query_fn, *args, **kwargs)
            success = True
            return result
        except Exception as e:
            rate_limited = is_rate_limit_error(e)
            raise
        finally:
            self._cubic.release(rate_limited=rate_limited, success=success)

    def _execute(self, query_fn: Callable, *args, **kwargs) -> Any:
        """Execute the query, track stats, and update circuit breaker."""
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
                cache_read = getattr(result, 'cache_read_input_tokens', 0) or 0
                cache_write = getattr(result, 'cache_creation_input_tokens', 0) or 0
                if cache_read > 0 or cache_write > 0:
                    logger.info(f"Pool tracking cache: read={cache_read}, write={cache_write}")
                self._stats.total_cache_read_tokens += cache_read
                self._stats.total_cache_write_tokens += cache_write

            self._breaker.record_success()
            return result
        except Exception as e:
            # In adaptive mode, CUBIC handles 429s — don't double-count in circuit breaker.
            # In non-adaptive mode, all errors go to circuit breaker (no CUBIC to handle 429s).
            if self._adaptive and is_rate_limit_error(e):
                pass  # CUBIC handles this in _submit_adaptive()
            else:
                self._breaker.record_failure()
            raise
        finally:
            with self._stats_lock:
                self._stats.active_requests -= 1

    def get_stats(self) -> dict:
        """Get current pool statistics including circuit breaker and CUBIC state."""
        breaker_stats = self._breaker.get_stats()
        with self._stats_lock:
            stats = {
                "total_requests": self._stats.total_requests,
                "active_requests": self._stats.active_requests,
                "peak_concurrent": self._stats.peak_concurrent,
                "total_cost": self._stats.total_cost,
                "max_concurrent": self.max_concurrent,
                "adaptive": self._adaptive,
                "total_cache_read_tokens": self._stats.total_cache_read_tokens,
                "total_cache_write_tokens": self._stats.total_cache_write_tokens,
                "circuit_breaker_state": breaker_stats.current_state,
                "circuit_breaker_trips": breaker_stats.total_trips,
                "circuit_breaker_total_sleep_seconds": breaker_stats.total_sleep_seconds,
            }
        if self._cubic:
            cubic_stats = self._cubic.get_stats()
            stats["cubic_window"] = cubic_stats.window
            stats["cubic_in_slow_start"] = cubic_stats.in_slow_start
            stats["cubic_w_max"] = cubic_stats.w_max
            stats["cubic_total_rate_limits"] = cubic_stats.total_rate_limits
        return stats

    def reconfigure(self, max_concurrent: int) -> None:
        """Reconfigure the pool with a new max_concurrent value.

        WARNING: This is a "soft" reconfigure that does NOT drain existing requests.
        In-flight requests continue on the old controller. Threads blocked in
        the old CUBIC acquire() are released so they can reacquire on the new one.
        """
        with self._lock:
            old_max = self.max_concurrent
            old_cubic = self._cubic
            self.max_concurrent = max_concurrent
            self._adaptive = max_concurrent != 1

            if self._adaptive:
                ceiling = AUTO_MODE_CEILING if max_concurrent == 0 else max_concurrent
                self._cubic = CubicConcurrency(ceiling=ceiling)
                self._semaphore = None
            else:
                self._cubic = None
                self._semaphore = threading.Semaphore(1)

            # Release any threads blocked in the old CUBIC's acquire()
            if old_cubic is not None:
                old_cubic.reset()

            logger.warning(
                f"LLMPool reconfigured: max_concurrent {old_max} -> {max_concurrent}. "
                f"In-flight requests may temporarily exceed new limit."
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
        max_concurrent: Maximum concurrent API calls.
            1 = sequential (fixed semaphore)
            >1 = CUBIC adaptive with this as ceiling
            0 = CUBIC adaptive with no artificial ceiling (auto mode)

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
