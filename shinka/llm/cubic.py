"""CUBIC-inspired adaptive concurrency control for LLM API calls.

Automatically discovers and tracks the optimal concurrency level by reacting
to HTTP 429 (rate limit) responses. Uses two phases:

1. Slow Start: Exponential growth from 1 until the first 429. Each success
   increments the window by 1; with N concurrent calls succeeding per round,
   the window doubles per round (~7 seconds to reach 100).

2. CUBIC Recovery: After the first 429, uses the TCP CUBIC formula to quickly
   recover to the last known-good concurrency (W_max), then cautiously probe
   above it.

   W(t) = C * (t - K)^3 + W_max

   Where t is time since last 429, K is the inflection point, and W_max is
   the window before the last 429. Constants C=0.4 and beta=0.7 are well-
   studied from 15+ years of TCP research.

Thread-safe — designed to replace threading.Semaphore in LLMPool.
"""

import logging
import threading
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CubicStats:
    """Observable statistics for monitoring."""
    window: int = 1
    active: int = 0
    ceiling: int = 10000
    in_slow_start: bool = True
    w_max: float = 0.0
    total_rate_limits: int = 0
    total_successes: int = 0


class CubicConcurrency:
    """CUBIC-inspired adaptive concurrency control.

    Drop-in replacement for threading.Semaphore in LLMPool. Dynamically adjusts
    how many concurrent requests are allowed based on 429 feedback.

    Args:
        ceiling: Hard maximum concurrency (never exceeded).
        beta: Multiplicative decrease factor on 429 (default 0.7).
        C: CUBIC scaling constant (default 0.4).
    """

    def __init__(
        self,
        ceiling: int = 10000,
        beta: float = 0.7,
        C: float = 0.4,
    ):
        self._ceiling = ceiling
        self._beta = beta
        self._C = C

        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)

        # Concurrency state
        self._active = 0
        self._window = 1.0  # Start at 1 for slow start
        self._in_slow_start = True

        # CUBIC state
        self._w_max = 0.0  # Window before last 429
        self._epoch_start = 0.0  # Time of last 429

        # Stats
        self._total_rate_limits = 0
        self._total_successes = 0

    @property
    def window(self) -> int:
        """Current effective concurrency window."""
        with self._lock:
            return max(1, int(self._window))

    @property
    def active(self) -> int:
        """Number of currently active requests."""
        with self._lock:
            return self._active

    def acquire(self) -> None:
        """Block until a concurrency slot is available."""
        with self._condition:
            while self._active >= max(1, int(self._window)):
                # Periodic wake-up to detect window changes from CUBIC time function
                self._condition.wait(timeout=1.0)
            self._active += 1

    def release(self, rate_limited: bool = False, success: bool = True) -> None:
        """Release a concurrency slot and update CUBIC state.

        Args:
            rate_limited: True if this request got a 429 after retries exhausted.
            success: True if the request succeeded. Ignored when rate_limited=True.
        """
        with self._condition:
            self._active -= 1

            if rate_limited:
                self._on_rate_limit()
            elif success:
                self._on_success()
            # Non-rate-limit failures: just release the slot, no window change

            self._condition.notify_all()

    def _on_rate_limit(self) -> None:
        """Multiplicative decrease on 429."""
        self._total_rate_limits += 1
        old_window = self._window
        self._w_max = self._window
        self._window = max(1.0, self._window * self._beta)
        self._epoch_start = time.monotonic()
        self._in_slow_start = False

        logger.warning(
            "CUBIC [rate-limited]: window %.0f -> %.0f (W_max=%.0f)",
            old_window, self._window, self._w_max,
        )

    def _on_success(self) -> None:
        """Increase window on success."""
        self._total_successes += 1

        if self._in_slow_start:
            # Exponential growth: +1 per success (doubles per round)
            self._window = min(self._window + 1, float(self._ceiling))
            return

        # CUBIC: W(t) = C * (t - K)^3 + W_max
        t = time.monotonic() - self._epoch_start
        K = self._compute_K()
        w_cubic = self._C * ((t - K) ** 3) + self._w_max

        self._window = max(1.0, min(w_cubic, float(self._ceiling)))

    def _compute_K(self) -> float:
        """Compute K: time to reach W_max after a decrease."""
        # K = ((W_max * (1 - beta)) / C) ^ (1/3)
        numerator = self._w_max * (1.0 - self._beta)
        if numerator <= 0:
            return 0.0
        return (numerator / self._C) ** (1.0 / 3.0)

    def get_stats(self) -> CubicStats:
        """Get current CUBIC statistics."""
        with self._lock:
            return CubicStats(
                window=max(1, int(self._window)),
                active=self._active,
                ceiling=self._ceiling,
                in_slow_start=self._in_slow_start,
                w_max=self._w_max,
                total_rate_limits=self._total_rate_limits,
                total_successes=self._total_successes,
            )

    def reset(self) -> None:
        """Reset to initial slow-start state."""
        with self._condition:
            self._window = 1.0
            self._active = 0
            self._in_slow_start = True
            self._w_max = 0.0
            self._epoch_start = 0.0
            self._total_rate_limits = 0
            self._total_successes = 0
            logger.info("CUBIC: reset to slow-start")
            self._condition.notify_all()


def is_rate_limit_error(exc: BaseException) -> bool:
    """Check if an exception indicates rate limiting (HTTP 429).

    Uses duck typing to avoid importing SDK-specific exception classes.
    Works with openai.RateLimitError, anthropic.RateLimitError, and any
    exception with status_code=429.
    """
    # Check exception class name
    if "RateLimitError" in type(exc).__name__:
        return True
    # Check for HTTP status code attribute
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    if status == 429:
        return True
    return False
