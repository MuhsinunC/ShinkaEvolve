"""Circuit breaker for API backpressure under high concurrency.

When an API endpoint starts failing (consecutive failures exceed a threshold),
the circuit breaker pauses all requests for a cooldown period, then probes
with a few requests before fully resuming. This prevents the thundering herd
problem where hundreds of concurrent retries make congestion worse.

States:
    CLOSED    - Normal operation. Failures are counted.
    OPEN      - Endpoint is overwhelmed. Callers wait on a Condition until
                recovery_timeout expires.
    HALF_OPEN - After cooldown, allow a limited number of probe requests
                through. Other callers wait until probes confirm recovery.

Thread-safe — designed to be shared across all callers hitting the same endpoint.

Usage:
    breaker = CircuitBreaker("my-api")

    breaker.before_request()   # blocks if circuit is open
    try:
        result = api_call()
        breaker.record_success()
    except SomeAPIError:
        breaker.record_failure()
        raise
"""

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# States
CLOSED = "closed"
OPEN = "open"
HALF_OPEN = "half_open"


@dataclass
class CircuitBreakerStats:
    """Observable statistics for monitoring."""
    total_failures: int = 0
    total_trips: int = 0
    total_sleeps: int = 0
    total_sleep_seconds: float = 0.0
    current_state: str = CLOSED
    current_consecutive_failures: int = 0


class CircuitBreaker:
    """Thread-safe circuit breaker with exponential recovery timeout.

    Uses a Condition variable to avoid the thundering herd problem: when
    the circuit opens, waiting threads block on Condition.wait(timeout)
    rather than time.sleep(). Only a limited number of probe requests
    pass through during HALF_OPEN state.

    Args:
        name: Human-readable identifier for logging.
        failure_threshold: Consecutive failures before opening the circuit.
        recovery_timeout: Initial cooldown seconds when circuit opens.
        max_recovery_timeout: Cap on recovery timeout after repeated trips.
        half_open_successes: Successes needed in half-open to close circuit.
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        max_recovery_timeout: float = 120.0,
        half_open_successes: int = 3,
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.base_recovery_timeout = recovery_timeout
        self.max_recovery_timeout = max_recovery_timeout
        self.half_open_success_threshold = half_open_successes

        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._state = CLOSED
        self._consecutive_failures = 0
        self._opened_at: float = 0.0
        self._current_recovery_timeout = recovery_timeout
        self._half_open_successes = 0
        self._half_open_permits = 0  # Probe slots remaining in HALF_OPEN

        # Stats
        self._total_failures = 0
        self._total_trips = 0
        self._total_sleeps = 0
        self._total_sleep_seconds = 0.0

    @property
    def state(self) -> str:
        with self._lock:
            return self._state

    @property
    def is_open(self) -> bool:
        with self._lock:
            return self._state == OPEN

    def get_stats(self) -> CircuitBreakerStats:
        with self._lock:
            return CircuitBreakerStats(
                total_failures=self._total_failures,
                total_trips=self._total_trips,
                total_sleeps=self._total_sleeps,
                total_sleep_seconds=self._total_sleep_seconds,
                current_state=self._state,
                current_consecutive_failures=self._consecutive_failures,
            )

    def before_request(self) -> None:
        """Called before making an API request. Blocks if circuit is open.

        Uses Condition.wait() to avoid thundering herd:
        - OPEN: threads wait until recovery_timeout expires, then one
          transitions to HALF_OPEN and proceeds as the first probe.
        - HALF_OPEN: only a limited number of probe requests pass through;
          remaining threads wait for the circuit to close or re-open.
        - CLOSED: returns immediately.
        """
        with self._condition:
            while True:
                if self._state == CLOSED:
                    return

                if self._state == HALF_OPEN:
                    if self._half_open_permits > 0:
                        self._half_open_permits -= 1
                        return  # Allowed through as a probe
                    # No probe permits left — wait for state change
                    self._condition.wait()
                    continue

                # OPEN state — check if cooldown has expired
                elapsed = time.monotonic() - self._opened_at
                if elapsed >= self._current_recovery_timeout:
                    # Cooldown expired — this thread transitions to HALF_OPEN
                    self._state = HALF_OPEN
                    self._half_open_successes = 0
                    # Allow (threshold - 1) more probes; this thread is the first
                    self._half_open_permits = self.half_open_success_threshold - 1
                    logger.info(
                        "Circuit breaker [%s] -> HALF_OPEN after cooldown",
                        self.name,
                    )
                    return

                # Cooldown not yet expired — wait with timeout
                wait_time = self._current_recovery_timeout - elapsed
                self._total_sleeps += 1
                self._total_sleep_seconds += wait_time
                logger.warning(
                    "Circuit breaker [%s] OPEN — waiting %.1fs",
                    self.name, wait_time,
                )
                # Releases lock, blocks until notified or timeout
                self._condition.wait(timeout=wait_time)
                # Loop re-checks state after waking

    def record_success(self) -> None:
        """Record a successful API call."""
        with self._condition:
            if self._state == HALF_OPEN:
                self._half_open_successes += 1
                if self._half_open_successes >= self.half_open_success_threshold:
                    self._state = CLOSED
                    self._consecutive_failures = 0
                    self._current_recovery_timeout = self.base_recovery_timeout
                    logger.info(
                        "Circuit breaker [%s] -> CLOSED (recovered)",
                        self.name,
                    )
                    # Wake all waiting threads — circuit is closed
                    self._condition.notify_all()
            else:
                self._consecutive_failures = 0

    def record_failure(self) -> None:
        """Record a failed API call."""
        with self._condition:
            self._consecutive_failures += 1
            self._total_failures += 1

            if self._state == HALF_OPEN:
                # Probe failed — re-open with longer timeout
                self._current_recovery_timeout = min(
                    self._current_recovery_timeout * 2,
                    self.max_recovery_timeout,
                )
                self._state = OPEN
                self._opened_at = time.monotonic()
                self._total_trips += 1
                logger.warning(
                    "Circuit breaker [%s] -> OPEN (probe failed, cooldown %.0fs)",
                    self.name, self._current_recovery_timeout,
                )
                # Wake waiters so they re-evaluate with the new OPEN timeout
                self._condition.notify_all()
            elif (
                self._state == CLOSED
                and self._consecutive_failures >= self.failure_threshold
            ):
                self._state = OPEN
                self._opened_at = time.monotonic()
                self._total_trips += 1
                logger.warning(
                    "Circuit breaker [%s] -> OPEN (%d consecutive failures, cooldown %.0fs)",
                    self.name,
                    self._consecutive_failures,
                    self._current_recovery_timeout,
                )
                # Wake waiters so they see the new OPEN state
                self._condition.notify_all()

    def reset(self) -> None:
        """Manually reset the circuit breaker to CLOSED state."""
        with self._condition:
            self._state = CLOSED
            self._consecutive_failures = 0
            self._current_recovery_timeout = self.base_recovery_timeout
            self._half_open_successes = 0
            self._half_open_permits = 0
            logger.info("Circuit breaker [%s] manually reset to CLOSED", self.name)
            self._condition.notify_all()


class EvalCircuitBreaker:
    """Adaptive concurrency control for evaluation jobs.

    Tracks recent eval job outcomes and reduces the effective concurrency
    when failure rate exceeds a threshold. Gradually ramps back up after
    sustained success.

    This is different from the API-level CircuitBreaker: it doesn't block
    individual requests. Instead, it adjusts how many concurrent eval jobs
    the scheduler should launch.

    Args:
        max_concurrent: The configured maximum concurrent evals.
        window_size: Number of recent jobs to consider for failure rate.
        failure_rate_threshold: Fraction of failures to trigger reduction.
        min_concurrent: Floor for the effective concurrency.
        ramp_up_interval: Seconds between ramp-up attempts after recovery.
    """

    def __init__(
        self,
        max_concurrent: int,
        window_size: int = 20,
        failure_rate_threshold: float = 0.5,
        min_concurrent: int = 1,
        ramp_up_interval: float = 60.0,
    ):
        self.max_concurrent = max_concurrent
        self.window_size = window_size
        self.failure_rate_threshold = failure_rate_threshold
        self.min_concurrent = max(1, min_concurrent)
        self.ramp_up_interval = ramp_up_interval

        self._lock = threading.Lock()
        self._outcomes: deque[bool] = deque(maxlen=window_size)
        self._effective_concurrent = max_concurrent
        self._last_ramp_up = 0.0
        self._last_reduction_time = 0.0

    @property
    def effective_concurrent(self) -> int:
        """Current effective concurrency level."""
        with self._lock:
            return self._effective_concurrent

    def record_outcome(self, success: bool) -> None:
        """Record an eval job outcome and possibly adjust concurrency."""
        with self._lock:
            self._outcomes.append(success)

            if len(self._outcomes) < 3:
                return  # Not enough data

            failure_rate = sum(1 for o in self._outcomes if not o) / len(self._outcomes)

            if failure_rate >= self.failure_rate_threshold:
                # Reduce concurrency
                new_concurrent = max(
                    self.min_concurrent,
                    self._effective_concurrent // 2,
                )
                if new_concurrent < self._effective_concurrent:
                    logger.warning(
                        "EvalCircuitBreaker: failure rate %.0f%% >= %.0f%% — "
                        "reducing concurrency %d -> %d",
                        failure_rate * 100,
                        self.failure_rate_threshold * 100,
                        self._effective_concurrent,
                        new_concurrent,
                    )
                    self._effective_concurrent = new_concurrent
                    self._last_reduction_time = time.monotonic()
                    # Clear outcomes after reduction to avoid immediate re-trigger
                    self._outcomes.clear()
            else:
                # Consider ramping up if we've been stable
                now = time.monotonic()
                if (
                    self._effective_concurrent < self.max_concurrent
                    and now - self._last_ramp_up >= self.ramp_up_interval
                    and now - self._last_reduction_time >= self.ramp_up_interval
                    and failure_rate < self.failure_rate_threshold / 2  # Only ramp up when well below threshold
                ):
                    new_concurrent = min(
                        self.max_concurrent,
                        self._effective_concurrent + max(1, self._effective_concurrent // 4),
                    )
                    logger.info(
                        "EvalCircuitBreaker: failure rate %.0f%% — "
                        "ramping up concurrency %d -> %d",
                        failure_rate * 100,
                        self._effective_concurrent,
                        new_concurrent,
                    )
                    self._effective_concurrent = new_concurrent
                    self._last_ramp_up = now

    def get_stats(self) -> dict:
        with self._lock:
            outcomes = list(self._outcomes)
            return {
                "effective_concurrent": self._effective_concurrent,
                "max_concurrent": self.max_concurrent,
                "window_size": len(outcomes),
                "failure_rate": (
                    sum(1 for o in outcomes if not o) / len(outcomes)
                    if outcomes else 0.0
                ),
            }

    def reset(self) -> None:
        """Reset to maximum concurrency."""
        with self._lock:
            self._effective_concurrent = self.max_concurrent
            self._outcomes.clear()
            logger.info(
                "EvalCircuitBreaker: reset to max_concurrent=%d",
                self.max_concurrent,
            )
