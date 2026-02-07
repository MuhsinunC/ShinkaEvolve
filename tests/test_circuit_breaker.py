"""Tests for the circuit breaker module."""

import time
import threading

import pytest

from shinka.llm.circuit_breaker import (
    CircuitBreaker,
    EvalCircuitBreaker,
    CLOSED,
    OPEN,
    HALF_OPEN,
)


class TestCircuitBreaker:
    def test_starts_closed(self):
        cb = CircuitBreaker("test")
        assert cb.state == CLOSED
        assert not cb.is_open

    def test_stays_closed_below_threshold(self):
        cb = CircuitBreaker("test", failure_threshold=5)
        for _ in range(4):
            cb.record_failure()
        assert cb.state == CLOSED

    def test_opens_at_threshold(self):
        cb = CircuitBreaker("test", failure_threshold=5)
        for _ in range(5):
            cb.record_failure()
        assert cb.state == OPEN
        assert cb.is_open

    def test_success_resets_failure_count(self):
        cb = CircuitBreaker("test", failure_threshold=5)
        for _ in range(4):
            cb.record_failure()
        cb.record_success()
        # After success, counter resets — 4 more failures shouldn't trip it
        for _ in range(4):
            cb.record_failure()
        assert cb.state == CLOSED

    def test_half_open_after_timeout(self):
        cb = CircuitBreaker("test", failure_threshold=2, recovery_timeout=0.01)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == OPEN
        time.sleep(0.02)
        cb.before_request()
        assert cb.state == HALF_OPEN

    def test_closes_after_half_open_successes(self):
        cb = CircuitBreaker(
            "test", failure_threshold=2, recovery_timeout=0.01, half_open_successes=3
        )
        cb.record_failure()
        cb.record_failure()
        time.sleep(0.02)
        cb.before_request()  # transitions to half-open
        cb.record_success()
        cb.record_success()
        cb.record_success()
        assert cb.state == CLOSED

    def test_reopens_on_half_open_failure(self):
        cb = CircuitBreaker("test", failure_threshold=2, recovery_timeout=0.01)
        cb.record_failure()
        cb.record_failure()
        time.sleep(0.02)
        cb.before_request()  # transitions to half-open
        cb.record_failure()
        assert cb.state == OPEN
        # Recovery timeout should have doubled
        assert cb._current_recovery_timeout == 0.02

    def test_max_recovery_timeout_cap(self):
        cb = CircuitBreaker(
            "test", failure_threshold=1, recovery_timeout=0.01, max_recovery_timeout=0.03
        )
        # Trip 1
        cb.record_failure()
        time.sleep(0.02)
        cb.before_request()
        cb.record_failure()  # re-open, timeout doubles to 0.02
        # Trip 2
        time.sleep(0.03)
        cb.before_request()
        cb.record_failure()  # re-open, timeout would be 0.04 but capped at 0.03
        assert cb._current_recovery_timeout == 0.03

    def test_before_request_blocks_when_open(self):
        cb = CircuitBreaker("test", failure_threshold=1, recovery_timeout=0.05)
        cb.record_failure()
        assert cb.state == OPEN

        start = time.monotonic()
        cb.before_request()  # Should block for ~0.05s
        elapsed = time.monotonic() - start

        assert elapsed >= 0.04  # Allow small tolerance
        assert cb.state == HALF_OPEN

    def test_before_request_no_block_when_closed(self):
        cb = CircuitBreaker("test")
        start = time.monotonic()
        cb.before_request()
        elapsed = time.monotonic() - start
        assert elapsed < 0.01  # Should be instant

    def test_reset(self):
        cb = CircuitBreaker("test", failure_threshold=2)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == OPEN
        cb.reset()
        assert cb.state == CLOSED

    def test_stats(self):
        cb = CircuitBreaker("test", failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        cb.record_failure()
        stats = cb.get_stats()
        assert stats.total_failures == 3
        assert stats.total_trips == 1
        assert stats.current_state == OPEN

    def test_thread_safety(self):
        """Multiple threads recording failures/successes concurrently."""
        cb = CircuitBreaker("test", failure_threshold=100)
        errors = []

        def record_many(is_failure: bool, count: int):
            try:
                for _ in range(count):
                    if is_failure:
                        cb.record_failure()
                    else:
                        cb.record_success()
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=record_many, args=(True, 50)),
            threading.Thread(target=record_many, args=(False, 50)),
            threading.Thread(target=record_many, args=(True, 50)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        stats = cb.get_stats()
        assert stats.total_failures == 100

    def test_half_open_limits_probe_count(self):
        """Only half_open_successes threads should pass through during HALF_OPEN."""
        cb = CircuitBreaker(
            "test", failure_threshold=1, recovery_timeout=0.01, half_open_successes=2
        )
        cb.record_failure()
        time.sleep(0.02)

        # First call transitions to HALF_OPEN and passes through (probe 1)
        cb.before_request()
        assert cb.state == HALF_OPEN

        # Second call should also pass (probe 2, from the permit)
        passed = [False]

        def second_probe():
            cb.before_request()
            passed[0] = True

        t = threading.Thread(target=second_probe)
        t.start()
        t.join(timeout=0.5)
        assert passed[0], "Second probe should pass through"

        # Third call should block (no permits left) — verify by timeout
        blocked = [True]

        def third_probe():
            cb.before_request()
            blocked[0] = False

        t = threading.Thread(target=third_probe)
        t.start()
        t.join(timeout=0.2)
        assert blocked[0], "Third probe should be blocked (no permits)"

        # Clean up: close the circuit so the blocked thread can proceed
        cb.record_success()
        cb.record_success()
        assert cb.state == CLOSED
        t.join(timeout=1.0)
        assert not blocked[0], "Blocked thread should have been released after CLOSED"

    def test_condition_wakes_waiters_on_close(self):
        """Threads waiting during OPEN should unblock when circuit closes."""
        cb = CircuitBreaker(
            "test", failure_threshold=1, recovery_timeout=0.01, half_open_successes=1
        )
        cb.record_failure()
        time.sleep(0.02)

        # One probe goes through
        cb.before_request()
        assert cb.state == HALF_OPEN

        # Another thread waits (no permits)
        unblocked = threading.Event()

        def waiter():
            cb.before_request()
            unblocked.set()

        t = threading.Thread(target=waiter)
        t.start()
        time.sleep(0.05)  # Give it time to block

        # Probe succeeds — circuit closes, waiter should be notified
        cb.record_success()
        assert cb.state == CLOSED
        assert unblocked.wait(timeout=1.0), "Waiter should have been unblocked"
        t.join(timeout=1.0)

    def test_condition_wakes_waiters_on_reopen(self):
        """Threads waiting during HALF_OPEN should re-evaluate when probe fails."""
        cb = CircuitBreaker(
            "test", failure_threshold=1, recovery_timeout=0.01, half_open_successes=1
        )
        cb.record_failure()
        time.sleep(0.02)

        cb.before_request()  # Transitions to HALF_OPEN
        assert cb.state == HALF_OPEN

        # Another thread tries — no permits, blocks
        started = threading.Event()
        finished = threading.Event()

        def waiter():
            started.set()
            cb.before_request()
            finished.set()

        t = threading.Thread(target=waiter)
        t.start()
        started.wait(timeout=1.0)
        time.sleep(0.05)  # Let it block

        # Probe fails — re-opens with 0.02s timeout
        cb.record_failure()
        assert cb.state == OPEN

        # The waiter should eventually proceed after the new recovery timeout
        # (0.02s). Give it some time.
        assert finished.wait(timeout=1.0), "Waiter should eventually proceed after re-open"
        t.join(timeout=1.0)


class TestEvalCircuitBreaker:
    def test_starts_at_max_concurrent(self):
        ecb = EvalCircuitBreaker(max_concurrent=100)
        assert ecb.effective_concurrent == 100

    def test_no_change_with_all_successes(self):
        ecb = EvalCircuitBreaker(max_concurrent=100, window_size=10)
        for _ in range(10):
            ecb.record_outcome(True)
        assert ecb.effective_concurrent == 100

    def test_reduces_on_high_failure_rate(self):
        ecb = EvalCircuitBreaker(
            max_concurrent=100, window_size=10, failure_rate_threshold=0.5
        )
        # 6/10 failures = 60% > 50% threshold
        for _ in range(4):
            ecb.record_outcome(True)
        for _ in range(6):
            ecb.record_outcome(False)
        assert ecb.effective_concurrent == 50  # Halved

    def test_min_concurrent_floor(self):
        ecb = EvalCircuitBreaker(
            max_concurrent=4, window_size=5, failure_rate_threshold=0.5, min_concurrent=2
        )
        # Trigger multiple reductions
        for _ in range(5):
            ecb.record_outcome(False)
        assert ecb.effective_concurrent >= 2

    def test_ramps_up_after_sustained_success(self):
        ecb = EvalCircuitBreaker(
            max_concurrent=100,
            window_size=10,
            failure_rate_threshold=0.5,
            ramp_up_interval=0.01,  # Very short for testing
        )
        # First, trigger a reduction
        for _ in range(10):
            ecb.record_outcome(False)
        reduced = ecb.effective_concurrent
        assert reduced < 100

        # Now succeed consistently
        time.sleep(0.02)
        for _ in range(10):
            ecb.record_outcome(True)
        assert ecb.effective_concurrent > reduced

    def test_no_reduction_below_threshold(self):
        ecb = EvalCircuitBreaker(
            max_concurrent=100, window_size=10, failure_rate_threshold=0.5
        )
        # 4/10 failures = 40% < 50% threshold
        for _ in range(6):
            ecb.record_outcome(True)
        for _ in range(4):
            ecb.record_outcome(False)
        assert ecb.effective_concurrent == 100

    def test_reset(self):
        ecb = EvalCircuitBreaker(max_concurrent=100, window_size=5)
        for _ in range(5):
            ecb.record_outcome(False)
        assert ecb.effective_concurrent < 100
        ecb.reset()
        assert ecb.effective_concurrent == 100

    def test_stats(self):
        ecb = EvalCircuitBreaker(max_concurrent=100, window_size=10)
        for _ in range(7):
            ecb.record_outcome(True)
        for _ in range(3):
            ecb.record_outcome(False)
        stats = ecb.get_stats()
        assert stats["effective_concurrent"] == 100
        assert stats["max_concurrent"] == 100
        assert stats["window_size"] == 10
        assert abs(stats["failure_rate"] - 0.3) < 0.01

    def test_not_enough_data(self):
        ecb = EvalCircuitBreaker(max_concurrent=100, window_size=10, failure_rate_threshold=0.5)
        # Only 2 outcomes — not enough to trigger
        ecb.record_outcome(False)
        ecb.record_outcome(False)
        assert ecb.effective_concurrent == 100  # No change with < 3 data points

    def test_cascading_reductions(self):
        """Repeated high failure rates halve concurrency multiple times down to floor."""
        ecb = EvalCircuitBreaker(
            max_concurrent=100, window_size=5, failure_rate_threshold=0.5, min_concurrent=5
        )
        # Sustained failures should cause multiple reductions.
        # Each reduction clears the window, so 3 new failures trigger the next.
        for _ in range(30):
            ecb.record_outcome(False)

        # Should have reduced multiple times but never below floor
        assert ecb.effective_concurrent >= 5
        assert ecb.effective_concurrent < 20  # Well below starting value

    def test_deque_maxlen_enforced(self):
        """Outcomes deque should never exceed window_size."""
        ecb = EvalCircuitBreaker(max_concurrent=100, window_size=5)
        for _ in range(20):
            ecb.record_outcome(True)
        assert len(ecb._outcomes) == 5

    def test_thread_safety(self):
        """Multiple threads recording outcomes concurrently."""
        ecb = EvalCircuitBreaker(max_concurrent=100, window_size=50, failure_rate_threshold=0.9)
        errors = []

        def record_many(success: bool, count: int):
            try:
                for _ in range(count):
                    ecb.record_outcome(success)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=record_many, args=(True, 25)),
            threading.Thread(target=record_many, args=(False, 25)),
            threading.Thread(target=record_many, args=(True, 25)),
            threading.Thread(target=record_many, args=(False, 25)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        # Should not have crashed and effective_concurrent should be valid
        assert ecb.effective_concurrent >= 1
