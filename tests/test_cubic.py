"""Tests for the CUBIC adaptive concurrency controller."""

import time
import threading

import pytest

from shinka.llm.cubic import CubicConcurrency, CubicStats, is_rate_limit_error


class TestCubicSlowStart:
    """Tests for the slow start phase (cold start discovery)."""

    def test_starts_at_window_1(self):
        cc = CubicConcurrency(ceiling=100)
        assert cc.window == 1

    def test_window_grows_on_success(self):
        cc = CubicConcurrency(ceiling=100)
        # Simulate 10 successes — window should be 11 (1 + 10)
        for _ in range(10):
            cc.acquire()
            cc.release(rate_limited=False, success=True)
        assert cc.window == 11

    def test_slow_start_is_exponential_with_concurrency(self):
        """With N concurrent successes, window doubles per round."""
        cc = CubicConcurrency(ceiling=1000)
        # Round 1: window=1, 1 success -> window=2
        cc.acquire()
        cc.release(success=True)
        assert cc.window == 2

        # Round 2: window=2, 2 successes -> window=4
        for _ in range(2):
            cc.acquire()
        for _ in range(2):
            cc.release(success=True)
        assert cc.window == 4

        # Round 3: window=4, 4 successes -> window=8
        for _ in range(4):
            cc.acquire()
        for _ in range(4):
            cc.release(success=True)
        assert cc.window == 8

    def test_slow_start_respects_ceiling(self):
        cc = CubicConcurrency(ceiling=5)
        for _ in range(20):
            cc.acquire()
            cc.release(success=True)
        assert cc.window == 5

    def test_starts_in_slow_start(self):
        cc = CubicConcurrency(ceiling=100)
        stats = cc.get_stats()
        assert stats.in_slow_start is True


class TestCubicRecovery:
    """Tests for CUBIC recovery after a rate limit event."""

    def test_rate_limit_exits_slow_start(self):
        cc = CubicConcurrency(ceiling=100)
        # Grow window to 10
        for _ in range(9):
            cc.acquire()
            cc.release(success=True)
        assert cc.window == 10

        # Hit rate limit
        cc.acquire()
        cc.release(rate_limited=True)

        stats = cc.get_stats()
        assert stats.in_slow_start is False

    def test_rate_limit_reduces_window(self):
        cc = CubicConcurrency(ceiling=100, beta=0.7)
        # Grow to 10 in slow start
        for _ in range(9):
            cc.acquire()
            cc.release(success=True)
        assert cc.window == 10

        # Rate limit: window should drop to 10 * 0.7 = 7
        cc.acquire()
        cc.release(rate_limited=True)
        assert cc.window == 7

    def test_w_max_set_on_rate_limit(self):
        cc = CubicConcurrency(ceiling=100)
        for _ in range(9):
            cc.acquire()
            cc.release(success=True)
        # Window is 10
        cc.acquire()
        cc.release(rate_limited=True)

        stats = cc.get_stats()
        assert stats.w_max == 10.0

    def test_cubic_recovers_toward_w_max(self):
        """After rate limit, CUBIC should recover toward W_max over time."""
        cc = CubicConcurrency(ceiling=200, beta=0.7, C=0.4)
        # Grow to 100 in slow start
        for _ in range(99):
            cc.acquire()
            cc.release(success=True)
        assert cc.window == 100

        # Rate limit: drops to 70
        cc.acquire()
        cc.release(rate_limited=True)
        post_decrease = cc.window
        assert post_decrease == 70

        # Wait and then do some successes — window should grow toward 100
        time.sleep(2.0)
        for _ in range(5):
            cc.acquire()
            cc.release(success=True)

        recovered = cc.window
        assert recovered > post_decrease, f"Window should recover: {recovered} > {post_decrease}"

    def test_cubic_recovers_to_w_max_eventually(self):
        """CUBIC should reach W_max at time K."""
        cc = CubicConcurrency(ceiling=200, beta=0.7, C=0.4)
        # Grow to 50
        for _ in range(49):
            cc.acquire()
            cc.release(success=True)

        # Rate limit
        cc.acquire()
        cc.release(rate_limited=True)

        # K = (W_max * (1-beta) / C)^(1/3) = (50 * 0.3 / 0.4)^(1/3) = (37.5)^(1/3) ≈ 3.35s
        time.sleep(3.5)
        cc.acquire()
        cc.release(success=True)

        # Should be very close to W_max (50)
        assert cc.window >= 45, f"Should be near W_max=50 after K seconds, got {cc.window}"

    def test_cubic_probes_above_w_max(self):
        """After reaching W_max, CUBIC should cautiously probe above."""
        cc = CubicConcurrency(ceiling=200, beta=0.7, C=0.4)
        for _ in range(49):
            cc.acquire()
            cc.release(success=True)

        cc.acquire()
        cc.release(rate_limited=True)

        # Wait well past K (≈3.35s) to be in the convex region
        time.sleep(5.0)
        for _ in range(5):
            cc.acquire()
            cc.release(success=True)

        assert cc.window > 50, f"Should probe above W_max=50, got {cc.window}"

    def test_multiple_rate_limits_converge(self):
        """Repeated rate limits should find the true capacity."""
        cc = CubicConcurrency(ceiling=200, beta=0.7)
        # Grow to 100
        for _ in range(99):
            cc.acquire()
            cc.release(success=True)

        # First rate limit: 100 -> 70
        cc.acquire()
        cc.release(rate_limited=True)
        assert cc.window == 70

        # Second rate limit: 70 -> 49
        cc.acquire()
        cc.release(rate_limited=True)
        assert cc.window == 49

        # Third rate limit: 49 -> 34
        cc.acquire()
        cc.release(rate_limited=True)
        assert cc.window == 34


class TestCubicCeiling:
    """Tests for the ceiling (hard maximum)."""

    def test_ceiling_enforced_in_slow_start(self):
        cc = CubicConcurrency(ceiling=10)
        for _ in range(20):
            cc.acquire()
            cc.release(success=True)
        assert cc.window == 10

    def test_ceiling_enforced_in_cubic(self):
        cc = CubicConcurrency(ceiling=50, beta=0.7, C=0.4)
        for _ in range(49):
            cc.acquire()
            cc.release(success=True)

        cc.acquire()
        cc.release(rate_limited=True)

        # Wait a long time and do many successes — should not exceed ceiling
        time.sleep(10.0)
        for _ in range(20):
            cc.acquire()
            cc.release(success=True)

        assert cc.window <= 50

    def test_auto_mode_ceiling(self):
        cc = CubicConcurrency(ceiling=10000)
        stats = cc.get_stats()
        assert stats.ceiling == 10000


class TestCubicConcurrencyControl:
    """Tests for acquire/release blocking behavior."""

    def test_acquire_blocks_at_window(self):
        cc = CubicConcurrency(ceiling=100)
        # Window is 1, acquire one slot
        cc.acquire()

        # Second acquire should block
        blocked = threading.Event()
        unblocked = threading.Event()

        def try_acquire():
            blocked.set()
            cc.acquire()
            unblocked.set()

        t = threading.Thread(target=try_acquire)
        t.start()
        blocked.wait(timeout=1.0)
        time.sleep(0.1)  # Give it time to actually block
        assert not unblocked.is_set(), "Should be blocked"

        # Release first slot — second should unblock
        cc.release(success=True)
        assert unblocked.wait(timeout=2.0), "Should have unblocked after release"
        cc.release(success=True)
        t.join(timeout=1.0)

    def test_concurrent_acquire_respects_window(self):
        """Multiple threads should respect the concurrency window."""
        cc = CubicConcurrency(ceiling=100)
        # Grow window to 5
        for _ in range(4):
            cc.acquire()
            cc.release(success=True)
        assert cc.window == 5

        # Acquire 5 slots concurrently
        acquired = threading.Event()
        barrier = threading.Barrier(5, timeout=5.0)
        errors = []

        def worker():
            try:
                cc.acquire()
                barrier.wait()  # All 5 should reach here
                acquired.set()
                cc.release(success=True)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)

        assert not errors, f"Got errors: {errors}"
        assert acquired.is_set()

    def test_window_never_below_1(self):
        cc = CubicConcurrency(ceiling=100, beta=0.1)
        # Grow then rate limit many times
        for _ in range(9):
            cc.acquire()
            cc.release(success=True)
        for _ in range(20):
            cc.acquire()
            cc.release(rate_limited=True)
        assert cc.window >= 1


class TestCubicStats:
    def test_stats_tracking(self):
        cc = CubicConcurrency(ceiling=100)
        for _ in range(5):
            cc.acquire()
            cc.release(success=True)
        cc.acquire()
        cc.release(rate_limited=True)

        stats = cc.get_stats()
        assert stats.total_successes == 5
        assert stats.total_rate_limits == 1
        assert stats.ceiling == 100
        assert stats.in_slow_start is False
        assert stats.w_max == 6.0

    def test_active_count(self):
        cc = CubicConcurrency(ceiling=100)
        # Grow window first
        for _ in range(4):
            cc.acquire()
            cc.release(success=True)

        cc.acquire()
        cc.acquire()
        assert cc.active == 2
        cc.release(success=True)
        assert cc.active == 1
        cc.release(success=True)
        assert cc.active == 0


class TestCubicReset:
    def test_reset(self):
        cc = CubicConcurrency(ceiling=100)
        for _ in range(10):
            cc.acquire()
            cc.release(success=True)
        cc.acquire()
        cc.release(rate_limited=True)

        cc.reset()
        stats = cc.get_stats()
        assert stats.window == 1
        assert stats.in_slow_start is True
        assert stats.w_max == 0.0
        assert stats.total_rate_limits == 0
        assert stats.total_successes == 0


class TestCubicNeutralRelease:
    """Non-rate-limit failures should not change the window."""

    def test_failure_does_not_change_window(self):
        cc = CubicConcurrency(ceiling=100)
        for _ in range(9):
            cc.acquire()
            cc.release(success=True)
        window_before = cc.window

        # Non-rate-limit failure
        cc.acquire()
        cc.release(rate_limited=False, success=False)
        assert cc.window == window_before


class TestCubicThreadSafety:
    def test_concurrent_operations(self):
        """Many threads acquiring and releasing concurrently."""
        cc = CubicConcurrency(ceiling=100)
        # Pre-grow window
        for _ in range(19):
            cc.acquire()
            cc.release(success=True)

        errors = []

        def worker(n):
            try:
                for _ in range(n):
                    cc.acquire()
                    time.sleep(0.001)
                    cc.release(success=True)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(20,)) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30.0)

        assert not errors
        assert cc.active == 0


class TestIsRateLimitError:
    def test_rate_limit_error_by_name(self):
        class RateLimitError(Exception):
            pass
        assert is_rate_limit_error(RateLimitError("too many"))

    def test_rate_limit_error_by_status_code(self):
        class APIError(Exception):
            def __init__(self):
                self.status_code = 429
        assert is_rate_limit_error(APIError())

    def test_rate_limit_error_by_status(self):
        class APIError(Exception):
            def __init__(self):
                self.status = 429
        assert is_rate_limit_error(APIError())

    def test_non_rate_limit_error(self):
        assert not is_rate_limit_error(ValueError("bad"))

    def test_server_error_not_rate_limit(self):
        class APIError(Exception):
            def __init__(self):
                self.status_code = 500
        assert not is_rate_limit_error(APIError())
