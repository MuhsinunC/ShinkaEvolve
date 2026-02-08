"""Exceptions for ShinkaEvolve evaluation scripts."""

from __future__ import annotations

import json
import logging
import os
from contextlib import contextmanager
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class ScorerFailure(Exception):
    """Raised when the scorer encounters a fatal, non-retryable error.

    Raising this exception signals ShinkaEvolve to halt evolution and
    preserve state for retry on resume.  Provide a human-readable message
    and an optional *error_type* for structured logging.

    Example::

        raise ScorerFailure("API credits exhausted",
                            error_type="CreditExhaustedError")
    """

    def __init__(self, message: str, *, error_type: str = "ScorerFailure"):
        super().__init__(message)
        self.error_type = error_type


_SCORER_FAILURE_FILE = "scorer_failure.json"


def _write_scorer_failure(results_dir: str, error_type: str, error: str):
    """Write scorer_failure.json marker.  Internal — use *scorer_failure_context*."""
    os.makedirs(results_dir, exist_ok=True)
    payload = {
        "error_type": error_type,
        "error": error,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    path = os.path.join(results_dir, _SCORER_FAILURE_FILE)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    logger.error("Scorer failure saved to %s", path)


@contextmanager
def scorer_failure_context(results_dir: str):
    """Context manager for eval scripts that may raise :class:`ScorerFailure`.

    Usage::

        def main(program_path, results_dir):
            with scorer_failure_context(results_dir):
                # ... scoring logic ...
                raise ScorerFailure("credits exhausted",
                                    error_type="CreditExhaustedError")

    On entry: cleans up any leftover ``scorer_failure.json`` from a previous
    attempt (ghost-recovery retry).

    On :class:`ScorerFailure`: writes ``scorer_failure.json`` (no
    ``metrics.json``), suppresses the exception so the script exits cleanly.

    On other exceptions: propagates normally.
    """
    marker = os.path.join(results_dir, _SCORER_FAILURE_FILE)
    if os.path.exists(marker):
        os.remove(marker)
        logger.info("Cleaned up previous %s", _SCORER_FAILURE_FILE)

    try:
        yield
    except ScorerFailure as e:
        logger.error("SCORER FAILURE: %s", e)
        _write_scorer_failure(results_dir, e.error_type, str(e))
        # Suppressed — script exits cleanly, marker file signals the runner
