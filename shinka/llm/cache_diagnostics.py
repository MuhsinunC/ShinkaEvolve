"""Prompt caching diagnostics for Anthropic models.

Prints a one-time startup report showing whether the system prompt is large
enough to benefit from Anthropic's prompt caching, along with cost analysis.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from .models.pricing import CLAUDE_MODELS, BEDROCK_MODELS

logger = logging.getLogger(__name__)

# Merge Claude + Bedrock models for lookup (Bedrock entries reference the same dicts)
_ALL_CLAUDE_MODELS: dict[str, dict] = {**CLAUDE_MODELS, **BEDROCK_MODELS}


def estimate_tokens(text: str) -> int:
    """Estimate token count for a text string.

    Uses a word-based heuristic (~1.3 tokens per word for English/code mix).
    The Anthropic SDK only supports server-side token counting via
    messages.count_tokens(), which requires an API call — too heavy for
    a synchronous startup diagnostic.
    """
    words = text.split()
    return max(1, int(len(words) * 1.3))


def get_cache_info(model_name: str) -> Optional[dict]:
    """Look up cache configuration for a model.

    Returns a dict with cache_min_tokens, cache_write_price, cache_read_price
    if the model supports caching, or None if it doesn't.
    """
    model_data = _ALL_CLAUDE_MODELS.get(model_name)
    if model_data is None:
        return None
    if "cache_min_tokens" not in model_data:
        return None
    return {
        "cache_min_tokens": model_data["cache_min_tokens"],
        "cache_write_price": model_data["cache_write_price"],
        "cache_read_price": model_data["cache_read_price"],
        "input_price": model_data["input_price"],
        "output_price": model_data["output_price"],
    }


def format_cache_diagnostics(
    model_names: List[str],
    system_prompt: str,
    estimated_calls_per_generation: int = 10,
    num_generations: int = 100,
) -> str:
    """Build the cache diagnostics report string.

    Args:
        model_names: List of model names being used for evolution.
        system_prompt: The system prompt text that gets cached.
        estimated_calls_per_generation: Approximate LLM calls per generation.
        num_generations: Total planned generations.

    Returns:
        Formatted diagnostics string, or empty string if no models support caching.
    """
    token_count = estimate_tokens(system_prompt)
    total_calls = estimated_calls_per_generation * num_generations

    lines: list[str] = []
    lines.append("")
    lines.append("=" * 70)
    lines.append("  PROMPT CACHING DIAGNOSTICS")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"  System prompt tokens (estimated): {token_count:,}")
    lines.append(f"  Planned API calls: ~{total_calls:,} ({num_generations} gens x {estimated_calls_per_generation} calls/gen)")
    lines.append("")

    any_cache_model = False

    for model_name in model_names:
        cache_info = get_cache_info(model_name)

        if cache_info is None:
            # Check if it's a known non-Anthropic model
            if model_name in _ALL_CLAUDE_MODELS:
                lines.append(f"  [{model_name}]")
                lines.append(f"    Caching: NOT SUPPORTED for this model")
                lines.append("")
            else:
                lines.append(f"  [{model_name}]")
                lines.append(f"    Caching: N/A (non-Anthropic model)")
                lines.append("")
            continue

        any_cache_model = True
        min_tokens = cache_info["cache_min_tokens"]
        input_price = cache_info["input_price"]
        write_price = cache_info["cache_write_price"]
        read_price = cache_info["cache_read_price"]

        meets_minimum = token_count >= min_tokens
        shortfall = max(0, min_tokens - token_count)

        lines.append(f"  [{model_name}]")
        lines.append(f"    Cache minimum: {min_tokens:,} tokens")

        if meets_minimum:
            lines.append(f"    Status: ACTIVE (system prompt meets minimum)")

            # Cost analysis: per-generation with caching vs without
            # First call per cache window: cache write (1.25x)
            # Subsequent calls: cache read (0.10x)
            # Assume ~1 write + (calls_per_gen - 1) reads per generation
            # (cache TTL is 5 min, calls within a generation are fast enough)
            cost_no_cache = total_calls * token_count * input_price
            # First call writes, rest read from cache
            writes_total = num_generations  # ~1 write per generation
            reads_total = total_calls - writes_total
            cost_with_cache = (
                writes_total * token_count * write_price
                + reads_total * token_count * read_price
            )
            savings = cost_no_cache - cost_with_cache
            savings_pct = (savings / cost_no_cache * 100) if cost_no_cache > 0 else 0

            lines.append(f"    Cost without caching: ${cost_no_cache:.4f}")
            lines.append(f"    Cost with caching:    ${cost_with_cache:.4f}")
            lines.append(f"    Estimated savings:    ${savings:.4f} ({savings_pct:.1f}%)")
        else:
            lines.append(f"    Status: BELOW MINIMUM ({token_count:,} < {min_tokens:,}, need {shortfall:,} more tokens)")

            # Cost analysis: what if we padded to meet the minimum?
            cost_current = total_calls * token_count * input_price
            padded_tokens = min_tokens
            writes_total = num_generations
            reads_total = total_calls - writes_total
            cost_padded_cache = (
                writes_total * padded_tokens * write_price
                + reads_total * padded_tokens * read_price
            )
            padded_savings = cost_current - cost_padded_cache
            would_save = padded_savings > 0

            lines.append(f"    Cost at current size (no cache): ${cost_current:.4f}")
            lines.append(f"    Cost if padded to {min_tokens:,} tokens:  ${cost_padded_cache:.4f}")
            if would_save:
                lines.append(f"    Padding would save: ${padded_savings:.4f}")
                lines.append(f"    Recommendation: PAD system prompt to {min_tokens:,} tokens to enable caching")
            else:
                lines.append(f"    Padding would cost MORE: ${-padded_savings:.4f} extra")
                lines.append(f"    Recommendation: DO NOT pad - caching not cost-effective at this scale")

        lines.append("")

    if not any_cache_model:
        lines.append("  No Anthropic models in use - prompt caching analysis skipped.")
        lines.append("")

    lines.append("=" * 70)
    lines.append("")

    return "\n".join(lines)


def print_cache_diagnostics(
    model_names: List[str],
    system_prompt: str,
    estimated_calls_per_generation: int = 10,
    num_generations: int = 100,
) -> None:
    """Print prompt caching diagnostics to stdout and log file.

    Called once at the start of an evolution run.
    """
    report = format_cache_diagnostics(
        model_names=model_names,
        system_prompt=system_prompt,
        estimated_calls_per_generation=estimated_calls_per_generation,
        num_generations=num_generations,
    )
    if report.strip():
        print(report)
        logger.info(report)
