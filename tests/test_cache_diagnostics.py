"""Tests for prompt caching diagnostics module."""

import pytest
from shinka.llm.cache_diagnostics import (
    estimate_tokens,
    get_cache_info,
    format_cache_diagnostics,
)
from shinka.llm.models.pricing import CLAUDE_MODELS


class TestEstimateTokens:
    def test_nonempty_text(self):
        text = "Hello world this is a test"
        tokens = estimate_tokens(text)
        assert tokens > 0

    def test_empty_text(self):
        tokens = estimate_tokens("")
        assert tokens >= 1  # Should return at least 1

    def test_long_text_scales(self):
        short = "Hello"
        long = "Hello " * 1000
        assert estimate_tokens(long) > estimate_tokens(short)

    def test_code_text(self):
        code = "def foo(x):\n    return x * 2\n\nfor i in range(10):\n    print(foo(i))"
        tokens = estimate_tokens(code)
        assert tokens > 5


class TestGetCacheInfo:
    def test_haiku_45_has_cache_info(self):
        info = get_cache_info("claude-haiku-4-5-20251001")
        assert info is not None
        assert info["cache_min_tokens"] == 4096
        assert info["cache_write_price"] > 0
        assert info["cache_read_price"] > 0

    def test_sonnet_45_has_cache_info(self):
        info = get_cache_info("claude-sonnet-4-5-20250929")
        assert info is not None
        assert info["cache_min_tokens"] == 1024

    def test_opus_46_has_cache_info(self):
        info = get_cache_info("claude-opus-4-6")
        assert info is not None
        assert info["cache_min_tokens"] == 4096

    def test_opus_41_has_cache_info(self):
        info = get_cache_info("claude-opus-4-1-20250805")
        assert info is not None
        assert info["cache_min_tokens"] == 1024

    def test_haiku_3_has_cache_info(self):
        info = get_cache_info("claude-3-haiku-20240307")
        assert info is not None
        assert info["cache_min_tokens"] == 2048

    def test_unknown_model_returns_none(self):
        info = get_cache_info("gpt-4o-mini")
        assert info is None

    def test_nonexistent_model_returns_none(self):
        info = get_cache_info("totally-fake-model")
        assert info is None

    def test_bedrock_model_has_cache_info(self):
        info = get_cache_info("bedrock/anthropic.claude-haiku-4-5-20251001-v1:0")
        assert info is not None
        assert info["cache_min_tokens"] == 4096

    def test_cache_write_price_is_125x_input(self):
        """Cache write should be 1.25x the base input price (verified from Anthropic docs)."""
        for model_name, model_data in CLAUDE_MODELS.items():
            if "cache_write_price" in model_data:
                expected = model_data["input_price"] * 1.25
                actual = model_data["cache_write_price"]
                assert abs(actual - expected) < 1e-15, (
                    f"{model_name}: cache_write_price {actual} != 1.25 * input_price {expected}"
                )

    def test_cache_read_price_is_010x_input(self):
        """Cache read should be 0.10x the base input price (verified from Anthropic docs)."""
        for model_name, model_data in CLAUDE_MODELS.items():
            if "cache_read_price" in model_data:
                expected = model_data["input_price"] * 0.10
                actual = model_data["cache_read_price"]
                assert abs(actual - expected) < 1e-15, (
                    f"{model_name}: cache_read_price {actual} != 0.10 * input_price {expected}"
                )


class TestFormatCacheDiagnostics:
    def test_anthropic_model_above_minimum(self):
        # Create a system prompt that's definitely above 1024 tokens
        long_prompt = "word " * 2000  # ~2000 words = ~2600 tokens
        report = format_cache_diagnostics(
            model_names=["claude-sonnet-4-5-20250929"],
            system_prompt=long_prompt,
        )
        assert "PROMPT CACHING DIAGNOSTICS" in report
        assert "claude-sonnet-4-5-20250929" in report
        assert "ACTIVE" in report
        assert "savings" in report.lower()

    def test_anthropic_model_below_minimum(self):
        # Very short prompt, below any model's minimum
        short_prompt = "You are a helpful assistant."
        report = format_cache_diagnostics(
            model_names=["claude-haiku-4-5-20251001"],
            system_prompt=short_prompt,
        )
        assert "BELOW MINIMUM" in report
        assert "4,096" in report  # Haiku 4.5's minimum

    def test_non_anthropic_model(self):
        report = format_cache_diagnostics(
            model_names=["gpt-4o-mini"],
            system_prompt="test prompt",
        )
        assert "non-Anthropic" in report

    def test_multiple_models(self):
        prompt = "word " * 2000
        report = format_cache_diagnostics(
            model_names=["claude-sonnet-4-5-20250929", "claude-haiku-4-5-20251001"],
            system_prompt=prompt,
        )
        assert "claude-sonnet-4-5-20250929" in report
        assert "claude-haiku-4-5-20251001" in report

    def test_empty_model_list(self):
        report = format_cache_diagnostics(
            model_names=[],
            system_prompt="test",
        )
        assert "No Anthropic models" in report

    def test_savings_positive_for_large_prompt(self):
        # With enough calls and a large prompt, savings should be positive
        prompt = "word " * 3000  # ~3900 tokens, above 1024 minimum
        report = format_cache_diagnostics(
            model_names=["claude-sonnet-4-5-20250929"],
            system_prompt=prompt,
            estimated_calls_per_generation=10,
            num_generations=100,
        )
        assert "ACTIVE" in report
        # The savings percentage should be substantial (cache reads are 0.10x)
        assert "savings" in report.lower()

    def test_custom_generation_count(self):
        prompt = "word " * 2000
        report = format_cache_diagnostics(
            model_names=["claude-sonnet-4-5-20250929"],
            system_prompt=prompt,
            estimated_calls_per_generation=5,
            num_generations=50,
        )
        assert "50 gens" in report
        assert "5 calls/gen" in report


class TestPricingIntegrity:
    """Verify that pricing.py has consistent data for all Claude models."""

    def test_all_claude_models_have_input_output_price(self):
        for name, data in CLAUDE_MODELS.items():
            assert "input_price" in data, f"{name} missing input_price"
            assert "output_price" in data, f"{name} missing output_price"
            assert data["input_price"] > 0, f"{name} input_price <= 0"
            assert data["output_price"] > 0, f"{name} output_price <= 0"

    def test_cache_fields_come_as_complete_set(self):
        """If a model has any cache field, it must have all three."""
        cache_fields = {"cache_min_tokens", "cache_write_price", "cache_read_price"}
        for name, data in CLAUDE_MODELS.items():
            has = cache_fields & set(data.keys())
            if has:
                assert has == cache_fields, (
                    f"{name} has incomplete cache fields: {has} (need all of {cache_fields})"
                )

    def test_cache_min_tokens_are_valid(self):
        valid_minimums = {1024, 2048, 4096}
        for name, data in CLAUDE_MODELS.items():
            if "cache_min_tokens" in data:
                assert data["cache_min_tokens"] in valid_minimums, (
                    f"{name} has unexpected cache_min_tokens: {data['cache_min_tokens']}"
                )
