"""
Repair strategy unit tests.
"""

import pytest

from llm_shield.models import ValidationErrorDetail
from llm_shield.providers import MockProvider
from llm_shield.repair import ChainedRepairStrategy, LLMRepairStrategy, RegexRepairStrategy


ERRORS = [
    ValidationErrorDetail(field="age", message="not an integer", validator_name="json_schema"),
]


class TestRegexRepairStrategy:
    @pytest.mark.asyncio
    async def test_strips_markdown_fences(self):
        strategy = RegexRepairStrategy()
        result = await strategy.repair(
            original_prompt="test",
            bad_response='```json\n{"name": "Alice"}\n```',
            errors=ERRORS,
            model="groq/test",
        )
        assert result == '{"name": "Alice"}'

    @pytest.mark.asyncio
    async def test_fixes_trailing_comma(self):
        strategy = RegexRepairStrategy()
        result = await strategy.repair(
            original_prompt="test",
            bad_response='{"name": "Alice",}',
            errors=ERRORS,
            model="groq/test",
        )
        assert result == '{"name": "Alice"}'

    @pytest.mark.asyncio
    async def test_passes_through_valid_json(self):
        strategy = RegexRepairStrategy()
        result = await strategy.repair(
            original_prompt="test",
            bad_response='{"name": "Alice"}',
            errors=ERRORS,
            model="groq/test",
        )
        assert result == '{"name": "Alice"}'


class TestLLMRepairStrategy:
    @pytest.mark.asyncio
    async def test_sends_repair_prompt(self):
        provider = MockProvider(responses=['{"name": "Alice", "age": 28}'])
        strategy = LLMRepairStrategy(provider)

        result = await strategy.repair(
            original_prompt="Generate a profile",
            bad_response='{"name": "Alice", "age": "bad"}',
            errors=ERRORS,
            model="groq/test",
        )

        assert result == '{"name": "Alice", "age": 28}'
        assert len(provider.call_history) == 1
        assert "VALIDATION ERRORS" in provider.call_history[0]["prompt"]


class TestChainedRepairStrategy:
    @pytest.mark.asyncio
    async def test_regex_fix_skips_llm(self):
        """If regex fix produces valid JSON, skip the LLM call."""
        provider = MockProvider(responses=["should not be called"])
        strategy = ChainedRepairStrategy(provider)

        result = await strategy.repair(
            original_prompt="test",
            bad_response='```json\n{"name": "Alice"}\n```',
            errors=ERRORS,
            model="groq/test",
        )

        assert result == '{"name": "Alice"}'
        # LLM should NOT have been called since regex fix was sufficient
        assert len(provider.call_history) == 0

    @pytest.mark.asyncio
    async def test_falls_back_to_llm_when_regex_fails(self):
        """If regex fix doesn't produce valid JSON, try LLM."""
        provider = MockProvider(responses=['{"name": "Alice", "age": 28}'])
        strategy = ChainedRepairStrategy(provider)

        result = await strategy.repair(
            original_prompt="test",
            bad_response="completely invalid garbage",
            errors=ERRORS,
            model="groq/test",
        )

        assert result == '{"name": "Alice", "age": 28}'
        assert len(provider.call_history) == 1
