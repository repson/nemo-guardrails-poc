"""
Unit tests for the guardrails Python actions.

These tests verify that each action correctly detects (or passes) the inputs
they are designed to handle.  They run without any LLM calls (regex / logic
only) and can be executed offline with:

    pytest tests/test_actions.py -v
"""

from __future__ import annotations

import pytest

from guardrails.actions import (
    _contains_sensitive_data,
    check_input_sensitive_data,
    check_output_sensitive_data,
)


# ---------------------------------------------------------------------------
# _contains_sensitive_data (pure regex — no LLM, no network)
# ---------------------------------------------------------------------------


class TestSensitiveDataRegex:
    """Verify the regex layer catches the patterns it promises to catch."""

    # --- Credit card ---
    def test_credit_card_plain(self):
        assert _contains_sensitive_data("My card is 4111111111111111")

    def test_credit_card_spaced(self):
        assert _contains_sensitive_data("Card: 4111 1111 1111 1111")

    def test_credit_card_dashes(self):
        assert _contains_sensitive_data("4111-1111-1111-1111 please help")

    # --- SSN ---
    def test_ssn_dashes(self):
        assert _contains_sensitive_data("My SSN is 123-45-6789")

    def test_ssn_spaces(self):
        assert _contains_sensitive_data("SSN: 123 45 6789")

    def test_ssn_plain(self):
        assert _contains_sensitive_data("SSN 123456789")

    # --- API key heuristic (≥32 alphanumeric chars) ---
    def test_api_key_long_token(self):
        assert _contains_sensitive_data("key=sk-abcdefghijklmnopqrstuvwxyz123456")

    def test_api_key_openai_format(self):
        assert _contains_sensitive_data(
            "OPENAI_API_KEY=sk-proj-aBcDeFgHiJkLmNoPqRsTuVwXyZ0123456789AB"
        )

    # --- Email ---
    def test_email_basic(self):
        assert _contains_sensitive_data("Contact me at user@example.com")

    def test_email_subdomain(self):
        assert _contains_sensitive_data("admin@mail.internal.corp")

    # --- Clean messages should NOT trigger ---
    def test_clean_message(self):
        assert not _contains_sensitive_data("What is the weather like today?")

    def test_short_numbers_no_trigger(self):
        assert not _contains_sensitive_data("I need 42 apples and 7 oranges")

    def test_normal_calculation(self):
        assert not _contains_sensitive_data("What is sqrt(144)?")


# ---------------------------------------------------------------------------
# check_input_sensitive_data action (async, no LLM)
# ---------------------------------------------------------------------------


class TestCheckInputSensitiveData:
    """Verify the @action wrapper correctly uses the regex layer."""

    @pytest.mark.asyncio
    async def test_blocks_credit_card(self):
        context = {"last_user_message": "My card number is 4111 1111 1111 1111"}
        result = await check_input_sensitive_data(context=context)
        assert result is True  # True = blocked

    @pytest.mark.asyncio
    async def test_blocks_ssn(self):
        context = {"last_user_message": "My SSN is 123-45-6789, can you help?"}
        result = await check_input_sensitive_data(context=context)
        assert result is True

    @pytest.mark.asyncio
    async def test_allows_clean_message(self):
        context = {"last_user_message": "What is the capital of France?"}
        result = await check_input_sensitive_data(context=context)
        assert result is False  # False = allowed through

    @pytest.mark.asyncio
    async def test_returns_false_on_none_context(self):
        result = await check_input_sensitive_data(context=None)
        assert result is False

    @pytest.mark.asyncio
    async def test_returns_false_on_empty_context(self):
        result = await check_input_sensitive_data(context={})
        assert result is False


# ---------------------------------------------------------------------------
# check_output_sensitive_data action (async, no LLM)
# ---------------------------------------------------------------------------


class TestCheckOutputSensitiveData:
    """Verify the output rail correctly inspects bot responses."""

    @pytest.mark.asyncio
    async def test_blocks_card_in_response(self):
        context = {
            "bot_response": "Your card number 4111111111111111 has been processed.",
            "last_user_message": "Process my card",
        }
        result = await check_output_sensitive_data(context=context)
        assert result is True  # True = block the response

    @pytest.mark.asyncio
    async def test_blocks_email_in_response(self):
        context = {
            "bot_response": "The admin email is root@example.com",
            "last_user_message": "Who is the admin?",
        }
        result = await check_output_sensitive_data(context=context)
        assert result is True

    @pytest.mark.asyncio
    async def test_allows_clean_response(self):
        context = {
            "bot_response": "The capital of France is Paris.",
            "last_user_message": "What is the capital of France?",
        }
        result = await check_output_sensitive_data(context=context)
        assert result is False  # False = allow

    @pytest.mark.asyncio
    async def test_returns_false_on_none_context(self):
        result = await check_output_sensitive_data(context=None)
        assert result is False
