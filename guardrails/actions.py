"""
Custom NeMo Guardrails actions.

Actions are Python functions decorated with @action that can be called
from Colang flows.  They extend the guardrails system with logic that
cannot be expressed in Colang alone.

Docs: https://docs.nvidia.com/nemo/guardrails/latest/configure-rails/actions/
"""

from __future__ import annotations

import os
import re
from typing import Optional

from nemoguardrails.actions import action
from openai import AsyncOpenAI

from .audit import log_event


# ---------------------------------------------------------------------------
# Patterns for detecting sensitive data
# ---------------------------------------------------------------------------

_SENSITIVE_PATTERNS = [
    # Credit card (Luhn-candidate 13-19 digit sequences, optionally separated)
    re.compile(r"\b(?:\d[ -]?){13,19}\b"),
    # US Social Security Number
    re.compile(r"\b\d{3}[- ]?\d{2}[- ]?\d{4}\b"),
    # Generic API key / secret token heuristic (long alphanumeric strings)
    re.compile(r"\b[A-Za-z0-9_\-]{32,}\b"),
    # Email address
    re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),
]


def _contains_sensitive_data(text: str) -> bool:
    return any(pattern.search(text) for pattern in _SENSITIVE_PATTERNS)


def _get_user_message(context: dict | None) -> str:
    """Extract the user message from context, handling NeMo 0.23 key names."""
    if context is None:
        return ""
    return context.get("user_message") or context.get("last_user_message") or ""


def _get_bot_response(context: dict | None) -> str:
    """Extract the bot response from context."""
    if context is None:
        return ""
    return context.get("bot_response") or ""


# ---------------------------------------------------------------------------
# Helper: secondary LLM call for self-check / hallucination
# ---------------------------------------------------------------------------


async def _llm_yes_no(prompt: str) -> bool:
    """
    Make a lightweight secondary LLM call and return True if the answer
    starts with 'yes' (case-insensitive).

    Uses gpt-4o-mini with temperature=0 for deterministic classification.
    """
    client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=5,
    )
    answer = (response.choices[0].message.content or "").strip().lower()
    return answer.startswith("yes")


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------


@action(name="check_input_sensitive_data")
async def check_input_sensitive_data(context: Optional[dict] = None) -> bool:
    """
    Returns True if the user's last message appears to contain sensitive data
    (credit card numbers, SSNs, API keys, email addresses, etc.).
    Logs the event to the audit trail when triggered.
    """
    if context is None:
        return False
    message = _get_user_message(context)
    result = _contains_sensitive_data(message)
    if result:
        log_event(
            event_type="input_blocked",
            rail="check_sensitive_data_input",
            user_input=message,
            details={"reason": "sensitive_data_detected"},
        )
    return result


@action(name="check_output_sensitive_data")
async def check_output_sensitive_data(context: Optional[dict] = None) -> bool:
    """
    Returns True if the bot's last response appears to contain sensitive data.
    Logs the event to the audit trail when triggered.
    """
    if context is None:
        return False
    message = _get_bot_response(context)
    result = _contains_sensitive_data(message)
    if result:
        log_event(
            event_type="output_blocked",
            rail="check_sensitive_data_output",
            user_input=_get_user_message(context),
            details={"reason": "sensitive_data_in_response"},
        )
    return result


@action(name="self_check_input")
async def self_check_input(context: Optional[dict] = None) -> bool:
    """
    Secondary LLM call that checks whether the user message complies with
    the policy defined in config.yml (task: self_check_input).

    Returns True (allowed) or False (blocked).
    Logs a blocked event to the audit trail.
    """
    if context is None:
        return True

    user_input = _get_user_message(context)
    prompt = context.get("self_check_input_prompt") or ""

    if not prompt:
        # Fallback: build a minimal prompt if NeMo didn't inject one
        prompt = (
            f"Does this user message comply with safety policies? "
            f'Answer only Yes or No.\nMessage: "{user_input}"'
        )

    allowed = await _llm_yes_no(prompt)

    if not allowed:
        log_event(
            event_type="input_blocked",
            rail="self_check_input",
            user_input=user_input,
            details={"reason": "self_check_llm_rejected"},
        )
    return allowed


@action(name="self_check_output")
async def self_check_output(context: Optional[dict] = None) -> bool:
    """
    Secondary LLM call that checks whether the bot response complies with
    the policy defined in config.yml (task: self_check_output).

    Returns True (allowed) or False (blocked).
    Logs a blocked event to the audit trail.
    """
    if context is None:
        return True

    bot_response = _get_bot_response(context)
    prompt = context.get("self_check_output_prompt") or ""

    if not prompt:
        prompt = (
            f"Does this assistant response comply with safety policies? "
            f'Answer only Yes or No.\nResponse: "{bot_response}"'
        )

    allowed = await _llm_yes_no(prompt)

    if not allowed:
        log_event(
            event_type="output_blocked",
            rail="self_check_output",
            user_input=_get_user_message(context),
            details={
                "reason": "self_check_llm_rejected",
                "bot_response_snippet": bot_response[:200],
            },
        )
    return allowed


@action(name="check_hallucination")
async def check_hallucination(context: Optional[dict] = None) -> bool:
    """
    Secondary LLM call that detects whether the bot response contains
    fabricated or unverifiable facts (hallucinations).

    Returns True if hallucination is detected (response should be blocked),
    False if the response appears grounded.
    Logs detected hallucinations to the audit trail.
    """
    if context is None:
        return False

    user_input = _get_user_message(context)
    bot_response = _get_bot_response(context)
    prompt = context.get("check_hallucination_prompt") or ""

    if not prompt:
        prompt = (
            f"Does the following assistant response contain fabricated or "
            f"unverifiable facts?\n"
            f'User asked: "{user_input}"\n'
            f'Assistant responded: "{bot_response}"\n'
            f"Answer only Yes or No."
        )

    has_hallucination = await _llm_yes_no(prompt)

    if has_hallucination:
        log_event(
            event_type="hallucination_detected",
            rail="check_hallucination",
            user_input=user_input,
            details={"bot_response_snippet": bot_response[:200]},
        )
    return has_hallucination


@action(name="log_guardrail_event")
async def log_guardrail_event(
    event_type: str = "unknown",
    rail: str = "unknown",
    context: Optional[dict] = None,
) -> None:
    """
    Generic action to log any guardrail event from a Colang flow.
    Can be called manually from flows that don't have a dedicated action.
    """
    user_message = _get_user_message(context or {})
    log_event(
        event_type=event_type,
        rail=rail,
        user_input=user_message,
    )


# ---------------------------------------------------------------------------
# Keyword-based intent detection actions
# ---------------------------------------------------------------------------
# These replace Colang intent matching (user xxx / bot xxx) which in
# NeMo >=0.23.0 blocks generation when intent does not match.
# Instead we use Python keyword checks via @action and if/else in Colang.

_JAILBREAK_KEYWORDS = [
    "ignore your previous instructions",
    "forget everything you were told",
    "you are now dan",
    "pretend you have no restrictions",
    "act as if you were trained differently",
    "disregard your guidelines",
    "ignore your safety limits",
    "override your safety",
    "new directive",
]

_EXCESSIVE_AGENCY_KEYWORDS = [
    "delete all records",
    "run a shell command",
    "execute this script",
    "access the file system",
    "list all files",
    "send an email to all",
    "make an http request",
    "loop through all entries",
    "perform bulk operations",
    "export them",
    "rm -rf",
    "mass email",
    "access the server",
]

_HARMFUL_CONTENT_KEYWORDS = [
    "how do i make a bomb",
    "how can i hack into",
    "how to synthesize drugs",
    "how to hurt someone",
    "tell me how to commit",
]


@action(name="detect_jailbreak")
async def detect_jailbreak(context: Optional[dict] = None) -> bool:
    if context is None:
        return False
    message = _get_user_message(context).lower()
    return any(kw in message for kw in _JAILBREAK_KEYWORDS)


@action(name="detect_excessive_agency")
async def detect_excessive_agency(context: Optional[dict] = None) -> bool:
    if context is None:
        return False
    message = _get_user_message(context).lower()
    return any(kw in message for kw in _EXCESSIVE_AGENCY_KEYWORDS)


@action(name="detect_harmful_content")
async def detect_harmful_content(context: Optional[dict] = None) -> bool:
    if context is None:
        return False
    message = _get_user_message(context).lower()
    return any(kw in message for kw in _HARMFUL_CONTENT_KEYWORDS)
