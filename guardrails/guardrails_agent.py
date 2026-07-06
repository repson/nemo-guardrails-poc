"""
Guarded agent: wraps the base Agent with NeMo Guardrails.

NeMo Guardrails sits in front of the agent as a middleware layer:

  User input
      │
      ▼
  ┌─────────────────────────┐
  │  NeMo Guardrails        │  ← input rails (jailbreak, sensitive data)
  │  (LLMRails)             │
  │         │               │
  │         ▼               │
  │    LLM generation       │  ← NodeMo generates the response using its
  │         │               │     own configured model (config.yml)
  │         ▼               │
  │  output rails           │  ← sensitive data, off-topic
  └─────────────────────────┘
      │
      ▼
  Final response

Note: The base ``Agent`` class is currently instantiated only to keep
conversation history in sync with the unprotected REPL (e.g. for the
``/reset`` command).  NeMo generates responses with its own configured
LLM; the tools exposed by ``agent.tools`` are NOT available in the
protected path yet (planned: register them as NeMo ``@action`` calls).
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from nemoguardrails import LLMRails, RailsConfig

from agent.agent import Agent

load_dotenv()

# Path to the guardrails config directory (contains config.yml and *.co files)
_CONFIG_DIR = Path(__file__).parent / "config"


class GuardedAgent:
    """
    Conversational agent protected by NeMo Guardrails.

    NeMo Guardrails handles input and output rails; the underlying
    Agent class handles tool orchestration and LLM communication.
    """

    def __init__(self) -> None:
        # Load the Colang + YAML configuration
        config = RailsConfig.from_path(str(_CONFIG_DIR))
        self._rails = LLMRails(config)

        # Register custom Python actions so Colang flows can call them
        from guardrails import actions  # noqa: F401 — registers @action decorators

        self._rails.register_action(actions.check_input_sensitive_data)
        self._rails.register_action(actions.check_output_sensitive_data)
        self._rails.register_action(actions.self_check_input)
        self._rails.register_action(actions.self_check_output)
        self._rails.register_action(actions.check_hallucination)
        self._rails.register_action(actions.log_guardrail_event)
        self._rails.register_action(actions.detect_jailbreak)
        self._rails.register_action(actions.detect_excessive_agency)
        self._rails.register_action(actions.detect_harmful_content)

        # Kept for parity with Agent.reset() in the REPL; NeMo generates
        # responses with its own LLM, so this instance is not used in chat().
        self._agent = Agent()

    # ------------------------------------------------------------------
    # Public API  (mirrors Agent)
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear base agent conversation history."""
        self._agent.reset()

    def chat(self, user_message: str) -> str:
        """
        Send *user_message* through the full guardrails pipeline and
        return the final (safe) response.

        Flow:
          1. NeMo Guardrails evaluates input rails (Colang flows).
          2. NeMo calls its configured LLM (gpt-4o-mini) to generate
             a response.
          3. NeMo Guardrails evaluates output rails on the response.
          4. The (possibly blocked) response is returned.
        """
        messages = [{"role": "user", "content": user_message}]
        response = self._rails.generate(messages=messages)

        if isinstance(response, dict):
            return response.get("content", "")
        return str(response)