"""LLM-backed operator-note interpretation.

The rest of the application depends only on the :class:`NoteInterpreter`
interface, so a provider can be swapped (or mocked in tests) without touching
the guardrails, optimizer or service layer.

The shipped implementation targets any OpenAI-compatible ``/chat/completions``
endpoint — OpenAI, Groq, Together, OpenRouter, Fireworks, Google Gemini, or a
local vLLM or Ollama server — selected purely through environment variables.
"""

from __future__ import annotations

import abc
import json
import logging
import re
import time
from typing import Any, Dict, List, Optional, Sequence

import httpx

from app.config import Settings, get_settings
from app.exceptions import LLMInterpretationError, LLMUnavailableError
from app.llm.prompts import (
    build_repair_prompt,
    build_response_schema,
    build_system_prompt,
    build_user_prompt,
)

logger = logging.getLogger(__name__)

_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)

# Providers that do not reliably support json_schema response_format.
# For these we send json_object directly — one request per case instead of
# two (schema attempt + fallback), which halves rate-limit pressure.
_SIMPLE_JSON_HOSTS = (
    "groq.com",
    "generativelanguage.googleapis.com",
    "openrouter.ai",
    "together.xyz",
)

# Transient HTTP statuses worth retrying with exponential backoff.
_RETRY_STATUSES = {408, 425, 429, 500, 502, 503, 504}
_RETRY_DELAYS = (1.0, 2.0, 4.0, 8.0, 15.0)  # total <= ~30s, fits judge timeout


class NoteInterpreter(abc.ABC):
    """Turns free-text operator notes into raw interpretation payloads.

    Implementations return *unvalidated* data. Trusting it is the job of the
    deterministic guardrail layer, never of this class.
    """

    @abc.abstractmethod
    def interpret(
        self,
        notes: Sequence[str],
        context: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Return one raw interpretation mapping per note, in order.

        ``context`` carries scenario facts a note may refer to proportionally
        (battery ratings), so wording like "half the battery capacity" can be
        converted into absolute kWh.
        """

    @abc.abstractmethod
    def repair(
        self,
        notes: Sequence[str],
        previous_payload: List[Dict[str, Any]],
        problems: List[str],
        context: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Re-attempt interpretation after guardrails rejected a payload."""


class OpenAICompatibleInterpreter(NoteInterpreter):
    """Calls a chat-completions endpoint that returns structured JSON."""

    def __init__(
        self,
        settings: Settings | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._client = client
        self._system_prompt = build_system_prompt(
            self._settings.hour_range_inclusive_end
        )

    # ---------------------------------------------------------------- public

    def interpret(
        self,
        notes: Sequence[str],
        context: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": build_user_prompt(notes, context)},
        ]
        return self._complete(messages, len(notes))

    def repair(
        self,
        notes: Sequence[str],
        previous_payload: List[Dict[str, Any]],
        problems: List[str],
        context: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": build_user_prompt(notes, context)},
            {
                "role": "assistant",
                "content": json.dumps({"interpretations": previous_payload}),
            },
            {
                "role": "user",
                "content": build_repair_prompt(problems, notes, context),
            },
        ]
        return self._complete(messages, len(notes))

    # --------------------------------------------------------------- private

    def _wants_simple_json(self) -> bool:
        """True when the provider should receive ``json_object`` directly."""
        base_url = (self._settings.llm_base_url or "").lower()
        return any(host in base_url for host in _SIMPLE_JSON_HOSTS)

    def _build_response_format(self, note_count: int) -> Dict[str, Any]:
        if self._wants_simple_json():
            return {"type": "json_object"}
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "operator_note_interpretations",
                "strict": True,
                "schema": build_response_schema(note_count),
            },
        }

    def _complete(
        self,
        messages: List[Dict[str, str]],
        note_count: int,
    ) -> List[Dict[str, Any]]:
        if not self._settings.llm_api_key:
            logger.error(
                "LLM_API_KEY is not configured; cannot interpret operator notes"
            )
            raise LLMUnavailableError(
                "The operator-note interpretation service is not configured."
            )

        simple_json = self._wants_simple_json()

        payload: Dict[str, Any] = {
            "model": self._settings.llm_model,
            "messages": messages,
            "temperature": self._settings.llm_temperature,
            "max_tokens": self._settings.llm_max_output_tokens,
            "response_format": self._build_response_format(note_count),
        }

        raw = self._post(payload, allow_schema_fallback=not simple_json)

        if raw is None:
            # Provider rejected json_schema on a host we did not pre-classify.
            # Retry once with the widely supported json_object mode.
            fallback = {**payload, "response_format": {"type": "json_object"}}
            raw = self._post(fallback, allow_schema_fallback=False)

        return _extract_interpretations(raw)

    def _post(
        self,
        payload: Dict[str, Any],
        allow_schema_fallback: bool = True,
    ) -> str | None:
        """POST to the provider with retries on transient failures.

        Returns ``None`` when the provider rejected the structured-output mode
        and a fallback should be attempted by the caller.

        Retries automatically on 408/425/429/5xx with exponential backoff.
        """
        url = f"{self._settings.llm_base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._settings.llm_api_key}",
            "Content-Type": "application/json",
        }

        response: httpx.Response | None = None
        attempts = [0.0] + list(_RETRY_DELAYS)

        for attempt_idx, delay in enumerate(attempts):
            if delay:
                logger.warning(
                    "LLM provider transient failure (HTTP %s); retrying in %.1fs "
                    "(attempt %d/%d)",
                    response.status_code if response is not None else "?",
                    delay,
                    attempt_idx,
                    len(_RETRY_DELAYS),
                )
                time.sleep(delay)

            try:
                if self._client is not None:
                    response = self._client.post(
                        url,
                        json=payload,
                        headers=headers,
                        timeout=self._settings.llm_timeout_seconds,
                    )
                else:
                    with httpx.Client(
                        timeout=self._settings.llm_timeout_seconds
                    ) as client:
                        response = client.post(url, json=payload, headers=headers)
            except httpx.TimeoutException as exc:
                logger.error(
                    "LLM request timed out after %ss: %s",
                    self._settings.llm_timeout_seconds,
                    exc,
                )
                if attempt_idx >= len(_RETRY_DELAYS):
                    raise LLMUnavailableError(
                        "The operator-note interpretation service timed out."
                    ) from exc
                continue
            except httpx.HTTPError as exc:
                logger.error("LLM transport error: %s", exc)
                if attempt_idx >= len(_RETRY_DELAYS):
                    raise LLMUnavailableError() from exc
                continue

            # --- Success path ---------------------------------------------
            if response.status_code < 400:
                break

            # --- Structured-output rejected: let caller fall back ----------
            if response.status_code == 400 and allow_schema_fallback:
                logger.warning(
                    "Provider rejected json_schema response_format; falling back"
                )
                return None

            # --- Transient failures: retry --------------------------------
            if (
                response.status_code in _RETRY_STATUSES
                and attempt_idx < len(_RETRY_DELAYS)
            ):
                continue

            # --- Permanent failure ----------------------------------------
            logger.error("LLM provider returned HTTP %s", response.status_code)
            raise LLMUnavailableError()

        if response is None or response.status_code >= 400:
            logger.error("LLM provider exhausted retries")
            raise LLMUnavailableError()

        try:
            body = response.json()
            return body["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            logger.error("Unexpected LLM envelope shape: %s", type(exc).__name__)
            raise LLMInterpretationError() from exc


def _extract_interpretations(content: str | None) -> List[Dict[str, Any]]:
    """Parse the model's message text into a list of interpretation mappings."""
    if not content or not content.strip():
        logger.error("LLM returned an empty message")
        raise LLMInterpretationError()

    cleaned = _FENCE_RE.sub("", content.strip())

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"[\[{].*[\]}]", cleaned, re.DOTALL)
        if not match:
            logger.error("LLM response contained no JSON payload")
            raise LLMInterpretationError()
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            logger.error("LLM response was not valid JSON")
            raise LLMInterpretationError() from exc

    if isinstance(parsed, dict):
        for key in ("interpretations", "results", "notes", "data"):
            if isinstance(parsed.get(key), list):
                parsed = parsed[key]
                break
        else:
            # A single interpretation object returned bare.
            parsed = [parsed] if "directive_type" in parsed else []

    if not isinstance(parsed, list):
        logger.error("LLM JSON payload was not a list of interpretations")
        raise LLMInterpretationError()

    items: List[Dict[str, Any]] = []
    for item in parsed:
        if not isinstance(item, dict):
            logger.error("LLM interpretation entry was not an object")
            raise LLMInterpretationError()
        items.append(item)
    return items


def build_interpreter(settings: Settings | None = None) -> NoteInterpreter:
    """Factory used by the API layer; honours ``LLM_PROVIDER``."""
    settings = settings or get_settings()
    provider = settings.llm_provider.strip().lower()
    if provider in {
        "openai_compatible",
        "openai",
        "groq",
        "together",
        "openrouter",
        "gemini",
        "",
    }:
        primary: NoteInterpreter = OpenAICompatibleInterpreter(settings)
        if settings.llm_fallback_to_rules:
            from app.llm.fallback import ResilientInterpreter

            logger.warning(
                "LLM_FALLBACK_TO_RULES is enabled: rule-based interpretation "
                "will be used if the provider is unreachable. This is a "
                "degraded mode."
            )
            return ResilientInterpreter(primary)
        return primary
    raise LLMUnavailableError(
        "The configured operator-note interpretation provider is not supported."
    )