"""Application configuration, loaded from environment variables.

Nothing here is hardcoded to a particular scenario or provider: the LLM
provider is addressed through an OpenAI-compatible base URL so that OpenAI,
Groq, Together, OpenRouter, Fireworks or a local vLLM server can all be used
by changing environment variables only.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()

HOURS_IN_DAY = 24

# Absolute tolerance used when checking the raw solver solution.
SOLVER_TOLERANCE = 1e-6
# Looser tolerance used when re-checking the rounded, client-facing schedule.
ROUNDED_TOLERANCE = 1e-2
# Decimal places used for every energy/cost figure in the response.
ROUND_DP = 3


def _get_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _get_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    """Runtime settings. Secrets are never logged or returned to clients."""

    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "openai_compatible"))
    llm_api_key: str = field(default_factory=lambda: os.getenv("LLM_API_KEY", ""))
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "gpt-4o-mini"))
    llm_base_url: str = field(
        default_factory=lambda: os.getenv("LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    )
    llm_timeout_seconds: float = field(default_factory=lambda: _get_float("LLM_TIMEOUT_SECONDS", 25.0))
    llm_temperature: float = field(default_factory=lambda: _get_float("LLM_TEMPERATURE", 0.0))
    llm_max_output_tokens: int = field(default_factory=lambda: _get_int("LLM_MAX_OUTPUT_TOKENS", 1200))

    # Hour-range convention for phrases like "from 1 PM to 3 PM".
    # False (default) -> half-open interval [13:00, 15:00) -> hours 13, 14.
    # True            -> inclusive of the end hour          -> hours 13, 14, 15.
    hour_range_inclusive_end: bool = field(
        default_factory=lambda: _get_bool("HOUR_RANGE_INCLUSIVE_END", False)
    )

    # Tie-breaking penalty per kWh of battery throughput. Keeps the optimizer
    # from cycling the battery pointlessly among equally-priced optima. It is
    # small enough never to change the cost-optimal grid schedule.
    battery_cycle_penalty: float = field(
        default_factory=lambda: _get_float("BATTERY_CYCLE_PENALTY", 1e-6)
    )

    # Degraded mode: if the LLM provider is unreachable, interpret notes with
    # deterministic rules instead of returning 503. Off by default -- the
    # challenge requires LLM interpretation, and rules cannot match a model on
    # paraphrase. Turn on only for availability during a live demo.
    llm_fallback_to_rules: bool = field(
        default_factory=lambda: _get_bool("LLM_FALLBACK_TO_RULES", False)
    )

    solver_time_limit_seconds: int = field(
        default_factory=lambda: _get_int("SOLVER_TIME_LIMIT_SECONDS", 20)
    )
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    port: int = field(default_factory=lambda: _get_int("PORT", 8000))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
