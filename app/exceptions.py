"""Domain exceptions.

Each exception carries a client-safe message and an HTTP status code. Internal
detail (provider responses, stack traces, credentials) stays in the logs and is
never attached to these objects.
"""

from __future__ import annotations


class GridwiseError(Exception):
    """Base class for all application errors."""

    status_code: int = 500
    client_message: str = "An unexpected error occurred."

    def __init__(self, client_message: str | None = None) -> None:
        if client_message is not None:
            self.client_message = client_message
        super().__init__(self.client_message)


class LLMUnavailableError(GridwiseError):
    """The LLM provider could not be reached, timed out, or is misconfigured."""

    status_code = 503
    client_message = "The operator-note interpretation service is currently unavailable."


class LLMInterpretationError(GridwiseError):
    """The LLM responded, but no trustworthy interpretation could be derived."""

    status_code = 502
    client_message = "Operator notes could not be interpreted reliably."


class InfeasibleScheduleError(GridwiseError):
    """The combined constraints admit no valid 24-hour schedule."""

    status_code = 422
    client_message = "No feasible energy schedule exists for the supplied constraints."


class OptimizerError(GridwiseError):
    """The solver failed for a reason other than infeasibility."""

    status_code = 500
    client_message = "The energy optimizer failed to produce a schedule."


class ScheduleValidationError(GridwiseError):
    """The produced schedule failed independent post-solve validation."""

    status_code = 500
    client_message = "The generated energy schedule failed internal validation and was rejected."

    def __init__(self, violations: list[str] | None = None) -> None:
        self.violations = violations or []
        super().__init__(self.client_message)
