"""HTTP routes.

Handlers are declared with ``def`` rather than ``async def`` on purpose: both
the LLM call and the CBC solve are blocking, so FastAPI runs them in its
threadpool instead of stalling the event loop for every concurrent request.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, status

from app.llm.interpreter import NoteInterpreter, build_interpreter
from app.models.request import OptimizationRequest
from app.models.response import ErrorResponse, HealthResponse, OptimizationResponse
from app.services.energy_service import EnergyService

logger = logging.getLogger(__name__)

router = APIRouter()


def get_interpreter() -> NoteInterpreter:
    """Dependency seam; tests override this with a mock interpreter."""
    return build_interpreter()


def get_service(interpreter: NoteInterpreter = Depends(get_interpreter)) -> EnergyService:
    return EnergyService(interpreter)


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    tags=["operations"],
    summary="Liveness probe",
)
def health() -> HealthResponse:
    """Unauthenticated liveness check; performs no external calls."""
    return HealthResponse(status="ok")


@router.post(
    "/optimize-energy",
    response_model=OptimizationResponse,
    status_code=status.HTTP_200_OK,
    tags=["optimization"],
    summary="Produce a cost-optimal 24-hour campus energy schedule",
    responses={
        400: {"model": ErrorResponse, "description": "Malformed request body"},
        422: {"model": ErrorResponse, "description": "Invalid scenario, or no feasible schedule"},
        502: {"model": ErrorResponse, "description": "Operator notes could not be interpreted"},
        503: {"model": ErrorResponse, "description": "Interpretation service unavailable"},
    },
)
def optimize_energy(
    request: OptimizationRequest,
    service: EnergyService = Depends(get_service),
) -> OptimizationResponse:
    """Interpret operator notes, then solve the constrained energy schedule."""
    logger.info("Optimizing scenario %s", request.scenario_id)
    return service.optimize(request)
