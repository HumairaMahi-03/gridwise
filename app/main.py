"""Application entrypoint.

Every error path is funnelled through a handler that returns a
``{"detail": ...}`` envelope. Stack traces and provider responses are logged,
never serialised to the client.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.routes import router
from app.config import get_settings
from app.exceptions import GridwiseError

logger = logging.getLogger("gridwise")


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)

    app = FastAPI(
        title="Gridwise Campus Energy Optimizer",
        version="1.0.0",
        description=(
            "Interprets natural-language operator notes with an LLM, validates the "
            "resulting directives deterministically, and solves a cost-optimal "
            "24-hour campus energy schedule with linear programming."
        ),
    )

    app.include_router(router)

    @app.exception_handler(GridwiseError)
    def handle_domain_error(_: Request, exc: GridwiseError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.client_message})

    @app.exception_handler(RequestValidationError)
    def handle_validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        # FastAPI's default 422 body is already client-safe and points at the
        # offending field, which is genuinely useful for scenario debugging.
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"detail": exc.errors()},
        )

    @app.exception_handler(StarletteHTTPException)
    def handle_http_error(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

    @app.exception_handler(Exception)
    def handle_unexpected_error(_: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled error: %s", type(exc).__name__)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "An unexpected error occurred."},
        )

    return app


app = create_app()


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=get_settings().port)
