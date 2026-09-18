"""pytest fixtures.

``client_factory`` builds a TestClient whose LLM dependency is replaced by a
mock, so the HTTP layer is exercised end to end without a provider.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence

import pytest
from fastapi.testclient import TestClient

from app.api.routes import get_interpreter
from app.llm.interpreter import NoteInterpreter
from app.main import create_app
from app.models.request import OptimizationRequest
from tests.helpers import MockNoteInterpreter, make_scenario, no_op_payload


@pytest.fixture
def scenario() -> Dict[str, Any]:
    return make_scenario()


@pytest.fixture
def battery():
    return OptimizationRequest(**make_scenario()).battery


@pytest.fixture
def client_factory() -> Callable[[Optional[Sequence[Dict[str, Any]]]], TestClient]:
    """Return a factory that builds a TestClient backed by a given interpreter."""
    created: List[TestClient] = []

    def build(
        payload: Optional[Sequence[Dict[str, Any]]] = None,
        interpreter: Optional[NoteInterpreter] = None,
    ) -> TestClient:
        app = create_app()
        stub = interpreter or MockNoteInterpreter(payload if payload is not None else no_op_payload(1))
        app.dependency_overrides[get_interpreter] = lambda: stub
        client = TestClient(app, raise_server_exceptions=False)
        client.interpreter = stub  # type: ignore[attr-defined]
        created.append(client)
        return client

    yield build

    for client in created:
        client.close()


@pytest.fixture
def client(client_factory) -> TestClient:
    """A client whose interpreter marks every note as no_op."""
    return client_factory(no_op_payload(1))
