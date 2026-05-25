"""Cross-surface parity test — PyO3 engine ↔ HTTP bridge.

Gates Phase 2 of the contract source-of-truth foundation. Asserts
that every engine method the HTTP routes depend on actually exists
on :class:`tardigrade_db.Engine`, and every Pydantic model the routes
bind actually exists in :mod:`tardigrade_http.models`. Drift on
either side (rename, removal) fails the test with a structured
message naming the offender.

Why this shape (and not field-level parity): the HTTP bridge is a
*translation layer*, not a pass-through. ``POST /mem/store`` takes
``fact_text: str`` and turns it into a retrieval key via the
engine's KV function before calling ``mem_write_pack``. Asserting
field-name parity between PyO3 kwargs and Pydantic fields would
fight the actual architecture. The right invariant for today is
existence: the registry names what the bridge needs, and this test
confirms each name still resolves. When the eventual unification
plan replaces hand-written models with PyO3-attribute codegen,
field-level parity becomes a separate (and tighter) check.
"""

from __future__ import annotations

from typing import Final

import pytest

import tardigrade_db
from tardigrade_http import models as http_models
from tardigrade_http._http_exposed import HTTP_EXPOSED, RouteContract


_REGISTRY_NOT_EMPTY: Final[str] = (
    "_http_exposed.HTTP_EXPOSED is empty — the registry must list every "
    "HTTP route or the parity gate enforces nothing."
)


def _models_named(contract) -> list[str]:
    """Flatten a contract's request, response, and nested model names."""
    names: list[str] = []
    if contract.request_model is not None:
        names.append(contract.request_model)
    if contract.response_model is not None:
        names.append(contract.response_model)
    names.extend(contract.nested_models)
    return names


class TestRegistryIsPopulated:
    """The registry itself is non-empty."""

    def test_at_least_one_route_is_registered(self) -> None:
        assert HTTP_EXPOSED, _REGISTRY_NOT_EMPTY


class TestEngineMethodsExist:
    """Every engine method named in the registry resolves on ``Engine``.

    Renaming a method in the Rust source without updating the
    registry would silently break the HTTP route at runtime. This
    test catches the drift at build time.
    """

    @pytest.mark.parametrize(
        ("route", "method"),
        [
            (route, method)
            for route, contract in HTTP_EXPOSED.items()
            for method in contract.engine_methods
        ],
    )
    def test_engine_method_is_callable(self, route: str, method: str) -> None:
        attr = getattr(tardigrade_db.Engine, method, None)
        assert attr is not None, (
            f"Route {route!r} declares dependency on "
            f"`tardigrade_db.Engine.{method}` but no such attribute "
            f"exists on the class. Either restore the method in "
            f"`crates/tdb-python/src/lib.rs` or update "
            f"`python/tardigrade_http/_http_exposed.py`."
        )
        assert callable(attr), (
            f"`tardigrade_db.Engine.{method}` exists but is not "
            f"callable (got {type(attr).__name__}). Routes need a "
            f"method to dispatch to."
        )


class TestPydanticModelsExist:
    """Every Pydantic model named in the registry exists in ``models``.

    Renaming or removing a model in :mod:`tardigrade_http.models`
    without updating the registry would let the HTTP bridge import
    successfully but break OpenAPI schema generation downstream. This
    test catches the drift before that.
    """

    @pytest.mark.parametrize(
        ("route", "model_name"),
        [
            (route, name)
            for route, contract in HTTP_EXPOSED.items()
            for name in _models_named(contract)
        ],
    )
    def test_pydantic_model_resolves(self, route: str, model_name: str) -> None:
        model = getattr(http_models, model_name, None)
        assert model is not None, (
            f"Route {route!r} declares Pydantic model "
            f"`tardigrade_http.models.{model_name}` but no such "
            f"class exists. Either restore the model in "
            f"`python/tardigrade_http/models.py` or update "
            f"`python/tardigrade_http/_http_exposed.py`."
        )


class TestParityGateActuallyCatchesDrift:
    """Meta-test: the parity gate above must FAIL on deliberate drift.

    A gate that never fails isn't a gate. These tests construct a
    synthetic drifted contract and verify the same assertion logic
    used by the parity tests above rejects it. Gives confidence the
    real gate is load-bearing without manually breaking the real
    registry.
    """

    def test_missing_engine_method_is_detected(self) -> None:
        drifted = RouteContract(
            engine_methods=("this_method_does_not_exist_on_engine",),
            request_model="StoreRequest",
            response_model="StoreResponse",
        )
        method = drifted.engine_methods[0]
        attr = getattr(tardigrade_db.Engine, method, None)
        assert attr is None, (
            "Sentinel name happens to exist on Engine — pick a more "
            "obviously-fake name for this meta-test."
        )

    def test_missing_pydantic_model_is_detected(self) -> None:
        drifted = RouteContract(
            engine_methods=("mem_write_pack",),
            request_model="ThisModelDoesNotExistInModels",
            response_model="StoreResponse",
        )
        assert drifted.request_model is not None
        model = getattr(http_models, drifted.request_model, None)
        assert model is None, (
            "Sentinel name happens to exist in models.py — pick a "
            "more obviously-fake name for this meta-test."
        )
