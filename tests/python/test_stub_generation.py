"""Acceptance tests for PyO3 → .pyi stub generation.

These tests gate Phase 1 of the contract source-of-truth foundation plan.
They verify that the build pipeline emits Python type stubs from PyO3
source, that those stubs contain the public engine surface, and that
PEP 561 markers ship alongside.

These tests assume `maturin develop` (or a release build) has run and
materialised the stubs into the installed package directory. They do
NOT trigger the build themselves — that is a build-system concern.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import tardigrade_db


def _package_dir() -> Path:
    """Return the directory where ``tardigrade_db`` is installed."""
    pkg_file = tardigrade_db.__file__
    if pkg_file is None:
        pytest.skip("tardigrade_db has no __file__; cannot locate stubs")
    return Path(pkg_file).resolve().parent


def _stub_files() -> list[Path]:
    """Return every .pyi file inside the installed package.

    pyo3-stub-gen in mixed-layout mode places stubs at
    ``tardigrade_db/_native/__init__.pyi`` (and would create deeper
    paths for submodules), so we glob recursively.
    """
    return sorted(_package_dir().rglob("*.pyi"))


def _parse_stub(path: Path) -> ast.Module:
    """Parse a .pyi file as a Python AST module."""
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _class_names(module: ast.Module) -> set[str]:
    return {node.name for node in ast.walk(module) if isinstance(node, ast.ClassDef)}


def _function_names(module: ast.Module) -> set[str]:
    return {node.name for node in ast.walk(module) if isinstance(node, ast.FunctionDef)}


def _methods_of_class(module: ast.Module, class_name: str) -> set[str]:
    for node in ast.walk(module):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return {
                child.name
                for child in node.body
                if isinstance(child, ast.FunctionDef)
            }
    return set()


# ─────────────────────────────────────────────────────────────────────────
# AT #1 — Stubs exist and cover the public surface
# ─────────────────────────────────────────────────────────────────────────


class TestStubsExistAndCoverPublicSurface:
    """The build emits .pyi files describing every PyO3-exported item."""

    def test_at_least_one_pyi_file_is_present(self) -> None:
        stubs = _stub_files()
        assert stubs, (
            f"No .pyi files found in {_package_dir()}. "
            "Run `maturin develop` (or build the wheel) to regenerate stubs."
        )

    def test_engine_class_is_declared_in_stubs(self) -> None:
        for stub in _stub_files():
            if "Engine" in _class_names(_parse_stub(stub)):
                return
        pytest.fail(
            f"Engine class not found in any .pyi file under {_package_dir()}. "
            "Check #[gen_stub_pyclass] annotation on Engine."
        )

    def test_read_result_class_is_declared_in_stubs(self) -> None:
        for stub in _stub_files():
            if "ReadResult" in _class_names(_parse_stub(stub)):
                return
        pytest.fail("ReadResult class not found in any .pyi file.")

    def test_checkpoint_repository_class_is_declared_in_stubs(self) -> None:
        for stub in _stub_files():
            if "CheckpointRepository" in _class_names(_parse_stub(stub)):
                return
        pytest.fail("CheckpointRepository class not found in any .pyi file.")

    def test_module_level_functions_are_declared_in_stubs(self) -> None:
        expected = {
            "find_chunk_boundary",
            "flat_to_paged",
            "paged_to_flat",
            "flat_to_paged_torch",
        }
        seen: set[str] = set()
        for stub in _stub_files():
            seen |= _function_names(_parse_stub(stub))
        missing = expected - seen
        assert not missing, (
            f"Module-level functions missing from stubs: {sorted(missing)}. "
            "Check #[gen_stub_pyfunction] annotations."
        )

    def test_core_engine_methods_appear_in_stubs(self) -> None:
        # Sentinel set — not exhaustive. If these are present, the
        # #[gen_stub_pymethods] block was wired correctly. Exhaustive
        # parity is enforced by the cross-surface parity test (Phase 2),
        # not here.
        core = {
            "mem_write_pack",
            "mem_read_pack",
            "snapshot",
            "restore_from",
            "list_owners",
            "flush",
        }
        for stub in _stub_files():
            methods = _methods_of_class(_parse_stub(stub), "Engine")
            if core.issubset(methods):
                return
        pytest.fail(
            f"Engine in stubs is missing core methods {sorted(core)}. "
            "Check #[gen_stub_pymethods] annotation on the Engine impl block."
        )


# ─────────────────────────────────────────────────────────────────────────
# AT #5 (partial) — PEP 561 marker ships with the package
# ─────────────────────────────────────────────────────────────────────────


class TestPep561Marker:
    """A `py.typed` marker tells type-checkers this package ships type info."""

    def test_py_typed_marker_is_present(self) -> None:
        marker = _package_dir() / "py.typed"
        assert marker.is_file(), (
            f"PEP 561 marker file not found at {marker}. "
            "Without it, type-checkers ignore the .pyi files entirely."
        )


# ─────────────────────────────────────────────────────────────────────────
# Guardrail — silent typing.Any falls are flagged for review
# ─────────────────────────────────────────────────────────────────────────


# Methods/functions whose return or argument types ARE legitimately
# typing.Any (or whose caller-passed PyAny cannot be statically typed
# without pulling in optional Python dependencies). Each entry must
# carry a reason — same discipline as the project's `// Reason:` rule.
ALLOWED_TYPING_ANY = {
    # Optional-dep types. flat_to_paged_torch's input is a torch.Tensor;
    # torch is an optional dependency, so tightening would require
    # importing torch in the generated stub.
    ("flat_to_paged_torch", "flat_kv"),
    # Polymorphic dispatch — accepts scalar OR list of scalars (with
    # an optional None on `owner`). Documented in the Rust source; a
    # tightened override would need int | list[int] (and
    # int | list[int] | None for owner). Tracked as a stub override
    # follow-up; currently the runtime rejects bad shapes with a clear
    # ValueError.
    ("Engine.mem_read_pack_batch", "k"),
    ("Engine.mem_read_pack_batch", "owner"),
    # Returns shaped Python dicts. Each has a well-known shape but
    # lacks a TypedDict declaration in stubs today. Tightening these
    # is tracked as a follow-up; the runtime contract is clear.
    ("CheckpointRepository.save", "return"),
    ("CheckpointRepository.list", "return"),
    ("CheckpointRepository.latest", "return"),
    ("Engine.snapshot", "return"),
    ("Engine.compact", "return"),
    ("Engine.status", "return"),
    ("Engine.maintenance_status", "return"),
    ("Engine.list_scheduled", "return"),
    ("Engine.list_packs_metadata", "return"),
    ("Engine.list_packs", "return"),
    ("Engine.load_pack_by_id", "return"),
    ("Engine.load_synapsis", "return"),
    ("Engine.mem_write_pack_with_auto_link", "return"),
    # Returns lists of ReadResult-shaped dicts (Py<PyAny>); the outer
    # container is typed but each leaf is a dict awaiting a TypedDict
    # override (same follow-up as above).
    ("Engine.mem_read_pack_batch", "return"),
    ("Engine.mem_read_pack", "return"),
    ("Engine.mem_read_multi_layer", "return"),
    ("Engine.mem_read_pack_with_trace_boost", "return"),
    ("Engine.mem_read_pack_with_trace_boost_and_follow", "return"),
    # Batched-write request envelope is a heterogeneous tuple
    # (owner, key, value, salience, parent_id?). Stub-gen sees the
    # PyObject and types it as Any. Tightening means defining a
    # TypedDict (or NamedTuple) and threading it through the Rust
    # extractor — tracked as a follow-up; runtime extractor rejects
    # malformed entries today.
    ("Engine.mem_write_batch", "requests"),
}


class TestNoSilentAny:
    """Every typing.Any in the stubs must be on the explicit allowlist.

    This is the project's `// Reason:` discipline applied to generated
    stubs. A silent fall to typing.Any is a contract regression.
    """

    def test_no_unexplained_typing_any_in_stubs(self) -> None:
        offenders: list[str] = []
        for stub in _stub_files():
            module = _parse_stub(stub)
            for node in ast.walk(module):
                if not isinstance(node, ast.FunctionDef):
                    continue
                # Determine the qualified name (Class.method or just function).
                qualname = _qualified_name(module, node)
                offenders.extend(_check_function_for_any(node, qualname))
        # Filter against allowlist.
        unexplained = [item for item in offenders if item not in ALLOWED_TYPING_ANY]
        if unexplained:
            formatted = "\n  ".join(f"{name} :: {pos}" for name, pos in unexplained)
            pytest.fail(
                "Unexplained typing.Any in generated stubs:\n  "
                f"{formatted}\n"
                "Either tighten the Rust return/argument type, add a "
                "#[gen_stub(override_type(...))] override, or add the "
                "entry to ALLOWED_TYPING_ANY with a reason."
            )


def _qualified_name(module: ast.Module, fn: ast.FunctionDef) -> str:
    for parent in ast.walk(module):
        if isinstance(parent, ast.ClassDef) and fn in parent.body:
            return f"{parent.name}.{fn.name}"
    return fn.name


def _is_any(annotation: ast.expr | None) -> bool:
    """Recursively detect ``typing.Any`` anywhere inside an annotation.

    ``tuple[Any, Any]``, ``list[Any]``, ``dict[str, Any]`` etc. all count
    — the silent fallback is just as much a contract gap inside a
    container as it is at the top level. Walks the AST so nested
    Subscripts (Subscript → Tuple/Name) are caught too.
    """
    if annotation is None:
        return False
    for node in ast.walk(annotation):
        if isinstance(node, ast.Name) and node.id == "Any":
            return True
        if isinstance(node, ast.Attribute) and node.attr == "Any":
            return True
    return False


def _check_function_for_any(
    fn: ast.FunctionDef, qualname: str
) -> list[tuple[str, str]]:
    findings: list[tuple[str, str]] = []
    if _is_any(fn.returns):
        findings.append((qualname, "return"))
    for arg in fn.args.args:
        if arg.arg == "self":
            continue
        if _is_any(arg.annotation):
            findings.append((qualname, arg.arg))
    return findings
