"""Structural verification for `_check_stage2_quality_resume`'s call site.

Mirrors a standard learned the hard way on this same class of guard: a test that calls the guard
function directly (`data_loader_test.py`'s `test_resume_with_a_schedule_raises` and its siblings)
stays GREEN if the guard's call site inside `create_torch_data_loader` is ever deleted, and
`data_loader_cfg_stage2_test.py`'s `test_a_resume_is_refused_through_the_real_loader` -- which
drives the real loader end to end -- catches a DELETED call site but cannot distinguish a healthy
one from a call site that is duplicated (present twice, once on a branch nothing reaches) or moved
to run AFTER the work it is meant to gate (`wrap_presentations`): both produce the same raised
exception on the happy path that test drives. An `any(ast.Raise)` check over the guard's body is
not enough either -- it does not catch a `raise` downgraded to `logging.warning` sitting beside an
untaken `raise` on some other branch.

This file inspects the SOURCE via `ast` instead of executing it, and checks four things no
exception-based test can: the guard raises (never merely warns), it is called from
`create_torch_data_loader` EXACTLY once, and that one call textually precedes the call to
`wrap_presentations` it exists to gate.

CORE tier: stdlib only (`ast`, `inspect`). Importing `openpi.training.data_loader` to read its own
source still pulls in jax/torch at module scope, so this file runs in the online venv like its
siblings -- it is not core-tier by residence, only by what it inspects.
"""

from __future__ import annotations

import ast
import inspect

from openpi.training import data_loader as _data_loader

GUARD = "_check_stage2_quality_resume"
HOST_FUNCTION = "create_torch_data_loader"
GATED_CALL = "wrap_presentations"


def _module_ast() -> ast.Module:
    return ast.parse(inspect.getsource(_data_loader))


def _find_function(tree: ast.AST, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name!r} is not defined in data_loader.py")


def _calls_named(tree: ast.AST, name: str) -> list[ast.Call]:
    """Every `ast.Call` anywhere under `tree` whose callee is literally `name` -- as a bare name
    (`name(...)`) or an attribute access ending in it (`module.name(...)`)."""
    calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id == name:
            calls.append(node)
        elif isinstance(func, ast.Attribute) and func.attr == name:
            calls.append(node)
    return calls


def test_the_guard_function_exists():
    guard = _find_function(_module_ast(), GUARD)
    assert isinstance(guard, ast.FunctionDef)


def test_the_guard_raises():
    """A guard that never raises anything cannot refuse a resume no matter where it is called."""
    guard = _find_function(_module_ast(), GUARD)
    raises = [n for n in ast.walk(guard) if isinstance(n, ast.Raise)]
    assert raises, f"{GUARD} contains no `raise` statement"


def test_the_guard_has_no_warning_only_path():
    """Catches a `raise` downgraded to `logging.warning`/`warnings.warn`. An `any(ast.Raise)`
    check alone would miss a warning call sitting alongside an untaken `raise`, or a future edit
    that swaps the raise for a warning without touching the surrounding `if`."""
    guard = _find_function(_module_ast(), GUARD)
    for node in ast.walk(guard):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        callee = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        assert callee not in ("warning", "warn"), (
            f"{GUARD} calls {callee!r} -- a resume must be REFUSED, not merely logged"
        )


def test_the_guard_is_called_exactly_once_inside_the_loader():
    """Zero calls is the orphaned-guard failure the real-loader test already catches by raising
    nothing when it should. More than one call is not extra coverage -- it usually means a
    duplicate sitting on a branch the real construction path never reaches, which reads as
    'guarded' while doing nothing there."""
    host = _find_function(_module_ast(), HOST_FUNCTION)
    calls = _calls_named(host, GUARD)
    assert len(calls) == 1, (
        f"expected exactly one call to {GUARD} inside {HOST_FUNCTION}, found {len(calls)}"
    )


def test_the_guard_is_called_before_the_work_it_gates():
    """`wrap_presentations` builds the presentation-keyed dataset and sampler pair -- the
    'expensive work' a late guard would fail to prevent starting. The guard must run first, the
    same placement `_check_quality_resume` uses ahead of stage 1's artifact read."""
    host = _find_function(_module_ast(), HOST_FUNCTION)
    guard_calls = _calls_named(host, GUARD)
    gated_calls = _calls_named(host, GATED_CALL)
    assert guard_calls, f"{GUARD} is not called inside {HOST_FUNCTION}"
    assert gated_calls, f"{GATED_CALL} is not called inside {HOST_FUNCTION}"
    assert guard_calls[0].lineno < gated_calls[0].lineno, (
        f"{GUARD} (line {guard_calls[0].lineno}) must be called BEFORE "
        f"{GATED_CALL} (line {gated_calls[0].lineno}), not after"
    )
