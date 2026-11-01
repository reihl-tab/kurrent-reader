"""Guard tests for the versioned Colab runner script (issue #5)."""

from __future__ import annotations

import ast
from pathlib import Path

SCRIPTS = [
    Path(__file__).resolve().parents[1] / "scripts" / "colab_train.py",
    Path(__file__).resolve().parents[1] / "scripts" / "colab_train_v2.py",
    Path(__file__).resolve().parents[1] / "scripts" / "colab_train_v4.py",
    Path(__file__).resolve().parents[1] / "scripts" / "kaggle_kernel_v4.py",
    Path(__file__).resolve().parents[1] / "scripts" / "kaggle_kernel_v8.py",
    Path(__file__).resolve().parents[1] / "scripts" / "kaggle_kernel_v11.py",
    Path(__file__).resolve().parents[1] / "scripts" / "kaggle_kernel_v12.py",
]


def test_colab_runner_compiles() -> None:
    for script in SCRIPTS:
        compile(script.read_text(encoding="utf-8"), str(script), "exec")


def test_stdout_reconfigure_is_guarded() -> None:
    """sys.stdout.reconfigure must sit inside try/except AttributeError.

    Jupyter/Colab replaces sys.stdout with ipykernel OutStream, which has no
    reconfigure(); an unguarded call crashes the run at startup (issue #5).
    """
    for script in SCRIPTS:
        tree = ast.parse(script.read_text(encoding="utf-8"))
        assert _has_guard(tree), (
            f"{script.name}: no try/except AttributeError guard around sys.stdout.reconfigure"
        )


def _has_guard(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        if not any(isinstance(h.type, ast.Name) and h.type.id == "AttributeError" for h in node.handlers):
            continue
        for inner in ast.walk(node):
            if (
                isinstance(inner, ast.Call)
                and isinstance(inner.func, ast.Attribute)
                and inner.func.attr == "reconfigure"
            ):
                return True
    return False


V4_RUNNERS = [
    Path(__file__).resolve().parents[1] / "scripts" / "colab_train_v4.py",
    Path(__file__).resolve().parents[1] / "scripts" / "kaggle_kernel_v4.py",
    Path(__file__).resolve().parents[1] / "scripts" / "kaggle_kernel_v8.py",
    Path(__file__).resolve().parents[1] / "scripts" / "kaggle_kernel_v11.py",
    Path(__file__).resolve().parents[1] / "scripts" / "kaggle_kernel_v12.py",
]


def _load_fn(path: Path, name: str, ns: dict):
    """Exec a single top-level function out of a runner script into ns."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            mod = ast.Module(body=[node], type_ignores=[])
            exec(compile(mod, str(path), "exec"), ns)
            return ns[name]
    raise AssertionError(f"{path.name}: function {name} not found")


def test_combined_score_excludes_inf_from_divisor() -> None:
    """inf entries must not inflate the divisor (spurious NEW BEST, issue #24).

    Applies to the v4-lineage combined_score; kernel v8's frozen variant
    returns the names CER directly and has no divisor.
    """
    for script in V4_RUNNERS[:2]:
        ns = {"score12": lambda r: r["s"]}
        fn = _load_fn(script, "combined_score", ns)
        assert fn({"s": float("inf")}, {"n": 1, "cer": 0.5}) == 0.5
        assert abs(fn({"s": 0.2}, {"n": 1, "cer": 0.4}) - 0.3) < 1e-9
        assert fn({"s": float("inf")}, {}) == float("inf")


def test_held_out_leak_assert_normalizes_idnr() -> None:
    """The row-level held-out check must catch int or str IdNr (issue #24)."""
    for script in V4_RUNNERS:
        src = script.read_text(encoding="utf-8")
        assert '_held_idnrs = {"33739", "35238"}' in src, f"{script.name}: held-out set not normalized"
        assert 'str(s.get("IdNr"))' in src, f"{script.name}: IdNr not normalized to str in leak check"


def test_evaluate12_rejects_traversal() -> None:
    """evaluate12 must resolve eval file paths under DATA only (issue #24)."""
    import tempfile

    for script in V4_RUNNERS:
        ns: dict = {}
        fn = _load_fn(script, "_safe_data_path", ns)
        src = script.read_text(encoding="utf-8")
        assert '_safe_data_path(DATA, r["file"])' in src, f"{script.name}: evaluate12 not using the guard"
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            assert fn(root, "a/b.png") == (root / "a/b.png").resolve()
            for bad in ("../escape.png", "a/../../escape.png", "/etc/passwd"):
                try:
                    fn(root, bad)
                except ValueError:
                    pass
                else:
                    raise AssertionError(f"{script.name}: traversal accepted: {bad}")


def _load_assignments(path: Path, names: set, ns: dict) -> dict:
    """Exec top-level assignments to the given names, in source order, into ns."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = [
        n
        for n in tree.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id in names for t in n.targets)
    ]
    found = {t.id for n in nodes for t in n.targets if isinstance(t, ast.Name)}
    missing = names - found
    if missing:
        raise AssertionError(f"{path.name}: assignments not found: {sorted(missing)}")
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), ns)
    return ns


def test_held_out_leak_check_catches_int_and_str() -> None:
    """Behaviorally pin the row-level held-out check (issue #24).

    Execs the runner's own _held_idnrs/_leaked lines against synthetic
    samples: the held-out IdNrs must be caught whether the manifest stores
    them as int or str, and clean samples must pass.
    """
    for script in V4_RUNNERS:
        for idnr in (33739, "33739", 35238, "35238"):
            ns = {"samples": [{"IdNr": idnr}, {"IdNr": 1}, {"IdNr": "2"}, {"IdNr": None}]}
            _load_assignments(script, {"_held_idnrs", "_leaked"}, ns)
            assert ns["_leaked"] == [str(idnr)], f"{script.name}: missed held-out IdNr {idnr!r}"
        ns = {"samples": [{"IdNr": 1}, {"IdNr": "2"}, {"IdNr": None}]}
        _load_assignments(script, {"_held_idnrs", "_leaked"}, ns)
        assert ns["_leaked"] == [], f"{script.name}: false positive on clean samples"


V11 = Path(__file__).resolve().parents[1] / "scripts" / "kaggle_kernel_v11.py"


def _load_heartbeat(ns: dict) -> dict:
    """Exec the heartbeat block (constants + 3 functions) out of kernel v11."""
    import json as _json
    import os as _os
    import time as _time

    tree = ast.parse(V11.read_text(encoding="utf-8"))
    want_fns = {"progress_payload", "progress_push", "_json_safe"}
    want_assigns = {"PROGRESS_DATASET", "PROGRESS_ENABLED", "_progress_state"}
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in want_fns:
            nodes.append(node)
        elif isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id in want_assigns for t in node.targets
        ):
            nodes.append(node)
    names = {n.name for n in nodes if isinstance(n, ast.FunctionDef)} | {
        t.id for n in nodes if isinstance(n, ast.Assign) for t in n.targets if isinstance(t, ast.Name)
    }
    assert want_fns | want_assigns <= names, f"missing heartbeat pieces: {want_fns | want_assigns - names}"
    ns.setdefault("json", _json)
    ns.setdefault("os", _os)
    ns.setdefault("time", _time)
    ns.setdefault("T0", 0.0)
    ns.setdefault("Path", Path)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(V11), "exec"), ns)
    return ns


class _FakeApi:
    """Captures call args AND file contents during the call (the heartbeat
    cleans up its temp dir right after, per F3)."""

    def __init__(self) -> None:
        self.calls = []

    def dataset_create_version(self, folder, version_notes, dir_mode=None):
        self.calls.append(
            (
                folder,
                version_notes,
                dir_mode,
                (Path(folder) / "progress.json").read_text(encoding="utf-8"),
                (Path(folder) / "dataset-metadata.json").read_text(encoding="utf-8"),
            )
        )


def test_progress_payload_shape() -> None:
    ns = _load_heartbeat({})
    p = ns["progress_payload"]("step500", step=500, combined=0.5)
    assert p["stage"] == "step500"
    assert p["step"] == 500
    assert p["combined"] == 0.5
    assert isinstance(p["elapsed_sec"], float)


def test_progress_push_success_writes_progress_json() -> None:
    fake = _FakeApi()
    ns = _load_heartbeat({})
    ns["_progress_api"] = lambda: fake
    ok = ns["progress_push"]({"stage": "boot", "train_samples": 42})
    assert ok is True
    assert ns["_progress_state"]["pushes"] == 1
    folder, notes, dir_mode, progress_text, meta_text = fake.calls[0]
    assert notes == "boot" and dir_mode == "skip"
    import json as _json

    assert _json.loads(progress_text) == {"stage": "boot", "train_samples": 42}
    meta = _json.loads(meta_text)
    assert meta["id"] == ns["PROGRESS_DATASET"] == "reihlinstinct/kurrent-run-progress"
    assert not Path(folder).exists(), "temp dir must be cleaned up after the push (F3)"


def test_progress_push_never_raises() -> None:
    def _boom():
        raise RuntimeError("no auth in this environment")

    ns = _load_heartbeat({})
    ns["_progress_api"] = _boom
    ok = ns["progress_push"]({"stage": "boot"})
    assert ok is False
    assert ns["_progress_state"]["last_error"] == "RuntimeError", "log the exception type only, not the message (F4)"


def test_progress_push_disabled_skips_api() -> None:
    fake = _FakeApi()
    ns = _load_heartbeat({})
    ns["PROGRESS_ENABLED"] = False
    ns["_progress_api"] = lambda: fake
    assert ns["progress_push"]({"stage": "boot"}) is False
    assert fake.calls == []


def test_heartbeat_hooked_at_boot_eval_and_end() -> None:
    src = V11.read_text(encoding="utf-8")
    assert src.count("def progress_push") == 1
    assert 'progress_push(progress_payload("boot"' in src, "no boot heartbeat after the data-stage asserts"
    assert 'f"step{state.global_step}", step=state.global_step' in src, "no per-eval heartbeat in KeepBestCallback"
    assert "eval12=r" not in src and "eval_names=rn" not in src, "per-eval payload must carry aggregates only, no per-row data (F1)"
    assert 'names_cer=rn.get("cer")' in src, "per-eval payload must use aggregate fields (F1)"
    assert "missing_pages=len(_missing)" in src, "boot payload must carry counts, not filename lists (F1)"
    assert 'progress_push(progress_payload("train-done"' in src, "no final heartbeat"
    # definition must precede every call site
    def_line = src.index("def progress_push")
    assert def_line < src.index('progress_push(progress_payload("boot"')


def test_progress_push_sanitizes_non_finite() -> None:
    """F2: inf/nan must become null in progress.json (json.dumps would
    otherwise emit bare Infinity, which strict JSON parsers reject)."""
    fake = _FakeApi()
    ns = _load_heartbeat({})
    ns["_progress_api"] = lambda: fake
    ok = ns["progress_push"]({"stage": "step500", "combined": float("inf"), "nested": {"x": float("nan")}, "ok": 1.5})
    assert ok is True
    import json as _json

    payload = _json.loads(fake.calls[0][3])  # strict json.loads rejects bare Infinity
    assert payload["combined"] is None
    assert payload["nested"] == {"x": None}
    assert payload["ok"] == 1.5


def test_progress_push_survives_mkdtemp_failure() -> None:
    """F5: a mkdtemp failure must be swallowed like any other push error."""
    import tempfile as _tempfile

    ns = _load_heartbeat({})
    ns["_progress_api"] = lambda: _FakeApi()
    real_mkdtemp = _tempfile.mkdtemp
    _tempfile.mkdtemp = lambda *a, **k: (_ for _ in ()).throw(OSError("disk full"))
    try:
        ok = ns["progress_push"]({"stage": "boot"})
    finally:
        _tempfile.mkdtemp = real_mkdtemp
    assert ok is False
    assert ns["_progress_state"]["last_error"] == "OSError"


V12 = Path(__file__).resolve().parents[1] / "scripts" / "kaggle_kernel_v12.py"

_V12_DOCSTRING_ADDITIONS = {
    '"""Kaggle GPU run v12: v11 pipeline, single-lever recipe change (issue #32).',
    "v12 = v11 EXACT except REPEAT 40 -> 4. v10 showed the x40 oversample drives",
    "perfect memorization (seeds_fit CER 0.0) while held-out names got WORSE than",
    "the untrained base (0.9067 -> best 1.2173) and the model recites training-set",
    "names on held-out crops. x4 keeps target gradient signal (836 target samples",
    "per epoch, 1.3% of mix) with 10x less repetition pressure. Everything else",
    "frozen: 2 epochs, from base, LR 5e-6, batch 8, eval-every-500, keep-best,",
    "frozen metrics, heartbeat.",
    "Original v11 header follows.",
    "Kaggle GPU run v11: v10 pipeline + run-progress heartbeat (issue #30).",
}


def test_v12_single_lever_repeat_only() -> None:
    """v12 (issue #32) is v11 with ONE lever changed: REPEAT 40 -> 4.

    Pins the recipe change so a later edit cannot silently move a second
    lever: v12's lines must be v11's lines plus only the new docstring
    lines, with 'REPEAT = 40' replaced by the v12 REPEAT assignment.
    """
    v11 = V11.read_text(encoding="utf-8").splitlines()
    v12 = V12.read_text(encoding="utf-8").splitlines()
    ns: dict = {}
    _load_assignments(V12, {"REPEAT"}, ns)
    assert ns["REPEAT"] == 4, f"v12 REPEAT must be 4, got {ns['REPEAT']}"
    only12 = set(v12) - set(v11)
    assert only12 <= _V12_DOCSTRING_ADDITIONS | {next(l for l in v12 if l.startswith("REPEAT = 4"))}, (
        f"v12 adds lines beyond the docstring + REPEAT change: {sorted(only12 - _V12_DOCSTRING_ADDITIONS)}"
    )
    only11 = set(v11) - set(v12)
    assert only11 == {"REPEAT = 40", '"""Kaggle GPU run v11: v10 pipeline + run-progress heartbeat (issue #30).'}, (
        f"v12 dropped v11 lines other than REPEAT and the re-homed docstring opener: {sorted(only11)}"
    )
    assert "issue #32" in V12.read_text(encoding="utf-8")
